import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_norm = RMSNorm(head_dim, q_norm, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, k_norm, rms_norm_eps)

        self.rms_norm_eps = rms_norm_eps

        self.scale = head_dim ** -0.5
        self.rope = RoPE(dims=head_dim, seq_len=max_seq_len,base=theta,traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        original_dtype = x.dtype

        q = linear(x, self.wq)
        k = linear(x, self.wk)
        v = linear(x, self.wv)

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, slice(0, L))
        
        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        v = v.astype(mx.float32)

        out = scaled_dot_product_attention_grouped(
            q, k, v, self.scale, mask=mask
        )
        out = out.astype(original_dtype)

        out = out.swapaxes(1, 2)
        out = out.reshape(
            B, L, self.num_heads * self.head_dim
        )

        out = linear(out, self.wo)
        return out





class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        original_dtype = x.dtype
        x = x.astype(mx.float32)
        gate = linear(x, self.w_gate)
        up = linear(x, self.w_up) 

        out = silu(gate)
        out = out * up
        out = linear(out, self.w_down) 
        out = out.astype(original_dtype)
        return out


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.input_layernorm = RMSNorm(
            hidden_size,
            w_input_layernorm,
            eps=rms_norm_eps
        ) 

        self.self_attn = Qwen3MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            max_seq_len,
            theta,
            rms_norm_eps
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            w_post_attention_layernorm,
            rms_norm_eps
        )

        self.mlp = Qwen3MLP(
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down
        )


    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        atten_input = self.input_layernorm(x)
        atten_output = self.self_attn(atten_input, mask=mask)
        h = x + atten_output
        mlp_input = self.post_attention_layernorm(h)
        mlp_output = self.mlp(mlp_input)
        out = h + mlp_output
        return out



class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        args = mlx_model.args
        model = mlx_model.model

        self.layers = []

        for src_layer in model.layers:
            block = Qwen3TransformerBlock(
                num_attention_heads=args.num_attention_heads,
                num_kv_heads=args.num_key_value_heads,
                hidden_size=args.hidden_size,
                head_dim=args.head_dim,
                intermediate_size=args.intermediate_size,
                rms_norm_eps=args.rms_norm_eps,
                wq=dequantize_linear(
                    src_layer.self_attn.q_proj
                ).astype(mx.bfloat16),
                wk=dequantize_linear(
                    src_layer.self_attn.k_proj
                ).astype(mx.bfloat16),

                wv=dequantize_linear(
                    src_layer.self_attn.v_proj
                ).astype(mx.bfloat16),

                wo=dequantize_linear(
                    src_layer.self_attn.o_proj
                ).astype(mx.bfloat16),

                # Q/K RMSNorm
                q_norm=src_layer.self_attn.q_norm.weight.astype(
                    mx.bfloat16
                ),
                 k_norm=src_layer.self_attn.k_norm.weight.astype(
                    mx.bfloat16
                ),

                # MLP
                w_gate=dequantize_linear(
                    src_layer.mlp.gate_proj
                ).astype(mx.bfloat16),

                w_up=dequantize_linear(
                    src_layer.mlp.up_proj
                ).astype(mx.bfloat16),

                w_down=dequantize_linear(
                    src_layer.mlp.down_proj
                ).astype(mx.bfloat16),

                # Block RMSNorm
                w_input_layernorm=(
                    src_layer.input_layernorm.weight.astype(
                        mx.bfloat16
                    )
                ),

                w_post_attention_layernorm=(
                    src_layer.post_attention_layernorm.weight.astype(
                        mx.bfloat16
                    )
                ),

                max_seq_len=args.max_position_embeddings,
                theta=args.rope_theta,
            )
            self.layers.append(block)
        embedding_weight = dequantize_linear(
                model.embed_tokens
            ).astype(mx.bfloat16)

        self.embed_tokens = Embedding(
            vocab_size=args.vocab_size,
            embedding_dim=args.hidden_size,
            weight=embedding_weight,
        )
        self.norm = RMSNorm(
            args.hidden_size,
            model.norm.weight.astype(mx.bfloat16),
            eps=args.rms_norm_eps,
        )
        self.tie_word_embeddings = args.tie_word_embeddings


    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        x = self.embed_tokens(inputs)
        for layer in self.layers:
            x = layer(
                x,
                mask="causal",
            )
        x = self.norm(x)
        if self.tie_word_embeddings:
            x = self.embed_tokens.as_linear(x)
        else:
            x = linear(x, self.lm_head)
        return x