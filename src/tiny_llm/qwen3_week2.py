from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mlx.core as mx

from tiny_llm.attention import scaled_dot_product_attention_grouped
from tiny_llm.basics import linear, silu
from tiny_llm.layer_norm import RMSNorm
from tiny_llm.positional_encoding import RoPE

from .embedding import Embedding  # noqa: F401 - learner checkpoint dependency
from .kv_cache import TinyKvCache
from .quantize import QuantizedWeights, dequantize_linear  # noqa: F401
from .week2_kernels import (
    FastRMSNorm,  # noqa: F401 - learner checkpoint dependency
    FastRoPE,  # noqa: F401 - learner checkpoint dependency
    scaled_dot_product_attention,  # noqa: F401 - learner checkpoint dependency
    swiglu,  # noqa: F401 - learner checkpoint dependency
)


@dataclass(frozen=True)
class Week2CheckpointFeatures:
    bounded_kv_capacity: bool = False
    quantized_weights: bool = False
    fast_rms_norm: bool = False
    fast_rope: bool = False
    fast_swiglu: bool = False
    simdgroup_matmul: bool = False
    tiled_prefill_attention: bool = False


WEEK2_CHECKPOINT_FEATURES = MappingProxyType(
    {
        "kv-cache": Week2CheckpointFeatures(),
        "capacity-cache": Week2CheckpointFeatures(bounded_kv_capacity=True),
        "quantized-matvec": Week2CheckpointFeatures(
            bounded_kv_capacity=True, quantized_weights=True
        ),
        "simd-matmul": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
        ),
        "rmsnorm": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
        ),
        "rope": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
        ),
        "swiglu": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
        ),
        "tiled-prefill": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
        ),
        "selected": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
        ),
    }
)
WEEK2_CHECKPOINTS = tuple(WEEK2_CHECKPOINT_FEATURES)


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_tiled_prefill_attention: bool = False,
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

        # Feature flags, kept as attributes so benchmarks/profilers can replay
        # this layer's operators without re-deriving them from the checkpoint.
        self.use_fast_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        self.use_tiled_prefill_attention = use_tiled_prefill_attention

        self.scale = head_dim ** -0.5
        self.rope = RoPE(dims=head_dim, seq_len=max_seq_len, base=theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
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

        rope_range = slice(
            offsets, offsets + L
        )

        q = self.rope(q, offset=rope_range)
        k = self.rope(k, offset=rope_range)

        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        k, v, _, mask = cache.update_and_fetch(
            k, v, mask_length=L, mask=mask
        )

        out = scaled_dot_product_attention_grouped(
            query=q.astype(mx.float32), key=k.astype(mx.float32), value=v.astype(mx.float32), scale=self.scale, mask=mask
        ).astype(original_dtype)

        out = out.swapaxes(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        out = linear(out, self.wo)

        return out


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        use_fast_swiglu: bool = True,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu

    def __call__(self, x: mx.array) -> mx.array:
        original_dtype = x.dtype
        gate = linear(x, self.w_gate)
        up = linear(x, self.w_up)

        if self.use_fast_swiglu:
            out = swiglu(gate, up)
        else:
            out = silu(gate) * up
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
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_tiled_prefill_attention: bool = False,
    ):
        # Shape probes: benchmarks read these off a block instead of the args.
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

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
            rms_norm_eps,
            use_fast_rms_norm=use_fast_rms_norm,
            use_fast_rope=use_fast_rope,
            use_tiled_prefill_attention=use_tiled_prefill_attention,
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
            w_down,
            use_fast_swiglu=use_fast_swiglu,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        atten_input = self.input_layernorm(x)
        atten_output = self.self_attn(atten_input, offset, cache, mask=mask)

        h = x + atten_output
        mlp_input = self.post_attention_layernorm(h)
        mlp_ouput = self.mlp(mlp_input)
        out = h + mlp_ouput
        return out
        


class Qwen3ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        checkpoint: str = "selected",
        use_mlx_quantized_linear: bool = False,
        use_bounded_kv_capacity: bool | None = None,
        use_register_cached_rms_norm: bool | None = None,
        use_tiled_prefill_attention: bool | None = None,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        args = mlx_model.args
        model = mlx_model.model
        self.checkpoint = checkpoint
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.precision = mx.bfloat16
        self.num_hidden_layers = args.num_hidden_layers

        self.use_bounded_kv_capacity = (
            checkpoint == "capacity-cache"
        )

        features = WEEK2_CHECKPOINT_FEATURES[checkpoint]
        use_fast_rms_norm = features.fast_rms_norm
        use_fast_rope = features.fast_rope
        use_fast_swiglu = features.fast_swiglu
        use_tiled_prefill_attention = features.tiled_prefill_attention

        # Profilers and benchmarks replay this layer's operators, so the
        # resolved flags must be readable off the model, not just the module.
        self.use_register_cached_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        self.use_tiled_prefill_attention = use_tiled_prefill_attention

        # `layers_inner` / `embedding` are the names benchmarks and profilers
        # look up on the model object.
        self.layers_inner = []

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
                ),

                q_norm=src_layer.self_attn.q_norm.weight.astype(mx.bfloat16),
                k_norm=src_layer.self_attn.k_norm.weight.astype(mx.bfloat16),

                w_gate=dequantize_linear(
                    src_layer.mlp.gate_proj
                ).astype(mx.bfloat16),

                w_up=dequantize_linear(
                    src_layer.mlp.up_proj
                ).astype(mx.bfloat16),

                w_down=dequantize_linear(
                    src_layer.mlp.down_proj
                ).astype(mx.bfloat16),

                w_input_layernorm=(
                    src_layer.input_layernorm.weight.astype(mx.bfloat16)
                ),

                w_post_attention_layernorm=(
                    src_layer.post_attention_layernorm.weight.astype(mx.bfloat16)
                ),

                max_seq_len=args.max_position_embeddings,
                theta=args.rope_theta,
                use_fast_rms_norm=use_fast_rms_norm,
                use_fast_rope=use_fast_rope,
                use_fast_swiglu=use_fast_swiglu,
                use_tiled_prefill_attention=use_tiled_prefill_attention,
            )
            self.layers_inner.append(block)
        embedding_weight = dequantize_linear(model.embed_tokens).astype(mx.bfloat16)
        self.embedding = Embedding(
            vocab_size=args.vocab_size,
            embedding_dim=args.hidden_size,
            weight=embedding_weight
        )

        self.norm = RMSNorm(
            args.hidden_size,
            model.norm.weight.astype(mx.bfloat16),
            eps=args.rms_norm_eps
        )
        if args.tie_word_embeddings:
            self.w_lm_head = None
        else:
            self.w_lm_head = dequantize_linear(
                mlx_model.lm_head
            ).astype(mx.bfloat16)
        self.mlx_model = mlx_model

    def create_kv_cache(self, capacity: int | None = None) -> list[TinyKvCache]:
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        pass
