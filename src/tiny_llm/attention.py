import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d = query.shape[-1]
    if scale is None:
        scale = 1 / (d ** 0.5)
    score = query @ key.swapaxes(-1, -2)
    score = score * scale
    if mask is not None:
        score = score + mask
    weights = softmax(score, axis=-1)
    output = weights @ value
    return output 


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        B, L, _ = query.shape
        q = linear(query, self.wq)
        k = linear(key, self.wk)
        v = linear(value, self.wv)

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)

        q = q.swapaxes(1, 2)
        k = k.swapaxes(1, 2)
        v = v.swapaxes(1, 2)

        out = scaled_dot_product_attention_simple(
            q, k, v, mask=mask
        )

        out = out.swapaxes(1, 2)
        out = out.reshape(B, L, self.hidden_size)
        out = linear(out, self.wo)
        return out
    


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    if L > S:
        raise ValueError("casual mask requires S >= L.")

    query_pos = mx.arange(L) + (S - L)
    key_pos = mx.arange(S)

    allowd = key_pos[None, :] <= query_pos[:, None]

    mask = mx.where(
        allowd,
        mx.array(0.0, dtype=dtype),
        mx.array(-mx.inf, dtype=dtype)
    )

    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    H_q = query.shape[-3]
    L = query.shape[-2]
    D = query.shape[-1]

    H = key.shape[-3]
    S = key.shape[-2]

    if H_q % H != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    n_repeats = H_q // H
    if scale is None:
        scale = 1 / (D ** 0.5)

    batch_shape = query.shape[:-3]
    q = query.reshape(
        *batch_shape,
        H,
        n_repeats,
        L,
        D
    )

    k = key.reshape(
        *batch_shape,
        H,
        1,
        S,
        D
    )

    v = value.reshape(
        *batch_shape,
        H,
        1,
        S,
        D
    )
    scores = q @ k.swapaxes(-1, -2)
    scores = scores * scale

    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unsupported mask: {mask}")
        scores = scores + causal_mask(L, S, scores.dtype) 
    elif mask is not None:
        m = mask.reshape(
            *batch_shape,
            H,
            n_repeats,
            L,
            S
        )
        scores += m
    weights = softmax(scores, axis=-1)
    output = weights @ v
    output = output.reshape(
        *batch_shape,
        H_q,
        L,
        D
    )

    return output

def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    """Attend to paged K/V storage without reconstructing a dense cache.

    Week 3 Day 4 owns the correctness-first decode and long-prefill paths for
    both float32 and BF16. Day 5 optimizes the same public boundary.
    """
    pass
