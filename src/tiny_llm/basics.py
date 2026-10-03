import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # Supplied for Day 1; a manual implementation is an optional bonus exercise.
    orginal_dtype = x.dtype
    x = x.astype(mx.float32)
    x_max = mx.max(x, axis=axis, keepdims=True)
    e = mx.exp(x - x_max)
    out = e / mx.sum(e, axis=axis, keepdims=True)
    out = out.astype(orginal_dtype)
    return out


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    output = x @ w.T
    if bias is not None:
        output = output + bias
    return output


def silu(x: mx.array) -> mx.array:
    z = mx.exp(-mx.abs(x))
    sigmoid = mx.where(x < 0, z / (1 + z), 1 / (1 + z))
    return x * sigmoid