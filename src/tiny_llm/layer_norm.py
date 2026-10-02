import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        original_dtype = x.dtype
        x = x.astype(mx.float32)
        rms = mx.sqrt(
            mx.mean(x * x, axis=-1, keepdims=True) + self.eps
        )
        x_norm = x / rms
        x_norm = x_norm.astype(original_dtype)
        return x_norm * self.weight
