import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

        half_dim = dims // 2
        i = mx.arange(half_dim)
        angular_rate = mx.power(base, -i / half_dim)
        positions = mx.arange(seq_len)

        angles = mx.outer(positions, angular_rate)
        self.cos_freqs = mx.cos(angles)
        self.sin_freqs = mx.sin(angles)

        
    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        original_dtype = x.dtype

        if offset is None:
            cos = self.cos_freqs[:L]
            sin = self.sin_freqs[:L]
        else:
            cos = self.cos_freqs[offset]
            sin = self.sin_freqs[offset]

        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        if self.traditional:
            x_pair = x.reshape(N, L, H, D//2, 2)
            x1 = x_pair[..., 0] 
            x2 = x_pair[..., 1]

            out1 = x1 * cos - x2 * sin
            out2 = x1 * sin + x2 * cos

            out = mx.stack([out1, out2], axis=-1)
            out = out.reshape(N, L, H, D)
        else:
            half_dim = D // 2
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]

            out1 = x1 * cos - x2 * sin
            out2 = x1 * sin + x2 * cos
            out = mx.concatenate([out1, out2], axis=-1)
        return out.astype(original_dtype)
