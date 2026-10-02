import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        filterd = copy.copy(logprobs)
        vocab_size = filterd.shape[-1]

        if top_k is not None and top_k > 0:
            if top_k > vocab_size:
                raise ValueError(
                    f"top_k ({top_k}) cannot not be larger than"
                    f"vocab_size({vocab_size})"
                )
            if top_k < vocab_size:
                indices = mx.argpartition(
                    -filterd,
                    kth=top_k - 1,
                    axis=-1
                )
                remove_indices = indices[..., top_k:]

                filterd = mx.put_along_axis(
                    filterd,
                    remove_indices,
                    mx.array(-mx.inf, dtype=filterd.dtype),
                    axis=-1
                )
        if top_p is not None and 0 < top_p < 1:
            sorted_indices = mx.argsort(
                -filterd,
                axis=-1,
            )

            sorted_logprobs = mx.take_along_axis(
                filterd,
                sorted_indices,
                axis=-1
            )

            sorted_probs = mx.exp(
                sorted_logprobs.astype(mx.float32)
            )

            cumulative_probs = mx.cumsum(
                sorted_probs,
                axis=-1
            )

            cumulative_before = (
                cumulative_probs - sorted_probs
            )
            keep = cumulative_before < top_p

            sorted_logprobs = mx.where(
                keep,
                sorted_logprobs,
                mx.array(-mx.inf, dtype=filterd.dtype)
            )

            inverse_indices = mx.argsort(
                sorted_indices,
                axis=-1,
            )

            filterd = mx.take_along_axis(
                sorted_logprobs,
                inverse_indices,
                axis=-1
            )

        filterd = filterd / temp
        token = mx.random.categorical(
            filterd,
            axis=-1
        )
        return token.astype(mx.uint32)
    return sample
