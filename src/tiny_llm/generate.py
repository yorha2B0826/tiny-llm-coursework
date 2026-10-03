import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _release_kv_cache(kv_cache):
    if kv_cache is None:
        return
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
    max_tokens: int = 256,
) -> None:
    def _step(model, y):
        x = y[None, :]
        output_logits = model(x)
        logits = output_logits[:, -1, :]
        log_probs = logits - mx.logsumexp(
            logits,
            axis=-1,
            keepdims=True
        )

        if sampler is None:
            next_token = mx.argmax(
                log_probs,
                axis=-1,
            )
        else:
            next_token = sampler(log_probs)
        return next_token
    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    if len(token_ids) == 0:
        raise ValueError("empty token sequence")

    y = mx.array(token_ids)

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    for _ in range(max_tokens):
        next_token = _step(model, y)

        token_id = int(next_token.item())

        if token_id == tokenizer.eos_token_id:
            break

        detokenizer.add_token(token_id)

        print(
            detokenizer.last_segment,
            end="",
            flush=True
        )

        y=mx.concatenate(
            [y, next_token],
            axis=0
        )

    detokenizer.finalize()

    print(
        detokenizer.last_segment,
        end="",
        flush=True
    )

        


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int = 256,
    use_bounded_kv_capacity: bool | None = None,
) -> str:
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 0
    ):
        raise ValueError("max_tokens must be a non-negative integer")
    if max_tokens == 0:
        return ""
    def _step(model, y, offset, kv_cache):
        logits = model(
            y[None, :],
            offset,
            kv_cache,
            logits_to_keep=1
        )

        logits = logits[:, -1, :]

        next_token = mx.argmax(
            logits,
            axis=-1
        ).astype(mx.int32)

        return next_token
    tokens = mx.array(
        tokenizer.encode(
            prompt,
            add_special_tokens=False
        ),
        dtype=mx.int32
    )

    if tokens.size == 0:
        raise ValueError("prompt must encode to at least one token")

    if use_bounded_kv_capacity:
        capacity=(
            int(tokens.size) + max_tokens
        )
    else:
        capacity = None

    kv_cache = model.create_kv_cache(
        capacity=capacity
    )

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    output_parts = []

    offset = 0

    try:
        for step in range(max_tokens):
            next_token = _step(model, tokens, offset, kv_cache)
            mx.eval(next_token)

            token_id = int(next_token.item())
            if(token_id == tokenizer.eos_token_id):
                break
            detokenizer.add_token(token_id)
            segment = detokenizer.last_segment
            output_parts.append(segment)

            print(segment, end="", flush=True)

            if step + 1 == max_tokens:
                break
            offset += int(tokens.size)

            tokens = next_token

        detokenizer.finalize()
        tail = detokenizer.last_segment
        output_parts.append(tail)

        print(tail, end="", flush=True)

        return "".join(output_parts)
    finally:
        _release_kv_cache(kv_cache)


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
    proposal_length: int = 4,
    max_tokens: int = 256,
) -> str:
    pass
