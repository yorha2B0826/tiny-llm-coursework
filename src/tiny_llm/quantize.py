from typing import Any

import mlx.core as mx
from extensions.tiny_llm_ext import (
    _ext as tiny_llm_ext
)


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w.astype(mx.bfloat16)


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight
        self.use_simdgroup_matmul = use_simdgroup_matmul
        self.use_simdgroup_matvec = use_simdgroup_matvec
        self.use_split_k_matmul = use_split_k_matmul
        self.use_mlx_quantized_linear = use_mlx_quantized_linear

    @staticmethod
    def from_mlx_layer(
        mlx_layer: Any,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_mlx_quantized_linear: bool = False,
    ) -> "QuantizedWeights":
        biases = mlx_layer.biases
        return QuantizedWeights(
            scales=mlx_layer.scales.astype(mx.bfloat16),
            biases=None if biases is None else biases.astype(mx.bfloat16),
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
            use_simdgroup_matmul=use_simdgroup_matmul,
            use_simdgroup_matvec=use_simdgroup_matvec,
            use_split_k_matmul=use_split_k_matmul,
            use_mlx_quantized_linear=use_mlx_quantized_linear,
        )


def mlx_quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    result = mx.quantized_matmul(
        x,
        w.weight,
        scales=w.scales,
        biases=w.biases,
        transpose=True,
        group_size=w.group_size,
        bits=w.bits
    )

    if bias is not None:
        result += bias

    return result


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
    use_simdgroup: bool = False,
    use_split_k: bool = False,
) -> mx.array:
    *leading, D = a.shape
    a = a.reshape(-1, D)
    result = tiny_llm_ext.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(a),
        mx.contiguous(b),
        transpose_b,
        use_simdgroup,
        use_split_k,
    )

    return result.reshape(
        *leading,
        -1,
    )


def dequantize_weights(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    group_size: int,
    bits: int,
) -> mx.array:
    values_per_word = 32 // bits

    shifts = mx.arange(
        0,
        32,
        bits,
        dtype=mx.uint32
    )

    mask = (1 << bits) - 1
    values = (
        weight[..., None] >> shifts
    ) & mask

    values = values.reshape(
        *weight.shape[:-1],
        weight.shape[-1] * values_per_word,
    )

    values = values.astype(mx.float32)

    expanded_scales = mx.repeat(
        scales,
        group_size,
        axis=-1
    ).astype(mx.float32)

    if biases is None:
        return (
            values * expanded_scales
        ).astype(scales.dtype)

    expanded_biases = mx.repeat(
        biases,
        group_size,
        axis=-1
    ).astype(mx.float32)

    return (values * expanded_scales + expanded_biases).astype(scales.dtype)


def quantized_matvec_custom(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    *leading, D = a.shape
    flat_a = a.reshape(-1, D)

    if flat_a.shape[0] > 8:
        raise ValueError(
            "quantized_matvec_custom supports "
            "at most 8 input rows"
        )

    result = tiny_llm_ext.quantized_matmul(
        mx.contiguous(scales),
        mx.contiguous(biases),
        group_size,
        bits,
        mx.contiguous(flat_a),
        mx.contiguous(b),
        transpose_b,
        True,
        False
    )

    return result.reshape(
        *leading,
        -1,
    )


def quantized_matmul_vanilla(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    return quantized_matmul(
        scales,
        biases,
        group_size,
        bits,
        a,
        b,
        transpose_b,
        use_simdgroup=False
    )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    if w.use_mlx_quantized_linear:
        return mlx_quantized_linear(
            x,
            w,
            bias,
        )

    rows = 1
    for size in x.shape[:-1]:
        rows *= size

    if(rows <= 8 and w.use_simdgroup_matvec):
        result = quantized_matvec_custom(
            w.scales,
            w.biases,
            w.group_size,
            w.bits,
            x,
            w.weight,
            transpose_b=True
        )

    else:
        result = quantized_matmul(
            w.scales,
            w.biases,
            w.group_size,
            w.bits,
            x,
            w.weight,
            transpose_b=True,
            use_simdgroup=False,
        )

    if bias is not None:
        result = result + bias

    return result
