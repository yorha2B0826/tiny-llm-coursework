#include <stdexcept>
#include <string>

#include "tiny_llm_ext.h"
#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace tiny_llm_ext {

namespace {

[[noreturn]] void checkpoint_todo(const char *function, const char *checkpoint) {
    throw std::runtime_error(std::string(function) + " is a starter stub; implement it in " + checkpoint);
}

}  // namespace

// Week 2, Day 2. Later days extend the dispatch policy behind this API.
mx::array quantized_matmul(
    const mx::array &scales,
    const mx::array &biases,
    const int group_size,
    const int bits,
    const mx::array &a,
    const mx::array &b,
    const bool transpose_b,
    const bool use_simdgroup,
    const bool use_split_k,
    mx::StreamOrDevice s
) {
    if (
        scales.dtype() != mx::float16 &&
        scales.dtype() != mx::bfloat16
    ) {
        throw std::runtime_error(
            "quantized_matmul: scales must be float16 or bfloat16"
        );
    }

    if (scales.dtype() != biases.dtype()) {
        throw std::runtime_error(
            "quantized_matmul: scales and biases must have the same dtype"
        );
    }

    if (a.dtype() != scales.dtype()) {
        throw std::runtime_error(
            "quantized_matmul: a must have the same dtype as scales"
        );
    }

    if (b.dtype() != mx::uint32) {
        throw std::runtime_error(
            "quantized_matmul: b must be uint32"
        );
    }

    if (a.shape().size() != 2) {
        throw std::runtime_error(
            "quantized_matmul: a must be 2D"
        );
    }

    if (b.shape().size() != 2) {
        throw std::runtime_error(
            "quantized_matmul: b must be 2D"
        );
    }

    if (bits != 4) {
        throw std::runtime_error(
            "quantized_matmul: bits must be 4"
        );
    }

    if (group_size != 128) {
        throw std::runtime_error(
            "quantized_matmul: group_size must be 128"
        );
    }

    if (!transpose_b) {
        throw std::runtime_error(
            "quantized_matmul: transpose_b must be true"
        );
    }

    const int values_per_word = 32 / bits;

    if (scales.shape() != biases.shape()) {
        throw std::runtime_error(
            "quantized_matmul: scales and biases shape mismatch"
        );
    }

    if (b.shape()[0] != scales.shape()[0]) {
        throw std::runtime_error(
            "quantized_matmul: output dimension mismatch"
        );
    }

    if (a.shape()[1] % group_size != 0) {
        throw std::runtime_error(
            "quantized_matmul: N must be divisible by group_size"
        );
    }

    if (
        scales.shape()[1]
        != a.shape()[1] / group_size
    ) {
        throw std::runtime_error(
            "quantized_matmul: invalid scale shape"
        );
    }

    if (
        b.shape()[1]
        != a.shape()[1] / values_per_word
    ) {
        throw std::runtime_error(
            "quantized_matmul: invalid packed weight shape"
        );
    }

    auto out_shape = a.shape();
    out_shape[1] = b.shape()[0];

    return mx::array(
        out_shape,
        a.dtype(),
        std::make_shared<QuantizedMatmul>(
            to_stream(s),
            use_simdgroup,
            use_split_k
        ),
        {
            scales,
            biases,
            a,
            b,
        }
    );
}

void QuantizedMatmul::eval_cpu(
    const std::vector<mx::array> &,
    std::vector<mx::array> &
) {
    throw std::runtime_error(
        "quantized_matmul: GPU-only operator"
    );
}

void QuantizedMatmul::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs
) {
    auto& scales = inputs[0];
    auto& biases = inputs[1];
    auto& a = inputs[2];
    auto& b = inputs[3];

    auto& out = outputs[0];

    if (!a.flags().row_contiguous) {
        throw std::runtime_error(
            "quantized_matmul: a must be contiguous"
        );
    }

    if(!b.flags().row_contiguous){
        throw std::runtime_error(
            "quantized_matmul: b must be contiguous"
        );
    }

    const int M = a.shape()[0];
    const int N = a.shape()[1];
    const int K = b.shape()[0];

    auto& s = stream();
    auto& d = mx::metal::device(s.device);

    auto library = d.get_library("tiny_llm_ext");

    out.set_data(
        mx::allocator::malloc(out.nbytes())
    );

    const bool use_matvec = use_simdgroup_ && M <= 8;
    const char* kernel_name;
    if(use_matvec) {
        kernel_name = out.dtype() == mx::float16 ? "quantized_matvec_x4_fast_w4a16_g128_f16" : "quantized_matvec_x4_fast_w4a16_g128_bf16";
    }else{
        kernel_name = out.dtype() == mx::float16 ? "quantized_matmul_vanilla_w4a16_g128_f16" : "quantized_matmul_vanilla_w4a16_g128_bf16"; 
    }

    auto kernel = d.get_kernel(kernel_name, library);
    auto& encoder = mx::metal::get_command_encoder(s);

    encoder.set_compute_pipeline_state(
        kernel
    );

    encoder.set_input_array(scales, 0);

    encoder.set_input_array(biases, 1);

    encoder.set_input_array(a, 2);

    encoder.set_input_array(b, 3);

    encoder.set_output_array(out, 4); 
    encoder.set_bytes(M, 5);
    encoder.set_bytes(N, 6);
    encoder.set_bytes(K, 7);

    if(use_matvec){
        constexpr int outputs_per_simdgroup = 4;
        constexpr int simdgroup_per_threadgroup = 2;

        constexpr int outputs_per_threadgroup = outputs_per_simdgroup * simdgroup_per_threadgroup;

        const int column_tiles = (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;

        encoder.dispatch_threadgroups(MTL::Size(
            M * column_tiles,
            1,
            1
        ),
        MTL::Size(
            simdgroup_per_threadgroup * 32,
            1,
            1
        )
    );

    return;

    }

    const size_t max_threads = kernel->maxTotalThreadsPerThreadgroup();
    const int x_size = M <= 16 ? 16 : 32;
    const int y_size = max_threads / x_size;
    encoder.dispatch_threadgroups(
        MTL::Size(
            (M + x_size - 1) / x_size,
            (K + y_size - 1) / y_size,
            1
        ),

        MTL::Size(
            x_size,
            y_size,
            1
        )
    );
}

// Week 3, Day 4. The earlier Week 2 checkpoints keep the readable row lookup.
mx::array quantized_embedding(const mx::array &, const mx::array &, const mx::array &, const mx::array &, int, int,
                              mx::StreamOrDevice) {
    checkpoint_todo("quantized_embedding", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_cpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_cpu", "Week 3, Day 4");
}

void QuantizedEmbedding::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
    checkpoint_todo("QuantizedEmbedding::eval_gpu", "Week 3, Day 4");
}

}  // namespace tiny_llm_ext
