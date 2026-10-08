#include <metal_stdlib>
#include <metal_simdgroup_matrix>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Starter interface map. Implement the named kernels at these checkpoints;
// their argument lists are defined by the matching C++ encoder you complete.
//
// Week 2, Day 2:
//   quantized_matmul_vanilla_w4a16_g128
//   quantized_matvec_x4_fast_w4a16_g128
// Week 2, Day 3:
//   quantized_matmul_simdgroup_w4a16_g128
// Week 3, Day 4:
//   quantized_embedding_w4a16_g128
//
// The x2/x8 tuning variants in the reference extension are deliberately not
// starter interfaces. Add an experimental variant only while running the
// optional scheduling comparison, then keep the selected course path.

template <typename T>
[[kernel]] void quantized_matmul_vanilla_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int& M [[buffer(5)]],
    device const int& N [[buffer(6)]],
    device const int& K [[buffer(7)]],

    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int values_per_word = 8;
    constexpr uint32_t mask = 0xF;

    const int i = group_id.x * threads_per_group.x + thread_id.x;
    const int k = group_id.y * threads_per_group.y + thread_id.y;

    if (i >= M || k >= K){
        return;
    }

    const int group_per_row = N / group_size;
    const int packed_cols = N / values_per_word;

    const int a_base = i * N;
    const int b_base = k * packed_cols;
    const int param_base = k * group_per_row;

    float sum = 0.0f;

    for (int g = 0; g < group_per_row; ++g){
        const float scale = static_cast<float>(scales[param_base + g]);
        const float bias = static_cast<float>(biases[param_base + g]);

        const int packed_per_group = group_size / values_per_word;

        for (int p = 0; p < packed_per_group; ++p){
            const int packed_idx = b_base + g * packed_per_group + p;
            uint32_t packed = b[packed_idx];

            const int a_idx = a_base + g * group_size + p * values_per_word;

            #pragma clang loop unroll(full)
            for(int j = 0; j < 8; ++j){
                const uint32_t q = (packed >> (j * bits)) & mask;
                const float weight = static_cast<float>(q) * scale + bias;
            
                sum += static_cast<float>(a[a_idx + j]) * weight;
            }
        }
    }

    out[i * K + k] = static_cast<T>(sum);

}

template <typename T>
[[kernel]] void quantized_matvec_x4_fast_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],

    device const int& M [[buffer(5)]],
    device const int& N [[buffer(6)]],
    device const int& K [[buffer(7)]],

    uint output_tile  [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
){
    constexpr int bits = 4;
    constexpr int group_size = 128;
    constexpr int values_per_word = 8;
    constexpr int packs_per_lane = 2;
    constexpr int values_per_lane = packs_per_lane * values_per_word;

    constexpr int outputs_per_simdgroup = 4;
    // Must match the threadgroup size the C++ encoder dispatches (2 simdgroups).
    constexpr int simdgroups_per_threadgroup = 2;

    constexpr int outputs_per_threadgroup = outputs_per_simdgroup * simdgroups_per_threadgroup;

    const int column_tiles = (K + outputs_per_threadgroup - 1) / outputs_per_threadgroup;

    const int row = output_tile / column_tiles;
    const int tile = output_tile % column_tiles;

    const int column_base = tile * outputs_per_threadgroup + simdgroup * outputs_per_simdgroup;

    if (row >= M || column_base >= K){
        return;
    }

    const int packed_cols = N / values_per_word;

    const int group_per_row = N / group_size;

    const int activation_base = row * N;

    float sums[outputs_per_simdgroup] = {
        0.0f, 0.0f, 0.0f, 0.0f
    };

    for( int packed_col = lane * packs_per_lane; packed_col < packed_cols; packed_col += 32 * packs_per_lane){
        const int group = packed_col / (group_size / values_per_word);
        float activation[values_per_lane];

        float activation_sum = 0.0f;

        #pragma clang loop unroll(full)
        for(int p = 0; p < packs_per_lane; ++p){
            const int a_offset = activation_base + (packed_col + p) * values_per_word;

            #pragma clang loop unroll(full)
            for(int j = 0; j < values_per_word; ++j){
                const int local = p * values_per_word + j;
                const float value = static_cast<float>(a[a_offset + j]);
                activation[local] = value;
                activation_sum += value;
            }
        }
        #pragma clang loop unroll(full)
        for(int output = 0; output < outputs_per_simdgroup; ++output){
            const int column = column_base + output;

            if(column >= K){
                continue;
            }

            const int param_idx = column * group_per_row + group;
            const float scale = static_cast<float>(scales[param_idx]);

            const float bias = static_cast<float>(biases[param_idx]);

            float quantized_dot = 0.0f;

            #pragma clang loop unroll(full)
            for(int p = 0; p < packs_per_lane; ++p){
                const uint32_t packed = b[column * packed_cols + packed_col + p];
                #pragma clang loop unroll(full)
                for(int j = 0; j < values_per_word; ++j){
                    const uint32_t q = (packed >> (j * bits)) & 0xF;
                    const int local = p * values_per_word + j;
                    quantized_dot += activation[local] * static_cast<float>(q);
                }
            }

            sums[output] += scale * quantized_dot + bias * activation_sum;
        }
    }

    #pragma clang loop unroll(full)
    for(int output = 0; output < outputs_per_simdgroup; ++output){
        sums[output] = simd_sum(sums[output]);
    }

    if (lane == 0) {
        #pragma clang loop unroll(full)
        for(int output = 0; output < outputs_per_simdgroup; ++output){
            const int column = column_base + output;

            if(column < K) {
                out[row * K + column] = static_cast<T>(sums[output]);
            }
        }
    }
}

instantiate_kernel(
    "quantized_matmul_vanilla_w4a16_g128_f16",
    quantized_matmul_vanilla_w4a16_g128,
    half
);

instantiate_kernel(
    "quantized_matmul_vanilla_w4a16_g128_bf16",
    quantized_matmul_vanilla_w4a16_g128,
    bfloat16_t
);

instantiate_kernel(
    "quantized_matvec_x4_fast_w4a16_g128_f16",
    quantized_matvec_x4_fast_w4a16_g128,
    half
);

instantiate_kernel(
    "quantized_matvec_x4_fast_w4a16_g128_bf16",
    quantized_matvec_x4_fast_w4a16_g128,
    bfloat16_t
);