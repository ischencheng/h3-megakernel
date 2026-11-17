#pragma once

#include "../h3.cuh"
#include <cuda_bf16.h>

namespace h3::qknorm_rope {
constexpr int HEAD_DIM = 128;
constexpr int ROTARY_DIM = 96;
constexpr int HEADS = 56;

template <typename G>
__device__ inline void execute(const G &g, state &s) {
    // All 12 Hazy warps participate. One warp owns one 128-channel head;
    // every lane accumulates four channels before the warp-wide RMS reduction.
    __syncthreads();
    const __nv_bfloat16 *inputs[2] = {
        reinterpret_cast<const __nv_bfloat16 *>(g.q_input.raw_ptr),
        reinterpret_cast<const __nv_bfloat16 *>(g.k_input.raw_ptr)};
    __nv_bfloat16 *outputs[2] = {
        reinterpret_cast<__nv_bfloat16 *>(g.q_output.raw_ptr),
        reinterpret_cast<__nv_bfloat16 *>(g.k_output.raw_ptr)};
    const __nv_bfloat16 *weights[2] = {
        reinterpret_cast<const __nv_bfloat16 *>(g.q_weight.raw_ptr),
        reinterpret_cast<const __nv_bfloat16 *>(g.k_weight.raw_ptr)};
    const auto *cos = reinterpret_cast<const __nv_bfloat16 *>(g.rope_cos.raw_ptr);
    const auto *sin = reinterpret_cast<const __nv_bfloat16 *>(g.rope_sin.raw_ptr);
    const auto *tile_index = reinterpret_cast<const int64_t *>(g.tile_index.raw_ptr);
    const __nv_bfloat16 *other_inputs[2] = {
        reinterpret_cast<const __nv_bfloat16 *>(g.v_input.raw_ptr),
        reinterpret_cast<const __nv_bfloat16 *>(g.gate_input.raw_ptr)};
    __nv_bfloat16 *other_outputs[2] = {
        reinterpret_cast<__nv_bfloat16 *>(g.v_output.raw_ptr),
        reinterpret_cast<__nv_bfloat16 *>(g.gate_output.raw_ptr)};
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int rows = s.instruction()[1];
    float eps = __int_as_float(s.instruction()[2]);
    bool tile = s.instruction()[3] != 0;
    for (int row = blockIdx.x * 12 + warp; row < rows * HEADS;
         row += gridDim.x * 12) {
        int token = row / HEADS;
        int base = row * HEAD_DIM;
        int output_base = tile ? (int)(tile_index[token] * HEADS + row % HEADS) * HEAD_DIM : base;
        for (int tensor = 0; tensor < 2; ++tensor) {
            float x[4];
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                x[j] = __bfloat162float(inputs[tensor][base + lane + 32 * j]);
                sum += x[j] * x[j];
            }
#pragma unroll
            for (int offset = 16; offset; offset >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / HEAD_DIM + eps);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int channel = lane + 32 * j;
                float normalized = x[j] * inv * __bfloat162float(weights[tensor][channel]);
                float result = normalized;
                if (channel < ROTARY_DIM) {
                    int partner = channel < ROTARY_DIM / 2 ? channel + ROTARY_DIM / 2
                                                             : channel - ROTARY_DIM / 2;
                    float px = __bfloat162float(inputs[tensor][base + partner]);
                    float pn = px * inv * __bfloat162float(weights[tensor][partner]);
                    float rotated = channel < ROTARY_DIM / 2 ? -pn : pn;
                    int table = token * ROTARY_DIM + channel;
                    result = fmaf(rotated, __bfloat162float(sin[table]),
                                  normalized * __bfloat162float(cos[table]));
                }
                outputs[tensor][output_base + channel] = __float2bfloat16_rn(result);
            }
        }
        if (tile) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int channel = lane + 32 * j;
                other_outputs[0][output_base + channel] = other_inputs[0][base + channel];
                other_outputs[1][output_base + channel] = other_inputs[1][base + channel];
            }
        }
    }
    __syncthreads();
}
} // namespace h3::qknorm_rope

namespace h3 {
struct qknorm_rope_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::qknorm_rope_stage;
    struct controller : megakernel::NoOp<config>::controller {
        template <typename G> static __device__ void run(const G &g, state &s) {
            qknorm_rope::execute(g, s);
        }
    };
    struct loader {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::loader::run(g, s);
            qknorm_rope::execute(g, s);
        }
    };
    struct launcher {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::launcher::run(g, s);
            qknorm_rope::execute(g, s);
        }
    };
    struct consumer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            qknorm_rope::execute(g, s);
        }
    };
    struct storer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            qknorm_rope::execute(g, s);
        }
    };
};
} // namespace h3
