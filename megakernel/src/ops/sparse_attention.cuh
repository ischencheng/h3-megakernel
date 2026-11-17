#pragma once

#include "../h3.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>

namespace h3::sparse_attention {
constexpr int HEADS = 56;
constexpr int HEAD_DIM = 128;
constexpr int TILE = 64;
constexpr int TILES = 668;
constexpr int SEQ = TILE * TILES;

template <typename G>
__device__ inline void execute(const G &g, state &s) {
    // Correctness-first BF16 VSA fine attention. Every Hazy warp owns one
    // query row and streams only the key tiles selected by the exact mask.
    // No device launch, allocation, or call into a host-launched Triton kernel.
    __syncthreads();
    const auto *q = reinterpret_cast<const __nv_bfloat16 *>(g.q_output.raw_ptr);
    const auto *k = reinterpret_cast<const __nv_bfloat16 *>(g.k_output.raw_ptr);
    const auto *v = reinterpret_cast<const __nv_bfloat16 *>(g.v_output.raw_ptr);
    const auto *mask = reinterpret_cast<const uint8_t *>(g.sparse_mask.raw_ptr);
    const auto *valid = reinterpret_cast<const int64_t *>(g.valid_lengths.raw_ptr);
    auto *output = reinterpret_cast<__nv_bfloat16 *>(g.sparse_output.raw_ptr);
    const int first = s.instruction()[1];
    const int count = s.instruction()[2];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr float scale = 0.08838834764831845f; // 1/sqrt(128)
    for (int row = first + blockIdx.x * 12 + warp; row < first + count;
         row += gridDim.x * 12) {
        const int head = row / SEQ;
        const int query = row - head * SEQ;
        const int qtile = query / TILE;
        const int qbase = (query * HEADS + head) * HEAD_DIM;
        float qv[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 4; ++j)
            qv[j] = __bfloat162float(q[qbase + lane + j * 32]);
        float maximum = -CUDART_INF_F;
        float denominator = 0.0f;
        for (int ktile = 0; ktile < TILES; ++ktile) {
            if (!mask[(head * TILES + qtile) * TILES + ktile]) continue;
            const int kcount = static_cast<int>(valid[ktile]);
            for (int kr = 0; kr < kcount; ++kr) {
                const int kbase = ((ktile * TILE + kr) * HEADS + head) * HEAD_DIM;
                float partial = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    partial = fmaf(qv[j], __bfloat162float(k[kbase + lane + j * 32]), partial);
#pragma unroll
                for (int offset = 16; offset; offset >>= 1)
                    partial += __shfl_down_sync(0xffffffff, partial, offset);
                const float score = __shfl_sync(0xffffffff, partial, 0) * scale;
                const float next_max = fmaxf(maximum, score);
                const float old_weight = expf(maximum - next_max);
                const float new_weight = expf(score - next_max);
                denominator = denominator * old_weight + new_weight;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float value = __bfloat162float(v[kbase + lane + j * 32]);
                    acc[j] = fmaf(new_weight, value, acc[j] * old_weight);
                }
                maximum = next_max;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            output[qbase + lane + j * 32] = __float2bfloat16_rn(
                denominator > 0.0f ? acc[j] / denominator : 0.0f);
    }
    __syncthreads();
}
} // namespace h3::sparse_attention

namespace h3 {
struct sparse_attention_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::sparse_attention_stage;
    struct controller : megakernel::NoOp<config>::controller {
        template <typename G> static __device__ void run(const G &g, state &s) {
            sparse_attention::execute(g, s);
        }
    };
    struct loader {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::loader::run(g, s);
            sparse_attention::execute(g, s);
        }
    };
    struct launcher {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::launcher::run(g, s);
            sparse_attention::execute(g, s);
        }
    };
    struct consumer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            sparse_attention::execute(g, s);
        }
    };
    struct storer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            sparse_attention::execute(g, s);
        }
    };
};
} // namespace h3
