#pragma once

#include "../h3.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>

namespace h3::vsa_compress {
constexpr int HEADS = 56;
constexpr int HEAD_DIM = 128;
constexpr int TILE = 64;
constexpr int TILES = 668;

template <typename G>
__device__ inline void execute(const G &g, state &s) {
    // FastVideo's trained coarse branch: softmax(pooled QK) @ pooled V,
    // cast to BF16, then addcmul into the fine-attention output.
    __syncthreads();
    const auto *scores = reinterpret_cast<const float *>(g.scores.raw_ptr);
    const auto *v_pool = reinterpret_cast<const float *>(g.v_pool.raw_ptr);
    const auto *gate = reinterpret_cast<const __nv_bfloat16 *>(g.gate_output.raw_ptr);
    auto *out = reinterpret_cast<__nv_bfloat16 *>(g.sparse_output.raw_ptr);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tiles = s.instruction()[1];
    for (int pair = blockIdx.x * 12 + warp; pair < tiles * HEADS;
         pair += gridDim.x * 12) {
        const int head = pair % HEADS;
        const int tile = pair / HEADS;
        const int scores_base = (head * TILES + tile) * TILES;
        float maximum = -CUDART_INF_F;
        for (int key = 0; key < TILES; ++key)
            maximum = fmaxf(maximum, scores[scores_base + key]);
        float denom = 0.0f;
        for (int key = 0; key < TILES; ++key)
            denom += expf(scores[scores_base + key] - maximum);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int key = 0; key < TILES; ++key) {
            const float probability = expf(scores[scores_base + key] - maximum) / denom;
            const int value_base = (head * TILES + key) * HEAD_DIM;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                acc[j] = fmaf(probability, v_pool[value_base + lane + j * 32], acc[j]);
        }
        __nv_bfloat16 coarse[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            coarse[j] = __float2bfloat16_rn(acc[j]);
        for (int row = 0; row < TILE; ++row) {
            const int base = ((tile * TILE + row) * HEADS + head) * HEAD_DIM;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int channel = lane + j * 32;
                const float fine = __bfloat162float(out[base + channel]);
                const float gval = __bfloat162float(gate[base + channel]);
                const float cval = __bfloat162float(coarse[j]);
                out[base + channel] = __float2bfloat16_rn(fmaf(cval, gval, fine));
            }
        }
    }
    __syncthreads();
}
} // namespace h3::vsa_compress

namespace h3 {
struct vsa_compress_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::vsa_compress_stage;
    struct controller : megakernel::NoOp<config>::controller {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_compress::execute(g, s);
        }
    };
    struct loader {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::loader::run(g, s);
            vsa_compress::execute(g, s);
        }
    };
    struct launcher {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::launcher::run(g, s);
            vsa_compress::execute(g, s);
        }
    };
    struct consumer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_compress::execute(g, s);
        }
    };
    struct storer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_compress::execute(g, s);
        }
    };
};
} // namespace h3
