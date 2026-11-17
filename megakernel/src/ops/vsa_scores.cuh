#pragma once

#include "../h3.cuh"

namespace h3::vsa_scores {
constexpr int HEAD_DIM = 128;
constexpr int TILES = 668;
constexpr int HEADS = 56;
constexpr float SCALE = 0.08838834764831845f;

template <typename G>
__device__ inline void execute(const G &g, state &s) {
    // Each warp computes one pooled Q·K score with FP32 accumulation.
    __syncthreads();
    const auto *q = reinterpret_cast<const float *>(g.q_pool.raw_ptr);
    const auto *k = reinterpret_cast<const float *>(g.k_pool.raw_ptr);
    auto *scores = reinterpret_cast<float *>(g.scores.raw_ptr);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int heads = s.instruction()[1];
    const int total = heads * TILES * TILES;
    for (int index = blockIdx.x * 12 + warp; index < total;
         index += gridDim.x * 12) {
        const int head = index / (TILES * TILES);
        const int query_tile = (index / TILES) % TILES;
        const int key_tile = index % TILES;
        const int q_base = (head * TILES + query_tile) * HEAD_DIM;
        const int k_base = (head * TILES + key_tile) * HEAD_DIM;
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int channel = lane + j * 32;
            sum = fmaf(q[q_base + channel], k[k_base + channel], sum);
        }
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0)
            scores[index] = sum * SCALE;
    }
    __syncthreads();
}
} // namespace h3::vsa_scores

namespace h3 {
struct vsa_scores_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::vsa_scores_stage;
    struct controller : megakernel::NoOp<config>::controller {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_scores::execute(g, s);
        }
    };
    struct loader {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::loader::run(g, s);
            vsa_scores::execute(g, s);
        }
    };
    struct launcher {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::launcher::run(g, s);
            vsa_scores::execute(g, s);
        }
    };
    struct consumer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_scores::execute(g, s);
        }
    };
    struct storer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_scores::execute(g, s);
        }
    };
};
} // namespace h3
