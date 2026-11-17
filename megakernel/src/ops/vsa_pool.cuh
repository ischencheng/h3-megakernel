#pragma once

#include "../h3.cuh"
#include <cuda_bf16.h>

namespace h3::vsa_pool {
constexpr int HEADS = 56;
constexpr int HEAD_DIM = 128;
constexpr int TILE = 64;
constexpr int TILES = 668;

template <typename G>
__device__ inline void execute(const G &g, state &s) {
    // FastVideo's _pool_tiles: FP32 sum of valid BF16 tokens, divided by the
    // actual tile length. One Hazy warp owns one (tile, head) pair.
    __syncthreads();
    const auto *q = reinterpret_cast<const __nv_bfloat16 *>(g.q_output.raw_ptr);
    const auto *k = reinterpret_cast<const __nv_bfloat16 *>(g.k_output.raw_ptr);
    const auto *valid = reinterpret_cast<const int64_t *>(g.valid_lengths.raw_ptr);
    auto *q_pool = reinterpret_cast<float *>(g.q_pool.raw_ptr);
    auto *k_pool = reinterpret_cast<float *>(g.k_pool.raw_ptr);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tiles = s.instruction()[1];
    for (int pair = blockIdx.x * 12 + warp; pair < tiles * HEADS;
         pair += gridDim.x * 12) {
        const int head = pair % HEADS;
        const int tile = pair / HEADS;
        const int count = static_cast<int>(valid[tile]);
        float q_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float k_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int row = 0; row < count; ++row) {
            const int base = ((tile * TILE + row) * HEADS + head) * HEAD_DIM;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int channel = lane + j * 32;
                q_sum[j] += __bfloat162float(q[base + channel]);
                k_sum[j] += __bfloat162float(k[base + channel]);
            }
        }
        const int out = (head * TILES + tile) * HEAD_DIM;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int channel = lane + j * 32;
            q_pool[out + channel] = count ? q_sum[j] / count : 0.0f;
            k_pool[out + channel] = count ? k_sum[j] / count : 0.0f;
        }
    }
    __syncthreads();
}
} // namespace h3::vsa_pool

namespace h3 {
struct vsa_pool_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::vsa_pool_stage;
    struct controller : megakernel::NoOp<config>::controller {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_pool::execute(g, s);
        }
    };
    struct loader {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::loader::run(g, s);
            vsa_pool::execute(g, s);
        }
    };
    struct launcher {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::launcher::run(g, s);
            vsa_pool::execute(g, s);
        }
    };
    struct consumer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_pool::execute(g, s);
        }
    };
    struct storer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_pool::execute(g, s);
        }
    };
};
} // namespace h3
