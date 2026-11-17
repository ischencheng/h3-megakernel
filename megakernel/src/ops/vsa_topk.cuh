#pragma once

#include "../h3.cuh"
#include <cstdint>

namespace h3::vsa_topk {
constexpr int TILES = 668;

__device__ inline uint32_t ordered_key(float score) {
    const uint32_t bits = __float_as_uint(score);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

template <typename G>
__device__ inline void execute(const G &g, state &s) {
    // Exact-order radix selection of the kth FP32 score. This first recipe
    // has one exempt video span; prefix rows and keys remain dense.
    __syncthreads();
    const auto *scores = reinterpret_cast<const float *>(g.scores.raw_ptr);
    auto *mask = reinterpret_cast<uint8_t *>(g.sparse_mask.raw_ptr);
    const int heads = s.instruction()[1];
    const int prefix = s.instruction()[2];
    const int video_start = s.instruction()[3];
    const int video_end = s.instruction()[4];
    const int keep = s.instruction()[5];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int row = blockIdx.x * 12 + warp; row < heads * TILES;
         row += gridDim.x * 12) {
        const int qtile = row % TILES;
        const int base = row * TILES;
        if (qtile < prefix || keep >= video_end - video_start) {
            for (int col = lane; col < TILES; col += 32)
                mask[base + col] = 1;
            continue;
        }
        uint32_t low = 0, high = 0xffffffffu;
        while (low < high) {
            const uint32_t mid = low + ((high - low) >> 1) + 1;
            int count = 0;
            for (int col = video_start + lane; col < video_end; col += 32)
                count += ordered_key(scores[base + col]) >= mid;
#pragma unroll
            for (int offset = 16; offset; offset >>= 1)
                count += __shfl_down_sync(0xffffffff, count, offset);
            count = __shfl_sync(0xffffffff, count, 0);
            if (count >= keep)
                low = mid;
            else
                high = mid - 1;
        }
        int greater = 0, equal = 0;
        for (int col = video_start + lane; col < video_end; col += 32) {
            const uint32_t key = ordered_key(scores[base + col]);
            greater += key > low;
            equal += key == low;
        }
#pragma unroll
        for (int offset = 16; offset; offset >>= 1) {
            greater += __shfl_down_sync(0xffffffff, greater, offset);
            equal += __shfl_down_sync(0xffffffff, equal, offset);
        }
        greater = __shfl_sync(0xffffffff, greater, 0);
        equal = __shfl_sync(0xffffffff, equal, 0);
        int tie_cutoff = video_end - 1;
        if (greater + equal > keep) {
            // The frozen FastVideo torch.topk resolves its observed boundary
            // tie toward the lower block ID. Select exactly keep entries.
            int left = video_start, right = video_end - 1;
            const int ties_to_keep = keep - greater;
            while (left < right) {
                const int mid = left + (right - left) / 2;
                int before = 0;
                for (int col = video_start + lane; col <= mid; col += 32)
                    before += ordered_key(scores[base + col]) == low;
#pragma unroll
                for (int offset = 16; offset; offset >>= 1)
                    before += __shfl_down_sync(0xffffffff, before, offset);
                before = __shfl_sync(0xffffffff, before, 0);
                if (before >= ties_to_keep)
                    right = mid;
                else
                    left = mid + 1;
            }
            tie_cutoff = left;
        }
        for (int col = lane; col < TILES; col += 32)
            mask[base + col] = static_cast<uint8_t>(
                col < prefix || (col >= video_start && col < video_end &&
                                 (ordered_key(scores[base + col]) > low ||
                                  (ordered_key(scores[base + col]) == low &&
                                   col <= tie_cutoff))));
    }
    __syncthreads();
}
} // namespace h3::vsa_topk

namespace h3 {
struct vsa_topk_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::vsa_topk_stage;
    struct controller : megakernel::NoOp<config>::controller {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_topk::execute(g, s);
        }
    };
    struct loader {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::loader::run(g, s);
            vsa_topk::execute(g, s);
        }
    };
    struct launcher {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::launcher::run(g, s);
            vsa_topk::execute(g, s);
        }
    };
    struct consumer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_topk::execute(g, s);
        }
    };
    struct storer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            vsa_topk::execute(g, s);
        }
    };
};
} // namespace h3
