#pragma once

// Shared pieces of the full-DiT instruction set.
//
// Every DiT operation is an all-warp instruction: the twelve Hazy warps of a
// CTA meet at the instruction boundary, stride a global work list by
// (blockIdx, warp), and leave through a CTA barrier. Buffers are addressed by
// index into a host-prepared device pointer table (globals::buffers), so no
// tensor shape, checkpoint name, or weight layout appears in the dispatch
// code. Instruction word 31 optionally names a single-use cross-SM wave
// counter: all resident CTAs finish the stage before any starts the next one.

#include "../h3.cuh"
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cstdint>

namespace h3::dit {
constexpr int WARPS = config::NUM_WARPS;
constexpr int WAVE_WORD = config::INSTRUCTION_WIDTH - 1;
// Roughly a minute of SM clock ticks. A CTA that cannot see every resident
// peer arrive reports the wave instead of hanging the launch.
constexpr unsigned long long WAVE_TIMEOUT_TICKS = 120000000000ULL;

template <typename T, typename G>
__device__ inline T *buffer(const G &g, int index) {
    const auto *table = reinterpret_cast<const uint64_t *>(g.buffers.raw_ptr);
    return reinterpret_cast<T *>(table[index]);
}

__device__ inline int lane() { return threadIdx.x & 31; }
__device__ inline int warp() { return threadIdx.x >> 5; }
// First work item of this warp and the stride that visits every item once
// across the resident grid.
__device__ inline long long first_item() {
    return static_cast<long long>(blockIdx.x) * WARPS + warp();
}
__device__ inline long long item_stride() {
    return static_cast<long long>(gridDim.x) * WARPS;
}

__device__ inline float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ inline float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ inline float silu(float x) { return x / (1.0f + expf(-x)); }

template <typename G>
__device__ inline void wave_barrier(const G &g, state &s) {
    const int wave = s.instruction()[WAVE_WORD] - 1;
    if (wave < 0) return;
    // Publish this CTA's stage writes before the arrival becomes visible.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> arrivals(
            g.wave_counts.raw_ptr[wave]);
        arrivals.fetch_add(1, cuda::memory_order_release);
        const unsigned long long start = clock64();
        while (arrivals.load(cuda::memory_order_acquire) < g.sm_count) {
            if (clock64() - start > WAVE_TIMEOUT_TICKS) {
                atomicCAS(g.graph_error.raw_ptr, 0, wave + 1);
                break;
            }
            __nanosleep(128);
        }
    }
    __syncthreads();
}

// Wrap an all-thread body as a Hazy operation. `Body::execute(g, s)` runs on
// every thread of the CTA between two CTA barriers; the loader and launcher
// first perform NoOp's page and tensor release so the page protocol of the
// surrounding ordinary instructions is unchanged.
template <int Opcode, typename Body>
struct all_warp_op : megakernel::NoOp<config> {
    static constexpr int opcode = Opcode;
    template <typename G>
    static __device__ inline void join(const G &g, state &s) {
        __syncthreads();
        Body::execute(g, s);
        __syncthreads();
        wave_barrier(g, s);
    }
    struct controller : megakernel::NoOp<config>::controller {
        template <typename G> static __device__ void run(const G &g, state &s) {
            join(g, s);
        }
    };
    struct loader {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::loader::run(g, s);
            join(g, s);
        }
    };
    struct launcher {
        template <typename G> static __device__ void run(const G &g, state &s) {
            megakernel::NoOp<config>::launcher::run(g, s);
            join(g, s);
        }
    };
    struct consumer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            join(g, s);
        }
    };
    struct storer {
        template <typename G> static __device__ void run(const G &g, state &s) {
            join(g, s);
        }
    };
};
} // namespace h3::dit
