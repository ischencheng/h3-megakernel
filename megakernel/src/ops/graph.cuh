#pragma once
#include "../h3.cuh"
#include <cuda/atomic>

namespace h3 {
// Instruction words: opcode, kind (1=affine, 2=add), src0, src1, dst,
// tile_start, tile_length, affine_scale_bits, affine_bias_bits, wave_id.
// Offsets and lengths are float elements in the preallocated global arena.
struct graph_task_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::graph_task;
    struct storer {
        static __device__ void run(const globals &g, state &s) {
            const auto &r = s.instruction();
            const int kind = r[1], src0 = r[2], src1 = r[3], dst = r[4];
            const int begin = r[5], count = r[6];
            const float scale = __int_as_float(r[7]);
            const float bias = __int_as_float(r[8]);
            float *arena = g.graph_arena.raw_ptr;
            for (int i = kittens::laneid(); i < count; i += 32) {
                const int p = begin + i;
                const float a = arena[src0 + p];
                arena[dst + p] = kind == 1 ? a * scale + bias
                                           : a + arena[src1 + p];
            }
        }
    };
};

// A whole-stage wave: all resident CTAs finish the same instruction before
// any begins the next one. The host validates one CTA per measured physical SM
// and rejects a grid that exceeds the SM count. CUDA release/acquire orders
// arena writes before the next wave reads them on a different SM.
struct wave_barrier_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::wave_barrier;
    struct storer {
        template <typename G>
        static __device__ void run(const G &g, state &s) {
            if (kittens::laneid() != 0) return;
            const int wave = s.instruction()[1];
            cuda::atomic_ref<int, cuda::thread_scope_device> arrivals(
                g.wave_counts.raw_ptr[wave]);
            arrivals.fetch_add(1, cuda::memory_order_release);
            const unsigned long long start = clock64();
            while (arrivals.load(cuda::memory_order_acquire) < g.sm_count) {
                if (clock64() - start > 10000000000ULL) {
                    atomicCAS(g.graph_error.raw_ptr, 0, wave + 1);
                    return;
                }
                __nanosleep(128);
            }
        }
    };
};
} // namespace h3
