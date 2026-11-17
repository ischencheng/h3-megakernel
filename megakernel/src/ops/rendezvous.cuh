#pragma once
#include "../h3.cuh"

namespace h3 {
// A bounded cross-SM barrier probe, not an H3 scheduling primitive. It catches
// the case where the grid cannot become fully resident before later ops rely
// on a global producer/consumer dependency.
struct rendezvous_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::rendezvous;
    struct storer {
        static __device__ void run(const globals &g, state &s) {
            if (kittens::laneid() != 0) return;
            // Each SM wrote its physical ID in the preceding smoke operation.
            // Publish those writes before incrementing the arrival counter.
            __threadfence();
            atomicAdd(g.rendezvous_count.raw_ptr, 1);
            const unsigned long long start = clock64();
            while (atomicAdd(g.rendezvous_count.raw_ptr, 0) < g.sm_count) {
                if (clock64() - start > 10000000000ULL) {
                    g.completion_counts.raw_ptr[blockIdx.x] = -1;
                    return;
                }
                __nanosleep(128);
            }
            for (int block = 0; block < g.sm_count; ++block) {
                if (g.physical_sm_ids.raw_ptr[block] < 0) {
                    g.completion_counts.raw_ptr[blockIdx.x] = -2;
                    return;
                }
            }
        }
    };
};
} // namespace h3
