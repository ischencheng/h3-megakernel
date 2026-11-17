#pragma once
#include "../h3.cuh"

namespace h3 {
// Inherit NoOp's page/tensor release protocol. This operation touches no
// shared pages. Real operations must replace these hooks as appropriate.
struct smoke_op : megakernel::NoOp<config> {
    static constexpr int opcode = h3::smoke;
    struct storer {
        template <typename G>
        static __device__ void run(const G &g, state &s) {
            if (kittens::laneid() == 0)
                atomicAdd(g.completion_counts.raw_ptr + blockIdx.x, 1);
            if (kittens::laneid() == 0)
                g.physical_sm_ids.raw_ptr[blockIdx.x] = megakernel::get_smid();
        }
    };
};
} // namespace h3
