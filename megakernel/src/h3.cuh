#pragma once
#include <iostream>
#include "megakernel.cuh"
#include "h3_config.cuh"

namespace h3 {
enum opcode : int { noop = 0, smoke = 1 };
// Add H3 tensor layouts, weight views, dependency counters and scalar
// parameters here as operations are implemented. No Llama dimensions apply.
struct globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};
using state = megakernel::state<config>;
} // namespace h3
