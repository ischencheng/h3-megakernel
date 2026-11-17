#pragma once
#include <iostream>
#include "megakernel.cuh"
#include "h3_config.cuh"

namespace h3 {
enum opcode : int { noop = 0, smoke = 1, rendezvous = 2, graph_task = 3, wave_barrier = 4,
                    projection_gemm = 5, qknorm_rope_stage = 6,
                    sparse_attention_stage = 7, vsa_pool_stage = 8,
                    vsa_scores_stage = 9, vsa_topk_stage = 10,
                    vsa_value_pool_stage = 11, vsa_compress_stage = 12 };
// Add H3 tensor layouts, weight views, dependency counters and scalar
// parameters here as operations are implemented. No Llama dimensions apply.
struct globals {
    megakernel::instruction_layout<config> instructions;
    megakernel::timing_layout<config> timings;
    kittens::gl<int, 1, 1, 1, -1> completion_counts;
    kittens::gl<int, 1, 1, 1, -1> physical_sm_ids;
    kittens::gl<int, 1, 1, 1, -1> rendezvous_count;
    kittens::gl<float, 1, 1, 1, -1> graph_arena;
    kittens::gl<int, 1, 1, 1, -1> wave_counts;
    kittens::gl<int, 1, 1, 1, -1> graph_error;
    int sm_count;
    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};
using state = megakernel::state<config>;
} // namespace h3
