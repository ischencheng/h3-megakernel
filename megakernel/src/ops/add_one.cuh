#pragma once

#include "../h3.cuh"

using namespace kittens;
using namespace megakernel;

// use templates when to reuse and fill variables at compile time

struct AddOneOp : megakernel::NoOp<config>{
    static constexpr int opcode = OPCODE_ADDONE;
    static constexpr int prev_opcode = 0;

    struct consumer {
        template <typename G>
        static __device__ void run(const G &g, megakernel::state<config> &s) {
            int num_weights = g.weights.cols();
            int beg = (long long)num_weights * blockIdx.x / gridDim.x;
            int end = (long long)num_weights * (blockIdx.x + 1) / gridDim.x;

            int NUM_THREADS = config::NUM_CONSUMER_WARPS * kittens::WARP_THREADS;
            int thread_no = kittens::warpid() * kittens::WARP_THREADS + kittens::laneid();

            for (int i = beg + thread_no; i < end; i += NUM_THREADS) {
                g.weights.raw_ptr[i] += 1.0f;
            }
        }
    };
}

// template <typename Config, typename Globals>
