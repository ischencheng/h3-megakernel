#pragma once

#include "../h3.cuh"
#include <cstdint>

/*
* accessor for specific area in HBM that the func requires
* use the area from the op instruction to get the index to lookup
* returns pointer to array of type T (used for floats for weights)
*/
template <typename T, typename G>
__device__ inline T *h3_buffer(const G &g, int index) {
    const auto *table = reinterpret_cast<const uint64_t *>(g.buffers.raw_ptr);
    return reinterpret_cast<T *>(table[index]);
}

struct ConsumerRange {
    int first;
    int end;
    int stride;
};

/*
* This helper function splits up the work for each consumer/worker warp based on the index 
*/
__device__ inline ConsumerRange consumer_range(int start, int end) {
    int num_threads = config::NUM_CONSUMER_WARPS * kittens::WARP_THREADS;
    int thread_no = kittens::warpid() * kittens::WARP_THREADS + kittens::laneid();
    return {start + thread_no, end, num_threads};
}

struct SmIdOp : megakernel::NoOp<config> {
    static constexpr int opcode = OPCODE_SM_ID;

    struct consumer {
        template <typename G>
        static __device__ void run(const G &g, megakernel::state<config> &s) {
            if (kittens::warpid() == 0 && kittens::laneid() == 0) {
                h3_buffer<int>(g, s.instruction()[1])[blockIdx.x] = megakernel::get_smid();
            }
        }
    };
};
