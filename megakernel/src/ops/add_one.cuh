#pragma once
#include "../h3.cuh"


struct AddOneOp : megakernel::NoOp<config> {
    static constexpr int opcode = OPCODE_ADD_ONE;

    struct controller {
        template <typename globals>
        static __device__ int
        release_lid(const globals &g,
                    typename config::instruction_t &instruction, int &query) {
            return query;
        }
        template <typename globals>
        static __device__ int init_semaphores(const globals &g,
                                              state<config> &s) {
            kittens::init_semaphore(s.semaphores()[0], 1); // loaded
            kittens::init_semaphore(s.semaphores()[1], config::NUM_CONSUMER_WARPS); // each consumer completed
            return 2; // 2 semaphores used
        }
    };

};