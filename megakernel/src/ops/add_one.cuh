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
                                              megakernel::state<config> &s) {
            kittens::init_semaphore(s.semaphores()[0], 1); // loaded
            kittens::init_semaphore(s.semaphores()[1], config::NUM_CONSUMER_WARPS); // each consumer completed
            return 2; // 2 semaphores used
        }
    };
    struct loader {
    template <typename globals>
    static __device__ void run(const globals &g, megakernel::state<config> &s) {
        int lane = kittens::laneid();
        if (lane == 0) {
            // logical page 0 holds the input tile
            int pid = s.pid(0);
            s.wait_page_ready(pid); // wait for the previous instruction to free it
            auto &tile = *reinterpret_cast<kittens::st_bf<128, 64> *>(s.pages[pid].data);
            kittens::tma::expect(s.semaphores()[0], tile); // expect one tile's worth of bytes
            kittens::tma::load_async(tile, g.input_weights,
                                     {0, 0, s.instruction()[1], s.instruction()[2]}, // tile row, tile col
                                     s.semaphores()[0]);
        } else if (lane < config::NUM_PAGES) {
            // release the 4 pages this op doesn't use, or the next instruction hangs
            int pid = s.pid(lane);
            s.wait_page_ready(pid);
            s.finish_page(pid, config::NUM_CONSUMER_WARPS);
        }
    }
    };



};