/*
First operation in the DiT block: covers the adaLN norm and fp4 quantization
*/

#pragma once
#include "../h3.cuh"

struct NormModQuantOp : megakernel::NoOp<config> {
    static constexpr int opcode = OPCODE_NORM_MOD_QUANT;

    // controller, gets semaphores and pages
    // every instruction gets fresh array of 32 semaphores
    struct controller {
        template <typename globals>
        static __device__ int
        release_lid(const globals &g,
                    typename config::instruction_t &instruction, int &query) {
            // pages in the order we free them: 1-4 right away, 0 after the store,
            // so the next instruction's logical page 0 is one that's already free
            return (query + 1) % config::NUM_PAGES;
        }
        template <typename globals>
        static __device__ int init_semaphores(const globals &g,
                                              megakernel::state<config> &s) {
            kittens::init_semaphore(s.semaphores()[0], 1); // loaded
            kittens::init_semaphore(s.semaphores()[1], config::NUM_CONSUMER_WARPS); // each consumer completed
            return 2; // 2 semaphores used
        }
    };

    /* consumer does computation*/
    struct consumer {
        template <typename globals>
        static __device__ void run(const globals &g, state<config> &s) {
            kittens::wait(s.semaphores()[0], 0); // wait for loader: first 128x64 chunk is in logical page 0
            auto &chunk = *reinterpret_cast<kittens::st_bf<128, 64> *>(s.pages[s.pid(0)].data);
            kittens::rt_fl<16, 64> xf; // this warp's 16 rows of the chunk, 64 cols out of H which fits into the shared page
            kittens::group<config::NUM_CONSUMER_WARPS>::load(xf, chunk); // warp w gets rows 16w..16w+15, bf16 -> FP32 on load
            kittens::warp::mul(xf, xf, xf); // x^2
            typename kittens::rt_fl<16, 64>::col_vec sum; // one FP32 value per row (16 rows)
            kittens::warp::zero(sum);
            kittens::warp::row_sum(sum, xf, sum); // sum += each row's x^2 over this chunk's 64 columns (2-arg form overwrites)
            // TODO: wrap the load..row_sum in a loop over the 84 chunks (zero sum once, before the loop)
        }
    };
};
