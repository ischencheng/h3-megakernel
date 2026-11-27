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

            constexpr int TOTAL_LOADS = 2 * NUM_CHUNKS; // 2 passes
            constexpr int FIRST_OF_LAST = TOTAL_LOADS - config::NUM_PAGES; // load 79: first of the last 5
            return (FIRST_OF_LAST + query) % config::NUM_PAGES; // Given a queried released order, returns a page
        }
        template <typename globals>
        static __device__ int init_semaphores(const globals &g,
                                              megakernel::state<config> &s) {
            // semaphores [0 .. 4] for the pages being full and ready to use
            // semaphores [5 .. 9] for the pages being empty and ready to load into
            // load each with NUM_CHUNKS * 2 / 5?
            #pragma unroll
            for (int i = 0; i < config::NUM_PAGES; ++i) {
                kittens::init_semaphore(s.semaphores()[i], 1); // loaded
            }
            #pragma unroll
            for (int i = config::NUM_PAGES; i < config::NUM_PAGES * 2; ++i) {
                kittens::init_semaphore(s.semaphores()[i], config::NUM_CONSUMER_WARPS); // computation completed, requires all consumer warps to finish
            }
            return config::NUM_PAGES * 2; // 2 semaphores used per shared memory page
        }
    };

    // loader streams this tile's chunks through the 5 pages as a ring: chunk i -> logical page i % NUM_PAGES
    // semaphores: full[slot] = semaphores()[slot]              (TMA landed; consumers wait on it)
    //             empty[slot] = semaphores()[NUM_PAGES + slot] (all consumer warps done reading; loader waits on it)
    struct loader {
        template <typename globals>
        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            if (kittens::laneid() != 0) return; // one thread issues every TMA load
            const int row_tile = s.instruction()[1]; // same 128 rows for every chunk; only the column tile changes
            for (int i = 0; i < 2 * NUM_CHUNKS; i++) { 
                const int slot = i % config::NUM_PAGES;
                const int page_id = s.pid(slot);
                if (i < config::NUM_PAGES) s.wait_page_ready(page_id); // wait page ready gets ownership of this page from a previous instruction
                else kittens::wait(s.semaphores()[config::NUM_PAGES + slot], (i / config::NUM_PAGES - 1) & 1); // consumers done with chunk i - 5, wait for the empty sempahore indicator.
                // lsb is enough to wait bc you use the semaphores in a ring

                auto &chunk = *reinterpret_cast<kittens::st_bf<128, CHUNK_COLS> *>(s.pages[pid].data);
                kittens::tma::expect(s.semaphores()[slot], chunk); // full[slot] completes when this many bytes land

                // load the relevant 128 row x 64 col chunk from the hidden_t x in globals into the page (shared memory)
                kittens::tma::load_async(chunk, g.x, {0, 0, row_tile, i % NUM_CHUNKS}, s.semaphores()[slot]);
            }
        }
    };

    /* consumer does computation*/
    struct consumer {
        template <typename globals>
        static __device__ void run(const globals &g, state<config> &s) {
            typename kittens::rt_fl<16, 64>::col_vec sum; // one FP32 value per row (16 rows)
            kittens::warp::zero(sum);
            int i = 0;
            for (; i < NUM_CHUNKS; ++i) {
                const int logical_slot = i % config::NUM_PAGES;
                const int page_id = s.pid(logical_slot);
                kittens::wait(s.semaphores()[logical_slot], (i / config::NUM_PAGES) & 1); // chunk i is ready

                auto &chunk = *reinterpret_cast<kittens::st_bf<128, 64> *>(s.pages[page_id].data);
                kittens::rt_fl<16, 64> xf; // this warp's 16 rows of the chunk, 64 cols out of H which fits into the shared page
                kittens::group<config::NUM_CONSUMER_WARPS>::load(xf, chunk); // load the rows for this consumer warp

                kittens::warp::mul(xf, xf, xf); // x^2
                kittens::warp::row_sum(sum, xf, sum); // accumulate by the 16 rows
            }

            // calculate inv = rsqrt(mean(x^2) + eps) per row, store in sum
            // each lane holds 2 rows sums per slot (.x = row lane/4, .y = row lane/4 + 8), so apply to both
            for (int o = 0; o < sum.outer_dim; ++o) {
                for (int n = 0; n < sum.inner_dim; ++n) {
                    sum.data[o][n].x = rsqrtf(sum.data[o][n].x / HIDDEN_DIM + NORM_EPS);
                    sum.data[o][n].y = rsqrtf(sum.data[o][n].y / HIDDEN_DIM + NORM_EPS);
                }
            }

            // here the x^2 sum is complete in sum, now we go to second pass for calculating normed value and doing mod/quant
            for (; i < 2 * NUM_CHUNKS; ++i) {
                const int logical_slot = i % config::NUM_PAGES;
                const int page_id = s.pid(logical_slot);
                kittens::wait(s.semaphores()[logical_slot], (i / config::NUM_PAGES) & 1); // chunk i is ready

                auto &chunk = *reinterpret_cast<kittens::st_bf<128, 64> *>(s.pages[page_id].data);
                kittens::rt_fl<16, 64> xf; // this warp's 16 rows of the chunk, 64 cols out of H which fits into the shared page
                kittens::group<config::NUM_CONSUMER_WARPS>::load(xf, chunk); // load the rows for this consumer warp

                // calculation for second phase: 

            }

        }
    };
};
