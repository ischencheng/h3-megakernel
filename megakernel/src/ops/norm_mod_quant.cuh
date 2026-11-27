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

    // loader streams this tile's chunks through the 5 pages as a ring: chunk i -> logical page i % NUM_PAGES
    // semaphores: full[slot] = semaphores()[slot]              (TMA landed; consumers wait on it)
    //             empty[slot] = semaphores()[NUM_PAGES + slot] (all consumer warps done reading; loader waits on it)
    struct loader {
        template <typename globals>
        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            if (kittens::laneid() != 0) return; // one thread issues every TMA load
            const int row_tile = s.instruction()[1]; // same 128 rows for every chunk; only the column tile changes
            for (int i = 0; i < NUM_CHUNKS; i++) { // TODO: 2 * NUM_CHUNKS once pass 2 reloads the row
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
