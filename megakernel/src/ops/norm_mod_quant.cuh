/*
First operation in the DiT block: covers the adaLN norm and fp4 quantization
*/

#pragma once
#include <cuda_fp4.h> // __nv_cvt_float2_to_fp4x2
#include "../h3.cuh"

struct NormModQuantOp : megakernel::NoOp<config> {
    static constexpr int opcode = OPCODE_NORM_MOD_QUANT;

    // approximate reciprocal, as FlashInfer's reciprocal_approximate_ftz; needed to match its scales bit-exactly
    __device__ static inline float rcp_approx(float a) {
        float b;
        asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
        return b;
    }

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

            // modulation setup, once per instruction: this lane's two rows and the table rows they use.
            // register layout of xf (per 16x16 sub-tile j, per lane): data[0] = top row, cols 2*(lane%4), +1
            //   data[1] = bottom row, same cols; data[2] = top row, cols +8, +9; data[3] = bottom row, cols +8, +9
            //   top row = lane/4, bottom row = lane/4 + 8 (of this warp's 16)
            const int lane = kittens::laneid();
            const int w = kittens::warpid();
            const int warp_row0 = s.instruction()[1] * 128 + 16 * (w / 4 + (w % 4) * 2); // group::load's warp -> rows mapping
            const int row_top = warp_row0 + lane / 4, row_bot = row_top + 8;
            const int S = g.x.rows();
            const int t_top = row_top < S ? g.adaln_index.raw_ptr[row_top] : 0; // rows past S (last tile): any valid table row, output never written
            const int t_bot = row_bot < S ? g.adaln_index.raw_ptr[row_bot] : 0;
            const int shift_off = s.instruction()[2], scale_off = s.instruction()[3]; // column offsets into a table row (norm1: base, base+H; norm2: base+3H, base+4H)
            const long long stride = g.adaln_table.cols();
            const kittens::bf16 *shift_top = g.adaln_table.raw_ptr + t_top * stride + shift_off;
            const kittens::bf16 *scale_top = g.adaln_table.raw_ptr + t_top * stride + scale_off;
            const kittens::bf16 *shift_bot = g.adaln_table.raw_ptr + t_bot * stride + shift_off;
            const kittens::bf16 *scale_bot = g.adaln_table.raw_ptr + t_bot * stride + scale_off;

            // here the x^2 sum is complete in sum, now we go to second pass for calculating normed value and doing mod/quant
            for (; i < 2 * NUM_CHUNKS; ++i) {
                const int logical_slot = i % config::NUM_PAGES;
                const int page_id = s.pid(logical_slot);
                kittens::wait(s.semaphores()[logical_slot], (i / config::NUM_PAGES) & 1); // chunk i is ready

                auto &chunk = *reinterpret_cast<kittens::st_bf<128, 64> *>(s.pages[page_id].data);
                kittens::rt_fl<16, 64> xf; // this warp's 16 rows of the chunk, 64 cols out of H which fits into the shared page
                kittens::group<config::NUM_CONSUMER_WARPS>::load(xf, chunk); // load the rows for this consumer warp

                // calculation for second phase: 
                kittens::warp::mul(xf, xf, sum); // sum = inv rms

                // multiply by the learned weight (final part of rms norm)
                kittens::rt_fl<16, 64>::row_vec wv; // one value per column, laid out to match xf
                kittens::warp::load(wv, g.norm_weight_vec, {0, 0, 0, i - NUM_CHUNKS}); // this chunk's 64 columns, bf16 -> FP32
                kittens::warp::mul_col(xf, xf, wv); // each column * its weight


                // look up table row and scale + shift (modulation) y = y * (1 + scale[t, col]) + shift[t, col]
                const int col0 = (i - NUM_CHUNKS) * CHUNK_COLS; // first column of this chunk
                #pragma unroll
                for (int j = 0; j < xf.width; ++j) { // 4 sub-tiles of 16 columns
                    auto &d = xf.tiles[0][j].data;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int c = col0 + 16 * j + 2 * (lane % 4) + (k >= 2 ? 8 : 0); // d[k].x is column c, d[k].y is c + 1
                        const kittens::bf16 *sc = (k & 1) ? scale_bot : scale_top; // entries for bottom row bc odd k
                        const kittens::bf16 *sh = (k & 1) ? shift_bot : shift_top;
                        d[k].x = d[k].x * (1.0f + __bfloat162float(sc[c]))     + __bfloat162float(sh[c]);
                        d[k].y = d[k].y * (1.0f + __bfloat162float(sc[c + 1])) + __bfloat162float(sh[c + 1]);
                    }
                }

                // round to bf16 then back to float (why does recipe have us do this?)
                kittens::rt_bf<16, 64> yb;
                kittens::warp::copy(yb, xf); // rounds to nearest even
                kittens::warp::copy(xf, yb); // converts back exact

                // find amax (abs value maximum) per row per block (16 col)
                // amax[j].data[0][0].x = top row's amax for block j, .y = bottom row's
                kittens::rt_fl<16, 16> blk;
                kittens::rt_fl<16, 16>::col_vec amax[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    blk.tiles[0][0] = xf.tiles[0][j];
                    kittens::warp::abs(blk, blk);
                    kittens::warp::row_max(amax[j], blk);
                }

                // compute scale and quantize
                // note: check FlashInfer's cvt_warp_fp16_to_fp4
                const float global_sf = __int_as_float(s.instruction()[4]); // whole tensor max, max / scale of output
                uint8_t sf_byte[4][2]; // [block j][top row, bottom row]: e4m3 scale bytes, for the scales write
                uint8_t q_byte[4][4];  // [block j][k] quantized value for the thread
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float out_scale[2];
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) { // h = 0: top row (.x), 1: bottom row (.y)
                        const float vmax = h ? amax[j].data[0][0].y : amax[j].data[0][0].x;
                        const __nv_fp8_e4m3 sf(global_sf * (vmax * rcp_approx(6.0f))); // vmax / 6 (largest e2m1)
                        sf_byte[j][h] = sf.__x;
                        out_scale[h] = sf.__x != 0 ? rcp_approx(float(sf) * rcp_approx(global_sf)) : 0.0f;
                    }
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        float2 v = xf.tiles[0][j].data[k];
                        const float os = out_scale[k & 1]; // odd k: bottom row
                        v.x *= os;
                        v.y *= os;
                        q_byte[j][k] = __nv_cvt_float2_to_fp4x2(v, __NV_E2M1, cudaRoundNearest); // .x (even column) -> low nibble
                    }
                }


            }

        }
    };
};
