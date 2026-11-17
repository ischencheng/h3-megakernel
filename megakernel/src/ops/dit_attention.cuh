#pragma once

// BF16 video-sparse attention for the full DiT instruction set. The
// arithmetic of the Q/K norm, pooling, score, top-k, warp-per-query fine
// attention and coarse compression stages is the arithmetic validated by the
// first-block gates (ops/qknorm_rope.cuh, ops/vsa_*.cuh, ops/sparse_attention
// .cuh); here the tensors come from the pointer table and the geometry from
// the instruction, so the same code serves every block, the text refiner and
// other resolutions. SPARSE_ATTENTION_MMA is the tensor-core fine attention.

#include "dit_common.cuh"
#include <math_constants.h>

namespace h3::dit {
constexpr int HEAD_DIM = 128;
constexpr int TILE = 64;
constexpr float ATTENTION_SCALE = 0.08838834764831845f; // 1 / sqrt(128)

// ---------------------------------------------------------------------------
// QKNORM_TILE
//   1 rows   2 eps bits   3 flags   4 heads   5 rotary channels
//   6 q in   7 k in   8 q out   9 k out  10 q norm weight  11 k norm weight
//  12 cos   13 sin   14 tile index (int64 [rows])
//  15 v in  16 gate in  17 v out  18 gate out
// Per (token, head): RMSNorm over 128 channels, partial RoPE on the leading
// rotary channels, and an optional scatter of Q/K/V/gate to the padded tile
// layout [tile row, head, channel].
// ---------------------------------------------------------------------------
namespace qk_flags {
constexpr int ROPE = 1;
constexpr int TILE_SCATTER = 2;
constexpr int COPY_V = 4;
constexpr int COPY_GATE = 8;
} // namespace qk_flags

struct qknorm_tile_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const int rows = r[1], flags = r[3], heads = r[4], rotary = r[5];
        const float eps = __int_as_float(r[2]);
        const __nv_bfloat16 *inputs[2] = {buffer<const __nv_bfloat16, G>(g, r[6]),
                                          buffer<const __nv_bfloat16, G>(g, r[7])};
        __nv_bfloat16 *outputs[2] = {buffer<__nv_bfloat16, G>(g, r[8]),
                                     buffer<__nv_bfloat16, G>(g, r[9])};
        const __nv_bfloat16 *weights[2] = {buffer<const __nv_bfloat16, G>(g, r[10]),
                                           buffer<const __nv_bfloat16, G>(g, r[11])};
        const bool rope = flags & qk_flags::ROPE;
        const auto *cos = rope ? buffer<const __nv_bfloat16, G>(g, r[12]) : nullptr;
        const auto *sin = rope ? buffer<const __nv_bfloat16, G>(g, r[13]) : nullptr;
        const bool tile = flags & qk_flags::TILE_SCATTER;
        const auto *tile_index = tile ? buffer<const long long, G>(g, r[14]) : nullptr;
        const __nv_bfloat16 *other_in[2] = {
            (flags & qk_flags::COPY_V) ? buffer<const __nv_bfloat16, G>(g, r[15]) : nullptr,
            (flags & qk_flags::COPY_GATE) ? buffer<const __nv_bfloat16, G>(g, r[16]) : nullptr};
        __nv_bfloat16 *other_out[2] = {
            (flags & qk_flags::COPY_V) ? buffer<__nv_bfloat16, G>(g, r[17]) : nullptr,
            (flags & qk_flags::COPY_GATE) ? buffer<__nv_bfloat16, G>(g, r[18]) : nullptr};
        const int l = lane();
        const long long total = static_cast<long long>(rows) * heads;
        for (long long row = first_item(); row < total; row += item_stride()) {
            const long long token = row / heads;
            const long long base = row * HEAD_DIM;
            const long long output_base =
                tile ? (tile_index[token] * heads + row % heads) * HEAD_DIM : base;
            for (int tensor = 0; tensor < 2; ++tensor) {
                float x[4];
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    x[j] = __bfloat162float(inputs[tensor][base + l + 32 * j]);
                    sum += x[j] * x[j];
                }
                const float inv = rsqrtf(warp_sum(sum) / HEAD_DIM + eps);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int channel = l + 32 * j;
                    const float normalized =
                        x[j] * inv * __bfloat162float(weights[tensor][channel]);
                    float result = normalized;
                    if (rope && channel < rotary) {
                        const int partner = channel < rotary / 2 ? channel + rotary / 2
                                                                 : channel - rotary / 2;
                        const float px = __bfloat162float(inputs[tensor][base + partner]);
                        const float pn = px * inv * __bfloat162float(weights[tensor][partner]);
                        const float rotated = channel < rotary / 2 ? -pn : pn;
                        const long long table = token * rotary + channel;
                        result = fmaf(rotated, __bfloat162float(sin[table]),
                                      normalized * __bfloat162float(cos[table]));
                    }
                    outputs[tensor][output_base + channel] = __float2bfloat16_rn(result);
                }
            }
#pragma unroll
            for (int other = 0; other < 2; ++other) {
                if (!other_in[other]) continue;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    other_out[other][output_base + l + 32 * j] =
                        other_in[other][base + l + 32 * j];
            }
        }
    }
};

// ---------------------------------------------------------------------------
// VSA_POOL  FP32 mean of the valid tokens of each (tile, head)
//   1 tiles   2 heads   3 tensor count (2: Q,K  3: Q,K,V)
//   4 valid lengths (int64 [tiles])
//   5..7 inputs (bf16 tiled)   8..10 pooled outputs (f32 [head, tile, 128])
// ---------------------------------------------------------------------------
struct vsa_pool_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const int tiles = r[1], heads = r[2], tensors = r[3];
        const auto *valid = buffer<const long long, G>(g, r[4]);
        const int l = lane();
        const long long total = static_cast<long long>(tiles) * heads;
        for (long long pair = first_item(); pair < total; pair += item_stride()) {
            const int head = static_cast<int>(pair % heads);
            const long long tile = pair / heads;
            const int count = static_cast<int>(valid[tile]);
            const long long out = (static_cast<long long>(head) * tiles + tile) * HEAD_DIM;
            for (int tensor = 0; tensor < tensors; ++tensor) {
                const auto *x = buffer<const __nv_bfloat16, G>(g, r[5 + tensor]);
                auto *pool = buffer<float, G>(g, r[8 + tensor]);
                float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int row = 0; row < count; ++row) {
                    const long long base = ((tile * TILE + row) * heads + head) * HEAD_DIM;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        sum[j] += __bfloat162float(x[base + l + j * 32]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    pool[out + l + j * 32] = count ? sum[j] / count : 0.0f;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// VSA_SCORES  scores[head, q tile, k tile] = <q_pool, k_pool> / sqrt(128)
//   1 tiles   2 heads   3 q pool   4 k pool   5 scores (f32)
// ---------------------------------------------------------------------------
struct vsa_scores_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const long long tiles = r[1], heads = r[2];
        const auto *q = buffer<const float, G>(g, r[3]);
        const auto *k = buffer<const float, G>(g, r[4]);
        auto *scores = buffer<float, G>(g, r[5]);
        const int l = lane();
        const long long total = heads * tiles * tiles;
        for (long long index = first_item(); index < total; index += item_stride()) {
            const long long head = index / (tiles * tiles);
            const long long query_tile = (index / tiles) % tiles;
            const long long key_tile = index % tiles;
            const long long q_base = (head * tiles + query_tile) * HEAD_DIM;
            const long long k_base = (head * tiles + key_tile) * HEAD_DIM;
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int channel = l + j * 32;
                sum = fmaf(q[q_base + channel], k[k_base + channel], sum);
            }
            sum = warp_sum(sum);
            if (l == 0) scores[index] = sum * ATTENTION_SCALE;
        }
    }
};

// ---------------------------------------------------------------------------
// VSA_TOPK  exempt-prefix block mask with one sparse video span
//   1 tiles   2 heads   3 prefix tiles   4 video start   5 video end
//   6 keep   7 scores   8 mask (u8 [head, q tile, k tile])
// Exact radix selection of the keep-th FP32 score; boundary ties resolve
// toward the lower tile ID, as the frozen torch.topk reference does.
// ---------------------------------------------------------------------------
__device__ inline uint32_t ordered_key(float score) {
    const uint32_t bits = __float_as_uint(score);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ inline int warp_count(int value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return __shfl_sync(0xffffffff, value, 0);
}

struct vsa_topk_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const int tiles = r[1], heads = r[2], prefix = r[3];
        const int video_start = r[4], video_end = r[5], keep = r[6];
        const auto *scores = buffer<const float, G>(g, r[7]);
        auto *mask = buffer<uint8_t, G>(g, r[8]);
        const int l = lane();
        const long long total = static_cast<long long>(heads) * tiles;
        for (long long row = first_item(); row < total; row += item_stride()) {
            const int qtile = static_cast<int>(row % tiles);
            const long long base = row * tiles;
            if (qtile < prefix || keep >= video_end - video_start) {
                for (int col = l; col < tiles; col += 32) mask[base + col] = 1;
                continue;
            }
            uint32_t low = 0, high = 0xffffffffu;
            while (low < high) {
                const uint32_t mid = low + ((high - low) >> 1) + 1;
                int count = 0;
                for (int col = video_start + l; col < video_end; col += 32)
                    count += ordered_key(scores[base + col]) >= mid;
                if (warp_count(count) >= keep)
                    low = mid;
                else
                    high = mid - 1;
            }
            int greater = 0, equal = 0;
            for (int col = video_start + l; col < video_end; col += 32) {
                const uint32_t key = ordered_key(scores[base + col]);
                greater += key > low;
                equal += key == low;
            }
            greater = warp_count(greater);
            equal = warp_count(equal);
            int tie_cutoff = video_end - 1;
            if (greater + equal > keep) {
                int left = video_start, right = video_end - 1;
                const int ties_to_keep = keep - greater;
                while (left < right) {
                    const int mid = left + (right - left) / 2;
                    int before = 0;
                    for (int col = video_start + l; col <= mid; col += 32)
                        before += ordered_key(scores[base + col]) == low;
                    if (warp_count(before) >= ties_to_keep)
                        right = mid;
                    else
                        left = mid + 1;
                }
                tie_cutoff = left;
            }
            for (int col = l; col < tiles; col += 32) {
                const uint32_t key = ordered_key(scores[base + col]);
                mask[base + col] = static_cast<uint8_t>(
                    col < prefix || (col >= video_start && col < video_end &&
                                     (key > low || (key == low && col <= tie_cutoff))));
            }
        }
    }
};

// ---------------------------------------------------------------------------
// SPARSE_ATTENTION (warp per query row; the validated correctness baseline)
// SPARSE_ATTENTION_MMA (tensor cores; four warps own one 64-row query tile)
//   1 tiles   2 heads   3 q   4 k   5 v (bf16 tiled [row, head, 128])
//   6 mask (u8 [head, q tile, k tile])   7 valid key lengths (int64 [tiles])
//   8 output (bf16 tiled)
// Padded keys are masked; padded query rows are still computed, as FastVideo
// does. Online softmax with FP32 accumulation.
// ---------------------------------------------------------------------------
struct sparse_attention_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const long long tiles = r[1], heads = r[2];
        const auto *q = buffer<const __nv_bfloat16, G>(g, r[3]);
        const auto *k = buffer<const __nv_bfloat16, G>(g, r[4]);
        const auto *v = buffer<const __nv_bfloat16, G>(g, r[5]);
        const auto *mask = buffer<const uint8_t, G>(g, r[6]);
        const auto *valid = buffer<const long long, G>(g, r[7]);
        auto *output = buffer<__nv_bfloat16, G>(g, r[8]);
        const long long seq = tiles * TILE;
        const int l = lane();
        for (long long row = first_item(); row < heads * seq; row += item_stride()) {
            const long long head = row / seq;
            const long long query = row - head * seq;
            const long long qtile = query / TILE;
            const long long qbase = (query * heads + head) * HEAD_DIM;
            float qv[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int j = 0; j < 4; ++j) qv[j] = __bfloat162float(q[qbase + l + j * 32]);
            float maximum = -CUDART_INF_F;
            float denominator = 0.0f;
            for (long long ktile = 0; ktile < tiles; ++ktile) {
                if (!mask[(head * tiles + qtile) * tiles + ktile]) continue;
                const int kcount = static_cast<int>(valid[ktile]);
                for (int kr = 0; kr < kcount; ++kr) {
                    const long long kbase = ((ktile * TILE + kr) * heads + head) * HEAD_DIM;
                    float partial = 0.0f;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        partial = fmaf(qv[j], __bfloat162float(k[kbase + l + j * 32]), partial);
                    const float score = warp_sum(partial) * ATTENTION_SCALE;
                    const float next_max = fmaxf(maximum, score);
                    const float old_weight = expf(maximum - next_max);
                    const float new_weight = expf(score - next_max);
                    denominator = denominator * old_weight + new_weight;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float value = __bfloat162float(v[kbase + l + j * 32]);
                        acc[j] = fmaf(new_weight, value, acc[j] * old_weight);
                    }
                    maximum = next_max;
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
                output[qbase + l + j * 32] = __float2bfloat16_rn(
                    denominator > 0.0f ? acc[j] / denominator : 0.0f);
        }
    }
};

// D(16x8, f32) += A(16x16, bf16, row major) * B(16x8, bf16, column major).
// Fragment layout for lane = 4 * group + t:
//   a0: A[group][2t, 2t+1]      a1: A[group+8][2t, 2t+1]
//   a2: A[group][2t+8, 2t+9]    a3: A[group+8][2t+8, 2t+9]
//   b0: B[2t, 2t+1][group]      b1: B[2t+8, 2t+9][group]
//   c0, c1: C[group][2t, 2t+1]  c2, c3: C[group+8][2t, 2t+1]
__device__ inline void mma_bf16(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ inline uint32_t pack_bf16(float low, float high) {
    return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(low))) |
           (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(high))) << 16);
}

__device__ inline uint32_t load_pair(const __nv_bfloat16 *p) {
    return *reinterpret_cast<const uint32_t *>(p);
}

__device__ inline uint32_t load_shared(uint32_t address) {
    uint32_t value;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(address));
    return value;
}

__device__ inline void load_shared4(uint32_t *value, uint32_t address) {
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
                 : "r"(address));
}

__device__ inline float group_max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
    return fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
}

__device__ inline float group_sum(float value) {
    value += __shfl_xor_sync(0xffffffff, value, 1);
    return value + __shfl_xor_sync(0xffffffff, value, 2);
}

// Tensor-core fine attention. Four warps form a group that owns one
// (head, query tile); each warp holds 16 of its 64 query rows in registers.
// For every selected key tile the group stages K, then V (transposed), in one
// 16-KiB page of the CTA's dynamic shared region with coalesced 16-byte
// loads, so the tile is fetched once per query tile instead of once per
// warp and the MMA operands come from shared memory. The three groups of a
// CTA use named barriers 1..3; barrier 0 stays the CTA barrier.
__device__ inline void group_sync(int group) {
    asm volatile("bar.sync %0, 128;" : : "r"(group + 1) : "memory");
}

struct sparse_attention_mma_body {
    static constexpr int GROUPS = WARPS / 4;
    static_assert(GROUPS * 16384 <= config::NUM_PAGES * config::PAGE_SIZE);

    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const long long tiles = r[1], heads = r[2];
        const auto *q = buffer<const __nv_bfloat16, G>(g, r[3]);
        const auto *k = buffer<const __nv_bfloat16, G>(g, r[4]);
        const auto *v = buffer<const __nv_bfloat16, G>(g, r[5]);
        const auto *mask = buffer<const uint8_t, G>(g, r[6]);
        const auto *valid = buffer<const long long, G>(g, r[7]);
        auto *output = buffer<__nv_bfloat16, G>(g, r[8]);
        // Word 9: diagnostic stage mask for timing (0 in every recipe).
        const int skip = r[9];
        const long long row_stride = heads * HEAD_DIM;
        const int group_id = warp() / 4;
        const int member = threadIdx.x - group_id * 128; // 0..127 within the group
        const int quarter = member >> 5;
        const int group = lane() >> 2, t = lane() & 3;
        // The page holds the K, then the V, tile in MMA fragment order.
        auto *stage = reinterpret_cast<__nv_bfloat16 *>(
            reinterpret_cast<char *>(&s.pages) + group_id * 16384);
        const uint32_t stage_shared = static_cast<uint32_t>(__cvta_generic_to_shared(stage));
        const long long total = heads * tiles;
        for (long long unit = static_cast<long long>(blockIdx.x) * GROUPS + group_id;
             unit < total; unit += static_cast<long long>(gridDim.x) * GROUPS) {
            const long long head = unit / tiles;
            const long long qtile = unit - head * tiles;
            const long long q0 = qtile * TILE + quarter * 16;
            const long long row0 = ((q0 + group) * heads + head) * HEAD_DIM;
            const long long row1 = row0 + 8 * row_stride;
            // The megakernel CTA leaves 168 registers per thread. Q fragments
            // are therefore re-read per channel chunk (16 rows stay cache
            // resident across the key tiles) instead of held in registers.
            const __nv_bfloat16 *q_even = q + row0 + 2 * t, *q_odd = q + row1 + 2 * t;
            float o[16][4];
#pragma unroll
            for (int dn = 0; dn < 16; ++dn)
                o[dn][0] = o[dn][1] = o[dn][2] = o[dn][3] = 0.0f;
            float max0 = -CUDART_INF_F, max1 = -CUDART_INF_F, sum0 = 0.0f, sum1 = 0.0f;
            const uint8_t *mask_row = mask + (head * tiles + qtile) * tiles;
            for (long long ktile = 0; ktile < tiles; ++ktile) {
                if (!mask_row[ktile]) continue;
                const int count = static_cast<int>(valid[ktile]);
                if (count <= 0 || (skip & 32)) continue;
                const long long key0 = (ktile * TILE * heads + head) * HEAD_DIM;
                // Stage K: 1024 chunks of 8 channels, 8 per thread.
                if (!(skip & 1))
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int chunk = member + 128 * j;
                    const int key = chunk >> 4, part = chunk & 15;
                    const uint4 word =
                        *reinterpret_cast<const uint4 *>(k + key0 + key * row_stride + part * 8);
                    // Fragment order [channel chunk][lane][key chunk][b0|b1]:
                    // channels 8*part + 2j, +1 are operand b(part & 1) of
                    // lane 4 * (key & 7) + j for (kk = part / 2, nc = key / 8).
                    uint32_t *dest = reinterpret_cast<uint32_t *>(stage) +
                        (((part >> 1) * 32 + (key & 7) * 4) * 8 + (key >> 3)) * 2 + (part & 1);
                    dest[0] = word.x;
                    dest[16] = word.y;
                    dest[32] = word.z;
                    dest[48] = word.w;
                }
                if (!(skip & 64)) group_sync(group_id);
                float sc[8][4];
                float tile_max0 = -CUDART_INF_F, tile_max1 = -CUDART_INF_F;
#pragma unroll
                for (int nc = 0; nc < 8; ++nc)
                    sc[nc][0] = sc[nc][1] = sc[nc][2] = sc[nc][3] = 0.0f;
                if (!(skip & 2))
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    uint32_t qa[4];
                    qa[0] = load_pair(q_even + kk * 16);
                    qa[1] = load_pair(q_odd + kk * 16);
                    qa[2] = load_pair(q_even + kk * 16 + 8);
                    qa[3] = load_pair(q_odd + kk * 16 + 8);
                    uint32_t kb[16];
#pragma unroll
                    for (int m = 0; m < 4; ++m)
                        load_shared4(kb + 4 * m, stage_shared + ((kk * 32 + lane()) * 16 + 4 * m) * 4);
#pragma unroll
                    for (int nc = 0; nc < 8; ++nc)
                        mma_bf16(sc[nc], qa, kb[2 * nc], kb[2 * nc + 1]);
                }
#pragma unroll
                for (int nc = 0; nc < 8; ++nc) {
                    sc[nc][0] *= ATTENTION_SCALE;
                    sc[nc][1] *= ATTENTION_SCALE;
                    sc[nc][2] *= ATTENTION_SCALE;
                    sc[nc][3] *= ATTENTION_SCALE;
                    if (count < TILE) { // partial tile: mask the padded keys
                        const int key = nc * 8 + 2 * t;
                        if (key >= count) sc[nc][0] = sc[nc][2] = -CUDART_INF_F;
                        if (key + 1 >= count) sc[nc][1] = sc[nc][3] = -CUDART_INF_F;
                    }
                    tile_max0 = fmaxf(tile_max0, fmaxf(sc[nc][0], sc[nc][1]));
                    tile_max1 = fmaxf(tile_max1, fmaxf(sc[nc][2], sc[nc][3]));
                }
                // Every warp of the group has read K; the page may now take V.
                if (!(skip & 64)) group_sync(group_id);
                if (!(skip & 4))
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int chunk = member + 128 * j;
                    const int key = chunk >> 4, part = chunk & 15;
                    const uint4 word =
                        *reinterpret_cast<const uint4 *>(v + key0 + key * row_stride + part * 8);
                    const auto *channels = reinterpret_cast<const __nv_bfloat16 *>(&word);
                    // Fragment order [channel chunk dn][lane][ks][b0|b1][half]:
                    // channel 8*part + d of this key is operand b((key & 15)
                    // / 8) of lane 4 * d + (key & 7) / 2 for (dn = part,
                    // ks = key / 16).
                    __nv_bfloat16 *dest = stage +
                        ((part * 32 + ((key & 7) >> 1)) * 8 + (key >> 4) * 2 + ((key & 15) >> 3)) * 2 +
                        (key & 1);
#pragma unroll
                    for (int d = 0; d < 8; ++d) dest[d * 4 * 16] = channels[d];
                }
                const float next0 = fmaxf(max0, group_max(tile_max0));
                const float next1 = fmaxf(max1, group_max(tile_max1));
                // count > 0 makes next finite; an empty history has max = -inf
                // and exp(-inf) = 0 discards its zero accumulators exactly.
                const float decay0 = expf(max0 - next0), decay1 = expf(max1 - next1);
                float tile_sum0 = 0.0f, tile_sum1 = 0.0f;
                if (!(skip & 8))
#pragma unroll
                for (int nc = 0; nc < 8; ++nc) {
                    sc[nc][0] = expf(sc[nc][0] - next0);
                    sc[nc][1] = expf(sc[nc][1] - next0);
                    sc[nc][2] = expf(sc[nc][2] - next1);
                    sc[nc][3] = expf(sc[nc][3] - next1);
                    tile_sum0 += sc[nc][0] + sc[nc][1];
                    tile_sum1 += sc[nc][2] + sc[nc][3];
                }
                sum0 = sum0 * decay0 + group_sum(tile_sum0);
                sum1 = sum1 * decay1 + group_sum(tile_sum1);
                max0 = next0;
                max1 = next1;
                // The running maxima rarely move after the first tiles; the
                // 64 accumulators are rescaled only when some row's did.
                if (__any_sync(0xffffffff, decay0 != 1.0f || decay1 != 1.0f)) {
#pragma unroll
                    for (int dn = 0; dn < 16; ++dn) {
                        o[dn][0] *= decay0;
                        o[dn][1] *= decay0;
                        o[dn][2] *= decay1;
                        o[dn][3] *= decay1;
                    }
                }
                if (!(skip & 64)) group_sync(group_id);
                uint32_t pa[4][4];
#pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    pa[ks][0] = pack_bf16(sc[2 * ks][0], sc[2 * ks][1]);
                    pa[ks][1] = pack_bf16(sc[2 * ks][2], sc[2 * ks][3]);
                    pa[ks][2] = pack_bf16(sc[2 * ks + 1][0], sc[2 * ks + 1][1]);
                    pa[ks][3] = pack_bf16(sc[2 * ks + 1][2], sc[2 * ks + 1][3]);
                }
                if (!(skip & 16))
#pragma unroll
                for (int dn = 0; dn < 16; ++dn) {
                    uint32_t vb[8];
                    load_shared4(vb, stage_shared + ((dn * 32 + lane()) * 8) * 4);
                    load_shared4(vb + 4, stage_shared + ((dn * 32 + lane()) * 8 + 4) * 4);
#pragma unroll
                    for (int ks = 0; ks < 4; ++ks)
                        mma_bf16(o[dn], pa[ks], vb[2 * ks], vb[2 * ks + 1]);
                }
                // V has been consumed before the next key tile overwrites it.
                if (!(skip & 64)) group_sync(group_id);
            }
            const float inv0 = sum0 > 0.0f ? 1.0f / sum0 : 0.0f;
            const float inv1 = sum1 > 0.0f ? 1.0f / sum1 : 0.0f;
#pragma unroll
            for (int dn = 0; dn < 16; ++dn) {
                *reinterpret_cast<uint32_t *>(output + row0 + dn * 8 + 2 * t) =
                    pack_bf16(o[dn][0] * inv0, o[dn][1] * inv0);
                *reinterpret_cast<uint32_t *>(output + row1 + dn * 8 + 2 * t) =
                    pack_bf16(o[dn][2] * inv1, o[dn][3] * inv1);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// VSA_COMPRESS  out += bf16(softmax(scores) @ v_pool) * gate   (per tile)
//   1 tiles   2 heads   3 scores   4 v pool   5 gate (bf16 tiled)
//   6 output (bf16 tiled, holds the fine attention result)
// ---------------------------------------------------------------------------
struct vsa_compress_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const long long tiles = r[1], heads = r[2];
        const auto *scores = buffer<const float, G>(g, r[3]);
        const auto *v_pool = buffer<const float, G>(g, r[4]);
        const auto *gate = buffer<const __nv_bfloat16, G>(g, r[5]);
        auto *out = buffer<__nv_bfloat16, G>(g, r[6]);
        const int l = lane();
        for (long long pair = first_item(); pair < tiles * heads; pair += item_stride()) {
            const long long head = pair % heads;
            const long long tile = pair / heads;
            const long long scores_base = (head * tiles + tile) * tiles;
            float maximum = -CUDART_INF_F;
            for (long long key = 0; key < tiles; ++key)
                maximum = fmaxf(maximum, scores[scores_base + key]);
            float denom = 0.0f;
            for (long long key = 0; key < tiles; ++key)
                denom += expf(scores[scores_base + key] - maximum);
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (long long key = 0; key < tiles; ++key) {
                const float probability = expf(scores[scores_base + key] - maximum) / denom;
                const long long value_base = (head * tiles + key) * HEAD_DIM;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[j] = fmaf(probability, v_pool[value_base + l + j * 32], acc[j]);
            }
            float coarse[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) coarse[j] = bf16_round(acc[j]);
            for (int row = 0; row < TILE; ++row) {
                const long long base = ((tile * TILE + row) * heads + head) * HEAD_DIM;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int channel = l + j * 32;
                    const float fine = __bfloat162float(out[base + channel]);
                    const float gval = __bfloat162float(gate[base + channel]);
                    out[base + channel] = __float2bfloat16_rn(fmaf(coarse[j], gval, fine));
                }
            }
        }
    }
};
} // namespace h3::dit
