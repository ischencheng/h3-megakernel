#pragma once

// Row-wise DiT operations: RMSNorm with AdaLN modulation and gated residual,
// NVFP4 activation quantization, a generic dense linear, timestep embedding,
// row copies and the Euler latent update. See dit_common.cuh for the shared
// all-warp contract. Instruction words are documented per operation and are
// mirrored by megakernel/h3_dit.py.

#include "dit_common.cuh"
#include <cuda_fp8.h>

namespace h3::dit {

// ---------------------------------------------------------------------------
// RMSNORM_MOD
//   1 rows   2 cols   3 eps bits   4 flags
//   5 x (bf16 [rows, cols]; updated in place when RESIDUAL)
//   6 out    7 norm weight (f32)   8 norm weight offset
//   9 row index (int32 [rows])    10 table (bf16)   11 table row stride
//  12 scale offset  13 shift offset  14 gate offset  15 branch (bf16)
// FastVideo's fused kernels: output = x * rsqrt(mean(x^2) + eps) * weight
// * (1 + scale[index]) + shift[index], with FP32 arithmetic on BF16 inputs.
// RESIDUAL first forms hidden = x + gate[index] * branch and stores it.
// ---------------------------------------------------------------------------
namespace norm_flags {
constexpr int MOD = 1;          // apply the scale/shift tables
constexpr int RESIDUAL = 2;     // hidden = x + gate * branch, stored to x
constexpr int NORM_ROUNDED = 4; // normalize the BF16-rounded hidden state
constexpr int DOUBLE_ROUND = 8; // BF16 product, then BF16 sum (eager BF16)
constexpr int OUT_F32 = 16;     // FP32 output instead of BF16
constexpr int NO_OUTPUT = 32;   // residual update only
constexpr int UNIT_GATE = 64;   // hidden = x + branch (no table lookup)
} // namespace norm_flags

struct rmsnorm_mod_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const int rows = r[1], cols = r[2], flags = r[4];
        const float eps = __int_as_float(r[3]);
        auto *x = buffer<__nv_bfloat16, G>(g, r[5]);
        void *out = (flags & norm_flags::NO_OUTPUT) ? nullptr : buffer<void, G>(g, r[6]);
        const float *weight = buffer<const float, G>(g, r[7]) + r[8];
        const bool gated = (flags & norm_flags::RESIDUAL) && !(flags & norm_flags::UNIT_GATE);
        const int *index = ((flags & norm_flags::MOD) || gated)
                               ? buffer<const int, G>(g, r[9]) : nullptr;
        const __nv_bfloat16 *table =
            index ? buffer<const __nv_bfloat16, G>(g, r[10]) : nullptr;
        const long long table_stride = r[11];
        const __nv_bfloat16 *branch = (flags & norm_flags::RESIDUAL)
                                          ? buffer<const __nv_bfloat16, G>(g, r[15]) : nullptr;
        const int l = lane();
        for (long long row = first_item(); row < rows; row += item_stride()) {
            const long long base = row * cols;
            const __nv_bfloat16 *trow =
                index ? table + static_cast<long long>(index[row]) * table_stride : nullptr;
            auto hidden_at = [&](int c) {
                float h = __bfloat162float(x[base + c]);
                if (flags & norm_flags::RESIDUAL) {
                    const float gate = gated ? __bfloat162float(trow[r[14] + c]) : 1.0f;
                    const float b = __bfloat162float(branch[base + c]);
                    h = (flags & norm_flags::DOUBLE_ROUND) ? h + bf16_round(gate * b)
                                                           : h + gate * b;
                }
                return h;
            };
            float sum = 0.0f;
            if (!(flags & norm_flags::NO_OUTPUT)) {
                for (int c = l; c < cols; c += 32) {
                    float h = hidden_at(c);
                    if (flags & norm_flags::NORM_ROUNDED) h = bf16_round(h);
                    sum += h * h;
                }
                sum = warp_sum(sum);
            }
            const float inv = rsqrtf(sum / cols + eps);
            for (int c = l; c < cols; c += 32) {
                float h = hidden_at(c);
                if (flags & norm_flags::RESIDUAL) x[base + c] = __float2bfloat16_rn(h);
                if (flags & norm_flags::NO_OUTPUT) continue;
                if (flags & norm_flags::NORM_ROUNDED) h = bf16_round(h);
                float value = h * inv * weight[c];
                if (flags & norm_flags::MOD)
                    value = value * (1.0f + __bfloat162float(trow[r[12] + c])) +
                            __bfloat162float(trow[r[13] + c]);
                if (flags & norm_flags::OUT_F32)
                    static_cast<float *>(out)[base + c] = value;
                else
                    static_cast<__nv_bfloat16 *>(out)[base + c] = __float2bfloat16_rn(value);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// NVFP4_QUANT
//   1 rows   2 cols (multiple of 64)   3 global scale bits   4 flags
//   5 input (bf16)   6 packed E2M1 out (u8 [rows, cols / 2])
//   7 scale out (u8, 128x4 swizzled, zero padded by the host)
//   8 gather index (int64 [rows])   9 input row stride in elements
// FlashInfer's default (fast-math) NVFP4 recipe per 16 channels:
//   sf = e4m3(global * amax * rcp(6)); code = e2m1(x * rcp(sf * rcp(global))).
// SWIGLU quantizes value * gate * sigmoid(gate) of packed (value, gate)
// halves after the same BF16 rounding FastVideo's fused SwiGLU stores.
// ---------------------------------------------------------------------------
namespace quant_flags {
constexpr int GATHER = 1;
constexpr int SWIGLU = 2;
} // namespace quant_flags

__device__ inline float rcp_approx_ftz(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Eight FP32 values to four packed E2M1 bytes; element 2j is the low nibble.
__device__ inline uint32_t e2m1x8(const float *v) {
    uint32_t packed;
    asm volatile(
        "{\n"
        ".reg .b8 b0;\n.reg .b8 b1;\n.reg .b8 b2;\n.reg .b8 b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}\n"
        : "=r"(packed)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
          "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return packed;
}

struct nvfp4_quant_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const int rows = r[1], cols = r[2], flags = r[4];
        const float global = __int_as_float(r[3]);
        const float rcp_global = rcp_approx_ftz(global);
        const float rcp_six = rcp_approx_ftz(6.0f);
        const auto *input = buffer<const __nv_bfloat16, G>(g, r[5]);
        auto *packed = buffer<uint32_t, G>(g, r[6]);
        auto *scales = buffer<uint8_t, G>(g, r[7]);
        const long long *gather = (flags & quant_flags::GATHER)
                                      ? buffer<const long long, G>(g, r[8]) : nullptr;
        const long long stride = r[9];
        const int blocks = cols / 16;
        const int k_tiles = (blocks + 3) / 4;
        const int l = lane();
        for (long long row = first_item(); row < rows; row += item_stride()) {
            const long long source = (gather ? gather[row] : row) * stride;
            const long long scale_row =
                ((row / 128) * k_tiles) * 512LL + (row % 32) * 16 + ((row % 128) / 32) * 4;
            for (int block = l; block < blocks; block += 32) {
                float v[16];
                float amax = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int c = block * 16 + i;
                    float value = __bfloat162float(input[source + c]);
                    if (flags & quant_flags::SWIGLU) {
                        const float gate = __bfloat162float(input[source + cols + c]);
                        value = bf16_round(value * (gate * (1.0f / (1.0f + expf(-gate)))));
                    }
                    v[i] = value;
                    amax = fmaxf(amax, fabsf(value));
                }
                const __nv_fp8_e4m3 sf(global * (amax * rcp_six));
                const float sf_value = static_cast<float>(sf);
                const float out_scale =
                    sf_value != 0.0f ? rcp_approx_ftz(sf_value * rcp_global) : 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) v[i] *= out_scale;
                const long long word = row * (cols / 8) + block * 2;
                packed[word] = e2m1x8(v);
                packed[word + 1] = e2m1x8(v + 8);
                scales[scale_row + (block / 4) * 512LL + (block % 4)] = sf.__x;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// DENSE_LINEAR  out[m, n] = sum_k in[m, k] * weight[n, k] + bias[n]
//   1 rows   2 in cols   3 out cols   4 flags
//   5 input   6 weight ([out cols, in cols])   7 bias (-1: none)   8 output
//   9 row index (int64)  10 input row stride  11 output row stride
//  12 output column offset
// FP32 accumulation; one warp owns one output element. Used for the dense
// (non-NVFP4) linears: embedders, time MLP, AdaLN, refiner, output heads.
// ---------------------------------------------------------------------------
namespace linear_flags {
constexpr int IN_BF16 = 1;
constexpr int W_BF16 = 2;
constexpr int OUT_BF16 = 4;
constexpr int BIAS_BF16 = 8;
constexpr int SCATTER = 16; // output row = index[m]
constexpr int GATHER = 32;  // input row = index[m]
} // namespace linear_flags

struct dense_linear_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const int rows = r[1], k_cols = r[2], n_cols = r[3], flags = r[4];
        const void *input = buffer<const void, G>(g, r[5]);
        const void *weight = buffer<const void, G>(g, r[6]);
        const void *bias = r[7] >= 0 ? buffer<const void, G>(g, r[7]) : nullptr;
        void *output = buffer<void, G>(g, r[8]);
        const long long *index = (flags & (linear_flags::SCATTER | linear_flags::GATHER))
                                     ? buffer<const long long, G>(g, r[9]) : nullptr;
        const long long in_stride = r[10], out_stride = r[11];
        const int out_offset = r[12];
        const int l = lane();
        const long long total = static_cast<long long>(rows) * n_cols;
        if (k_cols <= 128) {
            // Narrow inputs (latent patches): one thread owns one output.
            const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
            for (long long item = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
                 item < total; item += stride) {
                const long long m = item / n_cols;
                const int n = static_cast<int>(item - m * n_cols);
                const long long in_base =
                    ((flags & linear_flags::GATHER) ? index[m] : m) * in_stride;
                const long long w_base = static_cast<long long>(n) * k_cols;
                float sum = 0.0f;
                for (int k = 0; k < k_cols; ++k) {
                    const float a = (flags & linear_flags::IN_BF16)
                        ? __bfloat162float(static_cast<const __nv_bfloat16 *>(input)[in_base + k])
                        : static_cast<const float *>(input)[in_base + k];
                    const float w = (flags & linear_flags::W_BF16)
                        ? __bfloat162float(static_cast<const __nv_bfloat16 *>(weight)[w_base + k])
                        : static_cast<const float *>(weight)[w_base + k];
                    sum = fmaf(a, w, sum);
                }
                if (bias)
                    sum += (flags & linear_flags::BIAS_BF16)
                        ? __bfloat162float(static_cast<const __nv_bfloat16 *>(bias)[n])
                        : static_cast<const float *>(bias)[n];
                const long long out_at =
                    ((flags & linear_flags::SCATTER) ? index[m] : m) * out_stride + out_offset + n;
                if (flags & linear_flags::OUT_BF16)
                    static_cast<__nv_bfloat16 *>(output)[out_at] = __float2bfloat16_rn(sum);
                else
                    static_cast<float *>(output)[out_at] = sum;
            }
            return;
        }
        for (long long item = first_item(); item < total; item += item_stride()) {
            const long long m = item / n_cols;
            const int n = static_cast<int>(item - m * n_cols);
            const long long in_base =
                ((flags & linear_flags::GATHER) ? index[m] : m) * in_stride;
            const long long w_base = static_cast<long long>(n) * k_cols;
            float sum = 0.0f;
            for (int k = l; k < k_cols; k += 32) {
                const float a = (flags & linear_flags::IN_BF16)
                    ? __bfloat162float(static_cast<const __nv_bfloat16 *>(input)[in_base + k])
                    : static_cast<const float *>(input)[in_base + k];
                const float w = (flags & linear_flags::W_BF16)
                    ? __bfloat162float(static_cast<const __nv_bfloat16 *>(weight)[w_base + k])
                    : static_cast<const float *>(weight)[w_base + k];
                sum = fmaf(a, w, sum);
            }
            sum = warp_sum(sum);
            if (l != 0) continue;
            if (bias)
                sum += (flags & linear_flags::BIAS_BF16)
                    ? __bfloat162float(static_cast<const __nv_bfloat16 *>(bias)[n])
                    : static_cast<const float *>(bias)[n];
            const long long out_at =
                ((flags & linear_flags::SCATTER) ? index[m] : m) * out_stride + out_offset + n;
            if (flags & linear_flags::OUT_BF16)
                static_cast<__nv_bfloat16 *>(output)[out_at] = __float2bfloat16_rn(sum);
            else
                static_cast<float *>(output)[out_at] = sum;
        }
    }
};

// ---------------------------------------------------------------------------
// ELEMENTWISE
//   1 count   2 kind   3 flags (1: bf16 input, 2: bf16 output)
//   4 input   5 output   6 scalar bits   7 second input
// kind 1: out = silu(in)
// kind 2: rectified-flow update in FastVideo's evaluation order (FP32):
//         denoised = in + scalar * second; out = ratio * in + (1 - ratio)
//         * denoised, with ratio = sigma_next / sigma in word 8.
// kind 3: timestep embedding. count = timesteps * dim; scalar = dim;
//         out[t, :half] = cos(t * f), out[t, half:] = sin(t * f),
//         f_i = exp(-ln(10000) * i / half)   (flip_sin_to_cos, shift 0)
// ---------------------------------------------------------------------------
struct elementwise_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const long long count = r[1];
        const int kind = r[2], flags = r[3];
        const void *input = buffer<const void, G>(g, r[4]);
        void *output = buffer<void, G>(g, r[5]);
        const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
        for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += stride) {
            float value;
            if (kind == 3) {
                const int dim = r[6], half = dim / 2;
                const int t = static_cast<int>(i / dim), c = static_cast<int>(i % dim);
                const int f = c < half ? c : c - half;
                const float angle = static_cast<const float *>(input)[t] *
                    expf(-9.210340371976184f * static_cast<float>(f) / static_cast<float>(half));
                value = c < half ? cosf(angle) : sinf(angle);
            } else {
                value = (flags & 1)
                    ? __bfloat162float(static_cast<const __nv_bfloat16 *>(input)[i])
                    : static_cast<const float *>(input)[i];
                if (kind == 1)
                    value = silu(value);
                else {
                    const float ratio = __int_as_float(r[8]);
                    const float denoised = __fadd_rn(value, __fmul_rn(
                        __int_as_float(r[6]), buffer<const float, G>(g, r[7])[i]));
                    value = __fadd_rn(__fmul_rn(ratio, value),
                                      __fmul_rn(__fsub_rn(1.0f, ratio), denoised));
                }
            }
            if (flags & 2)
                static_cast<__nv_bfloat16 *>(output)[i] = __float2bfloat16_rn(value);
            else
                static_cast<float *>(output)[i] = value;
        }
    }
};

// ---------------------------------------------------------------------------
// COPY_ROWS  out[index[m]] = in[m]  (bf16 rows; index -1: identity)
//   1 rows   2 cols   3 input   4 output   5 scatter index (int64)
//   6 input row stride   7 output row stride   8 flags
// flags 1 (SWIGLU): out[m, c] = bf16(in[m, c] * silu(in[m, cols + c])).
// ---------------------------------------------------------------------------
struct copy_rows_body {
    template <typename G>
    static __device__ inline void execute(const G &g, state &s) {
        const auto &r = s.instruction();
        const int rows = r[1], cols = r[2];
        const auto *input = buffer<const __nv_bfloat16, G>(g, r[3]);
        auto *output = buffer<__nv_bfloat16, G>(g, r[4]);
        const long long *index = r[5] >= 0 ? buffer<const long long, G>(g, r[5]) : nullptr;
        const long long in_stride = r[6], out_stride = r[7];
        const int l = lane();
        for (long long row = first_item(); row < rows; row += item_stride()) {
            const long long out_base = (index ? index[row] : row) * out_stride;
            for (int c = l; c < cols; c += 32) {
                __nv_bfloat16 value = input[row * in_stride + c];
                if (r[8] & 1) {
                    const float gate = __bfloat162float(input[row * in_stride + cols + c]);
                    value = __float2bfloat16_rn(__bfloat162float(value) *
                                                (gate * (1.0f / (1.0f + expf(-gate)))));
                }
                output[out_base + c] = value;
            }
        }
    }
};
} // namespace h3::dit
