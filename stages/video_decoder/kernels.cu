// Fused kernels for the H3 video decoder blocks.
//
// FastVideo runs each INT8 ConvRot linear of a decoder block as
//   fp16 GEMM with the 256x256 Hadamard (rotation)
//   row amax, scale and round to int8 (quantize)
//   int8 GEMM to an int32 matrix (cuBLAS)
//   int32 * x_scale * w_scale + bias, then whatever follows (inductor)
// Here the norm, rotation and quantize of a row are one kernel (rowquant), and the dequant and what follows
// it are the epilogue of the int8 GEMM, so the fp16 rotation output and the int32 matrix never reach memory.
// The GEMM mainloop is CUTLASS's SM80 int8 mainloop with the 128x128x64 tile cuBLAS picks for these shapes.

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/layout/matrix.h"

namespace h3mk {

// ---------------------------------------------------------------------------------------------------------------
// rowquant: optional RMSNorm, then the Hadamard rotation in groups of 256, then per-row int8 quantization
// ---------------------------------------------------------------------------------------------------------------

constexpr int kRowThreads = 128;
constexpr int kChunk = 16;  // consecutive elements per thread, two base-4 digits of a 256 group

// h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]: y_a = sum_b h4[a][b] x_b = (x0+x1+x2+x3) - 2 x_(3-a)
__device__ __forceinline__ void h4(float& x0, float& x1, float& x2, float& x3) {
    float s = (x0 + x1) + (x2 + x3);
    float y0 = s - 2.f * x3, y1 = s - 2.f * x2, y2 = s - 2.f * x1, y3 = s - 2.f * x0;
    x0 = y0; x1 = y1; x2 = y2; x3 = y3;
}

// the same transform across the four lanes that differ in the bits of `step` (1 or 4) times 0..3
__device__ __forceinline__ float h4_lanes(float x, int lane, int shift) {
    int a = (lane >> shift) & 3, base = lane & ~(3 << shift);
    float v0 = __shfl_sync(0xffffffffu, x, base);
    float v1 = __shfl_sync(0xffffffffu, x, base | (1 << shift));
    float v2 = __shfl_sync(0xffffffffu, x, base | (2 << shift));
    float v3 = __shfl_sync(0xffffffffu, x, base | (3 << shift));
    float s = (v0 + v1) + (v2 + v3);
    float far = a == 0 ? v3 : a == 1 ? v2 : a == 2 ? v1 : v0;
    return s - 2.f * far;
}

template <typename T>
__device__ __forceinline__ void load16(const T* p, float* v);

template <>
__device__ __forceinline__ void load16<float>(const float* p, float* v) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float4 f = reinterpret_cast<const float4*>(p)[i];
        v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w;
    }
}

template <>
__device__ __forceinline__ void load16<__half>(const __half* p, float* v) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        uint4 u = reinterpret_cast<const uint4*>(p)[i];
        const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(h[j]);
            v[8 * i + 2 * j] = f.x; v[8 * i + 2 * j + 1] = f.y;
        }
    }
}

__device__ __forceinline__ float block_sum(float x, float* red) {
#pragma unroll
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    int warp = threadIdx.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[warp] = x;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kRowThreads / 32; ++w) total += red[w];
    return total;
}

__device__ __forceinline__ float block_max(float x, float* red) {
#pragma unroll
    for (int o = 16; o; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
    int warp = threadIdx.x >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[warp] = x;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kRowThreads / 32; ++w) total = fmaxf(total, red[w]);
    return total;
}

// One CTA per row. Each thread holds kChunk consecutive elements of a 256 group per step; the 16 lanes of a
// half-warp hold the 16 chunks of one group, so the two high base-4 digits of the rotation are lane shuffles.
// FastVideo's arithmetic, in the eager dtypes: norm in fp32 rounded to fp16 (autocast feeds the rotation
// GEMM fp16), the rotation summed in fp32 and rounded to fp16, scale = fp16(amax / 127), q = round(fp16(x / scale)).
template <typename T, int K, bool kNorm>
__global__ void __launch_bounds__(kRowThreads) rowquant_kernel(const T* __restrict__ x, const float* __restrict__ w,
                                                              float eps, int8_t* __restrict__ q,
                                                              float* __restrict__ scale) {
    constexpr int kSteps = K / kChunk / kRowThreads;
    static_assert(K % (kChunk * kRowThreads) == 0, "row length");
    __shared__ float red[kRowThreads / 32];
    const int row = blockIdx.x, lane = threadIdx.x & 31;
    const T* src = x + static_cast<size_t>(row) * K;
    float v[kSteps][kChunk];
#pragma unroll
    for (int s = 0; s < kSteps; ++s) load16<T>(src + (s * kRowThreads + threadIdx.x) * kChunk, v[s]);

    if constexpr (kNorm) {
        float ss = 0.f;
#pragma unroll
        for (int s = 0; s < kSteps; ++s)
#pragma unroll
            for (int i = 0; i < kChunk; ++i) ss += v[s][i] * v[s][i];
        float r = rsqrtf(block_sum(ss, red) / K + eps);
#pragma unroll
        for (int s = 0; s < kSteps; ++s) {
            const float* ws = w + (s * kRowThreads + threadIdx.x) * kChunk;
#pragma unroll
            for (int i = 0; i < kChunk; ++i) v[s][i] = __half2float(__float2half_rn(__fmul_rn(__fmul_rn(v[s][i], r), ws[i])));
        }
    } else {
#pragma unroll
        for (int s = 0; s < kSteps; ++s)
#pragma unroll
            for (int i = 0; i < kChunk; ++i) v[s][i] = __half2float(__float2half_rn(v[s][i]));
    }

    float amax = 0.f;
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        float* e = v[s];
#pragma unroll
        for (int i = 0; i < 4; ++i) h4(e[i], e[4 + i], e[8 + i], e[12 + i]);           // digit 1 (stride 4)
#pragma unroll
        for (int i = 0; i < 4; ++i) h4(e[4 * i], e[4 * i + 1], e[4 * i + 2], e[4 * i + 3]);  // digit 0 (stride 1)
#pragma unroll
        for (int i = 0; i < kChunk; ++i) {
            float y = h4_lanes(e[i], lane, 0);   // digit 2: lanes differing in bits 0-1
            y = h4_lanes(y, lane, 2);            // digit 3: lanes differing in bits 2-3
            e[i] = __half2float(__float2half_rn(y * 0.0625f));  // the 1/16 normalization is exact
            amax = fmaxf(amax, fabsf(e[i]));
        }
    }
    amax = block_max(amax, red);
    const float s16 = __half2float(__float2half_rn(amax / 127.f));
    int8_t* dst = q + static_cast<size_t>(row) * K;
#pragma unroll
    for (int s = 0; s < kSteps; ++s) {
        alignas(16) int8_t out[kChunk];
#pragma unroll
        for (int i = 0; i < kChunk; ++i) {
            float t = s16 > 0.f ? rintf(__half2float(__float2half_rn(__fdiv_rn(v[s][i], s16)))) : 0.f;
            out[i] = static_cast<int8_t>(fminf(fmaxf(t, -128.f), 127.f));
        }
        *reinterpret_cast<int4*>(dst + (s * kRowThreads + threadIdx.x) * kChunk) = *reinterpret_cast<int4*>(out);
    }
    if (threadIdx.x == 0) scale[row] = s16;
}

// ---------------------------------------------------------------------------------------------------------------
// int8 GEMM, D = A[M,K] * W[N,K]^T, CUTLASS SM80 mainloop, epilogue on the accumulator registers
// ---------------------------------------------------------------------------------------------------------------

constexpr int kTileM = 128, kTileN = 128, kTileK = 64, kGemmThreads = 128;

template <int kStages>
struct Gemm {
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
        int8_t, cutlass::layout::RowMajor, 16, int8_t, cutlass::layout::ColumnMajor, 16, int32_t,
        cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<kTileM, kTileN, kTileK>, cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>, kStages, cutlass::arch::OpMultiplyAddSaturate>::ThreadblockMma;
    static constexpr int kSmem = sizeof(typename Mma::SharedStorage);
};

struct Operands {
    const int8_t* a;      // [M, K] quantized rows
    const float* a_scale; // [M]
    const int8_t* w;      // [N, K]
    const float* w_scale; // [N]
    const float* bias;    // [N]
    int m, n, k;
};

// dequant exactly as FastVideo's eager path: (acc * x_scale) * w_scale + bias, each rounded
__device__ __forceinline__ float dq(int acc, float xs, float ws, float b) {
    return __fadd_rn(__fmul_rn(__fmul_rn(static_cast<float>(acc), xs), ws), b);
}

// q, k: RMSNorm over the 64-channel head without affine, then rotate-half RoPE on the first 48 channels with
// the eager rounding (a*cos + rotated*sin); v: dequant and bias. All three are written fp16 for the attention.
struct QkvEpilogue {
    __half* q; __half* k; __half* v;  // [M, 2048] each
    const float* cos; const float* sin;  // [M, 48]
    float eps;
};

// x += (dequant) * gate, the block's gated residual; x is the fp32 residual stream [M, N]
struct ResidualEpilogue {
    float* x;
    const float* gate;  // [N]
};

// rows of W were interleaved so that output 2j is the value half and 2j+1 the gate half of channel j
struct SwigluEpilogue {
    __half* h;  // [M, N / 2]
};

constexpr int kWarpRows = 4, kWarpCols = 8;  // 16x8 mma tiles per 64x64 warp tile

template <typename Mma, typename Epi>
__global__ void __launch_bounds__(kGemmThreads) int8_gemm_kernel(Operands op, typename Mma::IteratorA::Params pa,
                                                               typename Mma::IteratorB::Params pb, Epi epi) {
    extern __shared__ __align__(16) char smem[];
    auto& storage = *reinterpret_cast<typename Mma::SharedStorage*>(smem);
    const int tile_n = blockIdx.x, tile_m = blockIdx.y;
    const int warp = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0), lane = threadIdx.x % 32;

    typename Mma::IteratorA it_a(pa, const_cast<int8_t*>(op.a), {op.m, op.k}, threadIdx.x, {tile_m * kTileM, 0});
    typename Mma::IteratorB it_b(pb, const_cast<int8_t*>(op.w), {op.k, op.n}, threadIdx.x, {0, tile_n * kTileN});
    Mma mma(storage, threadIdx.x, warp, lane);
    typename Mma::FragmentC acc;
    acc.clear();
    mma((op.k + kTileK - 1) / kTileK, acc, it_a, it_b, acc);

    // accumulator (m, n) of the warp is acc[(m + n * kWarpRows) * 4 + i]; i = 0, 1 are row g, columns 2t and
    // 2t + 1 of the 16x8 tile, i = 2, 3 the same columns of row g + 8
    const int g = lane >> 2, t = lane & 3;
    const int row0 = tile_m * kTileM + (warp % 2) * 64 + g;
    const int col0 = tile_n * kTileN + (warp / 2) * 64 + 2 * t;

    if constexpr (std::is_same_v<Epi, QkvEpilogue>) {
        const int part = col0 / 2048;  // the whole CTA is in one of q, k, v
        __half* out = part == 0 ? epi.q : part == 1 ? epi.k : epi.v;
        const int head_col0 = col0 % 2048 - 2 * t;  // first channel of this warp's head
        float ws[kWarpCols][2], wb[kWarpCols][2];
#pragma unroll
        for (int n = 0; n < kWarpCols; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                ws[n][e] = op.w_scale[col0 + n * 8 + e];
                wb[n][e] = op.bias[col0 + n * 8 + e];
            }
#pragma unroll
        for (int m = 0; m < kWarpRows; ++m)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int r = row0 + m * 16 + half * 8;
                const bool live = r < op.m;
                const float xs = live ? op.a_scale[r] : 0.f;
                float val[kWarpCols][2];
                float ss = 0.f;
#pragma unroll
                for (int n = 0; n < kWarpCols; ++n)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        val[n][e] = dq(acc[(m + n * kWarpRows) * 4 + half * 2 + e], xs, ws[n][e], wb[n][e]);
                        ss += val[n][e] * val[n][e];
                    }
                if (part < 2) {
                    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
                    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
                    const float rn = rsqrtf(ss / 64.f + epi.eps);
                    float nv[kWarpCols][2];
#pragma unroll
                    for (int n = 0; n < kWarpCols; ++n)
#pragma unroll
                        for (int e = 0; e < 2; ++e) nv[n][e] = __fmul_rn(val[n][e], rn);
                    // channels n * 8 + 2t + e: tiles n < 3 are the first rotary half, 3 <= n < 6 the second
#pragma unroll
                    for (int n = 0; n < kWarpCols; ++n) {
                        if (n < 6 && live) {
                            const size_t at = static_cast<size_t>(r) * 48 + n * 8 + 2 * t;
                            const float2 c = *reinterpret_cast<const float2*>(epi.cos + at);
                            const float2 s = *reinterpret_cast<const float2*>(epi.sin + at);
                            const float cs[2] = {c.x, c.y}, sn[2] = {s.x, s.y};
#pragma unroll
                            for (int e = 0; e < 2; ++e) {
                                // rotated = cat(-second, first) over the 48 rotary channels
                                const float other = n < 3 ? -nv[n + 3 < kWarpCols ? n + 3 : n][e]
                                                          : nv[n >= 3 ? n - 3 : n][e];
                                val[n][e] = __fadd_rn(__fmul_rn(nv[n][e], cs[e]), __fmul_rn(other, sn[e]));
                            }
                        } else {
                            val[n][0] = nv[n][0];
                            val[n][1] = nv[n][1];
                        }
                    }
                }
                if (live) {
#pragma unroll
                    for (int n = 0; n < kWarpCols; ++n)
                        *reinterpret_cast<__half2*>(out + static_cast<size_t>(r) * 2048 + head_col0 + n * 8 + 2 * t) =
                            __floats2half2_rn(val[n][0], val[n][1]);
                }
            }
    } else {
        // this thread's 16 columns: their scales, biases and gates, loaded once
        float ws[kWarpCols][2], wb[kWarpCols][2], gt[kWarpCols][2];
#pragma unroll
        for (int n = 0; n < kWarpCols; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int c = col0 + n * 8 + e;
                ws[n][e] = op.w_scale[c];
                wb[n][e] = op.bias[c];
                if constexpr (std::is_same_v<Epi, ResidualEpilogue>) gt[n][e] = epi.gate[c];
            }
#pragma unroll
        for (int m = 0; m < kWarpRows; ++m)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int r = row0 + m * 16 + half * 8;
                if (r >= op.m) continue;
                const float xs = op.a_scale[r];
#pragma unroll
                for (int n = 0; n < kWarpCols; ++n) {
                    const int c = col0 + n * 8;
                    // index acc directly: taking an element's address would move the accumulators to local memory
                    const int i = (m + n * kWarpRows) * 4 + half * 2;
                    const float v0 = dq(acc[i], xs, ws[n][0], wb[n][0]);
                    const float v1 = dq(acc[i + 1], xs, ws[n][1], wb[n][1]);
                    if constexpr (std::is_same_v<Epi, ResidualEpilogue>) {
                        float2* p = reinterpret_cast<float2*>(epi.x + static_cast<size_t>(r) * op.n + c);
                        float2 x = *p;
                        x.x = __fadd_rn(x.x, __fmul_rn(v0, gt[n][0]));
                        x.y = __fadd_rn(x.y, __fmul_rn(v1, gt[n][1]));
                        *p = x;
                    } else {
                        // hidden * silu(gate), silu as torch computes it: g / (1 + exp(-g))
                        const float silu = __fdiv_rn(v1, __fadd_rn(1.f, expf(-v1)));
                        epi.h[static_cast<size_t>(r) * (op.n / 2) + c / 2] = __float2half_rn(__fmul_rn(v0, silu));
                    }
                }
            }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------------------------------------------

// 6 stages (96 KiB) is the stage count of cuBLAS's kernel for these shapes and leaves one CTA per SM; 3 stages
// (48 KiB) fit two CTAs per SM, so one CTA's epilogue can run beside the other's mainloop
using DecoderGemm = Gemm<6>;
int g_stages = 3;  // 14.2 ms per decoder block against 16.2 ms with 6 (baseline/vae_block_check.py)

template <int kStages, typename Epi>
void launch_gemm_stages(const Operands& op, const Epi& epi) {
    using G = Gemm<kStages>;
    using Mma = typename G::Mma;
    auto kernel = int8_gemm_kernel<Mma, Epi>;
    static bool opted = [&] {
        return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, G::kSmem) == cudaSuccess;
    }();
    TORCH_CHECK(opted, "shared memory opt-in failed");
    typename Mma::IteratorA::Params pa(cutlass::layout::RowMajor(op.k));
    typename Mma::IteratorB::Params pb(cutlass::layout::ColumnMajor(op.k));
    dim3 grid(op.n / kTileN, (op.m + kTileM - 1) / kTileM);
    kernel<<<grid, kGemmThreads, G::kSmem, at::cuda::getCurrentCUDAStream()>>>(op, pa, pb, epi);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Epi>
void launch_gemm(const Operands& op, const Epi& epi) {
    TORCH_CHECK(op.n % kTileN == 0 && op.k % kTileK == 0, "N must be a multiple of 128 and K of 64");
    if (g_stages == 3) launch_gemm_stages<3>(op, epi);
    else launch_gemm_stages<6>(op, epi);
}

void set_gemm_stages(int64_t stages) {
    TORCH_CHECK(stages == 3 || stages == 6, "3 or 6 stages");
    g_stages = static_cast<int>(stages);
}

Operands operands(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& w,
                  const torch::Tensor& w_scale, const torch::Tensor& bias) {
    TORCH_CHECK(a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8 && a.is_contiguous() && w.is_contiguous());
    TORCH_CHECK(a.size(1) == w.size(1), "K mismatch");
    for (auto* t : {&a_scale, &w_scale, &bias}) TORCH_CHECK(t->dtype() == torch::kFloat32 && t->is_contiguous());
    return {a.data_ptr<int8_t>(), a_scale.data_ptr<float>(), w.data_ptr<int8_t>(), w_scale.data_ptr<float>(),
            bias.data_ptr<float>(), static_cast<int>(a.size(0)), static_cast<int>(w.size(0)),
            static_cast<int>(a.size(1))};
}

std::vector<torch::Tensor> rowquant(const torch::Tensor& x, const c10::optional<torch::Tensor>& w, double eps) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous());
    const int64_t m = x.size(0), k = x.size(1);
    auto q = torch::empty({m, k}, x.options().dtype(torch::kInt8));
    auto s = torch::empty({m}, x.options().dtype(torch::kFloat32));
    const bool norm = w.has_value();
    if (norm) TORCH_CHECK(w->dtype() == torch::kFloat32 && w->is_contiguous() && w->numel() == k);
    const float* wp = norm ? w->data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
#define H3MK_ROWQUANT(T, KK, NORM)                                                                            \
    rowquant_kernel<T, KK, NORM><<<m, kRowThreads, 0, stream>>>(reinterpret_cast<const T*>(x.data_ptr()), wp, \
                                                               static_cast<float>(eps), q.data_ptr<int8_t>(),  \
                                                               s.data_ptr<float>())
    const bool f32 = x.dtype() == torch::kFloat32;
    TORCH_CHECK(f32 || x.dtype() == torch::kFloat16, "fp32 or fp16 rows");
    if (k == 2048 && f32 && norm) H3MK_ROWQUANT(float, 2048, true);
    else if (k == 2048 && f32) H3MK_ROWQUANT(float, 2048, false);
    else if (k == 2048 && norm) H3MK_ROWQUANT(__half, 2048, true);
    else if (k == 2048) H3MK_ROWQUANT(__half, 2048, false);
    else if (k == 8192 && f32) H3MK_ROWQUANT(float, 8192, false);
    else if (k == 8192 && !norm) H3MK_ROWQUANT(__half, 8192, false);
    else TORCH_CHECK(false, "unsupported rowquant shape");
#undef H3MK_ROWQUANT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q, s};
}

std::vector<torch::Tensor> gemm_qkv(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& w,
                                    const torch::Tensor& w_scale, const torch::Tensor& bias, const torch::Tensor& cos,
                                    const torch::Tensor& sin, double eps) {
    auto op = operands(a, a_scale, w, w_scale, bias);
    TORCH_CHECK(op.n == 3 * 2048, "q, k and v of 2048 channels");
    TORCH_CHECK(cos.dtype() == torch::kFloat32 && cos.is_contiguous() && cos.numel() == int64_t(op.m) * 48);
    TORCH_CHECK(sin.dtype() == torch::kFloat32 && sin.is_contiguous() && sin.numel() == int64_t(op.m) * 48);
    auto opts = a.options().dtype(torch::kFloat16);
    auto q = torch::empty({op.m, 2048}, opts), k = torch::empty({op.m, 2048}, opts), v = torch::empty({op.m, 2048}, opts);
    QkvEpilogue epi{reinterpret_cast<__half*>(q.data_ptr()), reinterpret_cast<__half*>(k.data_ptr()),
                    reinterpret_cast<__half*>(v.data_ptr()), cos.data_ptr<float>(), sin.data_ptr<float>(),
                    static_cast<float>(eps)};
    launch_gemm(op, epi);
    return {q, k, v};
}

void gemm_residual(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& w,
                   const torch::Tensor& w_scale, const torch::Tensor& bias, const torch::Tensor& gate,
                   torch::Tensor& x) {
    auto op = operands(a, a_scale, w, w_scale, bias);
    TORCH_CHECK(x.dtype() == torch::kFloat32 && x.is_contiguous() && x.size(0) == op.m && x.size(1) == op.n);
    TORCH_CHECK(gate.dtype() == torch::kFloat32 && gate.is_contiguous() && gate.numel() == op.n);
    launch_gemm(op, ResidualEpilogue{x.data_ptr<float>(), gate.data_ptr<float>()});
}

torch::Tensor gemm_swiglu(const torch::Tensor& a, const torch::Tensor& a_scale, const torch::Tensor& w,
                          const torch::Tensor& w_scale, const torch::Tensor& bias) {
    auto op = operands(a, a_scale, w, w_scale, bias);
    auto h = torch::empty({op.m, op.n / 2}, a.options().dtype(torch::kFloat16));
    launch_gemm(op, SwigluEpilogue{reinterpret_cast<__half*>(h.data_ptr())});
    return h;
}

}  // namespace h3mk

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowquant", &h3mk::rowquant, "norm (optional), rotation and int8 quantization of each row");
    m.def("gemm_qkv", &h3mk::gemm_qkv, "int8 qkv projection, dequant, qk norm and rope");
    m.def("gemm_residual", &h3mk::gemm_residual, "int8 projection, dequant and gated residual, in place");
    m.def("gemm_swiglu", &h3mk::gemm_swiglu, "int8 fc1, dequant and swiglu on interleaved rows");
    m.def("set_gemm_stages", &h3mk::set_gemm_stages, "pipeline stages of the int8 GEMM mainloop: 3 or 6");
    m.attr("gemm_smem") = h3mk::DecoderGemm::kSmem;
}
