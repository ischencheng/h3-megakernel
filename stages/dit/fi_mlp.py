# the dit mlp block in one launch, built from flashinfer's own sm120 kernels
# fuse() edits the installed flashinfer source at build time, so none of its code is in this repo
import functools
from pathlib import Path

import torch

FFN = 14336
SOURCE = "cake_minimax_h3_sm120_quant_mlp_sm120a.cu"
MEGA_SMEM = 94208  # max(FC1 93184, FC2 94208, norm 3 x 128)
MEGA_EXPORT = "h3mk_minimax_h3_sm120_nvfp4_mlp_mega"


def _deswizzle(s, rows, cols):  # 128x4 swizzled scales -> [rows, cols]
    return s.reshape(rows // 128, cols // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, cols)


def _swizzle(linear, rows, cols):
    return linear.reshape(rows // 128, 4, 32, cols // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1).contiguous()


# flashinfer row i holds checkpoint row perm[i]: gate rows first, 8 gate then 8 up rows per column group
def fc1_perm(device):
    c = torch.arange(FFN, device=device)
    base = 16 * (c // 8) + (c % 8)
    perm = torch.empty(2 * FFN, dtype=torch.long, device=device)
    perm[base] = FFN + c  # the checkpoint stores up rows first, then gate rows
    perm[base + 8] = c
    return perm


def _scalar(t):
    return t.reshape(1).float().contiguous() if torch.is_tensor(t) else torch.tensor([float(t)], device="cuda")


# checkpoint nvfp4 bytes (harness.weights.Linear fields) -> the operator's layout, no requantization
def pack(fc1, fc2):
    w1, s1 = fc1.w.view(torch.uint8), fc1.s.view(torch.uint8)
    cols = s1.numel() // (2 * FFN)
    perm = fc1_perm(w1.device)
    a_gs, y_gs = _scalar(fc1.input_sf), _scalar(fc2.input_sf)
    return {
        "fc1_q": w1[perm].contiguous(),
        "fc1_sf": _swizzle(_deswizzle(s1, 2 * FFN, cols)[perm], 2 * FFN, cols),
        "fc1_alpha": float(_scalar(fc1.alpha) / a_gs),
        "fc2_q": fc2.w.view(torch.uint8).contiguous(),
        "fc2_sf": fc2.s.view(torch.uint8).reshape(-1).contiguous(),
        "fc2_alpha": float(_scalar(fc2.alpha) / y_gs),
        "a_gs": a_gs, "y_gs": y_gs,
    }


def _workspaces(m, dev):
    u8 = torch.uint8
    return (torch.empty(m, 2688, dtype=u8, device=dev), torch.empty(m, 336, dtype=u8, device=dev),
            torch.empty(m, 7168, dtype=u8, device=dev), torch.empty(m, 896, dtype=u8, device=dev),
            torch.empty(-(-m // 256), dtype=torch.int32, device=dev))


# out = residual + gate[idx] * mlp(adaln(rmsnorm(x))) in flashinfer's three launches
# module=None goes through the public api; the compiled module skips its python checks, like run_one_launch()
def run(x, norm_w, scale, shift, idx, gate, residual, p, eps, module=None):
    if module is None:
        from flashinfer.diffusion_ops import minimax_h3_mlp_nvfp4_sm120

        return minimax_h3_mlp_nvfp4_sm120(x, norm_w, scale, shift, idx, gate, residual, p["a_gs"], p["fc1_q"],
                                          p["fc1_sf"], p["fc1_alpha"], p["y_gs"], p["fc2_q"], p["fc2_sf"],
                                          p["fc2_alpha"], eps=eps)
    out = torch.empty_like(x)
    module.minimax_h3_sm120_nvfp4_mlp(x, norm_w, scale, shift, idx, gate, residual, p["a_gs"], p["fc1_q"], p["fc1_sf"],
                                      p["y_gs"], p["fc2_q"], p["fc2_sf"], *_workspaces(x.shape[0], x.device), out,
                                      float(eps), p["fc1_alpha"], p["fc2_alpha"])
    return out


def flashinfer_module():
    from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_mlp import _get_module

    return _get_module()


# the same block in one launch; trace (int64 [SMs, 4]) gets each cta's start and phase ends in ns
def run_one_launch(x, norm_w, scale, shift, idx, gate, residual, p, eps, module=None, trace=None):
    module = module or build()
    m, dev = x.shape[0], x.device
    out = torch.empty_like(x)
    if trace is None:
        trace = torch.zeros(torch.cuda.get_device_properties(dev).multi_processor_count, 4, dtype=torch.int64,
                            device=dev)
    getattr(module, MEGA_EXPORT)(x, norm_w, scale, shift, idx, gate, residual, p["a_gs"], p["fc1_q"], p["fc1_sf"],
                                 p["y_gs"], p["fc2_q"], p["fc2_sf"], *_workspaces(m, dev),
                                 torch.empty(-(-m // 128), dtype=torch.int32, device=dev), trace, out, float(eps),
                                 p["fc1_alpha"], p["fc2_alpha"])
    return out


@functools.cache
def build():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec, sm120a_nvcc_flags

    csrc = Path(jit_env.FLASHINFER_CSRC_DIR)
    out = csrc / "h3mk_minimax_h3_sm120_mlp_one_launch.cu"  # next to the original so its quoted includes resolve
    out.write_text(fuse((csrc / SOURCE).read_text()))
    spec = gen_jit_spec("h3mk_minimax_h3_sm120_mlp_one_launch_v1", [str(out)],
                        extra_cuda_cflags=list(sm120a_nvcc_flags))
    return spec.build_and_load()


# source edits: each anchor must match exactly once, so a flashinfer update fails loudly

def _replace(text, old, new, count=1):
    found = text.count(old)
    if found != count:
        raise RuntimeError(f"expected {count} match(es) for patch anchor, found {found}: {old[:80]!r}")
    return text.replace(old, new)


def _block(src, ns):  # a namespace and the #undef lines that follow it
    start = src.index(f"namespace {ns} {{")
    close = f"}}  // namespace {ns}\n"
    end = src.index(close, start) + len(close)
    while src.startswith("#undef", end):
        end = src.index("\n", end) + 1
    return src[start:end]


# a copy of one kernel as a device function, with bid and num_bids as arguments
def _to_device(block, ns, new_ns, extra_params):
    b = block.replace(f"namespace {ns} {{", f"namespace {new_ns} {{")
    b = b.replace(f"}}  // namespace {ns}", f"}}  // namespace {new_ns}")
    b = _replace(b, "__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void", "__device__ __forceinline__ void")
    b = b.replace("const __grid_constant__ CUtensorMap ", "const CUtensorMap& ")
    b = _replace(b, "    const int bid = blockIdx.x;\n    const int num_bids = gridDim.x;",
                 "    const int bid = h3mk_bid;\n    const int num_bids = h3mk_nbids;")
    k = b.index("__device__ __forceinline__ void")
    sig_end = b.index(")\n{\n", k)
    return b[:sig_end] + ", int h3mk_bid, int h3mk_nbids" + extra_params + b[sig_end:]


# norm: three groups of 128 threads per cta, each with its own rows, shared memory slot and barrier
def _norm(src):
    ns = "h3_mlp_norm_adaln_quant_nvfp4_sm120a"
    b = _to_device(_block(src, ns), ns, "h3mk_norm", ", int h3mk_group, unsigned int* __restrict__ h3mk_flags_a")
    b = _replace(b, "    const int tid = threadIdx.x;\n", "    const int tid = threadIdx.x & 127;  // h3mk: one of three groups\n")
    b = _replace(b, "    extern __shared__ __align__(1024) char smem_raw[];\n",
                 "    extern __shared__ __align__(1024) char h3mk_smem[];\n"
                 "    char* smem_raw = h3mk_smem + h3mk_group * 128;\n")
    b = _replace(b, "        __syncthreads();\n    }\n}\n",
                 "        asm volatile(\"fence.acq_rel.gpu;\" ::: \"memory\");  // h3mk: this row of a is stored\n"
                 "        __syncthreads();\n"
                 "        if (tid == 0)\n"
                 "            asm volatile(\"red.release.gpu.global.add.u32 [%0], 1;\" :: "
                 "\"l\"(h3mk_flags_a + (row >> 7)) : \"memory\");\n"
                 "    }\n}\n")
    return b.replace("__syncthreads();",
                     "asm volatile(\"bar.sync %0, 128;\" :: \"r\"(1 + h3mk_group) : \"memory\");  // h3mk: one group")


# fc1's tiles go 16 m-tiles at a time (GROUP_M), so the producer waits once per group
_FC1_WAIT = '''                    const int h3mk_group_tile = tile / (GROUP_M * 112);
                    if (h3mk_group_tile != h3mk_last_group) {
                        if (elect_sync()) {  // h3mk: the norm phase has written the rows of this group
                            const int m_end = min(h3mk_group_tile * GROUP_M + GROUP_M, num_m_tiles);
                            for (int mt = h3mk_group_tile * GROUP_M; mt < m_end; ++mt) {
                                const unsigned int need = (unsigned int)min(128, M - mt * 128);
                                unsigned int seen;
                                do {
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(h3mk_flags_a + mt) : "memory");
                                } while (seen < need);
                            }
                            asm volatile("fence.proxy.async;" ::: "memory");
                        }
                        __syncwarp();
                        h3mk_last_group = h3mk_group_tile;
                    }
'''

# fc1 leaves 224 registers in warps 0-7 and 56 in warps 8-11 and fc2 wants the opposite, so go back to 168
# first; fc2 also puts its mbarriers where fc1's were
_FC1_EXIT = '''    __syncthreads();  // h3mk: every role is done with this phase
    if (warp == 0 && lane == 0) {
        for (int i = 0; i < 6; ++i)
            asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(smem + 8 * i) : "memory");
    }
    if (warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 168;");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
    }
    __syncthreads();
'''


# fc1: wait for the norm rows, release each stored 128 x 128 tile of y, hand over to fc2
def _fc1(src):
    ns = "h3_mlp_fc1_nvfp4_sm120a"
    b = _to_device(_block(src, ns), ns, "h3mk_fc1",
                   ", unsigned int* __restrict__ h3mk_flags_y, unsigned int* __restrict__ h3mk_flags_a")
    anchor = ("                    if (elect_sync()) {\n                        #pragma unroll 1\n"
              "                        for (int k_tile = 0; k_tile < 42; k_tile++) {")
    b = _replace(b, anchor, _FC1_WAIT + anchor)
    b = _replace(b, "                unsigned int load_kk = 0;\n                unsigned int _phase_ab_empty = 1;\n",
                 "                unsigned int load_kk = 0;\n                unsigned int _phase_ab_empty = 1;\n"
                 "                int h3mk_last_group = -1;\n")
    last_store = ("reinterpret_cast<int4*>(act_q_w + ((eq_grow_7 * 7168 + eq_tile_n * 64 + eq_c * 16) / 4))[0] = "
                  "reinterpret_cast<int4*>(eq_words)[0];\n                        }\n")
    b = _replace(b, last_store + "                    }\n", last_store +
                 "                        // h3mk: this 128 x 128 tile of y is stored; count it for its 256 rows\n"
                 "                        asm volatile(\"fence.acq_rel.gpu;\" ::: \"memory\");\n"
                 "                        asm volatile(\"bar.sync 12, 64;\" ::: \"memory\");\n"
                 "                        if (warp_id_in_role == 0 && lane == 0)\n"
                 "                            asm volatile(\"red.release.gpu.global.add.u32 [%0], 1;\" :: "
                 "\"l\"(h3mk_flags_y + (eq_tile_m >> 1)) : \"memory\");\n"
                 "                    }\n")
    return _replace(b, "    // Cleanup\n}\n", "    // Cleanup\n" + _FC1_EXIT + "}\n")


# fc2: its wait loop now counts fc1's 2 x 112 tiles per 256 rows (112 for a short last one)
def _fc2(src):
    ns = "h3_mlp_fc2_nvfp4_sm120a"
    b = _to_device(_block(src, ns), ns, "h3mk_fc2", "")
    return _replace(b, "unsigned int rows_ready = ((rows_left < 256) ? rows_left : 256);",
                    "unsigned int rows_ready = (rows_left > 128) ? 224u : 112u;  // h3mk: FC1 tiles of these rows")


_KERNEL = '''
namespace h3mk_one_launch {

__device__ __forceinline__ long long global_ns() {
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// one CTA per SM; trace[cta][0..3]: start, norm done, FC1 done, FC2 done (ns)
__global__ __launch_bounds__(384, 1) void kernel_h3mk_mlp(
    __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale,
    __nv_bfloat16* __restrict__ adaln_shift, long long* __restrict__ adaln_index, unsigned int* __restrict__ act_q,
    uint8_t* __restrict__ act_sf, float* __restrict__ a_g, int M, int adaln_rows, long long adaln_row_stride, float eps,
    const __grid_constant__ CUtensorMap fc1_a, const __grid_constant__ CUtensorMap fc1_b,
    const __grid_constant__ CUtensorMap fc1_sfa, const __grid_constant__ CUtensorMap fc1_sfb,
    unsigned int* __restrict__ y_q, uint8_t* __restrict__ y_sf, float* __restrict__ y_g, int fc1_num_m_tiles,
    int fc1_total_tiles, float fc1_alpha,
    const __grid_constant__ CUtensorMap fc2_a, const __grid_constant__ CUtensorMap fc2_b,
    const __grid_constant__ CUtensorMap fc2_sfa, const __grid_constant__ CUtensorMap fc2_sfb,
    __nv_bfloat16* __restrict__ gate, int gate_rows, long long gate_row_stride, __nv_bfloat16* __restrict__ residual,
    unsigned int* __restrict__ out, int fc2_num_m_tiles, int fc2_total_tiles, float fc2_alpha,
    unsigned int* __restrict__ flags_a, unsigned int* __restrict__ flags_y, long long* __restrict__ trace) {
    const int bid = blockIdx.x, nb = gridDim.x, group = threadIdx.x >> 7;
    long long* t = trace + bid * 4;
    if (threadIdx.x == 0) t[0] = global_ns();
    h3mk_norm::kernel_h3_mlp_norm_adaln_quant_nvfp4(x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, act_q,
                                                    act_sf, a_g, M, adaln_rows, adaln_row_stride, eps,
                                                    bid * 3 + group, nb * 3, group, flags_a);
    __syncthreads();
    if (threadIdx.x == 0) t[1] = global_ns();
    h3mk_fc1::kernel_h3_mlp_fc1_swiglu_gemm(fc1_a, fc1_b, fc1_sfa, fc1_sfb, nullptr, nullptr, y_q, y_q, y_sf, y_g, M,
                                            fc1_num_m_tiles, fc1_total_tiles, fc1_alpha, bid, nb, flags_y, flags_a);
    if (threadIdx.x == 0) t[2] = global_ns();
    h3mk_fc2::kernel_h3_mlp_fc2_gemm_fused(fc2_a, fc2_b, fc2_sfa, fc2_sfb, y_q, y_q, y_sf, y_g, y_g, y_g, gate,
                                           adaln_index, residual, out, flags_y, M, fc2_num_m_tiles, fc2_total_tiles,
                                           gate_rows, gate_row_stride, fc2_alpha, bid, nb);
    __syncthreads();
    if (threadIdx.x == 0) t[3] = global_ns();
}

}  // namespace h3mk_one_launch

'''

_LAUNCH = '''  // h3mk: norm, FC1 and FC2 in one persistent launch, one CTA per SM
  unsigned int* flags_a = static_cast<unsigned int*>(workspace_flags_a.data_ptr());
  TVM_FFI_CHECK(cudaMemsetAsync(flags_a, 0, sizeof(unsigned int) * workspace_flags_a.size(0), args.stream) == cudaSuccess,
                RuntimeError) << "flag zero-fill failed";
  static const bool h3mk_opted = cudaFuncSetAttribute(h3mk_one_launch::kernel_h3mk_mlp,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize, %d) == cudaSuccess;
  TVM_FFI_CHECK(h3mk_opted, RuntimeError) << "shared memory opt-in failed";
  int h3mk_sms = 0;
  cudaDeviceGetAttribute(&h3mk_sms, cudaDevAttrMultiProcessorCount, args.device.device_id);
  TVM_FFI_CHECK(trace.size(0) >= h3mk_sms, ValueError) << "trace needs one row per SM";
  h3mk_one_launch::kernel_h3mk_mlp<<<dim3(h3mk_sms), dim3(384), %d, args.stream>>>(
      const_cast<__nv_bfloat16*>(args.x), const_cast<__nv_bfloat16*>(args.x_norm_weight),
      const_cast<__nv_bfloat16*>(args.tables.scale), const_cast<__nv_bfloat16*>(args.tables.shift),
      const_cast<IndexT*>(args.index), static_cast<unsigned int*>(workspace_a_q.data_ptr()),
      static_cast<uint8_t*>(workspace_a_sf.data_ptr()), a_g, static_cast<int>(args.rows), args.tables.rows,
      args.tables.adaln_row_stride, static_cast<float>(eps), fc1_a, fc1_b, fc1_sfa, fc1_sfb, y_q_words, y_sf, y_g,
      args.plan.fc1_num_m_tiles, args.plan.fc1_total_tiles, static_cast<float>(fc1_alpha), fc2_a, fc2_b, fc2_sfa, fc2_sfb,
      const_cast<__nv_bfloat16*>(args.tables.gate), args.tables.rows, args.tables.gate_row_stride,
      const_cast<__nv_bfloat16*>(args.residual), args.out, args.plan.fc2_num_m_tiles, args.plan.fc2_total_tiles,
      static_cast<float>(fc2_alpha), flags_a, args.flags, static_cast<long long*>(trace.data_ptr()));
  CheckLaunch("h3mk MLP one launch");
''' % (MEGA_SMEM, MEGA_SMEM)


# host: a copy of flashinfer's nvfp4 entry point whose three launches become one
def _host(src):
    head = "void minimax_h3_sm120_nvfp4_mlp("
    start = src.index(head)
    host = src[start:src.index("\n}\n", start) + 3].replace(head, f"void {MEGA_EXPORT}(")
    host = _replace(host, "TensorView workspace_flags, TensorView out,",
                    "TensorView workspace_flags, TensorView workspace_flags_a, TensorView trace, TensorView out,")
    host = _replace(host, "FillFlags(args, true);", "FillFlags(args, false);  // h3mk: FC1 counts its tiles")
    for first, last in (("h3_mlp_norm_adaln_quant_nvfp4_sm120a::kernel_h3_mlp_norm_adaln_quant_nvfp4<<<",
                         'CheckLaunch("MiniMax-H3 NVFP4 norm/AdaLN quantization");\n'),
                        ("h3_mlp_fc1_nvfp4_sm120a::kernel_h3_mlp_fc1_swiglu_gemm<<<",
                         'CheckLaunch("MiniMax-H3 NVFP4 FC1+SwiGLU");\n')):
        a = host.index(first)
        host = host[:a] + host[host.index(last, a) + len(last):]
    a = host.index("h3_mlp_fc2_nvfp4_sm120a::kernel_h3_mlp_fc2_gemm_fused<<<")
    last = 'CheckLaunch("MiniMax-H3 NVFP4 FC2");\n'
    return host[:a] + _LAUNCH.lstrip() + host[host.index(last, a) + len(last):]


def fuse(src):
    kernels = _norm(src) + _fc1(src) + _fc2(src) + _KERNEL
    out = _replace(src, '#include "tvm_ffi_utils.h"\n', kernels + '#include "tvm_ffi_utils.h"\n')
    export = "TVM_FFI_DLL_EXPORT_TYPED_FUNC(minimax_h3_sm120_fp8_mlp, minimax_h3_sm120_fp8_mlp);"
    return _replace(out, export,
                    _host(src) + "\n" + export + f"\nTVM_FFI_DLL_EXPORT_TYPED_FUNC({MEGA_EXPORT}, {MEGA_EXPORT});")
