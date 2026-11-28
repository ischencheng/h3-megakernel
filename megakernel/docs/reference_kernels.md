# Reference kernels per opcode

Paths are relative to the repo root. "Write in" is what our Hazy-format op should use.

| # | Opcode | Reference kernel(s) | Reference language | Write in |
|---|---|---|---|---|
| 1 | `NORM_MOD_QUANT` | `third_party/flashinfer/csrc/cake_minimax_h3_sm120_norm_adaln_quant.cuh` (same op on SM120); `third_party/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh:230` `quantize_with_block_size` (NVFP4 + 128x4 scale swizzle) | generated CUDA; CUDA | CUDA + inline PTX |
| 2 | `GEMM_QKV` | mainloop: `third_party/flashinfer/3rdparty/cutlass/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu`, `.../cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`; epilogue (QK-norm + RoPE): `third_party/flashinfer/csrc/cake_minimax_h3_sm120_quant_pre_attention/` | CuTe C++; generated CUDA | CuTe C++ |
| 3 | `VSA_POOL` | none needed; semantics in `FastVideo/fastvideo/attention/backends/video_sparse_attn_h3.py:557` `_pool_tiles` | PyTorch | CUDA |
| 4 | `ATTN_QUANT` | `FastVideo/fastvideo-kernel/attn_qat_infer/quantization/fp4_quantization_4d.cu`; layouts in `FastVideo/fastvideo-kernel/attn_qat_infer/api.py` | CUDA | CUDA |
| 5 | `VSA_SELECT` | top-k: `third_party/flashinfer/include/flashinfer/topk.cuh` `RadixTopKKernel_Unified`; semantics: `video_sparse_attn_h3.py:572` `_build_block_mask`, `attn_qat_infer/api.py:203` `vsa_tile_mask_to_fp4_blocks` (output format the attention kernel expects) | CUDA; PyTorch | CUDA |
| 6 | `ATTN_SPARSE` | `FastVideo/fastvideo-kernel/attn_qat_infer/blackwell/` (SageAttention3 sparse FP4: `kernel_ws.h`, `mainloop_tma_ws.h`, `softmax_fused.h`, `epilogue_tma_ws.h`); baseline: `third_party/flashinfer/csrc/cake_vsa_sm120/` (BF16, 64-token tiles); coarse branch: `minimax_h3_vsa_fp4.py:419-441` | CuTe C++; generated CUDA | CuTe C++ |
| 7 | `GEMM_OUT` | mainloop as #2; epilogue (gated residual): `third_party/flashinfer/csrc/cake_minimax_h3_sm120_quant_out_proj_sm120a.cu` | CuTe C++; generated CUDA | CuTe C++ |
| 8 | `GEMM_FC1` | mainloop as #2; epilogue (SwiGLU + FP4 quant): `third_party/flashinfer/csrc/cake_minimax_h3_sm120_quant_fc1_swiglu_sm120a.cu`, CUTLASS `epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp` `Sm120BlockScaleFactorRowStore` | CuTe C++; generated CUDA | CuTe C++ |
| 9 | `GEMM_FC2` | mainloop as #2; epilogue (gated residual): `third_party/flashinfer/csrc/cake_minimax_h3_sm120_quant_mlp_sm120a.cu` | CuTe C++; generated CUDA | CuTe C++ |