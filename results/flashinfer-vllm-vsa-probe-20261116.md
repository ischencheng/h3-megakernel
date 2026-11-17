# FlashInfer and vLLM-Omni BF16 VSA reuse probe

The pinned first FastH3 V2 Consumer call uses BF16 Q/K/V, 56 heads of width
128, 668 logical 64-token tiles, and a per-head `[1,56,668,668]` sparse mask.
The frozen capture has per-tile valid lengths; query tile 0 for head 0 selects
all 668 KV tiles, including 122 partial tiles and 37,771 valid keys.

## Reusable pieces

- [vLLM-Omni's block-sparse adapter](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/attention/ops/block_sparse.py)
  sorts selected KV block IDs before `-1` padding and computes counts on the
  GPU. Its dispatch passes the exact per-block lengths to FlashInfer.
- [vLLM-Omni's FlashInfer adapter](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/attention/ops/flashinfer_block_sparse.py)
  calls `bsa_attn_sm120_blk64_fwd` directly with BF16 BSHD tensors, sorted
  indices, counts, and `block_sizes`. It keeps mask selection and trained
  compression in the model.
- [FlashInfer's SM120 entry point](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cute_dsl/sparse/bsa_attn_sm120.py)
  accepts variable valid KV lengths. Its
  [CuTe kernel body](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cute_dsl/sparse/sm120_blk64/flash_fwd_sm120.py)
  launches 128-thread CTAs. The high-level
  [BlockSparseAttentionWrapper](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/sparse.py)
  does not document `kv_block_lens` for `vsa_sm120_blk64`; its `cake` route
  does accept lengths but rejected this SM120 GPU in the installed version.

## Measured gate

The probe in `megakernel/tests/flashinfer_vsa_probe.py` reads hashed Q/K/V,
mask, and valid-length tensors from the verified reference capture. It compares
one full query tile to a direct FP32 softmax calculation using exactly the
selected, valid keys. GPU: RTX PRO 6000, FlashInfer 0.7.4rc2.

| Query tile | Selected KV blocks | Partial blocks | Elements | Max abs error | Outside BF16 tolerance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Head 0, tile 0 | 668 | 122 | 8,192 | 0.015625 | 0 |
| Head 0, tile 300 | 140 | 45 | 8,192 | 0.015625 | 0 |

[Passing direct-kernel run](https://modal.com/apps/r19morris/main/ap-QdAv7hGwHWN1Ll83GrLoKV).
[Sparse-tile and warmed timing run](https://modal.com/apps/r19morris/main/ap-g3fzRnvq2YSIsGQxdYgDxA).
[Direct FastVideo Triton comparison](https://modal.com/apps/r19morris/main/ap-fP0Ests9c3YVN8btUDM52f)
for head 0, tile 300 also passed on all 8,192 elements: max absolute error
0.015625, mean absolute error 0.0001051, zero outside tolerance.
The first tile-0 event measured 454.85 ms including first-use work; it is not
a steady-state throughput measurement. Warmed tile-300 calls measured
0.765 and 0.666 ms on GPU. These one-head samples are not a full-call
performance result.

This verifies the **host-launched FlashInfer path as a reference**, not a Hazy
integration. Hazy's existing one-launch, warp-per-query operation remains the
only verified in-megakernel implementation. To meet the one-kernel requirement,
adapt the FlashInfer SM120 MMA/softmax algorithm and vLLM-Omni index/length
contract to a Hazy device task, then revalidate the full 56-head output and
chained projection-to-attention gate.
