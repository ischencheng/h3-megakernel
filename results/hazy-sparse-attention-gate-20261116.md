# Hazy BF16 sparse fine-attention gate, step 0 block 0

The new Hazy opcode 7 computes BF16 block-sparse fine attention inside the
persistent CUDA kernel on one RTX PRO 6000 (SM120). It consumes the frozen
`[1,56,668,668]` boolean mask and 668 valid-key lengths from
`h3-megakernel-cache/runs/reference-bf16-vsa-20261115/`. Each of the twelve
CTA warps owns one query row and streams only selected 64-token key tiles.
The operation launches no Triton or other GPU kernel from the device.

The reference is FastVideo's tile-64 Triton fine-attention result using the
same frozen Q/K/V/mask. Triton is called by the **test oracle only**; it is
absent from the Hazy execution path. The recipe keeps NVFP4 transformer
linears and `FASTVIDEO_H3_VSA_FP4=0` for BF16 attention.

| Gate | Shape / rows | Repeats | Max absolute error | Mean absolute error | Outside tolerance | Hazy GPU time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense prefix sample | 64 query rows × 128 channels | 2 | 0.015625 | 0.0000803 | 0 | ~34 ms |
| Full first attention call | 2,394,112 head/query rows × 128 channels | 2 | 0.125 | 0.0001114 | 0 | ~4.61 s |

Tolerance is `atol=0.03125, rtol=0.015625`. The full output has
306,446,336 BF16 elements; 24,111,527 differ bitwise from Triton, so this
is a **numerical** pass, not bit-exact parity. The first failed sample was
diagnosed as a padded-query rule mismatch: FastVideo masks padded *keys* but
still computes padded query rows. The corrected Hazy operation follows that
rule.

- [Dense sample run](https://modal.com/apps/r19morris/main/ap-gMVUH4gICZUFim6vnqyAna)
- [Full-shape run](https://modal.com/apps/r19morris/main/ap-KojebkBzNki9SD4SSCrZpR)

The fine-attention operation subsequently passed **in the same Hazy kernel
launch** as four NVFP4 projections, Q/K norm, partial RoPE, and four-way tile
scatter. Two cross-SM release/acquire barriers connected the stages. Across
two repeats, all four projections remained bit-exact, Q/K tile inputs stayed
within their prior BF16 tolerance, V/gate tile inputs remained bit-exact, and
the 306,446,336-element fine-attention output had max absolute error 0.125,
mean absolute error 0.0001115, and zero elements outside the stated tolerance.
[Chained one-launch run](https://modal.com/apps/r19morris/main/ap-IjGzNipTVl9tXlpPK9cu46).

This gate validates **fine attention with the frozen exact mask**. Dynamic
pooling/top-k mask generation and the checkpoint's trained coarse compression
gate remain open. The warp-per-query implementation is a correctness baseline:
~4.61 seconds for one attention call is too slow for 400 block invocations,
so a tensor-core implementation or equivalent optimization is needed before
full-run performance is viable. A complete DiT block and all eight steps
remain open.
