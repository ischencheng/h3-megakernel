# Hazy dynamic BF16 VSA gate, step 0 block 0

The pinned FastH3 V2 Consumer reference uses NVFP4 transformer linears and
`FASTVIDEO_H3_VSA_FP4=0` on one RTX PRO 6000 (SM120). The new verified capture
is `h3-megakernel-cache/runs/reference-vsa-scores-20261116-a/`. It records
pooled Q/K, the `[1,56,668,668]` FP32 score matrix, exact mask, valid lengths,
and the VSA policy for step 0 block 0. The policy has eight dense prefix tiles,
one 660-tile video span, 80% sparsity (132 retained video tiles per video
query), and exempt prefix handling. Reconstructing the reference mask from
the captured scores and policy produced zero mismatches.

| Hazy operation | Elements compared | Result | GPU time |
| --- | ---: | --- | ---: |
| Q/K tile pooling | 4,788,224 each | max error Q 4.77e-7, K 2.38e-7; zero outside tolerance | 0.91–0.95 ms |
| Pooled Q/K score matrix | 24,988,544 | max error 1.24e-5; zero outside tolerance | 5.53–5.54 ms |
| Exempt-prefix top-k mask from captured scores | 24,988,544 | bit-exact on two launches | 0.74–0.76 ms |
| One Hazy launch: pool → score → top-k → sparse fine attention | Mask 24,988,544; output 306,446,336 | mask bit-exact; output max error 0.125, mean error 0.00011137, zero outside BF16 tolerance | 4,709.7 ms |
| One Hazy launch including V pooling and trained coarse compression | Mask 24,988,544; output 306,446,336 | mask bit-exact; post-compression output max error 0.125, mean error 0.00011138, zero outside BF16 tolerance | 4,451.4 ms |

The BF16 output tolerance is `atol=0.03125, rtol=0.015625`. The first mask
candidate selected one extra tile because two FP32 scores tied at the cutoff.
FastVideo's captured `torch.topk` kept the lower tile ID in that row. The Hazy
selector now limits ties by tile ID; the full captured mask then passed twice.

Runs:

- [Reference capture](https://modal.com/apps/r19morris/main/ap-SNHwg11HASjOuS9gUXoh58)
- [Capture and policy reconstruction](https://modal.com/apps/r19morris/main/ap-oCB9BeAqGsuhV0nDNrw7kx)
- [Hazy Q/K pooling](https://modal.com/apps/r19morris/main/ap-NMT7ocCR8ZjkXEPr6hSu8n)
- [Hazy pooled scores](https://modal.com/apps/r19morris/main/ap-woYocCdkZdIykR2RFtbJWW)
- [Hazy exact top-k mask](https://modal.com/apps/r19morris/main/ap-pAtpLwNUKPUrFLMkiutH4x)
- [One-launch dynamic fine attention](https://modal.com/apps/r19morris/main/ap-HjaXdEMppKXejzPz6UAGNx)
- [One-launch dynamic attention and compression](https://modal.com/apps/r19morris/main/ap-pC3mSH6FE8FjyS1ze9iN4K)

This gate covers one block at one step, with captured Q/K/V/gate tensors as
inputs. It does not chain the NVFP4 projections to dynamic mask generation,
and it does not untile or quantize the resulting attention output for `to_out`.
The warp-per-query fine-attention body dominates the 4.45-second call and is
too slow for 400 block invocations. A complete block, all 50 blocks, the
eight-step denoising loop, text encoder, video/audio decoders, and end-to-end
one-launch generation remain open.
