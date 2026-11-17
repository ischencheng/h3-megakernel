# SM120 CUTLASS NVFP4 projection result

On 2026-11-16, the standalone CUTLASS v4.4.2 probe built and passed on a
Modal RTX PRO 6000 Blackwell (SM120):
https://modal.com/apps/r19morris/main/ap-66JE7bAGA5gTk0RX7VM7FS

The weight came from the pinned FastH3 V2 Consumer checkpoint's
`transformer_blocks.0.attn.to_q` packed export. It was a uint8 tensor of shape `[7168, 2688]`,
representing a 7168-by-5376 NVFP4 weight. Activation and weight scales used
the FastVideo 128×4 format. The no-FP4-attention recipe still uses NVFP4
linears; attention remained outside this projection test.

| Input rows | CUTLASS grid | Threads/CTA | Shared bytes | Workspace | Max/mean absolute error vs FastVideo |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 × 56 | 384 | 51,200 | 0 | 0 / 0 |
| 37,771 | 56 × 296 | 384 | 51,200 | 0 | 0 / 0 |

The full sequence's scale tensor was padded to `[37888, 336]`, and its tail
tile passed. CUTLASS reported `can_implement=true` in both cases. The Hazy
runtime can provide at most 88,480 aligned dynamic shared bytes alongside its
static control state; this CUTLASS tile fits with 37,280 bytes to spare. The
result verifies a standalone CUTLASS launch, not execution inside the Hazy
persistent kernel. The test input was captured block-0 input data, before the
AdaLN/norm operations that produce the actual `attn.to_q` input.

The full grid contains 16,576 tiles, while Hazy keeps 188 CTAs resident.
Integration therefore needs a scheduler that assigns multiple CUTLASS tiles
to each persistent CTA; matching the 384-thread block shape alone is not
sufficient.
