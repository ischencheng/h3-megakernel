### Stage Contract

Every stage has the same interface: `run(inputs, weights) -> outputs`. This table shows what each stage hands to the next one at 1344x768, 5 s.

| From | To | Tensor | Shape |
|---|---|---|---|
| Text encoder | DiT | prompt embedding | `[Nt, 5120]` |
| DiT | Video decoder | video latents | `[24, 37, 48, 84]` (channels, frames, height, width) |
| DiT | Audio decoder | audio latents | 32 channels, 200 frames |

We fill in the dtypes and normalization when we store the reference tensors in Milestone 1.

A stage passes if its error against the stored FastVideo output is at most twice the error of the FastVideo path against FP32. The stored tensors are on the `h3-megakernel-cache` Modal Volume, not in git.
