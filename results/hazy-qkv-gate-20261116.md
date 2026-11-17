# Four first-block projections in one Hazy launch

Status: **Q/K/V and active VSA-gate projection gate passed; milestone 3 remains open.**

The pinned reference is FastVideo
`ce970eb715215c79a169658d7aa6c8a96e64891e` with checkpoint
`FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer` at
`9d0d0d3ae69365aded83fab5562f00f9f06c12aa`, one RTX PRO 6000 (SM120),
1344×768, 124 frames, seed 1101, prompt index 0, eight transformer forwards,
NVFP4 linears, and BF16 VSA (`FASTVIDEO_H3_VSA_FP4=0`).

The [exact capture](https://modal.com/apps/r19morris/main/ap-wV0QP116mgZkAFzZhELmwv)
at `/cache/runs/qkv-exact-20261116-r1/` exited with code 0 and wrote a verified
manifest. Its step-0 block-0 post-RMSNorm/AdaLN input has shape
`[1, 37771, 5376]` and SHA256
`2d3cfe899edae74256ef39646e4f5a1590c28a0eb2b6d55f6ca7765ee55dec1a`.
Each BF16 output has shape `[1, 37771, 7168]`:

| Projection | Captured output SHA256 |
| --- | --- |
| Q | `efeedbd9094ea6112f7e6d08344c5b91014564055ec7e6f419e99043c6d52e72` |
| K | `77e042dd7f8fc857544bc3163baaf3babb9f97388bf35b087dca980263502a5f` |
| V | `75062a99ab993fa0808b46cb0014657201d2f4a4e22080e2bfb8f43312b5f732` |
| Compression gate | `fec9e0e63c5c1531bfa65d0974fe9f627fef166dea3bbe8c24a9a7902b8d3363` |

The gate was active in this checkpoint. The
[Hazy validation](https://modal.com/apps/r19morris/main/ap-LYSaJ6SPf3zrmtrKDT3iqm)
reconstructed each FastVideo `mm_fp4` result from the captured input and pinned
packed weights/scales/alphas, then ran all four GEMMs in **one Hazy instruction
stream**. Every reconstructed and Hazy output matched its captured BF16 tensor
bit for bit: max/mean absolute error 0 and mismatch count 0 over 270,742,528
elements per projection. The Hazy stream passed three launches, with smoke
instructions before and after the GEMMs. It used 188 CTAs, 384 threads/CTA,
51,200 shared bytes for each exclusive CUTLASS task, and no global workspace.
The static persistent scheduler covered 16,576 tiles per projection, including
the 11-row tail.

Reproduce:

```sh
H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo \
  .venv/bin/modal run baseline/fastvideo_modal.py \
  --run-id qkv-exact-20261116-r1 --script projection_capture.py --args=--all-qkv

H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo \
  .venv/bin/modal run megakernel/cutlass_probe_modal.py \
  --mode qkv --capture-run-id qkv-exact-20261116-r1
```

These are the four linears at the start of block 0, step 0. The next
attention-boundary gate must account for Q/K normalization, partial RoPE,
tile permutation, the exact 64-token VSA mask and valid lengths, sparse BF16
attention, and the compression branch. The complete block and the remaining
49 blocks and seven steps have not passed.
