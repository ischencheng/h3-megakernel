"""Parity of every full-DiT Hazy operation against FastVideo on synthetic data.

Each case builds a small `Program`, launches it once on the resident SM120
grid, and compares with the FastVideo/flashinfer implementation the pinned
recipe uses. Quantization and the flow update must be bit exact; BF16
arithmetic is compared against the stated tolerance with mismatch counts.
"""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import h3_dit as hd
import mk_h3_dit

RESULTS = {}
ATOL, RTOL = 0.03125, 0.015625


def report(name, actual, expected, *, exact=False):
    a, e = actual.float(), expected.float()
    diff = (a - e).abs()
    mismatch = int((actual != expected).count_nonzero())
    outside = int((~torch.isclose(a, e, atol=ATOL, rtol=RTOL)).count_nonzero())
    row = {"elements": actual.numel(), "mismatched": mismatch, "outside_tolerance": outside,
           "max_abs_error": float(diff.max()), "mean_abs_error": float(diff.mean())}
    RESULTS[name] = row
    print(name, json.dumps(row), flush=True)
    if (exact and mismatch) or outside or not torch.isfinite(a).all():
        raise AssertionError(f"{name} differs from the reference")


def bf16(*shape, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(torch.bfloat16)


def test_quant():
    from flashinfer import SfLayout
    from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize
    from fastvideo.models.dits.minimax_h3_fusions import minimax_h3_swiglu

    rows, cols = 301, 5376
    x = bf16(rows, cols, scale=3.0)
    x[5, 64:96] = 0
    x[7] *= 400.0
    x[9] *= 1e-4
    for sf in (1.0, 2.33018159866333, 0.03411836549639702):
        p = hd.Program(mk_h3_dit)
        packed = p.alloc("packed", (384, cols // 2), torch.uint8)
        scales = p.alloc("scales", (384, cols // 16), torch.uint8)
        p.quantize("quant", rows=rows, cols=cols, global_sf=sf, flags=0, x=x, packed=packed,
                   packed_sf=scales, stride=cols)
        p.run()
        ref, ref_sf = _nvfp4_quantize(x, torch.tensor(sf, device="cuda"),
                                      sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        report(f"quant sf={sf:.4g} codes", packed[:rows], ref.reshape(rows, -1), exact=True)
        report(f"quant sf={sf:.4g} scales", scales.reshape(-1)[:ref_sf.numel()],
               ref_sf.reshape(-1), exact=True)
        if int(scales.reshape(-1)[ref_sf.numel():].count_nonzero()):
            raise AssertionError("scale padding was written")
    # gather + SwiGLU source
    perm = torch.randperm(rows, device="cuda")
    p = hd.Program(mk_h3_dit)
    packed = p.alloc("packed", (384, cols // 2), torch.uint8)
    scales = p.alloc("scales", (384, cols // 16), torch.uint8)
    p.quantize("gather", rows=rows, cols=cols, global_sf=1.0, flags=hd.QUANT_GATHER, x=x,
               packed=packed, packed_sf=scales, stride=cols, gather=perm)
    p.run()
    ref, ref_sf = _nvfp4_quantize(x[perm].contiguous(), torch.tensor(1.0, device="cuda"),
                                  sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    report("quant gather codes", packed[:rows], ref.reshape(rows, -1), exact=True)
    report("quant gather scales", scales.reshape(-1)[:ref_sf.numel()], ref_sf.reshape(-1), exact=True)
    ffn = 2048
    y = bf16(rows, 2 * ffn, scale=4.0)
    p = hd.Program(mk_h3_dit)
    packed = p.alloc("packed", (384, ffn // 2), torch.uint8)
    scales = p.alloc("scales", (384, ffn // 16), torch.uint8)
    p.quantize("swiglu", rows=rows, cols=ffn, global_sf=0.7079368233680725,
               flags=hd.QUANT_SWIGLU, x=y, packed=packed, packed_sf=scales, stride=2 * ffn)
    p.run()
    act = minimax_h3_swiglu(y)
    ref, ref_sf = _nvfp4_quantize(act, torch.tensor(0.7079368233680725, device="cuda"),
                                  sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    codes_equal = float((packed[:rows] == ref.reshape(rows, -1)).float().mean())
    RESULTS["quant swiglu byte agreement"] = codes_equal
    print("quant swiglu byte agreement", codes_equal, flush=True)
    if codes_equal < 0.999:
        raise AssertionError("SwiGLU-source quantization disagrees with FastVideo")


def test_norm():
    from fastvideo.models.dits.minimax_h3_fusions import (
        fused_residual_gate_rmsnorm_modulate, fused_rmsnorm_modulate)

    rows, cols = 997, 5376
    x, branch = bf16(rows, cols, scale=2.0), bf16(rows, cols)
    weight = bf16(cols).abs() + 0.5
    table = bf16(6, 6 * cols, scale=0.3)
    index = torch.randint(0, 6, (rows,), device="cuda")
    p = hd.Program(mk_h3_dit)
    out = p.alloc("out", (rows, cols), torch.bfloat16)
    w = p.const("w", weight, torch.float32)
    idx = p.const("idx", index, torch.int32)
    p.rmsnorm("mod", rows=rows, cols=cols, eps=1e-5, flags=hd.NORM_MOD, x=x, out=out, weight=w,
              index=idx, table=table, table_stride=6 * cols, scale=cols, shift=0)
    p.run()
    ref = fused_rmsnorm_modulate(x, weight, table[:, cols:2 * cols], table[:, :cols], index, 1e-5)
    report("rmsnorm_modulate", out, ref)

    hidden = x.clone()
    p = hd.Program(mk_h3_dit)
    out = p.alloc("out", (rows, cols), torch.bfloat16)
    w = p.const("w", weight, torch.float32)
    idx = p.const("idx", index, torch.int32)
    p.rmsnorm("res", rows=rows, cols=cols, eps=1e-5, flags=hd.NORM_MOD | hd.NORM_RESIDUAL,
              x=hidden, out=out, weight=w, index=idx, table=table, table_stride=6 * cols,
              scale=4 * cols, shift=3 * cols, gate=2 * cols, branch=branch)
    p.run()
    ref_hidden, ref = fused_residual_gate_rmsnorm_modulate(
        x, branch, table[:, 2 * cols:3 * cols], weight, table[:, 4 * cols:5 * cols],
        table[:, 3 * cols:4 * cols], index, 1e-5)
    report("residual_gate hidden", hidden, ref_hidden)
    report("residual_gate normed", out, ref)

    # eager BF16 MLP residual followed by the next block's fused norm
    hidden = x.clone()
    p = hd.Program(mk_h3_dit)
    out = p.alloc("out", (rows, cols), torch.bfloat16)
    w = p.const("w", weight, torch.float32)
    idx = p.const("idx", index, torch.int32)
    flags = hd.NORM_MOD | hd.NORM_RESIDUAL | hd.NORM_DOUBLE_ROUND | hd.NORM_ROUNDED
    p.rmsnorm("eager", rows=rows, cols=cols, eps=1e-5, flags=flags, x=hidden, out=out, weight=w,
              index=idx, table=table, table_stride=6 * cols, scale=cols, shift=0, gate=5 * cols,
              branch=branch)
    p.run()
    ref_hidden = x + table[:, 5 * cols:].index_select(0, index) * branch
    ref = fused_rmsnorm_modulate(ref_hidden, weight, table[:, cols:2 * cols], table[:, :cols],
                                 index, 1e-5)
    report("eager residual hidden", hidden, ref_hidden, exact=True)
    report("eager residual normed", out, ref)

    # plain RMSNorm with a unit-gate residual (text refiner)
    hidden = x.clone()
    p = hd.Program(mk_h3_dit)
    out = p.alloc("out", (rows, cols), torch.bfloat16)
    w = p.const("w", weight, torch.float32)
    p.rmsnorm("unit", rows=rows, cols=cols, eps=1e-5,
              flags=hd.NORM_RESIDUAL | hd.NORM_UNIT_GATE | hd.NORM_ROUNDED, x=hidden, out=out,
              weight=w, branch=branch)
    p.run()
    ref_hidden = x + branch
    norm = torch.nn.RMSNorm(cols, eps=1e-5, device="cuda", dtype=torch.bfloat16)
    norm.weight.data.copy_(weight)
    report("unit residual hidden", hidden, ref_hidden, exact=True)
    report("unit residual rmsnorm", out, norm(ref_hidden))


def test_linear_and_elementwise():
    from fastvideo.layers.visual_embedding import get_timestep_embedding
    from fastvideo.models.dits.minimax_h3_fusions import minimax_h3_swiglu

    rows, k, n = 211, 2688, 517
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(rows, k, device="cuda").to(dtype)
        weight = (torch.randn(n, k, device="cuda") / math.sqrt(k)).to(dtype)
        bias = torch.randn(n, device="cuda").to(dtype)
        p = hd.Program(mk_h3_dit)
        out = p.alloc("out", (rows, n), dtype)
        p.linear("linear", rows=rows, x=x, weight=weight, bias=bias, out=out)
        p.run()
        report(f"dense_linear {dtype}", out, F.linear(x.float(), weight.float(), bias.float()).to(dtype))
    index = torch.randperm(400, device="cuda")[:rows]
    x = torch.randn(rows, 96, device="cuda")
    weight, bias = torch.randn(n, 96, device="cuda") / 10, torch.randn(n, device="cuda")
    p = hd.Program(mk_h3_dit)
    out = p.alloc("out", (400, n), torch.bfloat16)
    p.linear("scatter", rows=rows, x=x, weight=weight, bias=bias, out=out, index=index,
             scatter=True, out_stride=n)
    p.run()
    report("dense_linear scatter", out[index], F.linear(x, weight, bias).to(torch.bfloat16))
    source = bf16(400, k)
    weight, bias = torch.randn(96, k, device="cuda") / math.sqrt(k), torch.randn(96, device="cuda")
    p = hd.Program(mk_h3_dit)
    out = p.alloc("out", (rows, 96), torch.float32)
    p.linear("gather", rows=rows, x=source, weight=weight, bias=bias, out=out, index=index,
             gather=True, in_stride=k)
    p.run()
    report("dense_linear gather", out, F.linear(source[index].float(), weight, bias))

    timesteps = torch.tensor([0.00010013580322265625, 0.4117646813392639, 0.7, 999.0],
                             device="cuda")
    x = torch.randn(4, 5376, device="cuda") * 3
    sample, velocity = torch.randn(3000, 96, device="cuda"), torch.randn(3000, 96, device="cuda")
    sigma, sigma_next = torch.tensor(0.9675752520561218), torch.tensor(0.9431681036949158)
    step, ratio = float(1 - (1 - sigma)), float(sigma_next / sigma)
    p = hd.Program(mk_h3_dit)
    emb = p.alloc("emb", (4, 256), torch.float32)
    act = p.alloc("act", (4, 5376), torch.bfloat16)
    updated = sample.clone()
    p.elementwise("timestep", hd.EW_TIMESTEP, emb.numel(), timesteps, emb, scalar=256)
    p.elementwise("silu", hd.EW_SILU, x.numel(), x, act)
    p.elementwise("flow", hd.EW_FLOW_UPDATE, updated.numel(), updated, updated,
                  scalar=hd.float_bits(step), second=velocity, ratio=ratio)
    p.run()
    report("timestep embedding", emb,
           get_timestep_embedding(timesteps, 256, flip_sin_to_cos=True, downscale_freq_shift=0))
    report("silu", act, F.silu(x).to(torch.bfloat16))
    denoised = sample + torch.tensor(step, device="cuda") * velocity
    ratio_t = torch.tensor(ratio, device="cuda")
    report("flow update", updated, ratio_t * sample + (1.0 - ratio_t) * denoised, exact=True)

    y = bf16(61, 2 * 14336, scale=3.0)
    p = hd.Program(mk_h3_dit)
    out = p.alloc("out", (61, 14336), torch.bfloat16)
    p.copy_rows("swiglu", rows=61, cols=14336, x=y, out=out, in_stride=2 * 14336,
                out_stride=14336, swiglu=True)
    p.run()
    report("swiglu rows", out, minimax_h3_swiglu(y))


def test_attention():
    from fastvideo.attention.backends import video_sparse_attn_h3 as vsa
    from fastvideo.models.dits.minimax_h3_fusions import fused_qknorm_rope
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn

    heads, tiles, prefix = 56, 40, 3
    padded = tiles * 64
    valid = torch.full((tiles,), 64, device="cuda", dtype=torch.int64)
    valid[0], valid[2], valid[-1] = 61, 30, 16
    tile_rows = torch.cat([torch.arange(int(c), device="cuda") + i * 64 for i, c in enumerate(valid)])
    rows = tile_rows.numel()
    q_in, k_in = bf16(rows, heads * 128), bf16(rows, heads * 128)
    v_in, gate_in = bf16(rows, heads * 128), bf16(rows, heads * 128, scale=0.2)
    wq, wk = bf16(128).abs() + 0.5, bf16(128).abs() + 0.5
    angle = torch.rand(rows, 96, device="cuda") * 6.28
    cos, sin = angle.cos().to(torch.bfloat16), angle.sin().to(torch.bfloat16)
    keep = vsa.compute_topk(0.8, tiles - prefix) if hasattr(vsa, "compute_topk") else math.ceil(
        0.2 * (tiles - prefix))

    def build(fast):
        p = hd.Program(mk_h3_dit)
        t = {n: p.alloc(f"t{n}", (padded, heads, 128), torch.bfloat16) for n in ("q", "k", "v", "gate")}
        pools = {n: p.alloc(f"p{n}", (heads, tiles, 128), torch.float32) for n in ("q", "k", "v")}
        scores = p.alloc("scores", (heads, tiles, tiles), torch.float32)
        mask = p.alloc("mask", (heads, tiles, tiles), torch.uint8)
        fine = p.alloc("fine", (padded, heads, 128), torch.bfloat16)
        out = p.alloc("out", (padded, heads, 128), torch.bfloat16)
        p.qknorm("qknorm", rows=rows, eps=1e-5,
                 flags=hd.QK_ROPE | hd.QK_TILE | hd.QK_COPY_V | hd.QK_COPY_GATE, heads=heads,
                 rotary=96, q=q_in, k=k_in, q_out=t["q"], k_out=t["k"], q_weight=wq, k_weight=wk,
                 cos=cos, sin=sin, tile_index=tile_rows, v=v_in, gate=gate_in, v_out=t["v"],
                 gate_out=t["gate"])
        p.emit("pool", "VSA_POOL", tiles, heads, 3, p.buf(valid), p.buf(t["q"]), p.buf(t["k"]),
               p.buf(t["v"]), p.buf(pools["q"]), p.buf(pools["k"]), p.buf(pools["v"]))
        p.emit("scores", "VSA_SCORES", tiles, heads, p.buf(pools["q"]), p.buf(pools["k"]),
               p.buf(scores))
        p.emit("topk", "VSA_TOPK", tiles, heads, prefix, prefix, tiles, keep, p.buf(scores),
               p.buf(mask))
        p.emit("attention", "SPARSE_ATTENTION_MMA" if fast else "SPARSE_ATTENTION", tiles, heads,
               p.buf(t["q"]), p.buf(t["k"]), p.buf(t["v"]), p.buf(mask), p.buf(valid), p.buf(fine))
        p.copy_rows("copy", rows=padded, cols=heads * 128, x=fine, out=out)
        p.emit("compress", "VSA_COMPRESS", tiles, heads, p.buf(scores), p.buf(pools["v"]),
               p.buf(t["gate"]), p.buf(out))
        ms = p.run()
        return t, pools, scores, mask, fine, out, ms

    t, pools, scores, mask, fine, out, slow_ms = build(False)
    q_ref = fused_qknorm_rope(q_in.view(1, rows, heads, 128), wq, cos, sin, 1e-5)
    k_ref = fused_qknorm_rope(k_in.view(1, rows, heads, 128), wk, cos, sin, 1e-5)

    def tile(x):
        buffer = torch.zeros(1, padded, heads, 128, device="cuda", dtype=torch.bfloat16)
        buffer[:, tile_rows] = x.reshape(1, rows, heads, 128)
        return buffer

    report("qknorm q tiled", t["q"], tile(q_ref)[0])
    report("qknorm k tiled", t["k"], tile(k_ref)[0])
    report("v tiled", t["v"], tile(v_in)[0], exact=True)
    report("gate tiled", t["gate"], tile(gate_in)[0], exact=True)
    tq, tk, tv, tg = (t[n].unsqueeze(0) for n in ("q", "k", "v", "gate"))
    q_pool, k_pool = vsa._pool_tiles(tq, valid, 64), vsa._pool_tiles(tk, valid, 64)
    v_pool = vsa._pool_tiles(tv, valid, 64)
    report("q pool", pools["q"], q_pool[0])
    report("v pool", pools["v"], v_pool[0])
    ref_scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) / 128 ** 0.5
    report("scores", scores, ref_scores[0])
    ref_mask = vsa._build_block_mask(scores.unsqueeze(0), prefix, 0.8, True, ((prefix, tiles),), (0.8,))
    report("mask", mask, ref_mask[0].to(torch.uint8), exact=True)
    ref_fine, _ = block_sparse_attn(tq.transpose(1, 2).contiguous(), tk.transpose(1, 2).contiguous(),
                                    tv.transpose(1, 2).contiguous(), mask.unsqueeze(0).bool(), valid)
    ref_fine = ref_fine.transpose(1, 2).contiguous()[0]
    report("fine attention (warp per query)", fine, ref_fine)
    coarse = torch.matmul(torch.softmax(scores.unsqueeze(0), dim=-1), v_pool).to(torch.bfloat16)
    coarse = coarse.permute(0, 2, 1, 3).repeat_interleave(64, dim=1)
    ref_out = torch.addcmul(ref_fine.unsqueeze(0), coarse, tg)[0]
    report("compressed attention (warp per query)", out, ref_out)
    slow_fine = fine.clone()

    t, pools, scores, mask2, fine, out, fast_ms = build(True)
    if not torch.equal(mask, mask2):
        raise AssertionError("mask changed between launches")
    report("fine attention (tensor core) vs FastVideo", fine, ref_fine)
    report("fine attention (tensor core) vs warp per query", fine, slow_fine)
    report("compressed attention (tensor core)", out, ref_out)
    RESULTS["attention_ms"] = {"warp_per_query_chain": slow_ms, "tensor_core_chain": fast_ms}
    print("attention chain ms", slow_ms, fast_ms, flush=True)


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 required")
    torch.manual_seed(1101)
    only = sys.argv[1:]
    for name, test in (("quant", test_quant), ("norm", test_norm),
                       ("linear", test_linear_and_elementwise), ("attention", test_attention)):
        if not only or name in only:
            test()
    print(json.dumps(RESULTS, indent=2), flush=True)
    print("PASS: full-DiT Hazy operations match FastVideo", flush=True)


if __name__ == "__main__":
    main()
