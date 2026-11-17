# checks and times the fused video decoder block on real checkpoint blocks and a random
# residual stream of the 1344x768 tile batch, against fastvideo's code on the same input
import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from harness import check
from stages.video_decoder import fused_block as fb

OUT = Path(os.environ.get("OUT_DIR", "out"))
MODEL = "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer"
REVISION = "9d0d0d3ae69365aded83fab5562f00f9f06c12aa"


def load_block(index, device="cuda"):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from fastvideo.models.vaes import minimax_h3_int8_convrot as ic
    from fastvideo.models.vaes.minimax_h3_video import MiniMaxH3VideoTransformerBlock

    root = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    path = ic.find_int8_convrot_vae_path(root / "vae") or next(root.glob(f"**/{ic.INT8_CONVROT_FILENAME}"))
    prefix = f"decoder.transformer_blocks.{index}"
    with safe_open(str(path), framework="pt") as f:
        t = {k.removeprefix(prefix + "."): f.get_tensor(k) for k in f.keys() if k.startswith(prefix + ".")}
    block = MiniMaxH3VideoTransformerBlock(dim=2048, heads=32, dim_head=64, ffn_mult=4, eps=1e-5)
    marker = lambda name: ic.parse_comfy_quant_marker(t[name + ic._COM_FY_QUANT_SUFFIX])
    block.attn.to_q, block.attn.to_k, block.attn.to_v = ic._split_qkv(
        t["attn.to_qkv.weight"], t["attn.to_qkv.weight_scale"], t.get("attn.to_qkv.bias"), marker("attn.to_qkv"),
        heads=32, dim_head=64)
    block.attn.to_out[0] = ic._int8_linear_from_tensors(t["attn.to_out.weight"], t["attn.to_out.weight_scale"],
                                                        t.get("attn.to_out.bias"), marker("attn.to_out"))
    swap = ic._swap_swiglu_halves
    block.ff.net[0].proj = ic._int8_linear_from_tensors(
        swap(t["ff.w1.weight"]), swap(t["ff.w1.weight_scale"]),
        swap(t["ff.w1.bias"]) if "ff.w1.bias" in t else None, marker("ff.w1"))
    block.ff.net[2] = ic._int8_linear_from_tensors(t["ff.w2.weight"], t["ff.w2.weight_scale"], t.get("ff.w2.bias"),
                                                   marker("ff.w2"))
    for name in ("norm1.weight", "norm2.weight"):
        block.get_submodule(name.rsplit(".", 1)[0]).weight.data.copy_(t[name])
    block.scale1.data.copy_(t["scale1"])
    block.scale2.data.copy_(t["scale2"])
    return block.to(device).eval()


def rotary(batch, device="cuda"):
    from fastvideo.models.vaes.minimax_h3_video import MiniMaxH3VideoRotaryPosEmbed

    frames, height, width = 7, 16, 16  # 1,792 patches per tile, plus 4 register tokens and the cls token
    grids = [2.0 * (torch.arange(0.5, s, dtype=torch.float32, device=device) / s) - 1.0
             for s in (frames, height, width)]
    pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    pos = torch.cat([pos, pos.new_zeros(5, 3)]).unsqueeze(0).expand(batch, -1, -1)
    return MiniMaxH3VideoRotaryPosEmbed(48, theta=100.0).to(device)(pos)


def int8_stats(ours, theirs):
    diff = (ours.int() - theirs.int()).abs()
    return {"mismatch": int((diff > 0).sum()), "max_diff": int(diff.max()), "elements": ours.numel()}


def close_stats(ours, ref):
    diff = (ours.float() - ref.float()).abs()
    return {"max_abs": diff.max().item(), "mean_abs": diff.mean().item(),
            "not_equal": int((ours != ref.to(ours.dtype)).sum()), "elements": ours.numel(),
            "rel_l2": (diff.norm() / ref.float().norm()).item()}


def timed(fn, reps):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    times = []
    for _ in range(reps):
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, nargs="+", default=[0, 13])
    parser.add_argument("--tiles", type=int, default=28)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--no-compile", action="store_true", help="skip timing FastVideo's block under inductor")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("H3MK_EXT_VERBOSE", "1")
    e = fb.ext()
    print("built, gemm smem", e.gemm_smem, flush=True)
    results = {"tiles": args.tiles, "tokens_per_tile": 1797, "blocks": []}
    autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    torch.manual_seed(0)
    for index in args.block:
        block = load_block(index)
        packed = fb.Packed(block)
        x = torch.randn(args.tiles, 1797, 2048, device="cuda")
        rot = rotary(args.tiles)
        m = x.shape[0] * x.shape[1]
        row = {"block": index, "rows": m}

        # rowquant against fastvideo's rotation and quantizer
        norm = F.rms_norm(x, (2048,), block.norm1.weight, block.norm1.eps).reshape(m, 2048)
        with autocast():
            fq, fs = block.attn.to_q.quantize_input(norm)
        a, s = e.rowquant(x.reshape(m, 2048), packed.norm1, packed.eps1)
        row["rowquant_norm_vs_fastvideo"] = {**int8_stats(a, fq[:m]), "scale": close_stats(s, fs[:m, 0].float())}
        h16 = torch.randn(m, 8192, device="cuda").half()
        with autocast():
            fq8, fs8 = block.ff.net[2].quantize_input(h16)
        a8, s8 = e.rowquant(h16, None, 0.0)
        row["rowquant_8192_vs_fastvideo"] = {**int8_stats(a8, fq8[:m]), "scale": close_stats(s8, fs8[:m, 0].float())}

        # each gemm against torch._int_mm and the eager dequant on our codes
        def dequant(a, s, w, ws, b):
            return torch._int_mm(a, w.t()).float() * s[:, None] * ws[None, :] + b[None, :]

        cos, sin = (t.reshape(m, 48).contiguous() for t in rot)
        q, k, v = e.gemm_qkv(a, s, packed.w_qkv, packed.s_qkv, packed.b_qkv, cos, sin, packed.qk_eps)
        full = dequant(a, s, packed.w_qkv, packed.s_qkv, packed.b_qkv)
        rq, rk, rv = (full[:, i * 2048:(i + 1) * 2048].reshape(m, 32, 64) for i in range(3))
        rq = F.rms_norm(rq, (64,), eps=packed.qk_eps)
        rk = F.rms_norm(rk, (64,), eps=packed.qk_eps)
        from fastvideo.models.vaes.minimax_h3_video import _apply_rotary

        c4, s4 = cos.view(m, 1, 48), sin.view(m, 1, 48)
        rq, rk = _apply_rotary(rq, rk, c4, s4)
        row["gemm_qkv"] = {"q": close_stats(q, rq.reshape(m, 2048).half()),
                           "k": close_stats(k, rk.reshape(m, 2048).half()),
                           "v": close_stats(v, rv.reshape(m, 2048).half())}
        xres = torch.randn(m, 2048, device="cuda")
        ours = xres.clone()
        e.gemm_residual(a, s, packed.w_out, packed.s_out, packed.b_out, packed.gate1, ours)
        ref = xres + dequant(a, s, packed.w_out, packed.s_out, packed.b_out) * packed.gate1
        row["gemm_residual"] = close_stats(ours, ref)
        h = e.gemm_swiglu(a, s, packed.w1, packed.s1, packed.b1)
        d = dequant(a, s, packed.w1, packed.s1, packed.b1)
        ref = (d[:, 0::2] * F.silu(d[:, 1::2])).half()
        row["gemm_swiglu"] = close_stats(h, ref)
        del full, rq, rk, rv, d, ref

        # the whole block against fp32, with fastvideo's eager block as the bar
        with autocast():
            ours = fb.forward(block, packed, x, rot)
            fastvideo = block(x, rot).float()
        ref = fb.reference_fp32(block, x, rot)
        row["block_vs_fp32"] = check.error_stats(ours - x, ref - x)
        row["fastvideo_block_vs_fp32"] = check.error_stats(fastvideo - x, ref - x)
        row["block_vs_fastvideo"] = check.error_stats(ours - x, fastvideo - x)
        row["passes"] = check.passes(row["block_vs_fp32"], row["fastvideo_block_vs_fp32"])
        del ref, fastvideo, ours

        # times: each kernel, our block, fastvideo's block eager and under inductor
        t = {}
        t["rowquant_norm_2048"] = timed(lambda: e.rowquant(x.reshape(m, 2048), packed.norm1, packed.eps1), args.reps)
        t["rowquant_8192"] = timed(lambda: e.rowquant(h16, None, 0.0), args.reps)
        t["gemm_qkv"] = timed(lambda: e.gemm_qkv(a, s, packed.w_qkv, packed.s_qkv, packed.b_qkv, cos, sin,
                                                 packed.qk_eps), args.reps)
        t["gemm_residual_2048"] = timed(lambda: e.gemm_residual(a, s, packed.w_out, packed.s_out, packed.b_out,
                                                                packed.gate1, xres), args.reps)
        t["gemm_swiglu"] = timed(lambda: e.gemm_swiglu(a, s, packed.w1, packed.s1, packed.b1), args.reps)
        t["gemm_residual_8192"] = timed(lambda: e.gemm_residual(a8, s8, packed.w2, packed.s2, packed.b2,
                                                                packed.gate2, xres), args.reps)
        with autocast():
            t["ours_block"] = timed(lambda: fb.forward(block, packed, x, rot), args.reps)
        t["int_mm_qkv_3x"] = timed(lambda: [torch._int_mm(a, w.t()) for w in packed.w_qkv.split(2048)], args.reps)
        t["int_mm_w1"] = timed(lambda: torch._int_mm(a, block.ff.net[0].proj.weight.t()), args.reps)
        with autocast():
            t["fastvideo_quantize_input"] = timed(lambda: block.attn.to_q.quantize_input(norm), args.reps)
            t["fastvideo_block_eager"] = timed(lambda: block(x, rot), max(3, args.reps // 4))
            if not args.no_compile:
                compiled = torch.compile(block)
                compiled(x, rot)
                t["fastvideo_block_inductor"] = timed(lambda: compiled(x, rot), args.reps)
        row["ms"] = t
        results["blocks"].append(row)
        print(json.dumps(row), flush=True)
        (OUT / "vae_block_check.json").write_text(json.dumps(results, indent=1))
        del block, packed


if __name__ == "__main__":
    main()
