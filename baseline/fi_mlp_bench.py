"""FlashInfer's MLP block in three launches vs in one launch (m1/fi_mlp.py), on block 0's weights.

Both must give bitwise the same output. The two take turns in each timed round, so clock drift on the unlocked
GPU hits them alike, and each round is compared as a pair. The one-launch build also records when each CTA ended
its norm rows, its FC1 tiles and its FC2 tiles; --plot draws that per CTA. With --profile, a few calls of each run
inside cudaProfilerStart/Stop for nsys.

    python fi_mlp_bench.py --tokens 256 1024 4096 14830 37730 --profile 4096 37730
    python fi_mlp_bench.py --plot ../results/fi-mlp-one-launch.json
"""
import argparse
import json
import os
from pathlib import Path

HIDDEN, TABLE_ROWS, EPS = 5376, 6, 1e-5
COLORS = {"norm": "#B4B2A9", "fc1": "#7F77DD", "fc2": "#1D9E75"}


def inputs(tokens, seed=0):
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed + tokens)
    r = lambda *shape, s=1.0: (torch.randn(*shape, generator=g, device="cuda") * s).to(torch.bfloat16)
    x = r(tokens, HIDDEN)
    norm_w = (1 + torch.randn(HIDDEN, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    scale, shift, gate = r(TABLE_ROWS, HIDDEN, s=0.1), r(TABLE_ROWS, HIDDEN, s=0.1), r(TABLE_ROWS, HIDDEN, s=0.1)
    idx = torch.randint(0, TABLE_ROWS, (tokens,), generator=g, device="cuda", dtype=torch.int64)
    return x, norm_w, scale, shift, idx, gate


def interleaved(names, call, iters, warmup=3):
    import torch

    from m1 import bench

    flusher = bench.L2Flusher()
    for _ in range(warmup):
        for n in names:
            call(n)
    events = {n: [] for n in names}
    for i in range(iters):
        for n in names[i % len(names):] + names[:i % len(names)]:
            flusher.flush()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call(n)
            end.record()
            events[n].append((start, end))
    torch.cuda.synchronize()
    ms = {n: [s.elapsed_time(e) for s, e in ev] for n, ev in events.items()}
    med = lambda v: sorted(v)[len(v) // 2]
    out = {n: {"median_ms": med(v), "p10_ms": sorted(v)[len(v) // 10], "p90_ms": sorted(v)[-len(v) // 10 - 1]}
           for n, v in ms.items()}
    diffs = [b - a for a, b in zip(ms[names[0]], ms[names[1]])]  # one_launch - flashinfer in the same round
    out["paired_saving_ms"] = -med(diffs)
    out["paired_saving"] = -med(diffs) / out[names[0]]["median_ms"]
    return out


def _panel(out, rows, x0, x1, y, h, left, width, label):
    scale = width / (x1 - x0)
    out.append(f'<text x="{left}" y="{y - 8}" font-size="13">{label}</text>')
    out.append(f'<rect x="{left}" y="{y}" width="{width}" height="{h}" fill="#F4F4F2" stroke="#BBB" stroke-width="0.5"/>')
    rh = h / len(rows)
    for i, p in enumerate(rows):
        for a, b, kind in ((p[0], p[1], "norm"), (p[1], p[2], "fc1"), (p[2], p[3], "fc2")):
            a, b = max(a, x0), min(b, x1)
            if b > a:
                out.append(f'<rect x="{left + (a - x0) * scale:.2f}" y="{y + i * rh:.2f}" '
                           f'width="{max(0.4, (b - a) * scale):.2f}" height="{rh:.2f}" fill="{COLORS[kind]}"/>')
    step = 10 ** max(0, len(str(int((x1 - x0) / 5))) - 1)
    while (x1 - x0) / step > 10:
        step *= 2
    t = (int(x0 / step) + 1) * step
    while t < x1:
        x = left + (t - x0) * scale
        out.append(f'<line x1="{x:.1f}" y1="{y + h}" x2="{x:.1f}" y2="{y + h + 4}" stroke="#666"/>'
                   f'<text x="{x:.1f}" y="{y + h + 16}" text-anchor="middle" font-size="11">{t:.0f}</text>')
        t += step


# one row per CTA, sorted by when its FC1 ended; the second panel zooms on the FC1 to FC2 hand-over
def plot(row, gpu):
    rows = sorted(row["one_launch_phases_us"], key=lambda p: p[2])
    end = max(p[3] for p in rows)
    lo, hi = min(p[2] for p in rows), max(p[2] for p in rows)
    w, left, width = 1100, 60, 1010
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="640" viewBox="0 0 {w} 640" '
           f'font-family="sans-serif" font-size="12">', f'<rect width="{w}" height="640" fill="white"/>',
           f'<text x="{left}" y="22" font-size="15">MLP block in one launch, {row["tokens"]} tokens, {len(rows)} CTAs '
           f'on one {gpu} (one row per CTA)</text>']
    for i, (kind, label) in enumerate((("norm", "norm"), ("fc1", "FC1 + SwiGLU"), ("fc2", "FC2 + residual"))):
        x = w - 330 + i * 100
        out.append(f'<rect x="{x}" y="12" width="12" height="12" fill="{COLORS[kind]}"/>'
                   f'<text x="{x + 16}" y="22">{label}</text>')
    _panel(out, rows, 0, end, 60, 230, left, width, f"whole launch: {end:.0f} us (time in us)")
    pad = (hi - lo) * 0.6 + 20
    _panel(out, rows, lo - pad, hi + pad, 360, 230, left, width,
           f"FC1 to FC2: each CTA starts FC2 when its own FC1 tiles end, up to {hi - lo:.0f} us before the last one")
    out.append("</svg>")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[256, 1024, 4096, 14830, 37730])
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--profile", type=int, nargs="*", default=None, help="token counts to record for nsys")
    parser.add_argument("--plot", default=None, help="only draw the svgs from a saved results json")
    args = parser.parse_args()

    if args.plot:
        data = json.loads(Path(args.plot).read_text())
        for row in data["rows"]:
            path = Path(args.plot).with_name(f"{Path(args.plot).stem}-{row['tokens']}.svg")
            path.write_text(plot(row, data["gpu"].replace("NVIDIA ", "")))
            print("wrote", path)
        return

    import torch

    from m1 import fi_mlp, weights

    w = weights.load("cuda")
    p = fi_mlp.pack(w.fc1, w.fc2)
    shipped, one = fi_mlp.flashinfer_module(), fi_mlp.build()
    names = ["flashinfer", "one_launch"]

    rows = []
    for tokens in args.tokens:
        a = inputs(tokens)
        calls = {"flashinfer": lambda: fi_mlp.run(*a, a[0], p, EPS, module=shipped),
                 "one_launch": lambda: fi_mlp.run_one_launch(*a, a[0], p, EPS, module=one)}
        ref = fi_mlp.run(*a, a[0], p, EPS)  # the public flashinfer api
        row = {"tokens": tokens}
        for n in names:
            out = calls[n]()
            torch.cuda.synchronize()
            row[f"{n}_equal"] = torch.equal(out, ref)
        row.update(interleaved(names, lambda n: calls[n](), args.iters))
        trace = torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count, 4, dtype=torch.int64,
                            device="cuda")
        fi_mlp.run_one_launch(*a, a[0], p, EPS, module=one, trace=trace)
        torch.cuda.synchronize()
        tr = trace.cpu()
        row["one_launch_phases_us"] = ((tr - int(tr[:, 0].min())).double() / 1e3).tolist()
        row["fc2_starts_before_last_fc1_ends_us"] = (int(tr[:, 2].max()) - int(tr[:, 2].min())) / 1e3
        rows.append(row)
        print(json.dumps({"tokens": tokens, **{f"{n}_equal": row[f"{n}_equal"] for n in names},
                          **{n: round(row[n]["median_ms"], 4) for n in names},
                          "paired_saving": round(row["paired_saving"], 4),
                          "fc2_starts_before_last_fc1_ends_us": row["fc2_starts_before_last_fc1_ends_us"]}),
              flush=True)
        del a, ref

    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for tokens in args.profile:
            a = inputs(tokens)
            for n, fn in (("flashinfer", lambda: fi_mlp.run(*a, a[0], p, EPS, module=shipped)),
                          ("one_launch", lambda: fi_mlp.run_one_launch(*a, a[0], p, EPS, module=one))):
                torch.cuda.nvtx.range_push(f"{n}/M{tokens}")
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()

    out = Path(os.environ.get("OUT_DIR", ".")) / "fi-mlp-one-launch.json"
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "rows": rows}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
