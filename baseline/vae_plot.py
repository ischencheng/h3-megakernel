"""Draw the video decoder charts from the run summaries.

  python baseline/vae_plot.py --requests <summary.json>... --block <vae_block_check.json> --out results/
writes video-decoder-request.svg (decoder stage and whole request, FastVideo against the fused blocks, per shape)
and video-decoder-block.svg (one decoder block, FastVideo's kernels against ours).
"""
import argparse
import json
from pathlib import Path

COLORS = {"fastvideo": "#7f8c8d", "fused": "#2e86c1"}
LABELS = {"fastvideo": "FastVideo", "fused": "fused blocks"}


def _bars(out, groups, top, height, left, width, title, unit):
    """groups: [(label, {arm: value})]; one cluster of bars per group, values labelled on top"""
    peak = max(v for _, arms in groups for v in arms.values()) * 1.15
    out.append(f'<text x="{left}" y="{top - 12}" font-size="14">{title}</text>')
    out.append(f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#333"/>')
    slot = width / len(groups)
    for i, (label, arms) in enumerate(groups):
        bar = slot * 0.7 / len(arms)
        x0 = left + i * slot + slot * 0.15
        for j, (arm, value) in enumerate(arms.items()):
            h = height * value / peak
            x = x0 + j * bar
            out.append(f'<rect x="{x:.1f}" y="{top + height - h:.1f}" width="{bar * 0.9:.1f}" height="{h:.1f}" '
                       f'fill="{COLORS[arm]}"/>')
            out.append(f'<text x="{x + bar * 0.45:.1f}" y="{top + height - h - 4:.1f}" text-anchor="middle">'
                       f'{value:.2f}{unit}</text>')
        out.append(f'<text x="{left + i * slot + slot / 2:.1f}" y="{top + height + 18}" text-anchor="middle">'
                   f'{label}</text>')


def _svg(width, height, body, legend_arms):
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
            f'<rect width="{width}" height="{height}" fill="white"/>']
    for i, arm in enumerate(legend_arms):
        x = width - 260 + i * 130
        head.append(f'<rect x="{x}" y="14" width="12" height="12" fill="{COLORS[arm]}"/>'
                    f'<text x="{x + 16}" y="24">{LABELS[arm]}</text>')
    return "\n".join(head + body + ["</svg>"])


def request_chart(summaries):
    decoder, request = [], []
    for path in summaries:
        runs = json.loads(Path(path).read_text())["runs"]
        by_arm = {r["arm"]: r for r in runs if r.get("arm") in COLORS and r["rep"] > 0}
        shape = runs[0]["shape"]
        decoder.append((shape, {a: by_arm[a]["stages"]["video_decoding_stage"] for a in COLORS if a in by_arm}))
        request.append((shape, {a: by_arm[a]["client_s_without_checks"] for a in COLORS if a in by_arm}))
    body = []
    _bars(body, decoder, 70, 220, 60, 420, "Video decoder stage (s)", " s")
    _bars(body, request, 70, 220, 560, 420, "Whole request (s)", " s")
    return _svg(1040, 340, body, list(COLORS))


def block_chart(check):
    row = check["blocks"][-1]["ms"]
    ours = min(v for k, v in row.items() if k.startswith("ours_block"))
    groups = [("one block", {"fastvideo": row["fastvideo_block_inductor"], "fused": ours})]
    body = []
    _bars(body, groups, 70, 220, 60, 300, f"One decoder block, {check['tiles']} tiles of {check['tokens_per_tile']} "
          "tokens (ms)", " ms")
    return _svg(420, 340, body, list(COLORS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", nargs="*", default=[])
    parser.add_argument("--block", default=None)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    if args.requests:
        (out / "video-decoder-request.svg").write_text(request_chart(args.requests))
    if args.block:
        (out / "video-decoder-block.svg").write_text(block_chart(json.loads(Path(args.block).read_text())))


if __name__ == "__main__":
    main()
