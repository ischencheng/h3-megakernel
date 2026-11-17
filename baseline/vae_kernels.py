"""Group the video decoder's kernel time by kind, per arm, from an nsys export (request.sqlite).

  python baseline/vae_kernels.py <request.sqlite> [--json out.json]
Each arm runs inside an "arm.<name>" NVTX range (fastvideo_capacity.py --arms) and its decoder inside
"minimax_h3.vae"; without arm ranges the whole report is one arm.
"""
import argparse
import json
import re
import sqlite3

# first match wins
KINDS = [
    ("int8 GEMM (ours, fused epilogue)", r"int8_gemm_kernel"),
    ("rotation + quantize (ours)", r"rowquant_kernel"),
    ("int8 GEMM", r"i16832gemm_s8|igemm|_int_mm"),
    ("rotation GEMM (fp16)", r"f16_s16816gemm_relu_f16_128x256_32x3_nn"),
    ("row quantize", r"abs_amax_clamp_clamp_min_div_round"),
    ("attention", r"flash_fwd|fmha|sdpa_kernel"),
    ("dequant + swiglu", r"silu_split"),
    ("norm / residual / dequant", r"rms_norm"),
    ("qk norm + rope", r"_cos_|_sin_"),
    ("attention input dequant + cast", r"scaled_dot_product"),
    ("proj_in / proj_out GEMM", r"gemm"),
]


def kind(name):
    for label, pattern in KINDS:
        if re.search(pattern, name):
            return label
    return "other"


def ranges(db, text):
    return db.execute("select start, end from NVTX_EVENTS where text = ? order by start", (text,)).fetchall()


def summarize(path):
    db = sqlite3.connect(path)
    arms = db.execute("select text, start, end from NVTX_EVENTS where text like 'arm.%' order by start").fetchall()
    if not arms:
        arms = [("all", 0, 1 << 62)]
    vae = ranges(db, "minimax_h3.vae")
    out = {}
    for arm, a0, a1 in arms:
        windows = [(s, e) for s, e in vae if s >= a0 and e <= a1]
        rows, total, count = {}, 0, 0
        for s, e in windows:
            for name, dur in db.execute(
                    "select sn.value, k.end - k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds sn "
                    "on k.demangledName = sn.id where k.start >= ? and k.end <= ?", (s, e)):
                label = kind(name)
                r = rows.setdefault(label, {"kernels": 0, "ms": 0.0})
                r["kernels"] += 1
                r["ms"] += dur / 1e6
                total += dur
                count += 1
        span = sum(e - s for s, e in windows) / 1e6
        out[arm.removeprefix("arm.")] = {"vae_span_ms": span, "kernel_ms": total / 1e6, "kernels": count,
                                          "idle_ms": span - total / 1e6, "by_kind": rows}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    out = summarize(args.sqlite)
    for arm, r in out.items():
        print(f"{arm}: decoder {r['vae_span_ms']:.1f} ms, {r['kernels']} kernels, {r['kernel_ms']:.1f} ms of kernels")
        for label, k in sorted(r["by_kind"].items(), key=lambda kv: -kv[1]["ms"]):
            print(f"  {label:36s} {k['kernels']:6d} {k['ms']:9.1f} ms")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
