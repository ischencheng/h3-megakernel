"""Milestone 1 driver: check every variant against FP32, then time the ones that pass.

  python -m m1.run                         # all token counts, all variants
  python -m m1.run --tokens 256 4096 --variants cublas split fused_g2048
  python -m m1.run --profile-once ...      # one call per variant inside an NVTX
                                           # range "mlp", for ncu --nvtx-include "mlp/"
"""

import argparse
import json
import os
import subprocess
import time

import torch

from . import bench, check, config, kernels, reference, weights


def _flops(tokens):
    return 2 * tokens * (config.FC1_OUT * config.HIDDEN + config.HIDDEN * config.FFN)


def _device_info():
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "l2_bytes": getattr(props, "L2_cache_size", None),
        "memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _commit():
    if os.environ.get("H3MK_COMMIT"):
        return os.environ["H3MK_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _first_line(exc):
    return f"{type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ''}"


def run_checked(args, table, w1, w2):
    rows = []
    for tokens in args.tokens:
        x = reference.make_input(tokens)
        ref = reference.mlp_fp32(x, w1, w2)
        bar = check.error_stats(reference.mlp_cublas(x, w1, w2), ref)
        for name in args.variants:
            fn, as_graph = table[name]
            call = lambda fn=fn: fn(x, w1, w2)
            row = {"tokens": tokens, "variant": name, "bar": bar}
            try:
                y = call()
                torch.cuda.synchronize()
            except kernels.NotImplementedKernel:
                rows.append({**row, "status": "not_implemented"})
                continue
            except Exception as exc:  # a build or launch failure should not stop the sweep
                rows.append({**row, "status": "error", "error": _first_line(exc)})
                continue
            stats = check.error_stats(y, ref)
            row.update(stats)
            if not check.passes(stats, bar):
                rows.append({**row, "status": "wrong"})
                continue
            timed = bench.time_fn(bench.capture_graph(call) if as_graph else call,
                                  name=f"{name}/M{tokens}", warmup=args.warmup,
                                  iters=args.iters, flush=not args.no_flush)
            timed["tflops"] = _flops(tokens) / (timed["median_ms"] * 1e-3) / 1e12
            rows.append({**row, **timed, "status": "ok"})
        del x, ref
        torch.cuda.empty_cache()
    return rows


def run_profile_once(args, table, w1, w2):
    for tokens in args.tokens:
        x = reference.make_input(tokens)
        for name in args.variants:
            fn, _ = table[name]
            try:
                fn(x, w1, w2)  # outside the range: builds, picks cuBLAS heuristics
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"skip {name} M={tokens}: {_first_line(exc)}")
                continue
            torch.cuda.nvtx.range_push("mlp")
            fn(x, w1, w2)
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            print(f"profiled {name} M={tokens}")


def print_table(rows):
    ok = {(r["tokens"], r["variant"]): r for r in rows if r["status"] == "ok"}
    print(f"{'tokens':>7} {'variant':<14} {'status':<16} {'rel_l2':>9} {'max/rms':>9} "
          f"{'median ms':>10} {'spread':>7} {'TFLOP/s':>8} {'vs cublas_graph':>16}")
    for r in rows:
        base = ok.get((r["tokens"], "cublas_graph"))
        line = f"{r['tokens']:>7} {r['variant']:<14} {r['status']:<16}"
        if "rel_l2" in r:
            line += f" {r['rel_l2']:>9.2e} {r['max_over_rms']:>9.2e}"
        if r["status"] == "ok":
            ratio = base["median_ms"] / r["median_ms"] if base else float("nan")
            line += f" {r['median_ms']:>10.3f} {r['spread']:>7.1%} {r['tflops']:>8.1f} {ratio:>15.3f}x"
        elif r["status"] == "error":
            line += f"  {r['error']}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(config.TOKENS))
    parser.add_argument("--groups", type=int, nargs="+", default=list(config.GROUP_TOKENS))
    parser.add_argument("--variants", nargs="+", default=None, help="default: all")
    parser.add_argument("--warmup", type=int, default=config.WARMUP)
    parser.add_argument("--iters", type=int, default=config.ITERS)
    parser.add_argument("--no-flush", action="store_true", help="keep L2 warm between runs")
    parser.add_argument("--profile-once", action="store_true")
    parser.add_argument("--out", default=None, help="results JSON path")
    args = parser.parse_args()

    table = kernels.variants(args.groups)
    args.variants = args.variants or list(table)
    unknown = [v for v in args.variants if v not in table]
    if unknown:
        parser.error(f"unknown variants {unknown}; choose from {list(table)}")

    device = _device_info()
    print(json.dumps(device))
    w1, w2 = weights.load("cuda")

    if args.profile_once:
        run_profile_once(args, table, w1, w2)
        return

    rows = run_checked(args, table, w1, w2)
    print_table(rows)
    result = {
        "device": device,
        "commit": _commit(),
        "config": {"tokens": args.tokens, "groups": args.groups, "warmup": args.warmup,
                   "iters": args.iters, "l2_flush": not args.no_flush,
                   "error_factor": config.ERROR_FACTOR, "hf_revision": config.HF_REVISION},
        "rows": rows,
    }
    if args.out:
        out = args.out
    else:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        slug = device["name"].replace("NVIDIA ", "").replace(" ", "-")
        out = config.RESULTS_DIR / f"m1-{slug}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
