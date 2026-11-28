"""Test for AddOneOp: output = input + 1, one 128x64 tile per instruction.

Run on the GPU machine (RTX PRO 6000):  python megakernel/src/add_one.py
The first run compiles the extension, which takes a minute or two.
"""

from dataclasses import dataclass
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

SRC = Path(__file__).resolve().parents[1]  # megakernel/src: h3.cu, instructions.py, scheduler.py
ROOT = SRC.parents[1]
sys.path.insert(0, str(SRC))

from instructions import AddOne, BaseGlobals, NoOp
from scheduler import DAG_Node, Schedule, assign_to_sms, tensorize_instructions


# must match megakernel/src/h3.cuh
HIDDEN_DIM = 5376
TILE_ROWS, TILE_COLS = 128, 64


def build():
    return load(
        name="h3_mk",
        sources=[str(SRC / "h3.cu")],
        extra_include_paths=[
            str(SRC),
            str(ROOT / "megakernel" / "include"),
            str(ROOT / "ThunderKittens" / "include"),
        ],
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-DKITTENS_HOPPER",  # TMA + mbarrier ops; sm_120 has no tcgen05, so no KITTENS_BLACKWELL
            "-gencode=arch=compute_120a,code=sm_120a",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-Xcompiler=-fno-strict-aliasing",
            "-Xcompiler=-Wno-psabi",
        ],
        extra_cflags=["-std=c++20"],
        extra_ldflags=["-lcuda"],
        verbose=True,
    )


@dataclass
class AddOneGlobals(BaseGlobals):
    input: torch.Tensor
    output: torch.Tensor


def make_schedule(globs, rows):
    """One AddOne per 128x64 tile. The tiles don't depend on each other,
    so every DAG node has an empty dependency list."""
    nodes = [DAG_Node(AddOne(r, c), [])
             for r in range(rows // TILE_ROWS)
             for c in range(HIDDEN_DIM // TILE_COLS)]
    end = DAG_Node(NoOp(), nodes)  # sink node; Schedule requires one
    return Schedule(globs, nodes, end)


def run(mk, rows, mode):
    assert rows % TILE_ROWS == 0
    # small integers, so input + 1 is exact in bf16
    x = torch.randint(-64, 64, (rows, HIDDEN_DIM), device="cuda").to(torch.bfloat16)
    out = torch.full_like(x, float("nan"))  # NaN marks any tile the kernel never wrote
    globs = AddOneGlobals(device=x.device, input=x, output=out)

    # these functions 
    schedule = make_schedule(globs, rows)
    queues = assign_to_sms(mode, schedule=schedule, memory_fraction=0.5)  # fraction only used by "pool"
    tensorize_instructions(globs, queues)  # pads with NoOps, sets globs.instructions / globs.timings

    # mk is pybind module from build() / h3.cu created in main
    mk.h3(globs.instructions, globs.timings, globs.input, globs.output) # 4 args
    torch.cuda.synchronize()

    missing = torch.isnan(out).sum().item()
    max_err = (out.float() - (x.float() + 1)).abs().nan_to_num(0).max().item()
    ok = missing == 0 and max_err == 0
    print(f"mode={mode:4s}  rows={rows:5d}  tiles={len(schedule.dag_nodes):5d}  "
          f"per_sm={globs.instructions.shape[1]:3d}  unwritten={missing}  "
          f"max_err={max_err}  {'PASS' if ok else 'FAIL'}")
    print_timings(globs)
    return ok


def print_timings(globs):
    """Per-warp cycle counts from globs.timings, only for AddOne instructions.

    Slots (megakernel/include/util.cuh:215): controller 0-4, loader start/end 5/6,
    storer 9/10, consumer warp w start/end 11+2w / 12+2w. Values are clock64()
    cycles since that SM's block started.
    """
    t = globs.timings.cpu().long()
    is_add_one = globs.instructions[..., 0].cpu() == AddOne.opcode()
    rows = t[is_add_one]  # [num_add_one_instructions, 128]
    if rows.numel() == 0:
        return

    def span(a, b):
        d = (rows[:, b] - rows[:, a]).float()
        return f"median={d.median().item():8.0f}  max={d.max().item():8.0f}"

    print(f"    timings over {rows.shape[0]} AddOne instructions (cycles):")
    print(f"      controller (0->4): {span(0, 4)}")
    print(f"      loader     (5->6): {span(5, 6)}")
    print(f"      storer    (9->10): {span(9, 10)}")
    for w in range(8):  # NUM_CONSUMER_WARPS
        print(f"      consumer {w} ({11 + 2*w}->{12 + 2*w}): {span(11 + 2*w, 12 + 2*w)}")
    print(f"      raw slots 0-26, first AddOne: {rows[0, :27].tolist()}")


def main():
    mk = build()
    # 128 rows: fewer tiles than SMs (one instruction each, the rest NoOp).
    # 2048 rows: several add-ones back to back per SM, which exercises page reuse.
    # every scheduler mode should give the same answer; only the SM placement differs
    results = [run(mk, rows, mode)
               for rows in (128, 2048)
               for mode in ("rr", "zz", "wave", "dag", "pool")]
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
