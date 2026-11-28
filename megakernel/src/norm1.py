"""Test for first op in DiT
Norm Mod Quant
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import torch
from fastvideo.models.dits.minimax_h3_fusions import fused_rmsnorm_modulate
from flashinfer import fp4_quantize
from torch.utils.cpp_extension import load

from instructions import BaseGlobals, NoOp, NormModQuant
from scheduler import DAG_Node, Schedule, assign_to_sms, tensorize_instructions

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]

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
class NormGlobals(BaseGlobals):
    # same order as globals_t in h3.cuh
    input_weights: torch.Tensor  # for add_one, dummy, maybe remove
    output_weights: torch.Tensor  # for add_one, dummy, maybe remove
    x: torch.Tensor
    norm_weight_vec: torch.Tensor
    adaln_table: torch.Tensor
    adaln_index: torch.Tensor
    packed: torch.Tensor
    scales: torch.Tensor


def make_schedule(globs, rows, gsf):
    """One NormModQuant per 128-row tile, no dependencies in DAG since there is one op"""
    gsf_bits = struct.unpack("i", struct.pack("f", gsf))[0]  # float32 bits for instruction word 4
    nodes = [DAG_Node(NormModQuant(r, 0, HIDDEN_DIM, gsf_bits), [])  # norm1: shift at 0, scale at H
             for r in range((rows + TILE_ROWS - 1) // TILE_ROWS)]
    end = DAG_Node(NoOp(), nodes)  # sink node; Schedule requires one
    return Schedule(globs, nodes, end)


def run(mk, rows, mode):
    H, T = HIDDEN_DIM, 6  # T table rows, e.g. 2 timesteps x 3 modalities
    x = torch.randn(rows, H, device="cuda").to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(H, device="cuda")).to(torch.bfloat16)
    table = (0.1 * torch.randn(T, 6 * H, device="cuda")).to(torch.bfloat16)  # shift1, scale1, gate1, shift2, scale2, gate2
    index = torch.randint(0, T, (rows,), device="cuda", dtype=torch.int32)

    # reference: FastVideo's norm + modulation, then FlashInfer's NVFP4 quantize (what the NVFP4 linear does)
    y = fused_rmsnorm_modulate(x, weight, table[:, H:2 * H], table[:, 0:H], index, 1e-5)
    gsf = 448.0 * 6.0 / y.float().abs().max().item()
    ref_packed, ref_scales = fp4_quantize(y, torch.tensor([gsf], device="cuda", dtype=torch.float32))

    num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS
    packed = torch.zeros(rows, H // 2, device="cuda", dtype=torch.uint8)
    scales = torch.zeros(num_tiles * (H // TILE_COLS), 512, device="cuda", dtype=torch.uint8)  # 84 chunks x 512 B per tile
    dummy = torch.zeros(TILE_ROWS, H, device="cuda", dtype=torch.bfloat16)
    globs = NormGlobals(device=x.device, input_weights=dummy, output_weights=dummy, x=x, norm_weight_vec=weight,
                        adaln_table=table, adaln_index=index, packed=packed, scales=scales)

    # these functions 
    schedule = make_schedule(globs, rows, gsf)
    queues = assign_to_sms(mode, schedule=schedule, memory_fraction=0.5)  # fraction only used by "pool"
    tensorize_instructions(globs, queues)  # pads with NoOps, sets globs.instructions / globs.timings

    # mk is pybind module from build() / h3.cu created in main
    mk.h3(globs.instructions, globs.timings, globs.input_weights, globs.output_weights, globs.x,
          globs.norm_weight_vec, globs.adaln_table, globs.adaln_index, globs.packed, globs.scales)  # 10 args
    torch.cuda.synchronize()

    # byte-level match; not exactly 1.0 since our sum of squares adds in a different order than Triton's
    packed_match = (packed == ref_packed.view(torch.uint8)).float().mean().item()
    scales_match = (scales.flatten() == ref_scales.view(torch.uint8).flatten()).float().mean().item()
    ok = packed_match > 0.999 and scales_match > 0.999
    print(f"mode={mode:4s}  rows={rows:5d}  tiles={len(schedule.dag_nodes):5d}  "
          f"per_sm={globs.instructions.shape[1]:3d}  packed_match={packed_match:.6f}  "
          f"scales_match={scales_match:.6f}  {'PASS' if ok else 'FAIL'}")
    print_timings(globs)
    return ok


def print_timings(globs):
    """Per-warp cycle counts from globs.timings, only for NormModQuant instructions
    Slots (megakernel/include/util.cuh:215): controller 0-4, loader start/end 5/6,
    storer 9/10, consumer warp w start/end 11+2w / 12+2w. Values are clock64()
    cycles since that SM's block started
    """
    t = globs.timings.cpu().long()
    is_op = globs.instructions[..., 0].cpu() == NormModQuant.opcode()
    rows = t[is_op]  # [num_norm_instructions, 128]
    if rows.numel() == 0:
        return

    def span(a, b):
        d = (rows[:, b] - rows[:, a]).float()
        return f"median={d.median().item():8.0f}  max={d.max().item():8.0f}"

    print(f"    timings over {rows.shape[0]} NormModQuant instructions (cycles):")
    print(f"      controller (0->4): {span(0, 4)}")
    print(f"      loader     (5->6): {span(5, 6)}")
    print(f"      storer    (9->10): {span(9, 10)}")
    for w in range(8):  # NUM_CONSUMER_WARPS
        print(f"      consumer {w} ({11 + 2*w}->{12 + 2*w}): {span(11 + 2*w, 12 + 2*w)}")
    print(f"      raw slots 0-26, first NormModQuant: {rows[0, :27].tolist()}")


def main():
    mk = build()
    # 128 rows is one tile,
    # 300 rows: partial last tile (rows past S must be clipped, not written).
    # 37710 rows: real H3 sequence (37296 video + 414 audio), 295 tiles > 188 SMs,
    #   so some SMs run two back to back, which exercises page handoff between instructions.
    # round robin: tiles are independent and equal cost, so spreading them evenly is all that matters
    results = [run(mk, rows, "rr") for rows in (128, 300, 37710)]
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
