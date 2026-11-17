"""Host compiler for the FastH3 DiT on the Hazy megakernel runtime.

`Program` is checkpoint independent: it registers device buffers in a pointer
table, emits typed 32-word instructions for the operations in
`src/ops/dit_*.cuh`, assigns one cross-SM wave per stage, and launches the
whole stream once. `DiTRecipe` describes one checkpoint and request (weights,
packed layout, VSA policy, schedule); `compile_dit` turns it into a program
covering text refinement, every denoising step and the latent updates.

Nothing here allocates or launches GPU work between launch and readback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct
from typing import Any

import torch

INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
WAVE_WORD = INSTRUCTION_WIDTH - 1

# Flags mirrored from src/ops/dit_ops.cuh and src/ops/dit_attention.cuh.
NORM_MOD, NORM_RESIDUAL, NORM_ROUNDED, NORM_DOUBLE_ROUND = 1, 2, 4, 8
NORM_OUT_F32, NORM_NO_OUTPUT, NORM_UNIT_GATE = 16, 32, 64
QUANT_GATHER, QUANT_SWIGLU = 1, 2
LIN_IN_BF16, LIN_W_BF16, LIN_OUT_BF16, LIN_BIAS_BF16, LIN_SCATTER, LIN_GATHER = 1, 2, 4, 8, 16, 32
QK_ROPE, QK_TILE, QK_COPY_V, QK_COPY_GATE = 1, 2, 4, 8
EW_SILU, EW_FLOW_UPDATE, EW_TIMESTEP = 1, 2, 3
HEAD_DIM, TILE = 128, 64


def float_bits(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", value))[0]


def _check_int32(words: list[int]) -> None:
    for word in words:
        if not -2**31 <= word < 2**31:
            raise ValueError(f"instruction word {word} does not fit int32")


class Program:
    """One Hazy instruction stream over a table of device buffers."""

    def __init__(self, module: Any, *, device: str = "cuda", max_projections: int = 512):
        self.module = module
        self.device = torch.device(device)
        self.op = module.opcodes()
        self.sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._tensors: list[torch.Tensor] = []
        self._index: dict[tuple[int, int], int] = {}
        self.named: dict[str, torch.Tensor] = {}
        self.instructions: list[list[int]] = []
        self.labels: list[str] = []
        self._params_words = module.params_words()
        self._params = torch.zeros(max_projections * self._params_words * 4,
                                   device=self.device, dtype=torch.int32)
        self._max_projections = max_projections
        self._projections = 0
        self._projection_index: dict[tuple, int] = {}
        self._sm_ids: list[int] | None = None
        self.last_gpu_ms: float | None = None
        # Consumer warp 0 start/end clocks per instruction and SM, int32 [slots, instructions, 2],
        # filled by run() when the module is built with TIMINGS=1.
        self.last_timings: torch.Tensor | None = None

    # -- buffers ------------------------------------------------------------
    def buf(self, tensor: torch.Tensor) -> int:
        """Register a contiguous device tensor (or view) and return its index."""
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("megakernel buffers must be contiguous CUDA tensors")
        key = (tensor.data_ptr(), tensor.numel() * tensor.element_size())
        if key not in self._index:
            self._index[key] = len(self._tensors)
            self._tensors.append(tensor)
        return self._index[key]

    def alloc(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Preallocate a zeroed activation or scratch buffer."""
        if name in self.named:
            raise ValueError(f"duplicate buffer {name}")
        tensor = torch.zeros(shape, device=self.device, dtype=dtype)
        self.named[name] = tensor
        self.buf(tensor)
        return tensor

    def const(self, name: str, tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Keep a host-prepared descriptor or weight view alive on the device."""
        tensor = tensor.detach().to(device=self.device, dtype=dtype or tensor.dtype).contiguous()
        self.named[name] = tensor
        self.buf(tensor)
        return tensor

    # -- instructions -------------------------------------------------------
    def emit(self, label: str, opcode: str, *words: int) -> None:
        row = [self.op[opcode], *[int(w) for w in words]]
        if len(row) > WAVE_WORD:
            raise ValueError(f"{label}: too many instruction words")
        row += [0] * (INSTRUCTION_WIDTH - len(row))
        row[WAVE_WORD] = len(self.instructions) + 1  # single-use wave counter
        _check_int32(row)
        self.instructions.append(row)
        self.labels.append(label)

    def projection(self, label: str, packed: torch.Tensor, packed_sf: torch.Tensor,
                   weight: torch.Tensor, weight_sf: torch.Tensor, out: torch.Tensor,
                   rows: int, cols: int, depth: int, alpha: float) -> None:
        """NVFP4 x NVFP4 -> BF16 GEMM; CUTLASS parameters are prepared here."""
        if cols % 128 or depth % 128:
            raise ValueError(f"{label}: projection needs 128-aligned columns and depth")
        if out.numel() < rows * cols or packed.numel() < rows * depth // 2:
            raise ValueError(f"{label}: projection buffer too small")
        for tensor in (packed, packed_sf, weight, weight_sf, out):
            self.buf(tensor)  # keep alive
        # One parameter record per distinct GEMM; every denoising step reuses it.
        key = (packed.data_ptr(), packed_sf.data_ptr(), weight.data_ptr(), weight_sf.data_ptr(),
               out.data_ptr(), rows, cols, depth, float(alpha))
        index = self._projection_index.get(key)
        if index is None:
            if self._projections == self._max_projections:
                raise ValueError("projection parameter table is full")
            index = self._projections
            self.module.prepare(self._params.data_ptr() + index * self._params_words * 4,
                                packed.data_ptr(), weight.data_ptr(), packed_sf.data_ptr(),
                                weight_sf.data_ptr(), out.data_ptr(), rows, cols, depth,
                                float(alpha), self.sm_count)
            self._projections += 1
            self._projection_index[key] = index
        self.emit(label, "PROJECTION", index)

    def rmsnorm(self, label: str, *, rows: int, cols: int, eps: float, flags: int,
                x: torch.Tensor, out: torch.Tensor | None, weight: torch.Tensor,
                index: torch.Tensor | None = None, table: torch.Tensor | None = None,
                table_stride: int = 0, scale: int = 0, shift: int = 0, gate: int = 0,
                branch: torch.Tensor | None = None) -> None:
        none = -1
        self.emit(label, "RMSNORM_MOD", rows, cols, float_bits(eps), flags, self.buf(x),
                  self.buf(out) if out is not None else none, self.buf(weight), 0,
                  self.buf(index) if index is not None else none,
                  self.buf(table) if table is not None else none, table_stride,
                  scale, shift, gate, self.buf(branch) if branch is not None else none)

    def quantize(self, label: str, *, rows: int, cols: int, global_sf: float, flags: int,
                 x: torch.Tensor, packed: torch.Tensor, packed_sf: torch.Tensor,
                 stride: int, gather: torch.Tensor | None = None) -> None:
        if cols % 64:
            raise ValueError(f"{label}: NVFP4 quantization needs 64-aligned columns")
        padded_rows = (rows + 127) // 128 * 128
        if packed_sf.numel() < padded_rows * (cols // 16) or packed.numel() < rows * cols // 2:
            raise ValueError(f"{label}: quantization scratch too small")
        self.emit(label, "NVFP4_QUANT", rows, cols, float_bits(global_sf), flags, self.buf(x),
                  self.buf(packed), self.buf(packed_sf),
                  self.buf(gather) if gather is not None else -1, stride)

    def linear(self, label: str, *, rows: int, x: torch.Tensor, weight: torch.Tensor,
               bias: torch.Tensor | None, out: torch.Tensor, in_stride: int | None = None,
               out_stride: int | None = None, index: torch.Tensor | None = None,
               scatter: bool = False, gather: bool = False, out_offset: int = 0) -> None:
        n_cols, k_cols = weight.shape
        flags = 0
        for tensor, flag in ((x, LIN_IN_BF16), (weight, LIN_W_BF16), (out, LIN_OUT_BF16),
                             (bias, LIN_BIAS_BF16)):
            if tensor is None:
                continue
            if tensor.dtype == torch.bfloat16:
                flags |= flag
            elif tensor.dtype != torch.float32:
                raise TypeError(f"{label}: dense linear supports BF16 and FP32 only")
        flags |= (LIN_SCATTER if scatter else 0) | (LIN_GATHER if gather else 0)
        self.emit(label, "DENSE_LINEAR", rows, k_cols, n_cols, flags, self.buf(x),
                  self.buf(weight), self.buf(bias) if bias is not None else -1, self.buf(out),
                  self.buf(index) if index is not None else -1,
                  in_stride if in_stride is not None else k_cols,
                  out_stride if out_stride is not None else n_cols, out_offset)

    def elementwise(self, label: str, kind: int, count: int, x: torch.Tensor,
                    out: torch.Tensor, *, scalar: int = 0, second: torch.Tensor | None = None,
                    ratio: float = 0.0) -> None:
        flags = (1 if x.dtype == torch.bfloat16 else 0) | (2 if out.dtype == torch.bfloat16 else 0)
        self.emit(label, "ELEMENTWISE", count, kind, flags, self.buf(x), self.buf(out), scalar,
                  self.buf(second) if second is not None else -1, float_bits(ratio))

    def copy_rows(self, label: str, *, rows: int, cols: int, x: torch.Tensor, out: torch.Tensor,
                  index: torch.Tensor | None = None, in_stride: int | None = None,
                  out_stride: int | None = None, swiglu: bool = False) -> None:
        self.emit(label, "COPY_ROWS", rows, cols, self.buf(x), self.buf(out),
                  self.buf(index) if index is not None else -1,
                  in_stride if in_stride is not None else cols,
                  out_stride if out_stride is not None else cols, int(swiglu))

    def qknorm(self, label: str, *, rows: int, eps: float, flags: int, heads: int, rotary: int,
               q: torch.Tensor, k: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor,
               q_weight: torch.Tensor, k_weight: torch.Tensor,
               cos: torch.Tensor | None = None, sin: torch.Tensor | None = None,
               tile_index: torch.Tensor | None = None, v: torch.Tensor | None = None,
               gate: torch.Tensor | None = None, v_out: torch.Tensor | None = None,
               gate_out: torch.Tensor | None = None) -> None:
        opt = lambda t: self.buf(t) if t is not None else -1  # noqa: E731
        self.emit(label, "QKNORM_TILE", rows, float_bits(eps), flags, heads, rotary, self.buf(q),
                  self.buf(k), self.buf(q_out), self.buf(k_out), self.buf(q_weight),
                  self.buf(k_weight), opt(cos), opt(sin), opt(tile_index), opt(v), opt(gate),
                  opt(v_out), opt(gate_out))

    # -- launch -------------------------------------------------------------
    def _launch(self, stream: torch.Tensor, sm_ids: list[int] | None):
        slots = 1024 if sm_ids is None else max(sm_ids) + 1
        count = stream.shape[0]
        instructions = torch.zeros((slots, count, INSTRUCTION_WIDTH), device=self.device,
                                   dtype=torch.int32)
        if sm_ids is None:
            instructions[:] = stream
        else:
            instructions[torch.tensor(sm_ids, device=self.device)] = stream
        timings = torch.zeros((slots, count, TIMING_WIDTH), device=self.device, dtype=torch.int32)
        counts = torch.zeros(self.sm_count, device=self.device, dtype=torch.int32)
        ids = torch.full((self.sm_count,), -1, device=self.device, dtype=torch.int32)
        waves = torch.zeros(count + 1, device=self.device, dtype=torch.int32)
        error = torch.zeros(1, device=self.device, dtype=torch.int32)
        table = torch.tensor([t.data_ptr() for t in self._tensors] or [0], device=self.device,
                             dtype=torch.int64)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        self.module.run(instructions, timings, counts, ids, waves, error, self._params, table,
                        self.sm_count, stream=torch.cuda.current_stream())
        end.record()
        torch.cuda.synchronize()
        self._timings = timings
        return counts, ids, waves, error, start.elapsed_time(end)

    def discover_sms(self) -> list[int]:
        """Measure the physical SM ID of every resident CTA with a smoke launch."""
        if self._sm_ids is None:
            probe = torch.zeros((1, INSTRUCTION_WIDTH), device=self.device, dtype=torch.int32)
            probe[0, 0] = self.op["SMOKE"]
            _, ids, _, _, _ = self._launch(probe, None)
            measured = ids.cpu().tolist()
            if len(set(measured)) != self.sm_count or min(measured) < 0 or max(measured) >= 1024:
                raise RuntimeError("physical SM discovery failed")
            self._sm_ids = measured
        return self._sm_ids

    def profile(self) -> list[tuple[str, float]]:
        """Diagnostic only: one launch per instruction, returning (label, ms).

        Buffers carry state between launches, so the result equals `run`.
        """
        sm_ids = self.discover_sms()
        smoke = [self.op["SMOKE"]] + [0] * (INSTRUCTION_WIDTH - 1)
        rows = []
        for label, instruction in zip(self.labels, self.instructions, strict=True):
            single = list(instruction)
            single[WAVE_WORD] = 1
            stream = torch.tensor([smoke, single, smoke], device=self.device, dtype=torch.int32)
            counts, _, waves, error, gpu_ms = self._launch(stream, sm_ids)
            if int(error[0]) or int(waves[0]) != self.sm_count or int(counts.min()) != 2:
                raise RuntimeError(f"profiled instruction failed: {label}")
            rows.append((label, gpu_ms))
        return rows

    def run(self) -> float:
        """Launch the whole stream once; return GPU milliseconds."""
        if not self.instructions:
            raise ValueError("empty program")
        sm_ids = self.discover_sms()
        smoke = [self.op["SMOKE"]] + [0] * (INSTRUCTION_WIDTH - 1)
        stream = torch.tensor([smoke, *self.instructions, smoke], device=self.device,
                              dtype=torch.int32)
        counts, _, waves, error, gpu_ms = self._launch(stream, sm_ids)
        failed = int(error[0])
        if failed:
            raise RuntimeError(f"cross-SM wave timed out at instruction {failed - 1}: "
                               f"{self.labels[failed - 1]}")
        expected = torch.full((len(self.instructions),), self.sm_count, device=self.device,
                              dtype=torch.int32)
        if not torch.equal(waves[:len(self.instructions)], expected):
            raise RuntimeError("a stage did not complete on every resident SM")
        if not torch.equal(counts, torch.full_like(counts, 2)):
            raise RuntimeError("Hazy instruction stream did not complete")
        self.last_gpu_ms = gpu_ms
        # Runtime event slots 11 and 12 are consumer warp 0's start and end.
        self.last_timings = self._timings[sm_ids][:, 1:len(self.instructions) + 1, 11:13].cpu()
        self._timings = None
        return gpu_ms

    def timing_summary(self, gpu_ms: float | None = None) -> dict:
        """Per-instruction stage time from the runtime's SM clocks.

        Each record is a truncated 32-bit clock difference, so durations are
        taken modulo 2^32 (valid below ~1.7 s per instruction at 2.4 GHz).
        An instruction's stage time is its longest per-SM duration; it
        includes the trailing wave barrier. Cycles are converted to ms with
        the measured launch time, since clocks are not locked.
        """
        import re

        if self.last_timings is None or not int(self.last_timings.abs().sum()):
            raise RuntimeError("no runtime timings; build the module with TIMINGS=1")
        start = self.last_timings[..., 0].to(torch.int64) & 0xFFFFFFFF
        end = self.last_timings[..., 1].to(torch.int64) & 0xFFFFFFFF
        cycles = (end - start) & 0xFFFFFFFF                      # [SMs, instructions]
        stage = cycles.max(dim=0).values.double()
        total = float(cycles.sum(dim=1).double().median())
        ms_per_cycle = (gpu_ms or self.last_gpu_ms) / total
        groups: dict[str, dict] = {}
        for label, value in zip(self.labels, stage.tolist(), strict=True):
            key = re.sub(r"step\d+", "step*", re.sub(r"block\d+", "block*", label))
            row = groups.setdefault(key, {"count": 0, "ms": 0.0})
            row["count"] += 1
            row["ms"] += value * ms_per_cycle
        ordered = dict(sorted(groups.items(), key=lambda item: -item[1]["ms"]))
        return {"ms_per_cycle": ms_per_cycle, "sum_of_stage_ms": float(stage.sum()) * ms_per_cycle,
                "launch_ms": gpu_ms or self.last_gpu_ms,
                "per_instruction": [{"label": label, "ms": value * ms_per_cycle}
                                    for label, value in zip(self.labels, stage.tolist(), strict=True)],
                "by_op": ordered}


# ---------------------------------------------------------------------------
# Checkpoint recipe
# ---------------------------------------------------------------------------
@dataclass
class NVFP4Linear:
    weight: torch.Tensor        # packed E2M1 [out, in / 2]
    weight_sf: torch.Tensor     # [out, in / 16]
    alpha: float                # checkpoint alpha / input global scale
    input_sf: float             # activation global scale

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1] * 2


@dataclass
class DenseLinear:
    weight: torch.Tensor
    bias: torch.Tensor | None


@dataclass
class BlockWeights:
    norm1: torch.Tensor
    norm2: torch.Tensor
    norm_q: torch.Tensor
    norm_k: torch.Tensor
    adaln: DenseLinear
    to_q: NVFP4Linear
    to_k: NVFP4Linear
    to_v: NVFP4Linear
    to_gate: NVFP4Linear | None
    to_out: NVFP4Linear
    fc_in: NVFP4Linear
    fc_out: NVFP4Linear


@dataclass
class RefinerBlockWeights:
    norm1: torch.Tensor
    norm2: torch.Tensor
    norm_q: torch.Tensor
    norm_k: torch.Tensor
    to_q: DenseLinear
    to_k: DenseLinear
    to_v: DenseLinear
    to_out: DenseLinear
    fc_in: DenseLinear
    fc_out: DenseLinear


@dataclass
class StepPlan:
    timesteps: torch.Tensor          # unique FP32 clean-time values [U]
    timestep_index: torch.Tensor     # row -> unique timestep [S]
    video_sigma_step: float          # 1 - video timestep (FP32)
    video_ratio: float               # sigma_next / sigma (FP32)
    audio_sigma_step: float
    audio_ratio: float
    keep: int                        # retained video tiles per video query


@dataclass
class DiTRecipe:
    hidden: int
    heads: int
    ffn: int
    rotary: int
    norm_eps: float
    qk_eps: float
    final_eps: float
    freq_dim: int
    blocks: list[BlockWeights]
    refiner: list[RefinerBlockWeights]
    refiner_final_norm: torch.Tensor
    proj_in: DenseLinear
    audio_proj_in: DenseLinear
    context_embedder: DenseLinear
    time_fc_in: DenseLinear
    time_fc_out: DenseLinear
    norm_out_weight: torch.Tensor
    norm_out_linear: DenseLinear
    proj_out: DenseLinear
    audio_proj_out: DenseLinear
    # request layout
    sequence: int
    token_tags: torch.Tensor
    text_index: torch.Tensor
    video_index: torch.Tensor
    audio_index: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    # VSA geometry
    tile_index: torch.Tensor         # token -> padded tile row
    valid: torch.Tensor              # valid tokens per tile
    prefix_tiles: int
    video_span: tuple[int, int]
    steps: list[StepPlan] = field(default_factory=list)
    modalities: int = 3


@dataclass
class CompiledDiT:
    program: Program
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    hidden: torch.Tensor
    buffers: dict[str, torch.Tensor]
    stage_counts: dict[str, int]


def compile_dit(module: Any, recipe: DiTRecipe, prompt: torch.Tensor,
                video_latents: torch.Tensor, audio_latents: torch.Tensor, *,
                steps: int | None = None, blocks: int | None = None,
                fast_attention: bool = True, finish: bool = True) -> CompiledDiT:
    """Compile text refinement and `steps` denoising steps into one program.

    `steps` / `blocks` truncate the graph for the block-level gates; with
    `finish=False` or a truncated block count the output heads and latent
    update are omitted and `hidden` holds the last block output.
    """
    r = recipe
    p = Program(module)
    S, H, heads, F = r.sequence, r.hidden, r.heads, r.ffn
    inner = heads * HEAD_DIM
    n_steps = len(r.steps) if steps is None else steps
    n_blocks = len(r.blocks) if blocks is None else blocks
    complete = finish and n_blocks == len(r.blocks)
    tiles = r.valid.numel()
    padded = tiles * TILE
    bf16, f32 = torch.bfloat16, torch.float32
    rows128 = (S + 127) // 128 * 128
    table_row = 6 * H
    U = r.steps[0].timesteps.numel()
    if any(step.timesteps.numel() != U for step in r.steps):
        raise ValueError("every step must have the same number of row timesteps")
    if prompt.dim() != 2 or prompt.shape[1] != r.context_embedder.weight.shape[1]:
        raise ValueError("prompt embeddings must be [text rows, text dim]")
    T = prompt.shape[0]
    if T > TILE or T != r.text_index.numel():
        raise ValueError("the text refiner path supports one text tile")
    counts: dict[str, int] = {}

    def count(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    # ---- request inputs and outputs ----------------------------------------
    prompt_b = p.const("prompt", prompt, bf16)
    video = p.const("video_latents", video_latents.clone(), f32)
    audio = p.const("audio_latents", audio_latents.clone(), f32)
    Nv, Cv = video.shape
    Na, Ca = audio.shape
    text_index = p.const("text_index", r.text_index, torch.int64)
    video_index = p.const("video_index", r.video_index, torch.int64)
    audio_index = p.const("audio_index", r.audio_index, torch.int64)
    tile_index = p.const("tile_index", r.tile_index, torch.int64)
    valid = p.const("valid", r.valid, torch.int64)
    cos = p.const("cos", r.cos, bf16)
    sin = p.const("sin", r.sin, bf16)
    tags = r.token_tags.to(torch.int64).cpu()

    # ---- activations and scratch (all preallocated) ------------------------
    hidden = p.alloc("hidden", (S, H), bf16)
    normed1 = p.alloc("normed1", (S, H), bf16)       # attention input, output-head input
    normed2 = p.alloc("normed2", (S, H), bf16)       # feed-forward input
    branch_attn = p.alloc("branch_attn", (S, H), bf16)
    branch_mlp = p.alloc("branch_mlp", (S, H), bf16)
    proj = {name: p.alloc(f"proj_{name}", (S, inner), bf16) for name in ("q", "k", "v", "gate")}
    tiled = {name: p.alloc(f"tiled_{name}", (padded, heads, HEAD_DIM), bf16)
             for name in ("q", "k", "v", "gate")}
    attention = p.alloc("attention", (padded, heads, HEAD_DIM), bf16)
    pools = {name: p.alloc(f"pool_{name}", (heads, tiles, HEAD_DIM), f32) for name in ("q", "k", "v")}
    scores = p.alloc("scores", (heads, tiles, tiles), f32)
    mask = p.alloc("mask", (heads, tiles, tiles), torch.uint8)
    ffn_mid = p.alloc("ffn_mid", (S, 2 * F), bf16)
    packed = {width: p.alloc(f"packed_{width}", (rows128, width // 2), torch.uint8)
              for width in (H, inner, F)}
    packed_sf = {width: p.alloc(f"packed_sf_{width}", (rows128, width // 16), torch.uint8)
                 for width in (H, inner, F)}
    tables = p.alloc("adaln_tables", (len(r.blocks), U * r.modalities, table_row), bf16)
    out_table = p.alloc("norm_out_table", (U, 2 * H), bf16)
    time_proj = p.alloc("time_proj", (U, r.freq_dim), f32)
    time_hidden = p.alloc("time_hidden", (U, r.time_fc_in.weight.shape[0]), f32)
    time_act = p.alloc("time_act", (U, r.time_fc_in.weight.shape[0]), f32)
    temb = p.alloc("temb", (U, r.time_fc_out.weight.shape[0]), f32)
    temb_act = p.alloc("temb_act", (U, r.time_fc_out.weight.shape[0]), bf16)
    velocity_v = p.alloc("velocity_video", (Nv, Cv), f32)
    velocity_a = p.alloc("velocity_audio", (Na, Ca), f32)

    # ---- text refiner (timestep free; once per generation) -----------------
    text = p.alloc("text_hidden", (TILE, H), bf16)
    text_normed = p.alloc("text_normed", (TILE, H), bf16)
    text_branch = p.alloc("text_branch", (TILE, H), bf16)
    text_refined = p.alloc("text_refined", (TILE, H), bf16)
    text_proj = {name: p.alloc(f"text_proj_{name}", (TILE, inner), bf16) for name in ("q", "k", "v")}
    text_q = p.alloc("text_q", (TILE, heads, HEAD_DIM), bf16)
    text_k = p.alloc("text_k", (TILE, heads, HEAD_DIM), bf16)
    text_attention = p.alloc("text_attention", (TILE, heads, HEAD_DIM), bf16)
    text_mid = p.alloc("text_mid", (TILE, 2 * F), bf16)
    text_act = p.alloc("text_act", (TILE, F), bf16)
    text_mask = p.const("text_mask", torch.ones((heads, 1, 1), dtype=torch.uint8))
    text_valid = p.const("text_valid", torch.tensor([T], dtype=torch.int64))

    def w32(name: str, tensor: torch.Tensor) -> torch.Tensor:
        return p.const(name, tensor, f32)

    p.linear("context_embedder", rows=T, x=prompt_b, weight=r.context_embedder.weight,
             bias=r.context_embedder.bias, out=text)
    residual = NORM_RESIDUAL | NORM_UNIT_GATE | NORM_ROUNDED
    for i, rb in enumerate(r.refiner):
        tag = f"refiner{i}"
        p.rmsnorm(f"{tag}.norm1", rows=T, cols=H, eps=r.norm_eps, flags=residual if i else 0,
                  x=text, out=text_normed, weight=w32(f"{tag}.norm1", rb.norm1),
                  branch=text_branch if i else None)
        for name, layer in (("q", rb.to_q), ("k", rb.to_k), ("v", rb.to_v)):
            p.linear(f"{tag}.to_{name}", rows=T, x=text_normed, weight=layer.weight,
                     bias=layer.bias, out=text_proj[name])
        p.qknorm(f"{tag}.qknorm", rows=T, eps=r.qk_eps, flags=0, heads=heads, rotary=0,
                 q=text_proj["q"], k=text_proj["k"], q_out=text_q, k_out=text_k,
                 q_weight=p.const(f"{tag}.norm_q", rb.norm_q, bf16),
                 k_weight=p.const(f"{tag}.norm_k", rb.norm_k, bf16))
        p.emit(f"{tag}.attention", "SPARSE_ATTENTION", 1, heads, p.buf(text_q), p.buf(text_k),
               p.buf(text_proj["v"]), p.buf(text_mask), p.buf(text_valid), p.buf(text_attention))
        p.linear(f"{tag}.to_out", rows=T, x=text_attention, weight=rb.to_out.weight,
                 bias=rb.to_out.bias, out=text_branch)
        p.rmsnorm(f"{tag}.norm2", rows=T, cols=H, eps=r.norm_eps, flags=residual, x=text,
                  out=text_normed, weight=w32(f"{tag}.norm2", rb.norm2), branch=text_branch)
        p.linear(f"{tag}.fc_in", rows=T, x=text_normed, weight=rb.fc_in.weight,
                 bias=rb.fc_in.bias, out=text_mid)
        p.copy_rows(f"{tag}.swiglu", rows=T, cols=F, x=text_mid, out=text_act,
                    in_stride=2 * F, out_stride=F, swiglu=True)
        p.linear(f"{tag}.fc_out", rows=T, x=text_act, weight=rb.fc_out.weight,
                 bias=rb.fc_out.bias, out=text_branch)
    p.rmsnorm("refiner.final_norm", rows=T, cols=H, eps=r.final_eps, flags=residual, x=text,
              out=text_refined, weight=w32("refiner.final_norm", r.refiner_final_norm),
              branch=text_branch)
    counts["refiner"] = len(p.instructions)

    # ---- per-block resident weights ----------------------------------------
    norm1 = [w32(f"block{i}.norm1", b.norm1) for i, b in enumerate(r.blocks)]
    norm2 = [w32(f"block{i}.norm2", b.norm2) for i, b in enumerate(r.blocks)]
    norm_q = [p.const(f"block{i}.norm_q", b.norm_q, bf16) for i, b in enumerate(r.blocks)]
    norm_k = [p.const(f"block{i}.norm_k", b.norm_k, bf16) for i, b in enumerate(r.blocks)]
    norm_out = w32("norm_out.weight", r.norm_out_weight)
    span_start, span_end = r.video_span
    attention_op = "SPARSE_ATTENTION_MMA" if fast_attention else "SPARSE_ATTENTION"

    def gemm(label: str, layer: NVFP4Linear, out: torch.Tensor) -> None:
        width = layer.in_features
        p.projection(label, packed[width], packed_sf[width], layer.weight, layer.weight_sf,
                     out, S, layer.out_features, width, layer.alpha)
        count("projection")

    for step_index in range(n_steps):
        step = r.steps[step_index]
        st = f"step{step_index}"
        timesteps = p.const(f"{st}.timesteps", step.timesteps, f32)
        row_time = step.timestep_index.to(torch.int64).cpu()
        time_index = p.const(f"{st}.time_index", row_time, torch.int32)
        adaln_index = p.const(f"{st}.adaln_index", row_time * r.modalities + tags, torch.int32)

        # timestep embedding and AdaLN tables
        p.elementwise(f"{st}.time_proj", EW_TIMESTEP, U * r.freq_dim, timesteps, time_proj,
                      scalar=r.freq_dim)
        p.linear(f"{st}.time_fc_in", rows=U, x=time_proj, weight=r.time_fc_in.weight,
                 bias=r.time_fc_in.bias, out=time_hidden)
        p.elementwise(f"{st}.time_silu", EW_SILU, time_hidden.numel(), time_hidden, time_act)
        p.linear(f"{st}.time_fc_out", rows=U, x=time_act, weight=r.time_fc_out.weight,
                 bias=r.time_fc_out.bias, out=temb)
        p.elementwise(f"{st}.temb_silu", EW_SILU, temb.numel(), temb, temb_act)
        for i in range(n_blocks):
            layer = r.blocks[i].adaln
            p.linear(f"{st}.block{i}.adaln", rows=U, x=temb_act, weight=layer.weight,
                     bias=layer.bias, out=tables[i])
        if complete:
            p.linear(f"{st}.norm_out.linear", rows=U, x=temb_act, weight=r.norm_out_linear.weight,
                     bias=r.norm_out_linear.bias, out=out_table)

        # modality packing
        p.linear(f"{st}.proj_in", rows=Nv, x=video, weight=r.proj_in.weight, bias=r.proj_in.bias,
                 out=hidden, index=video_index, scatter=True, out_stride=H)
        p.linear(f"{st}.audio_proj_in", rows=Na, x=audio, weight=r.audio_proj_in.weight,
                 bias=r.audio_proj_in.bias, out=hidden, index=audio_index, scatter=True,
                 out_stride=H)
        p.copy_rows(f"{st}.text", rows=T, cols=H, x=text_refined, out=hidden, index=text_index)

        block_table = tables.shape[1] * table_row
        for i in range(n_blocks):
            b = r.blocks[i]
            tag = f"{st}.block{i}"
            base = i * block_table
            flags = NORM_MOD
            if i:
                flags |= NORM_RESIDUAL | NORM_DOUBLE_ROUND | NORM_ROUNDED
            p.rmsnorm(f"{tag}.norm1", rows=S, cols=H, eps=r.norm_eps, flags=flags, x=hidden,
                      out=normed1, weight=norm1[i], index=adaln_index, table=tables,
                      table_stride=table_row, scale=base + H, shift=base,
                      gate=base - block_table + 5 * H if i else 0,
                      branch=branch_mlp if i else None)
            p.quantize(f"{tag}.quant_attn", rows=S, cols=H, global_sf=b.to_q.input_sf, flags=0,
                       x=normed1, packed=packed[H], packed_sf=packed_sf[H], stride=H)
            shared = [b.to_q, b.to_k, b.to_v] + ([b.to_gate] if b.to_gate else [])
            if any(layer.input_sf != b.to_q.input_sf for layer in shared):
                raise ValueError(f"{tag}: Q/K/V/gate must share one activation scale")
            gemm(f"{tag}.to_q", b.to_q, proj["q"])
            gemm(f"{tag}.to_k", b.to_k, proj["k"])
            gemm(f"{tag}.to_v", b.to_v, proj["v"])
            if b.to_gate:
                gemm(f"{tag}.to_gate", b.to_gate, proj["gate"])
            p.qknorm(f"{tag}.qknorm", rows=S, eps=r.qk_eps,
                     flags=QK_ROPE | QK_TILE | QK_COPY_V | (QK_COPY_GATE if b.to_gate else 0),
                     heads=heads, rotary=r.rotary, q=proj["q"], k=proj["k"], q_out=tiled["q"],
                     k_out=tiled["k"], q_weight=norm_q[i], k_weight=norm_k[i], cos=cos, sin=sin,
                     tile_index=tile_index, v=proj["v"], gate=proj["gate"] if b.to_gate else None,
                     v_out=tiled["v"], gate_out=tiled["gate"] if b.to_gate else None)
            p.emit(f"{tag}.vsa_pool", "VSA_POOL", tiles, heads, 3 if b.to_gate else 2,
                   p.buf(valid), p.buf(tiled["q"]), p.buf(tiled["k"]), p.buf(tiled["v"]),
                   p.buf(pools["q"]), p.buf(pools["k"]), p.buf(pools["v"]))
            p.emit(f"{tag}.vsa_scores", "VSA_SCORES", tiles, heads, p.buf(pools["q"]),
                   p.buf(pools["k"]), p.buf(scores))
            p.emit(f"{tag}.vsa_topk", "VSA_TOPK", tiles, heads, r.prefix_tiles, span_start,
                   span_end, step.keep, p.buf(scores), p.buf(mask))
            p.emit(f"{tag}.attention", attention_op, tiles, heads, p.buf(tiled["q"]),
                   p.buf(tiled["k"]), p.buf(tiled["v"]), p.buf(mask), p.buf(valid),
                   p.buf(attention))
            count("attention")
            if b.to_gate:
                p.emit(f"{tag}.vsa_compress", "VSA_COMPRESS", tiles, heads, p.buf(scores),
                       p.buf(pools["v"]), p.buf(tiled["gate"]), p.buf(attention))
            p.quantize(f"{tag}.quant_out", rows=S, cols=inner, global_sf=b.to_out.input_sf,
                       flags=QUANT_GATHER, x=attention, packed=packed[inner],
                       packed_sf=packed_sf[inner], stride=inner, gather=tile_index)
            gemm(f"{tag}.to_out", b.to_out, branch_attn)
            p.rmsnorm(f"{tag}.norm2", rows=S, cols=H, eps=r.norm_eps,
                      flags=NORM_MOD | NORM_RESIDUAL, x=hidden, out=normed2, weight=norm2[i],
                      index=adaln_index, table=tables, table_stride=table_row,
                      scale=base + 4 * H, shift=base + 3 * H, gate=base + 2 * H, branch=branch_attn)
            p.quantize(f"{tag}.quant_fc_in", rows=S, cols=H, global_sf=b.fc_in.input_sf, flags=0,
                       x=normed2, packed=packed[H], packed_sf=packed_sf[H], stride=H)
            gemm(f"{tag}.fc_in", b.fc_in, ffn_mid)
            p.quantize(f"{tag}.quant_fc_out", rows=S, cols=F, global_sf=b.fc_out.input_sf,
                       flags=QUANT_SWIGLU, x=ffn_mid, packed=packed[F], packed_sf=packed_sf[F],
                       stride=2 * F)
            gemm(f"{tag}.fc_out", b.fc_out, branch_mlp)

        # MLP residual of the last block, in FastVideo's eager BF16 order.
        p.rmsnorm(f"{st}.last_residual", rows=S, cols=H, eps=r.norm_eps,
                  flags=NORM_RESIDUAL | NORM_DOUBLE_ROUND | NORM_NO_OUTPUT, x=hidden, out=None,
                  weight=norm_out, index=adaln_index, table=tables, table_stride=table_row,
                  gate=(n_blocks - 1) * block_table + 5 * H, branch=branch_mlp)
        if not complete:
            continue
        p.rmsnorm(f"{st}.norm_out", rows=S, cols=H, eps=r.final_eps, flags=NORM_MOD, x=hidden,
                  out=normed1, weight=norm_out, index=time_index, table=out_table,
                  table_stride=2 * H, scale=H, shift=0)
        p.linear(f"{st}.proj_out", rows=Nv, x=normed1, weight=r.proj_out.weight,
                 bias=r.proj_out.bias, out=velocity_v, index=video_index, gather=True,
                 in_stride=H)
        p.linear(f"{st}.audio_proj_out", rows=Na, x=normed1, weight=r.audio_proj_out.weight,
                 bias=r.audio_proj_out.bias, out=velocity_a, index=audio_index, gather=True,
                 in_stride=H)
        p.elementwise(f"{st}.video_update", EW_FLOW_UPDATE, video.numel(), video, video,
                      scalar=float_bits(step.video_sigma_step), second=velocity_v,
                      ratio=step.video_ratio)
        p.elementwise(f"{st}.audio_update", EW_FLOW_UPDATE, audio.numel(), audio, audio,
                      scalar=float_bits(step.audio_sigma_step), second=velocity_a,
                      ratio=step.audio_ratio)
    counts["instructions"] = len(p.instructions)
    return CompiledDiT(p, video, audio, hidden, p.named, counts)


# ---------------------------------------------------------------------------
# FastVideo adapter (imports FastVideo lazily; used by the gates and pipeline)
# ---------------------------------------------------------------------------
def _dense(layer: Any) -> DenseLinear:
    return DenseLinear(layer.weight.detach(), None if layer.bias is None else layer.bias.detach())


def _nvfp4(layer: Any) -> NVFP4Linear:
    method = layer.quant_method
    if type(method).__name__ != "NVFP4QuantizeMethod":
        raise TypeError("expected an NVFP4 linear")
    if method._static_activation_global_sf() is not None or method._dynamic_activation_scale():
        raise ValueError("calibration-table and dynamic NVFP4 activation scales are not in this recipe")
    input_sf = getattr(layer, "_nvfp4_input_global_sf", None)
    input_sf = method.x_global_sf if input_sf is None else input_sf
    input_sf = input_sf.detach().float()
    alpha = float((layer._nvfp4_alpha.detach().float() / input_sf.to(layer._nvfp4_alpha.device)).item())
    if layer.bias is not None:
        raise ValueError("NVFP4 DiT linears are bias free")
    return NVFP4Linear(layer._nvfp4_weight, layer._nvfp4_weight_scale, alpha, float(input_sf.item()))


def recipe_from_fastvideo(transformer: Any, layout: Any, metadata: Any,
                          video_scheduler: Any, audio_scheduler: Any,
                          row_plan: list[tuple[torch.Tensor, torch.Tensor]]) -> DiTRecipe:
    """Read weights, layout, VSA policy and schedule from loaded FastVideo objects."""
    from fastvideo.attention.backends.video_sparse_attn import compute_topk

    t = transformer
    arch = t.config.arch_config
    if t.adaln_rank is not None or t.pdd_steps is not None or layout.pad_tokens or metadata.pad_tokens:
        raise ValueError("recipe supports the full-rank, single-head, padless T2VA checkpoint")
    if not metadata.exempt or len(metadata.video_tile_spans) != 1 or metadata.tile_elems != TILE:
        raise ValueError("recipe needs exempt VSA with one tile-64 video span")
    if metadata.dense_layers or layout.num_condition_video_rows or layout.num_condition_audio_rows:
        raise ValueError("dense layers and conditioning rows are not in this recipe")
    blocks = []
    for block in t.transformer_blocks:
        a = block.attn
        gate = a.to_gate_compress if a.to_gate_compress is not None and a._gate_active() else None
        blocks.append(BlockWeights(
            block.norm1.weight.detach(), block.norm2.weight.detach(), a.norm_q.weight.detach(),
            a.norm_k.weight.detach(), _dense(block.adaln_proj.linear), _nvfp4(a.to_q),
            _nvfp4(a.to_k), _nvfp4(a.to_v), _nvfp4(gate) if gate is not None else None,
            _nvfp4(a.to_out), _nvfp4(block.ff.fc_in), _nvfp4(block.ff.fc_out)))
    refiner = [RefinerBlockWeights(
        rb.norm1.weight.detach(), rb.norm2.weight.detach(), rb.attn.norm_q.weight.detach(),
        rb.attn.norm_k.weight.detach(), _dense(rb.attn.to_q), _dense(rb.attn.to_k),
        _dense(rb.attn.to_v), _dense(rb.attn.to_out), _dense(rb.ff.fc_in), _dense(rb.ff.fc_out))
        for rb in t.token_refiner.refiner_blocks]
    device = t.proj_in.weight.device
    cos, sin = t.rope(layout.position_ids.to(device))
    start, end = metadata.video_tile_spans[0]
    keep = compute_topk(metadata.span_sparsities[0], end - start)
    steps = []
    for i, (unique, inverse) in enumerate(row_plan):
        plan = []
        for scheduler in (video_scheduler, audio_scheduler):
            timestep = scheduler.timesteps[i].float().cpu()
            sigma = scheduler.sigmas[i].float().cpu()
            sigma_next = scheduler.sigmas[i + 1].float().cpu()
            plan += [float(1 - timestep), float(sigma_next / sigma)]
        steps.append(StepPlan(unique.float().cpu(), inverse.cpu(), plan[0], plan[1], plan[2],
                              plan[3], keep))
    return DiTRecipe(
        hidden=arch.hidden_size, heads=arch.num_attention_heads, ffn=arch.ffn_dim,
        rotary=2 * 3 * arch.rope_freq_dim, norm_eps=arch.norm_eps, qk_eps=arch.qk_norm_eps,
        final_eps=arch.final_norm_eps, freq_dim=arch.freq_dim, blocks=blocks, refiner=refiner,
        refiner_final_norm=t.token_refiner.final_norm.weight.detach(),
        proj_in=_dense(t.proj_in), audio_proj_in=_dense(t.audio_proj_in),
        context_embedder=_dense(t.context_embedder), time_fc_in=_dense(t.time_embedder.fc_in),
        time_fc_out=_dense(t.time_embedder.fc_out), norm_out_weight=t.norm_out.norm.weight.detach(),
        norm_out_linear=_dense(t.norm_out.linear), proj_out=_dense(t.proj_out),
        audio_proj_out=_dense(t.audio_proj_out), sequence=layout.sequence_length,
        token_tags=layout.token_tags, text_index=layout.text_indices,
        video_index=layout.video_indices, audio_index=layout.audio_indices, cos=cos, sin=sin,
        tile_index=metadata.untile_combined_index, valid=metadata.variable_block_sizes,
        prefix_tiles=metadata.num_prefix_tiles, video_span=(start, end), steps=steps)
