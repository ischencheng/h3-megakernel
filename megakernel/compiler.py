"""Checkpoint-independent whole-wave compiler for the Hazy SM120 runtime.

The synthetic affine/add operations exercise scheduling and memory ownership.
Real H3 operations will register their own opcodes and resource contracts here.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct

INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
GRAPH_TASK = 3
WAVE_BARRIER = 4
PROJECTION_GEMM = 5
QKNORM_ROPE = 6
SPARSE_ATTENTION = 7
VSA_POOL = 8
VSA_SCORES = 9
VSA_TOPK = 10
VSA_VALUE_POOL = 11
VSA_COMPRESS = 12
ALIGN_ELEMENTS = 64


@dataclass(frozen=True)
class Resources:
    shared_bytes: int = 0
    registers_per_thread: int = 0
    barriers: int = 0
    pages: int = 0
    scratch_bytes: int = 0


@dataclass(frozen=True)
class Operation:
    opcode: int
    kind: int
    arity: int
    resources: Resources


OPERATIONS = {
    "affine": Operation(GRAPH_TASK, 1, 1, Resources(registers_per_thread=32)),
    "add": Operation(GRAPH_TASK, 2, 2, Resources(registers_per_thread=32)),
}

# CUTLASS SM120's 384-thread projection owns the entire aligned dynamic
# region for one instruction. The five 16-KiB Hazy pages are unavailable
# until all twelve warps leave its completion barrier. It has no global
# workspace; its 51,200-byte shared storage aliases the page region.
PROJECTION_RESOURCES = Resources(shared_bytes=51200,
                                 registers_per_thread=232,
                                 barriers=32, pages=5, scratch_bytes=0)


@dataclass(frozen=True)
class ProjectionTaskParams:
    rows: int
    columns: int
    depth: int
    persistent_ctas: int
    parameter_index: int = 0

    @property
    def tile_count(self) -> int:
        return ((self.rows + 127) // 128) * ((self.columns + 127) // 128)

    def words(self) -> list[int]:
        return [PROJECTION_GEMM, self.parameter_index, self.rows, self.columns,
                self.depth, self.persistent_ctas, self.tile_count] + [0] * (INSTRUCTION_WIDTH - 7)


def compile_projection(params: ProjectionTaskParams, sm_ids: list[int] | tuple[int, ...],
                       slot_count: int) -> list[list[list[int]]]:
    """Emit smoke, exclusive all-warp GEMM, smoke for measured physical SMs."""
    return compile_projections((params,), sm_ids, slot_count)


def compile_projections(params: tuple[ProjectionTaskParams, ...],
                        sm_ids: list[int] | tuple[int, ...],
                        slot_count: int) -> list[list[list[int]]]:
    """Emit multiple exclusive GEMMs in one Hazy stream, bounded by smoke ops."""
    if not params:
        raise ValueError("at least one projection is required")
    ids = tuple(sm_ids)
    for index, task in enumerate(params):
        if (task.rows <= 0 or task.columns <= 0 or task.depth <= 0 or
                task.columns % 128 or task.depth % 128 or
                task.parameter_index != index or
                task.persistent_ctas != len(ids)):
            raise ValueError("invalid projection shape or parameter index")
    if (len(set(ids)) != len(ids) or
            not ids or min(ids) < 0 or max(ids) >= slot_count):
        raise ValueError("projection requires one measured, addressable SM per CTA")
    r = PROJECTION_RESOURCES
    if (r.shared_bytes > 88480 or r.registers_per_thread > 255 or
            r.barriers > 32 or r.pages != 5 or r.scratch_bytes != 0):
        raise ValueError("projection resources exceed Hazy SM120")
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(len(params) + 2)]
               for _ in range(slot_count)]
    for smid in ids:
        streams[smid][0][0] = 1
        for index, task in enumerate(params):
            streams[smid][index + 1] = task.words()
        streams[smid][-1][0] = 1
    return streams


def compile_qknorm_rope(rows: int, eps: float,
                        sm_ids: list[int] | tuple[int, ...],
                        slot_count: int, *, tile: bool = False) -> list[list[list[int]]]:
    """Emit a first-block Q/K norm and partial-RoPE gate for the resident grid."""
    ids = tuple(sm_ids)
    if rows <= 0 or not (0 < eps < 1) or not ids or len(set(ids)) != len(ids):
        raise ValueError("invalid Q/K norm geometry, epsilon, or SM IDs")
    if min(ids) < 0 or max(ids) >= slot_count:
        raise ValueError("instruction table cannot address every physical SM ID")
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(3)] for _ in range(slot_count)]
    for smid in ids:
        streams[smid][0][0] = 1
        streams[smid][1][:4] = [QKNORM_ROPE, rows, _float_bits(eps), int(tile)]
        streams[smid][2][0] = 1
    return streams


def compile_projection_attention(params: tuple[ProjectionTaskParams, ...],
                                 rows: int, eps: float,
                                 sm_ids: list[int] | tuple[int, ...],
                                 slot_count: int) -> list[list[list[int]]]:
    """Chain NVFP4 projections and the tiled BF16 VSA boundary in one launch."""
    if len(params) != 4 or any(task.rows != rows or task.columns != 7168 for task in params):
        raise ValueError("the first VSA boundary needs Q/K/V/gate projections")
    projections = compile_projections(params, sm_ids, slot_count)
    tiled = compile_qknorm_rope(rows, eps, sm_ids, slot_count, tile=True)
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(8)] for _ in range(slot_count)]
    for smid in sm_ids:
        streams[smid][:5] = projections[smid][:5]
        streams[smid][5][:2] = [WAVE_BARRIER, 0]
        streams[smid][6] = tiled[smid][1]
        streams[smid][7][0] = 1
    return streams


def compile_sparse_attention(first_row: int, row_count: int,
                             sm_ids: list[int] | tuple[int, ...],
                             slot_count: int) -> list[list[list[int]]]:
    """Emit a bounded BF16 VSA fine-attention gate for the Hazy resident grid."""
    ids = tuple(sm_ids)
    if (first_row < 0 or row_count <= 0 or first_row + row_count > 56 * 42752 or
            not ids or len(set(ids)) != len(ids) or min(ids) < 0 or
            max(ids) >= slot_count):
        raise ValueError("invalid sparse-attention row range or physical SM IDs")
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(3)] for _ in range(slot_count)]
    for smid in ids:
        streams[smid][0][0] = 1
        streams[smid][1][:3] = [SPARSE_ATTENTION, first_row, row_count]
        streams[smid][2][0] = 1
    return streams


def compile_vsa_pool(tile_count: int, sm_ids: list[int] | tuple[int, ...],
                     slot_count: int) -> list[list[list[int]]]:
    """Pool tiled Q/K inside one persistent Hazy launch."""
    ids = tuple(sm_ids)
    if (tile_count != 668 or not ids or len(set(ids)) != len(ids) or
            min(ids) < 0 or max(ids) >= slot_count):
        raise ValueError("invalid VSA tile count or physical SM IDs")
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(3)] for _ in range(slot_count)]
    for smid in ids:
        streams[smid][0][0] = 1
        streams[smid][1][:2] = [VSA_POOL, tile_count]
        streams[smid][2][0] = 1
    return streams


def compile_vsa_scores(head_count: int, sm_ids: list[int] | tuple[int, ...],
                       slot_count: int) -> list[list[list[int]]]:
    """Compute all pooled Q/K scores inside one persistent Hazy launch."""
    ids = tuple(sm_ids)
    if (head_count != 56 or not ids or len(set(ids)) != len(ids) or
            min(ids) < 0 or max(ids) >= slot_count):
        raise ValueError("invalid VSA head count or physical SM IDs")
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(3)] for _ in range(slot_count)]
    for smid in ids:
        streams[smid][0][0] = 1
        streams[smid][1][:2] = [VSA_SCORES, head_count]
        streams[smid][2][0] = 1
    return streams


def compile_vsa_topk(head_count: int, prefix_tiles: int, video_start: int,
                     video_end: int, keep: int,
                     sm_ids: list[int] | tuple[int, ...],
                     slot_count: int) -> list[list[list[int]]]:
    """Select one exempt video span of pooled scores in the Hazy launch."""
    ids = tuple(sm_ids)
    if (head_count != 56 or not (0 <= prefix_tiles == video_start < video_end == 668) or
            not (0 < keep <= video_end - video_start) or not ids or
            len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= slot_count):
        raise ValueError("invalid VSA selection policy or physical SM IDs")
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(3)] for _ in range(slot_count)]
    for smid in ids:
        streams[smid][0][0] = 1
        streams[smid][1][:6] = [VSA_TOPK, head_count, prefix_tiles,
                                video_start, video_end, keep]
        streams[smid][2][0] = 1
    return streams


def compile_vsa_dynamic(prefix_tiles: int, video_start: int,
                        video_end: int, keep: int,
                        sm_ids: list[int] | tuple[int, ...],
                        slot_count: int) -> list[list[list[int]]]:
    """Pool, score, select, and attend in one resident Hazy instruction stream."""
    pool = compile_vsa_pool(668, sm_ids, slot_count)
    scores = compile_vsa_scores(56, sm_ids, slot_count)
    topk = compile_vsa_topk(56, prefix_tiles, video_start, video_end,
                            keep, sm_ids, slot_count)
    fine = compile_sparse_attention(0, 56 * 42752, sm_ids, slot_count)
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(9)] for _ in range(slot_count)]
    for smid in sm_ids:
        streams[smid][0][0] = 1
        streams[smid][1] = pool[smid][1]
        streams[smid][2][:2] = [WAVE_BARRIER, 0]
        streams[smid][3] = scores[smid][1]
        streams[smid][4][:2] = [WAVE_BARRIER, 1]
        streams[smid][5] = topk[smid][1]
        streams[smid][6][:2] = [WAVE_BARRIER, 2]
        streams[smid][7] = fine[smid][1]
        streams[smid][8][0] = 1
    return streams


def compile_vsa_full(prefix_tiles: int, video_start: int,
                     video_end: int, keep: int,
                     sm_ids: list[int] | tuple[int, ...],
                     slot_count: int) -> list[list[list[int]]]:
    """Add pooled V and trained coarse compression to the dynamic VSA path."""
    base = compile_vsa_dynamic(prefix_tiles, video_start, video_end,
                               keep, sm_ids, slot_count)
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(12)] for _ in range(slot_count)]
    for smid in sm_ids:
        streams[smid][0:2] = base[smid][0:2]
        streams[smid][2][:2] = [VSA_VALUE_POOL, 668]
        streams[smid][3][:2] = [WAVE_BARRIER, 0]
        streams[smid][4] = base[smid][3]
        streams[smid][5][:2] = [WAVE_BARRIER, 1]
        streams[smid][6] = base[smid][5]
        streams[smid][7][:2] = [WAVE_BARRIER, 2]
        streams[smid][8] = base[smid][7]
        streams[smid][9][:2] = [WAVE_BARRIER, 3]
        streams[smid][10][:2] = [VSA_COMPRESS, 668]
        streams[smid][11][0] = 1
    return streams


def compile_projection_sparse(params: tuple[ProjectionTaskParams, ...],
                              rows: int, eps: float,
                              sm_ids: list[int] | tuple[int, ...],
                              slot_count: int) -> list[list[list[int]]]:
    """Chain four projections, Q/K tile stage, and BF16 VSA fine attention."""
    base = compile_projection_attention(params, rows, eps, sm_ids, slot_count)
    sparse = compile_sparse_attention(0, 56 * 42752, sm_ids, slot_count)
    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(10)] for _ in range(slot_count)]
    for smid in sm_ids:
        streams[smid][:7] = base[smid][:7]
        streams[smid][7][:2] = [WAVE_BARRIER, 1]
        streams[smid][8] = sparse[smid][1]
        streams[smid][9][0] = 1
    return streams


@dataclass(frozen=True)
class Node:
    name: str
    op: str
    inputs: tuple[str, ...]
    length: int
    scale: float = 1.0
    bias: float = 0.0


@dataclass(frozen=True)
class GraphTaskParams:
    """Typed form of src/ops/graph.cuh's 32-word instruction ABI."""
    kind: int
    src0: int
    src1: int
    dst: int
    tile_start: int
    tile_length: int
    scale: float
    bias: float
    wave: int

    def words(self, opcode: int = GRAPH_TASK) -> list[int]:
        return [opcode, self.kind, self.src0, self.src1, self.dst,
                self.tile_start, self.tile_length, _float_bits(self.scale),
                _float_bits(self.bias), self.wave] + [0] * (INSTRUCTION_WIDTH - 10)


@dataclass(frozen=True)
class Graph:
    inputs: dict[str, int]
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]


@dataclass
class Compiled:
    instructions: list[list[list[int]]]
    offsets: dict[str, int]
    lengths: dict[str, int]
    arena_elements: int
    waves: list[dict]
    sm_ids: tuple[int, ...]

    def tensors(self, device="cuda"):
        import torch

        return (
            torch.tensor(self.instructions, dtype=torch.int32, device=device),
            torch.zeros((len(self.instructions), len(self.instructions[0]), TIMING_WIDTH),
                        dtype=torch.int32, device=device),
            torch.zeros(self.arena_elements, dtype=torch.float32, device=device),
            torch.zeros(len(self.waves), dtype=torch.int32, device=device),
        )

    def describe(self) -> dict:
        return {
            "physical_sm_ids": list(self.sm_ids),
            "arena_elements": self.arena_elements,
            "buffers": {name: {"offset": offset, "length": self.lengths[name]}
                        for name, offset in self.offsets.items()},
            "waves": self.waves,
        }


def _align(n: int) -> int:
    return (n + ALIGN_ELEMENTS - 1) // ALIGN_ELEMENTS * ALIGN_ELEMENTS


def _float_bits(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", value))[0]


def compile_graph(graph: Graph, sm_ids: list[int] | tuple[int, ...],
                  sm_count: int, *, slot_count: int | None = None,
                  max_arena_elements: int | None = None,
                  operations: dict[str, Operation] = OPERATIONS) -> Compiled:
    """Compile one topologically ordered graph into per-physical-SM streams.

    Each graph node is one whole-stage wave. Every resident CTA executes a
    disjoint tile, then participates in a release/acquire global barrier.
    The one-CTA-per-SM residency condition is checked against measured IDs.
    """
    ids = tuple(sm_ids)
    if sm_count <= 0 or len(ids) != sm_count or len(set(ids)) != sm_count:
        raise ValueError("one unique measured physical SM ID is required per CTA")
    if min(ids) < 0:
        raise ValueError("physical SM IDs must be nonnegative")
    slots = slot_count if slot_count is not None else max(ids) + 1
    if slots <= max(ids):
        raise ValueError("instruction table cannot address every physical SM ID")
    if not graph.nodes or not graph.outputs:
        raise ValueError("graph needs nodes and outputs")
    if any(n <= 0 for n in graph.inputs.values()):
        raise ValueError("input lengths must be positive")

    lengths = dict(graph.inputs)
    seen = set(lengths)
    for node in graph.nodes:
        if node.name in seen or node.name == "":
            raise ValueError(f"duplicate or empty buffer name: {node.name!r}")
        if node.op not in operations:
            raise ValueError(f"unknown operation: {node.op}")
        spec = operations[node.op]
        if len(node.inputs) != spec.arity or any(x not in seen for x in node.inputs):
            raise ValueError(f"{node.name}: missing, cyclic, or wrong number of inputs")
        if node.length <= 0 or any(lengths[x] != node.length for x in node.inputs):
            raise ValueError(f"{node.name}: incompatible tensor length")
        r = spec.resources
        if (r.shared_bytes > 0 or r.registers_per_thread > 255 or
                r.barriers > 32 or r.pages > 5 or r.scratch_bytes > 4096 or
                min(r.shared_bytes, r.registers_per_thread, r.barriers,
                    r.pages, r.scratch_bytes) < 0):
            raise ValueError(f"{node.name}: exceeds RTX PRO 6000 runtime resources")
        lengths[node.name] = node.length
        seen.add(node.name)
    if any(x not in seen for x in graph.outputs):
        raise ValueError("unknown graph output")

    # Output allocation precedes release of inputs used by the current node.
    # This rules out accidental in-place aliasing, including branch fanout.
    last_use = {name: -1 for name in graph.inputs}
    for i, node in enumerate(graph.nodes):
        last_use[node.name] = i
        for name in node.inputs:
            last_use[name] = i
    for name in graph.outputs:
        last_use[name] = len(graph.nodes)
    offsets: dict[str, int] = {}
    free: list[tuple[int, int]] = []
    high_water = 0

    def allocate(name: str, length: int) -> None:
        nonlocal high_water
        size = _align(length)
        for j, (offset, available) in enumerate(free):
            if available >= size:
                offsets[name] = offset
                free.pop(j)
                if available > size:
                    free.append((offset + size, available - size))
                return
        offsets[name] = high_water
        high_water += size

    for name, length in graph.inputs.items():
        allocate(name, length)
    for i, node in enumerate(graph.nodes):
        allocate(node.name, node.length)
        for name in (*graph.inputs, *(n.name for n in graph.nodes[:i + 1])):
            if last_use[name] == i:
                free.append((offsets[name], _align(lengths[name])))
        free.sort()
        merged: list[tuple[int, int]] = []
        for offset, size in free:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((offset, size))
        free = merged
    if max_arena_elements is not None and high_water > max_arena_elements:
        raise ValueError("activation arena exceeds configured budget")

    streams = [[[0] * INSTRUCTION_WIDTH for _ in range(2 * len(graph.nodes))]
               for _ in range(slots)]
    waves = []
    for wave, node in enumerate(graph.nodes):
        spec = operations[node.op]
        chunk = (node.length + sm_count - 1) // sm_count
        for rank, smid in enumerate(ids):
            begin = min(rank * chunk, node.length)
            count = min(chunk, node.length - begin)
            streams[smid][2 * wave] = GraphTaskParams(
                spec.kind, offsets[node.inputs[0]],
                offsets[node.inputs[1]] if spec.arity == 2 else 0,
                offsets[node.name], begin, count, node.scale, node.bias,
                wave).words(spec.opcode)
            streams[smid][2 * wave + 1][:2] = [WAVE_BARRIER, wave]
        waves.append({"id": wave, "name": node.name, "op": node.op,
                      "inputs": list(node.inputs), "tiles": sm_count,
                      "resources": spec.resources.__dict__})
    return Compiled(streams, offsets, lengths, high_water, waves, ids)
