from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from math import ceil, prod


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 5376
    heads: int = 56
    head_dim: int = 128
    ffn: int = 14336
    blocks: int = 50
    steps: int = 8
    rope_dim: int = 96
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    sf_block: int = 16
    adaln_rows: int = 9

    @property
    def attn_dim(self) -> int:
        return self.heads * self.head_dim


@dataclass(frozen=True)
class RequestShape:
    text_tokens: int = 61
    audio_tokens: int = 414
    video_tokens: int = 37296
    vsa_tile: int = 64
    prefix_tiles: int = 8
    video_tiles: int = 660
    keep_tiles: int = 132

    @property
    def tokens(self) -> int:
        return self.text_tokens + self.audio_tokens + self.video_tokens

    @property
    def tiles(self) -> int:
        return self.prefix_tiles + self.video_tiles

    @property
    def tiled_rows(self) -> int:
        return self.tiles * self.vsa_tile

    @property
    def kept_tiles(self) -> int:
        return self.prefix_tiles + self.keep_tiles


@dataclass(frozen=True)
class Machine:
    sms: int = 188
    consumer_warps: int = 8
    other_warps: int = 4
    page_bytes: int = 16384
    pages: int = 5
    instruction_words: int = 32


class Opcode(IntEnum):
    NORM_MOD_QUANT = 1
    GEMM_QKV = 2
    VSA_POOL = 3
    ATTN_QUANT = 4
    VSA_SELECT = 5
    ATTN_SPARSE = 6
    GEMM_OUT = 7
    GEMM_FC1 = 8
    GEMM_FC2 = 9


@dataclass(frozen=True)
class Kernel:
    source: str | None
    port_from: str | None
    pages: int | None


# missing for now: Hazy-format kernel sources and the pages each one needs
KERNELS = {
    Opcode.NORM_MOD_QUANT: Kernel(None, "flashinfer csrc/cake_minimax_h3_sm120_norm_adaln_quant.cuh", None),
    Opcode.GEMM_QKV: Kernel(None, "cutlass examples/79_blackwell_geforce_gemm/79b + flashinfer "
                            "csrc/cake_minimax_h3_sm120_quant_pre_attention", None),
    Opcode.VSA_POOL: Kernel(None, None, None),
    Opcode.ATTN_QUANT: Kernel(None, "fastvideo-kernel attn_qat_infer/quantization/fp4_quantization_4d.cu", None),
    Opcode.VSA_SELECT: Kernel(None, None, None),
    Opcode.ATTN_SPARSE: Kernel(None, "fastvideo-kernel attn_qat_infer/blackwell", None),
    Opcode.GEMM_OUT: Kernel(None, "flashinfer csrc/cake_minimax_h3_sm120_quant_out_proj_sm120a.cu", None),
    Opcode.GEMM_FC1: Kernel(None, "flashinfer csrc/cake_minimax_h3_sm120_quant_mlp_sm120a.cu", None),
    Opcode.GEMM_FC2: Kernel(None, "flashinfer csrc/cake_minimax_h3_sm120_quant_mlp_sm120a.cu", None),
}


DTYPE_BYTES = {"bf16": 2, "f32": 4, "i32": 4, "e4m3": 1, "u8": 1, "fp4": 0.5}


@dataclass
class Buffer:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str = "activation"
    order: str = "token"
    offset: int | None = None  # missing for now: assigned by the memory planner

    @property
    def nbytes(self) -> int:
        return ceil(prod(self.shape) * DTYPE_BYTES[self.dtype])


@dataclass
class CounterRange:
    name: str
    count: int
    offset: int | None = None  # missing for now: assigned by the memory planner


@dataclass(frozen=True)
class WorkUnit:
    dims: tuple[str, ...]
    extent: tuple[int, ...]
    atom: tuple[int, ...]
    group: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(a * g for a, g in zip(self.atom, self.group))

    @property
    def count(self) -> int:
        return prod(ceil(e / s) for e, s in zip(self.extent, self.shape))


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str


@dataclass
class Input:
    buffer: Buffer
    producer: Op | None = None
    waits_per_unit: int | None = None  # missing for now: computed by the compiler


@dataclass
class Op:
    name: str
    opcode: Opcode
    unit: WorkUnit
    layers: tuple[Layer, ...]
    inputs: list[Input]
    outputs: list[Buffer]
    weights: list[Buffer]
    counters: CounterRange
    params: dict[str, float | int] = field(default_factory=dict)

    @property
    def kernel(self) -> Kernel:
        return KERNELS[self.opcode]


@dataclass
class TransformerBlock:
    index: int
    layers: list[Layer]
    ops: list[Op]


@dataclass
class SharedConstants:
    adaln_index: Buffer
    tile_index: Buffer
    untile_index: Buffer
    tile_valid: Buffer
    rope_cos: Buffer
    rope_sin: Buffer


@dataclass
class DiT:
    model: ModelConfig
    request: RequestShape
    machine: Machine
    shared: SharedConstants
    x_in: Buffer
    blocks: list[TransformerBlock]
    buffers: list[Buffer]
    counters: list[CounterRange]

    @property
    def ops(self) -> list[Op]:
        return [op for block in self.blocks for op in block.ops]


BLOCK_LAYERS = [
    ("norm1", "rmsnorm"),
    ("modulate_msa", "adaln"),
    ("quant_attn_in", "fp4_quantize"),
    ("to_q", "linear"),
    ("to_k", "linear"),
    ("to_v", "linear"),
    ("to_gate_compress", "linear"),
    ("norm_q", "rmsnorm"),
    ("norm_k", "rmsnorm"),
    ("rope", "rope"),
    ("vsa_pool", "tile_mean"),
    ("quant_qkv", "fp4_quantize"),
    ("vsa_scores", "matmul"),
    ("vsa_topk", "topk"),
    ("fine_attention", "block_sparse_attention"),
    ("coarse_attention", "attention"),
    ("untile_quant", "fp4_quantize"),
    ("to_out", "linear"),
    ("gate_residual_msa", "gated_residual"),
    ("norm2", "rmsnorm"),
    ("modulate_mlp", "adaln"),
    ("quant_mlp_in", "fp4_quantize"),
    ("fc_in", "linear"),
    ("swiglu", "swiglu"),
    ("quant_fc_out_in", "fp4_quantize"),
    ("fc_out", "linear"),
    ("gate_residual_mlp", "gated_residual"),
]


class Arena:
    def __init__(self) -> None:
        self.buffers: list[Buffer] = []
        self.counters: list[CounterRange] = []

    def buffer(self, name: str, shape: tuple[int, ...], dtype: str, role: str = "activation",
               order: str = "token") -> Buffer:
        buffer = Buffer(name, shape, dtype, role, order)
        self.buffers.append(buffer)
        return buffer

    def fp4(self, name: str, rows: int, cols: int, sf_block: int, role: str = "activation",
            order: str = "token") -> tuple[Buffer, Buffer]:
        data = self.buffer(name, (rows, cols), "fp4", role, order)
        scales = self.buffer(f"{name}_sf", (rows, cols // sf_block), "e4m3", role, order)
        return data, scales

    def counter_range(self, name: str, count: int) -> CounterRange:
        counters = CounterRange(name, count)
        self.counters.append(counters)
        return counters


def gemm_unit(rows: int, cols: int, group_cols: int) -> WorkUnit:
    return WorkUnit(("row", "col"), (rows, cols), (128, 128), (1, group_cols))


def build_shared_constants(m: ModelConfig, r: RequestShape, mem: Arena) -> SharedConstants:
    return SharedConstants(
        adaln_index=mem.buffer("adaln_index", (r.tokens, ), "i32", "constant"),
        tile_index=mem.buffer("tile_index", (r.tokens, ), "i32", "constant"),
        untile_index=mem.buffer("untile_index", (r.tiled_rows, ), "i32", "constant", "tile"),
        tile_valid=mem.buffer("tile_valid", (r.tiles, ), "i32", "constant"),
        rope_cos=mem.buffer("rope_cos", (r.tiled_rows, m.rope_dim), "bf16", "constant", "tile"),
        rope_sin=mem.buffer("rope_sin", (r.tiled_rows, m.rope_dim), "bf16", "constant", "tile"),
    )


def build_block(i: int, m: ModelConfig, r: RequestShape, mem: Arena, shared: SharedConstants, x: Buffer,
                x_producer: Op | None) -> tuple[TransformerBlock, Buffer, Op]:
    p = f"block{i}."
    layers = [Layer(name, kind) for name, kind in BLOCK_LAYERS]
    L = {layer.name: layer for layer in layers}
    tokens, tiled, H, D, sf = r.tokens, r.tiled_rows, m.heads, m.head_dim, m.sf_block

    # missing for now: producer of the per-step AdaLN tables (computed before the blocks)
    adaln = mem.buffer(p + "adaln_table", (m.adaln_rows, 6, m.hidden), "bf16", "constant")

    # missing for now: per-linear alpha and static input global scale from the checkpoint
    w_norm1 = mem.buffer(p + "norm1.weight", (m.hidden, ), "bf16", "weight")
    w_qkvg = mem.fp4(p + "attn.qkvg.weight", 4 * m.attn_dim, m.hidden, sf, "weight")
    w_norm_q = mem.buffer(p + "attn.norm_q.weight", (D, ), "bf16", "weight")
    w_norm_k = mem.buffer(p + "attn.norm_k.weight", (D, ), "bf16", "weight")
    w_out = mem.fp4(p + "attn.to_out.weight", m.hidden, m.attn_dim, sf, "weight")
    w_norm2 = mem.buffer(p + "norm2.weight", (m.hidden, ), "bf16", "weight")
    w_fc_in = mem.fp4(p + "ff.fc_in.weight", 2 * m.ffn, m.hidden, sf, "weight")
    w_fc_out = mem.fp4(p + "ff.fc_out.weight", m.hidden, m.ffn, sf, "weight")

    a, a_sf = mem.fp4(p + "a", tiled, m.hidden, sf, order="tile")
    q = mem.buffer(p + "q", (H, tiled, D), "bf16", order="tile")
    k = mem.buffer(p + "k", (H, tiled, D), "bf16", order="tile")
    v = mem.buffer(p + "v", (H, tiled, D), "bf16", order="tile")
    g = mem.buffer(p + "gate", (tiled, m.attn_dim), "bf16", order="tile")
    q_pool = mem.buffer(p + "q_pool", (H, r.tiles, D), "f32", order="tile")
    k_pool = mem.buffer(p + "k_pool", (H, r.tiles, D), "f32", order="tile")
    v_pool = mem.buffer(p + "v_pool", (H, r.tiles, D), "f32", order="tile")
    q4 = mem.buffer(p + "q4", (H, tiled, D), "fp4", order="tile")
    q4_sf = mem.buffer(p + "q4_sf", (H, tiled, D // sf), "e4m3", order="tile")
    k4 = mem.buffer(p + "k4", (H, tiled, D), "fp4", order="tile")
    k4_sf = mem.buffer(p + "k4_sf", (H, tiled, D // sf), "e4m3", order="tile")
    v4t = mem.buffer(p + "v4t", (H, D, tiled), "fp4", order="tile")
    v4t_sf = mem.buffer(p + "v4t_sf", (H, D, tiled // sf), "e4m3", order="tile")
    kv_index = mem.buffer(p + "kv_index", (H, r.tiles, r.kept_tiles), "i32", order="tile")
    kv_count = mem.buffer(p + "kv_count", (H, r.tiles), "i32", order="tile")
    o, o_sf = mem.fp4(p + "o", tokens, m.attn_dim, sf)
    x_mid = mem.buffer(p + "x_mid", (tokens, m.hidden), "bf16")
    b, b_sf = mem.fp4(p + "b", tokens, m.hidden, sf)
    h, h_sf = mem.fp4(p + "h", tokens, m.ffn, sf)
    x_out = mem.buffer(p + "x_out", (tokens, m.hidden), "bf16")

    def op(name: str, opcode: Opcode, unit: WorkUnit, covers: list[str], inputs: list[Input],
           outputs: list[Buffer], weights: list[Buffer], **params: float | int) -> Op:
        counters = mem.counter_range(p + name, unit.count)
        return Op(p + name, opcode, unit, tuple(L[n] for n in covers), inputs, outputs, weights, counters, params)

    norm1 = op("norm1", Opcode.NORM_MOD_QUANT,
               WorkUnit(("row", ), (tiled, ), (1, ), (128, )),
               ["norm1", "modulate_msa", "quant_attn_in"],
               [Input(x, x_producer), Input(adaln), Input(shared.adaln_index), Input(shared.tile_index)],
               [a, a_sf], [w_norm1],
               eps=m.norm_eps, table_shift=0, table_scale=1, gather_to_tiles=1)

    qkv = op("qkv", Opcode.GEMM_QKV,
             gemm_unit(tiled, 4 * m.attn_dim, 8),
             ["to_q", "to_k", "to_v", "to_gate_compress", "norm_q", "norm_k", "rope"],
             [Input(a, None), Input(a_sf, None), Input(shared.rope_cos), Input(shared.rope_sin)],
             [q, k, v, g], [*w_qkvg, w_norm_q, w_norm_k],
             eps=m.qk_norm_eps, rope_dim=m.rope_dim)
    qkv.inputs[0].producer = qkv.inputs[1].producer = norm1

    pool = op("vsa_pool", Opcode.VSA_POOL,
              WorkUnit(("head", "tile"), (H, r.tiles), (1, 1), (8, 1)),
              ["vsa_pool"],
              [Input(q, qkv), Input(k, qkv), Input(v, qkv), Input(shared.tile_valid)],
              [q_pool, k_pool, v_pool], [])

    attn_quant = op("attn_quant", Opcode.ATTN_QUANT,
                    WorkUnit(("head", "row"), (H, tiled), (1, 128), (1, 4)),
                    ["quant_qkv"],
                    [Input(q, qkv), Input(k, qkv), Input(v, qkv)],
                    [q4, q4_sf, k4, k4_sf, v4t, v4t_sf], [],
                    smooth_k=0)

    select = op("vsa_select", Opcode.VSA_SELECT,
                WorkUnit(("head", "q_tile"), (H, r.tiles), (1, 1), (1, 32)),
                ["vsa_scores", "vsa_topk"],
                [Input(q_pool, pool), Input(k_pool, pool)],
                [kv_index, kv_count], [],
                prefix_tiles=r.prefix_tiles, keep_tiles=r.keep_tiles)

    attn = op("attention", Opcode.ATTN_SPARSE,
              WorkUnit(("head", "q_tile"), (H, r.tiles), (1, 1), (1, 2)),
              ["fine_attention", "coarse_attention", "untile_quant"],
              [Input(q4, attn_quant), Input(q4_sf, attn_quant), Input(k4, attn_quant), Input(k4_sf, attn_quant),
               Input(v4t, attn_quant), Input(v4t_sf, attn_quant), Input(kv_index, select), Input(kv_count, select),
               Input(q_pool, pool), Input(k_pool, pool), Input(v_pool, pool), Input(g, qkv),
               Input(shared.untile_index), Input(shared.tile_valid)],
              [o, o_sf], [],
              scale=D**-0.5)

    out = op("to_out", Opcode.GEMM_OUT,
             gemm_unit(tokens, m.hidden, 6),
             ["to_out", "gate_residual_msa"],
             [Input(o, attn), Input(o_sf, attn), Input(x, x_producer), Input(adaln), Input(shared.adaln_index)],
             [x_mid], list(w_out),
             table_gate=2)

    norm2 = op("norm2", Opcode.NORM_MOD_QUANT,
               WorkUnit(("row", ), (tokens, ), (1, ), (128, )),
               ["norm2", "modulate_mlp", "quant_mlp_in"],
               [Input(x_mid, out), Input(adaln), Input(shared.adaln_index)],
               [b, b_sf], [w_norm2],
               eps=m.norm_eps, table_shift=3, table_scale=4, gather_to_tiles=0)

    fc1 = op("fc_in", Opcode.GEMM_FC1,
             gemm_unit(tokens, 2 * m.ffn, 8),
             ["fc_in", "swiglu", "quant_fc_out_in"],
             [Input(b, norm2), Input(b_sf, norm2)],
             [h, h_sf], list(w_fc_in),
             interleave_rows=8)

    fc2 = op("fc_out", Opcode.GEMM_FC2,
             gemm_unit(tokens, m.hidden, 6),
             ["fc_out", "gate_residual_mlp"],
             [Input(h, fc1), Input(h_sf, fc1), Input(x_mid, out), Input(adaln), Input(shared.adaln_index)],
             [x_out], list(w_fc_out),
             table_gate=5)

    ops = [norm1, qkv, pool, attn_quant, select, attn, out, norm2, fc1, fc2]
    return TransformerBlock(i, layers, ops), x_out, fc2


def build_dit(model: ModelConfig = ModelConfig(), request: RequestShape = RequestShape(),
              machine: Machine = Machine()) -> DiT:
    mem = Arena()
    shared = build_shared_constants(model, request, mem)
    # missing for now: proj_in / audio_proj_in / text copy that produce x_in, and norm_out / proj_out / flow update after
    x_in = mem.buffer("x_in", (request.tokens, model.hidden), "bf16")
    x, producer = x_in, None
    blocks = []
    for i in range(model.blocks):
        block, x, producer = build_block(i, model, request, mem, shared, x, producer)
        blocks.append(block)
    return DiT(model, request, machine, shared, x_in, blocks, mem.buffers, mem.counters)


def summary(dit: DiT) -> str:
    block = dit.blocks[0]
    lines = [f"{'op':<12} {'opcode':<15} {'unit shape':<14} {'units':>7}  layers"]
    for op in block.ops:
        name = op.name.split(".", 1)[1]
        shape = "x".join(str(s) for s in op.unit.shape)
        lines.append(f"{name:<12} {op.opcode.name:<15} {shape:<14} {op.unit.count:>7}  "
                     f"{', '.join(layer.name for layer in op.layers)}")
    per_block = sum(op.unit.count for op in block.ops)
    instr_bytes = dit.machine.instruction_words * 4
    acts = sum(b.nbytes for b in dit.buffers if b.role == "activation" and b.name.startswith("block0."))
    weights = sum(b.nbytes for b in dit.buffers if b.role == "weight" and b.name.startswith("block0."))
    steps = dit.model.steps
    lines += [
        "",
        f"instructions per block: {per_block:,}",
        f"instructions per step:  {per_block * dit.model.blocks:,} "
        f"({per_block * dit.model.blocks * instr_bytes / 2**30:.2f} GiB)",
        f"instructions per video: {per_block * dit.model.blocks * steps:,}",
        f"activations per block:  {acts / 2**30:.2f} GiB",
        f"activations per step:   {acts * dit.model.blocks / 2**30:.1f} GiB",
        f"weights per block:      {weights / 2**30:.2f} GiB",
        f"counters per block:     {sum(op.counters.count for op in block.ops):,}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(build_dit()))
