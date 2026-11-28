"""Python-side instructions for the H3 megakernel.

Ported from Hazy Research's Megakernels (megakernels/instructions.py, MIT).
Each Instruction subclass mirrors one C++ op: opcode() must match the op's
`opcode` constant, and the dataclass fields, in order, become instruction
words 1, 2, ... that the op reads with s.instruction()[1], [2], ...
"""

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor


@dataclass
class BaseGlobals:
    """What the scheduler needs to know about the run: the device and the
    instructions/timings tensors it fills in. Op buffers go in subclasses."""

    device: torch.device

    def __post_init__(self):
        self.instructions: Tensor | None = None
        self.timings: Tensor | None = None

    def sm_count(self) -> int:
        return torch.cuda.get_device_properties(self.device).multi_processor_count


class Instruction:
    @classmethod
    def opcode(cls) -> int:
        raise NotImplementedError

    @classmethod
    def tags(cls) -> dict[str, Any]:
        # used by the "pool" scheduler: {"pool": "memory"} or {"pool": "compute"}
        return {}

    def cost(self, globs: BaseGlobals) -> float:
        # relative runtime estimate, used by the "wave" and "dag" schedulers
        raise NotImplementedError

    def serialize(self) -> list[int]:
        # [opcode, field1, field2, ...]; tuples/lists become [len, *items]
        words = [self.opcode()]
        for field in fields(self):
            attr = getattr(self, field.name)
            if isinstance(attr, int):
                words.append(attr)
            elif isinstance(attr, (tuple, list)):
                words.append(len(attr))
                words.extend(attr)
            elif attr is None:
                words.append(0)
            else:
                raise ValueError(f"Unsupported field type: {attr}")
        return words


@dataclass
class NoOp(Instruction):
    @classmethod
    def opcode(cls) -> int:
        return 0

    def cost(self, globs: BaseGlobals) -> float:
        return 0


# ---- H3 ops ---------------------------------------------------------------

OPCODE_ADD_ONE = 2  # must match megakernel/src/h3.cuh


@dataclass
class AddOne(Instruction):
    """AddOneOp in ops/add_one.cuh: output tile = input tile + 1."""

    row_tile: int  # s.instruction()[1]
    col_tile: int  # s.instruction()[2]

    @classmethod
    def opcode(cls) -> int:
        return OPCODE_ADD_ONE

    @classmethod
    def tags(cls) -> dict[str, Any]:
        return {"pool": "memory"}  # one load, trivial math, one store

    def cost(self, globs: BaseGlobals) -> float:
        return 1.0  # every tile is the same size


OPCODE_NORM_MOD_QUANT = 3  # must match megakernel/src/h3.cuh


@dataclass
class NormModQuant(Instruction):
    """NormModQuantOp in ops/norm_mod_quant.cuh
    RMSNorm, AdaLN modulation and NVFP4 quant of one 128-row tile"""

    row_tile: int        # s.instruction()[1]
    shift_off: int       # s.instruction()[2]: column offset of shift in an adaln_table row
    scale_off: int       # s.instruction()[3]: column offset of scale in an adaln_table row
    global_sf_bits: int  # s.instruction()[4]: float32 bits of global_sf, e.g. struct.unpack("i", struct.pack("f", gsf))[0]

    @classmethod
    def opcode(cls) -> int:
        return OPCODE_NORM_MOD_QUANT

    @classmethod
    def tags(cls) -> dict[str, Any]:
        return {"pool": "memory"}

    def cost(self, globs: BaseGlobals) -> float:
        return 1.0
