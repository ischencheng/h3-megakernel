import json
import struct
import urllib.request
from typing import NamedTuple

import torch
from safetensors.torch import load_file, save_file

from . import config

_URL = f"https://huggingface.co/{config.HF_REPO}/resolve/{config.HF_REVISION}/{config.HF_FILE}"
_CHUNK = 64 << 20


class Linear(NamedTuple):
    w: torch.Tensor         # [out, in/2] packed fp4, low nibble first
    s: torch.Tensor         # [out, in/16] fp8 scales, swizzled for 128x4
    alpha: torch.Tensor     # weight scale
    input_sf: torch.Tensor  # calibrated activation scale


class Weights(NamedTuple):
    fc1: Linear  # up rows first, then gate
    fc2: Linear


def _read_range(start, end):
    out = bytearray()
    pos = start
    while pos < end:
        stop = min(pos + _CHUNK, end)
        for attempt in range(4):
            try:
                req = urllib.request.Request(_URL, headers={"Range": f"bytes={pos}-{stop - 1}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                break
            except OSError:
                if attempt == 3:
                    raise
        if len(data) != stop - pos:
            raise RuntimeError(f"short read at {pos}: got {len(data)} of {stop - pos} bytes")
        out += data
        pos = stop
    return bytes(out)


def fetch():
    header_len = struct.unpack("<Q", _read_range(0, 8))[0]
    header = json.loads(_read_range(8, 8 + header_len))
    data_start = 8 + header_len

    def tensor(prefix, field, dtype):
        info = header[f"{prefix}::{field}"]
        begin, end = info["data_offsets"]
        raw = _read_range(data_start + begin, data_start + end)
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(info["shape"]).clone()

    tensors = {}
    for name, prefix, rows, columns in (("1", config.FC1_PREFIX, config.FC1_OUT, config.HIDDEN),
                                        ("2", config.FC2_PREFIX, config.HIDDEN, config.FFN)):
        tensors[f"w{name}"] = tensor(prefix, "_nvfp4_weight", torch.uint8)
        tensors[f"s{name}"] = tensor(prefix, "_nvfp4_weight_scale", torch.uint8)
        tensors[f"alpha{name}"] = tensor(prefix, "_nvfp4_alpha", torch.float32)
        tensors[f"sf{name}"] = tensor(prefix, "_nvfp4_input_global_sf", torch.float32)
        if tuple(tensors[f"w{name}"].shape) != (rows, columns // 2) or tuple(tensors[f"s{name}"].shape) != (rows, columns // 16):
            raise RuntimeError(f"unexpected shapes for {prefix}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"repo": config.HF_REPO, "revision": config.HF_REVISION, "file": config.HF_FILE,
            "w1_layout": "up rows [0, FFN), gate rows [FFN, 2*FFN)"}
    save_file(tensors, str(config.WEIGHTS_FILE), metadata=meta)
    return config.WEIGHTS_FILE


def load(device="cuda"):
    if not config.WEIGHTS_FILE.exists():
        fetch()
    t = load_file(str(config.WEIGHTS_FILE), device=str(device))
    return Weights(*(Linear(t[f"w{n}"], t[f"s{n}"], t[f"alpha{n}"], t[f"sf{n}"]) for n in "12"))


if __name__ == "__main__":
    path = fetch()
    w = load("cpu")
    print(f"saved {path}: fc1 {tuple(w.fc1.w.shape)} alpha {w.fc1.alpha.item()} input_sf {w.fc1.input_sf.item()}, "
          f"fc2 {tuple(w.fc2.w.shape)} alpha {w.fc2.alpha.item()} input_sf {w.fc2.input_sf.item()}")
