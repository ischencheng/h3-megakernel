"""Fetch the block-0 MLP weights without downloading the whole checkpoint.

The shard is about 5 GB, but we only need two tensors (about 0.46 GB), so we
read the safetensors header and then only those byte ranges.

Layout trap: this Diffusers export stores ff.net.0.proj value-first (up, then
gate), while H3's native fc1 is gate-first. vLLM-Omni swaps the halves when it
loads the checkpoint (see `swap_halves` in minimax_h3_transformer.py). We save
the weights already in the native gate-first order, so every kernel and the
reference see rows [0, FFN) as gate and rows [FFN, 2*FFN) as up.

Run: python -m m1.weights
"""

import hashlib
import json
import struct
import urllib.request

import torch
from safetensors.torch import load_file, save_file

from . import config

_URL = f"https://huggingface.co/{config.HF_REPO}/resolve/{config.HF_REVISION}/{config.HF_SHARD}"
_CHUNK = 64 << 20


def _read_range(start, end):
    """Bytes [start, end) of the shard, in chunks with a few retries."""
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


def _fetch_tensor(header, data_start, key):
    info = header[key]
    if info["dtype"] != "BF16":
        raise RuntimeError(f"{key} is {info['dtype']}, expected BF16")
    begin, end = info["data_offsets"]
    raw = _read_range(data_start + begin, data_start + end)
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(info["shape"]).clone()
    return tensor, hashlib.sha256(raw).hexdigest()


def fetch():
    header_len = struct.unpack("<Q", _read_range(0, 8))[0]
    header = json.loads(_read_range(8, 8 + header_len))
    data_start = 8 + header_len

    fc1, fc1_sha = _fetch_tensor(header, data_start, config.FC1_KEY)
    fc2, fc2_sha = _fetch_tensor(header, data_start, config.FC2_KEY)
    if tuple(fc1.shape) != (config.FC1_OUT, config.HIDDEN) or tuple(fc2.shape) != (config.HIDDEN, config.FFN):
        raise RuntimeError(f"unexpected shapes: fc1 {tuple(fc1.shape)}, fc2 {tuple(fc2.shape)}")

    up, gate = fc1.chunk(2, dim=0)
    w1 = torch.cat((gate, up), dim=0).contiguous()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "repo": config.HF_REPO,
        "revision": config.HF_REVISION,
        "shard": config.HF_SHARD,
        "fc1_key": config.FC1_KEY,
        "fc2_key": config.FC2_KEY,
        "fc1_sha256": fc1_sha,
        "fc2_sha256": fc2_sha,
        "w1_layout": "gate rows [0, FFN), up rows [FFN, 2*FFN)",
    }
    save_file({"w1": w1, "w2": fc2.contiguous()}, str(config.WEIGHTS_FILE), metadata=meta)
    return config.WEIGHTS_FILE


def load(device="cuda"):
    """Return (w1, w2) on `device`: w1 [2*FFN, HIDDEN] gate-first, w2 [HIDDEN, FFN]."""
    if not config.WEIGHTS_FILE.exists():
        fetch()
    tensors = load_file(str(config.WEIGHTS_FILE), device=str(device))
    return tensors["w1"], tensors["w2"]


if __name__ == "__main__":
    path = fetch()
    w1, w2 = load("cpu")
    print(f"saved {path}: w1 {tuple(w1.shape)}, w2 {tuple(w2.shape)}")
