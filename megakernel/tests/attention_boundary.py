"""Verify the first FastH3 BF16 VSA boundary against the frozen reference."""

import hashlib
import json
import os
from pathlib import Path

import torch


def read_tensor(root: Path, manifest: dict, name: str) -> torch.Tensor:
    entry = manifest["tensors"][name]
    path = root / entry["file"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(f"{name}: capture hash mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if list(value.shape) != entry["shape"] or str(value.dtype) != entry["dtype"]:
        raise RuntimeError(f"{name}: capture metadata mismatch")
    return value


def main() -> None:
    capture_root = Path(os.environ["H3MK_ATTENTION_CAPTURE"])
    reference_root = Path(os.environ.get(
        "H3MK_REFERENCE_CAPTURE", "/cache/runs/reference-bf16-vsa-20261115"))
    capture = json.loads((capture_root / "manifest.json").read_text())
    reference = json.loads((reference_root / "reference_manifest.json").read_text())
    if not capture.get("verified") or not reference.get("verified"):
        raise RuntimeError("unverified capture or reference")
    for key in ("checkpoint", "checkpoint_revision", "seed"):
        if capture[key] != reference[key]:
            raise RuntimeError(f"capture/reference {key} differs")
    if not reference["fastvideo_revision"].startswith(capture["fastvideo_revision"][:7]):
        raise RuntimeError("capture/reference FastVideo revision differs")
    if capture["capture_mode"] != "attention_inputs" or capture["tile_elems"] != 64:
        raise RuntimeError("unexpected attention capture mode or tile size")

    index = read_tensor(capture_root, capture, "untile_combined_index").to(torch.long)
    sizes = read_tensor(capture_root, capture, "variable_block_sizes")
    if index.numel() != capture["total_seq_length"] or sizes.numel() * 64 != 42752:
        raise RuntimeError("unexpected first-block tile geometry")
    if torch.unique(index).numel() != index.numel():
        raise RuntimeError("duplicate destination in tile map")
    if not bool(((index % 64) < sizes[index // 64]).all()):
        raise RuntimeError("real token maps into tile padding")

    results = {}
    for name in ("q", "k", "v", "gate"):
        source = read_tensor(capture_root, capture, f"{name}_pre_tile")
        expected = read_tensor(reference_root, reference, f"step0_block0_{name}")
        if source.shape != (1, index.numel(), 56, 128) or expected.shape != (1, 42752, 56, 128):
            raise RuntimeError(f"{name}: unexpected VSA shapes")
        tiled = torch.zeros_like(expected)
        tiled[:, index] = source
        unequal = int(torch.count_nonzero(tiled != expected))
        results[name] = {"source_shape": list(source.shape), "tiled_shape": list(tiled.shape),
                         "mismatched_elements": unequal}
        if unequal:
            raise RuntimeError(f"{name}: {unequal} elements differ from frozen VSA input")
    print(json.dumps({"verified": True, "tile_count": int(sizes.numel()),
                      "real_tokens": int(index.numel()), "results": results}, indent=2))


if __name__ == "__main__":
    main()
