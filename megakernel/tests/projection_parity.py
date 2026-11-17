"""Parity and resource probe for one pinned FastH3 NVFP4 DiT projection.

Requires the FastVideo environment, checkpoint cache, reference tensor capture,
RTX PRO 6000, and the separately built cutlass_projection extension.
"""
import json
import os
import hashlib
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from fastvideo.layers.quantization.nvfp4_config import _mm_fp4, _nvfp4_quantize
from flashinfer import SfLayout

CHECKPOINT = "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer"
REVISION = "9d0d0d3ae69365aded83fab5562f00f9f06c12aa"
SUFFIX = "attn.to_q::_nvfp4_weight"


def captured_tensors(extra_labels=()):
    capture_path = os.environ.get("H3MK_PROJECTION_CAPTURE")
    if not capture_path:
        raise RuntimeError("H3MK_PROJECTION_CAPTURE must name the successful exact capture")
    reference = Path(capture_path)
    if (reference / "returncode").read_text().strip() != "0":
        raise RuntimeError("reference capture did not exit successfully")
    manifest = json.loads((reference / "manifest.json").read_text())
    prompts = json.loads((Path(__file__).resolve().parents[2] / "baseline/prompts.json").read_text())
    if (manifest.get("verified") is not True or
            manifest.get("checkpoint") != CHECKPOINT or
            manifest.get("checkpoint_revision") != REVISION or
            manifest.get("fastvideo_revision") != "ce970eb715215c79a169658d7aa6c8a96e64891e" or
            manifest.get("env", {}).get("FASTVIDEO_H3_VSA_FP4") != "0" or
            not manifest.get("layer_prefix", "").endswith("transformer_blocks.0.attn.to_q") or
            manifest.get("projection_input_stage") != "post_norm1_adaln" or
            manifest.get("step_index") != 0 or
            manifest.get("num_inference_steps") != 9 or
            manifest.get("prompt_index") != 0 or
            manifest.get("prompt_sha256") != hashlib.sha256(prompts[0].encode()).hexdigest() or
            manifest.get("seed") != 1101 or manifest.get("shape") != "1344x768" or
            manifest.get("frames") != 124):
        raise RuntimeError("capture manifest does not match the pinned exact projection")
    allowed_extra = {"to_k_output", "to_v_output", "to_gate_compress_output"}
    if set(extra_labels) - allowed_extra or (extra_labels and manifest.get("capture_mode") != "qkv"):
        raise RuntimeError("capture does not contain verified Q/K/V outputs")
    tensors = {}
    shapes = (("to_q_input", (1, 37771, 5376)),
              ("to_q_output", (1, 37771, 7168)),
              *((label, (1, 37771, 7168)) for label in extra_labels))
    for label, shape in shapes:
        meta = manifest["tensors"][label]
        path = reference / meta["file"]
        if meta["file"] != f"{label}.pt":
            raise RuntimeError(f"unexpected capture file for {label}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != meta["sha256"]:
            raise RuntimeError(f"invalid capture file or digest for {label}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if (tuple(tensor.shape) != tuple(meta["shape"]) or
                tuple(tensor.shape) != shape or
                str(tensor.dtype) != meta["dtype"] or tensor.dtype != torch.bfloat16 or
                list(tensor.stride()) != meta["stride"]):
            raise RuntimeError(f"invalid capture tensor contract for {label}")
        tensors[label] = tensor
    return tuple(tensors[label] for label, _ in shapes)


def compare(label, actual, expected):
    delta = (actual.float() - expected.float()).abs()
    # The same packed operands and SM120 CUTLASS BF16 path matched bit for bit
    # in the preceding block-input probe. Require that contract here as well.
    bad = actual != expected
    print(label, "max_abs_error", delta.max().item(), "mean_abs_error", delta.mean().item(),
          "mismatch_count", bad.count_nonzero().item(), "elements", delta.numel(), flush=True)
    if bad.any():
        raise AssertionError(f"{label}: {bad.count_nonzero().item()} BF16 elements differ (atol=rtol=0)")


def main():
    from cutlass_projection import resources, run
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("RTX PRO 6000 SM120 required")
    if os.environ.get("FASTVIDEO_H3_VSA_FP4", "0") != "0":
        raise RuntimeError("probe must use the no-FP4-attention FastVideo recipe")
    path = snapshot_download(CHECKPOINT, revision=REVISION, local_files_only=True)
    export = Path(path) / "transformer/nvfp4_weights.safetensors"
    with safe_open(export, framework="pt", device="cpu") as reader:
        keys = [key for key in reader.keys()
                if key.startswith("transformer_blocks.") and key.endswith(SUFFIX)]
        if not keys:
            raise RuntimeError("checkpoint has no packed DiT to_q projection")
        key = sorted(keys, key=lambda x: int(x.split(".")[1]))[0]
        prefix = key.removesuffix("::_nvfp4_weight")
        fields = {name: reader.get_tensor(f"{prefix}::{name}").cuda()
                  for name in ("_nvfp4_weight", "_nvfp4_weight_scale", "_nvfp4_alpha")}
        optional_key = f"{prefix}::_nvfp4_input_global_sf"
        input_sf = reader.get_tensor(optional_key).cuda() if optional_key in reader.keys() else None
    weight = fields["_nvfp4_weight"]
    weight_sf = fields["_nvfp4_weight_scale"]
    n, packed_k = weight.shape
    k = packed_k * 2
    source, captured = captured_tensors()
    if source.shape[-1] != k:
        raise RuntimeError(f"reference input K={source.shape[-1]} != weight K={k}")
    if input_sf is None:
        input_sf = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    alpha = (fields["_nvfp4_alpha"].float() / input_sf.float()).reshape(())
    for m in (128, source.numel() // k):
        x = source.reshape(-1, k)[:m].contiguous().cuda()
        a, a_sf = _nvfp4_quantize(x, input_sf, sfLayout=SfLayout.layout_128x4,
                                  do_shuffle=False)
        expected = _mm_fp4(a, weight.T, a_sf, weight_sf.T, alpha,
                           torch.bfloat16, None, backend="auto")
        reference = captured.reshape(-1, n)[:m].contiguous().cuda()
        compare("FastVideo mm_fp4 vs captured to_q", expected, reference)
        metadata = dict(resources(m, n, k))
        print("projection", prefix, "shape", (m, n, k), flush=True)
        print("packed", tuple(a.shape), tuple(weight.shape), "scales",
              tuple(a_sf.shape), tuple(weight_sf.shape), flush=True)
        print("CUTLASS", json.dumps(metadata), flush=True)
        if metadata["threads_per_cta"] != 384:
            raise RuntimeError("CUTLASS CTA thread count differs from the Hazy SM120 runtime")
        # Hazy uses 10,496 static bytes, reserves 100,000 total, and aligns its
        # dynamic page base upward by up to 1,023 bytes.
        available_shared = 100000 - 10496 - 1024
        print("Hazy dynamic storage available for CUTLASS", available_shared, flush=True)
        if metadata["shared_bytes"] > available_shared:
            raise RuntimeError("CUTLASS collective cannot fit in Hazy's dynamic shared region")
        if not metadata["can_implement"]:
            raise RuntimeError("CUTLASS cannot implement the checkpoint projection")
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        scratch = torch.empty(max(1, metadata["workspace_bytes"]), device="cuda", dtype=torch.uint8)
        run(a.data_ptr(), weight.data_ptr(), a_sf.data_ptr(), weight_sf.data_ptr(),
            out.data_ptr(), scratch.data_ptr(), m, n, k, float(alpha.item()),
            torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        compare("CUTLASS vs FastVideo mm_fp4", out, expected)
        compare("CUTLASS vs captured to_q", out, reference)
        del x, a, a_sf, expected, reference, out, scratch
    print("PASS: exact AdaLN/norm to_q parity at 128 and full sequence length", flush=True)


if __name__ == "__main__":
    main()
