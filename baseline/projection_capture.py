"""Capture the first FastH3 DiT Q projection, optionally including K/V/gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from fastvideo_capacity import ENV, MODEL, REVISION

FASTVIDEO_REVISION = "ce970eb715215c79a169658d7aa6c8a96e64891e"


class CapturedProjection(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-qkv", action="store_true",
                        help="also capture the K/V and active VSA-gate projections")
    parser.add_argument("--attention-inputs", action="store_true",
                        help="capture first-block Q/K norm and RoPE inputs, outputs, and VSA tile geometry")
    args = parser.parse_args()
    if args.attention_inputs:
        args.all_qkv = True
    prompts_path = Path(__file__).with_name("prompts.json")
    prompts = json.loads(prompts_path.read_text())
    prompt = prompts[0]
    out = Path(os.environ.get("OUT_DIR", "out/projection-capture"))
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update({**ENV, "FASTVIDEO_H3_VSA_FP4": "0",
                       "FASTVIDEO_VSA_PTX_BLACKWELL": "0"})

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.hooks.activation_trace import current_step_idx
    from fastvideo.models.dits import minimax_h3_vsa_fp4 as projection_module
    from fastvideo.models.dits import minimax_h3 as dit_module
    from fastvideo.attention.layer import DistributedAttention_VSA
    from fastvideo.forward_context import get_forward_context
    from huggingface_hub import snapshot_download

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("projection capture requires RTX PRO 6000 SM120")
    source = projection_module._shared_input_projections
    manifest = {
        "checkpoint": MODEL, "checkpoint_revision": REVISION,
        "fastvideo_revision": FASTVIDEO_REVISION,
        "prompt_index": 0, "seed": 1101, "shape": "1344x768", "frames": 124,
        "step_index": 0, "num_inference_steps": 9,
        "capture_mode": "attention_inputs" if args.attention_inputs else ("qkv" if args.all_qkv else "q"),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "env": {k: os.environ[k] for k in (*ENV, "FASTVIDEO_VSA_PTX_BLACKWELL")},
        "tensors": {},
    }

    def save(label, tensor):
        file = out / f"{label}.pt"
        saved = tensor.detach().contiguous().cpu()
        torch.save(saved, file)
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["tensors"][label] = {
            "file": file.name, "shape": list(tensor.shape),
            "dtype": str(saved.dtype), "stride": list(saved.stride()),
            "source_stride": list(tensor.stride()),
            "sha256": digest,
        }

    state = {"projection_seen": False, "qk_calls": 0, "attention_seen": False}

    def traced(linears, x):
        method = getattr(linears[0], "quant_method", None)
        prefix = getattr(method, "layer_prefix", "")
        step = current_step_idx()
        target = step == 0 and prefix.endswith("transformer_blocks.0.attn.to_q")
        result = source(linears, x)
        if target:
            if len(result) not in (3, 4) or x.dtype != torch.bfloat16 or result[0].dtype != torch.bfloat16:
                raise RuntimeError("unexpected first Q/K/V projection contract")
            if tuple(x.shape) != (1, 37771, 5376):
                raise RuntimeError(f"unexpected to_q input shape: {tuple(x.shape)}")
            if tuple(result[0].shape) != (1, 37771, 7168):
                raise RuntimeError(f"unexpected to_q output shape: {tuple(result[0].shape)}")
            save("to_q_input", x)
            save("to_q_output", result[0])
            if args.all_qkv:
                for label, tensor in zip(("to_k_output", "to_v_output"), result[1:3], strict=True):
                    if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (1, 37771, 7168):
                        raise RuntimeError(f"unexpected {label} shape or dtype: {tuple(tensor.shape)}, {tensor.dtype}")
                    save(label, tensor)
                manifest["gate_active"] = len(result) == 4
                if len(result) == 4:
                    if result[3].dtype != torch.bfloat16 or tuple(result[3].shape) != (1, 37771, 7168):
                        raise RuntimeError("unexpected active VSA gate projection contract")
                    save("to_gate_compress_output", result[3])
            manifest["layer_prefix"] = prefix
            manifest["projection_input_stage"] = "post_norm1_adaln"
            state["projection_seen"] = True
            if not args.attention_inputs:
                raise CapturedProjection
        return result

    fused_qknorm_rope = dit_module.fused_qknorm_rope
    distributed_forward = DistributedAttention_VSA.forward

    def traced_qknorm(x, weight, cos, sin, eps):
        if state["projection_seen"] and not state["attention_seen"]:
            index = state["qk_calls"]
            if index >= 2:
                raise RuntimeError("unexpected Q/K normalization call count")
            name = ("q", "k")[index]
            save(f"{name}_norm_weight", weight)
            if index == 0:
                save("rope_cos", cos)
                save("rope_sin", sin)
            elif cos.shape != tuple(manifest["tensors"]["rope_cos"]["shape"]):
                raise RuntimeError("Q/K rotary geometry differs")
            manifest[f"{name}_norm_eps"] = float(eps)
            state["qk_calls"] += 1
        return fused_qknorm_rope(x, weight, cos, sin, eps)

    def traced_attention(self, q, k, v, original_seq_len, *a, **kw):
        if state["projection_seen"] and not state["attention_seen"] and current_step_idx() == 0:
            if state["qk_calls"] != 2:
                raise RuntimeError("first attention did not use the expected fused Q/K norm and RoPE")
            meta = kw.get("attn_metadata") or get_forward_context().attn_metadata
            if meta is None or meta.tile_elems != 64:
                raise RuntimeError("first VSA metadata lacks 64-token tile geometry")
            for name, value in (("q_pre_tile", q), ("k_pre_tile", k), ("v_pre_tile", v),
                                ("gate_pre_tile", kw.get("gate_compress")),
                                ("untile_combined_index", meta.untile_combined_index),
                                ("variable_block_sizes", meta.variable_block_sizes)):
                if value is None:
                    raise RuntimeError(f"missing {name} at VSA boundary")
                save(name, value)
            manifest["tile_elems"] = int(meta.tile_elems)
            manifest["original_seq_len"] = int(original_seq_len)
            manifest["total_seq_length"] = int(meta.total_seq_length)
            manifest["pad_tokens"] = int(meta.pad_tokens)
            state["attention_seen"] = True
            raise CapturedProjection
        return distributed_forward(self, q, k, v, original_seq_len, *a, **kw)

    projection_module._shared_input_projections = traced
    if args.attention_inputs:
        dit_module.fused_qknorm_rope = traced_qknorm
        DistributedAttention_VSA.forward = traced_attention
    try:
        model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
        generator = VideoGenerator.from_config({
            "model_path": model_path,
            "engine": {
                "num_gpus": 1,
                "quantization": {"transformer_quant": "NVFP4", "layer_profile": "h3_dit_vsa"},
                "compile": {"enabled": False, "vae_enabled": True},
                "offload": {"dit": False, "dit_layerwise": False, "text_encoder": False,
                            "image_encoder": False, "vae": False},
            },
            "pipeline": {"experimental": {"attention_backend": "VIDEO_SPARSE_ATTN_H3",
                                           "VSA_sparsity": 0.8, "VSA_tile_size": 64}},
        })
        try:
            generator.generate({
                "prompt": prompt,
                "sampling": {"height": 768, "width": 1344, "num_frames": 124,
                             "num_inference_steps": 9, "guidance_scale": 1.0,
                             "seed": 1101, "fps": 24},
                "output": {"output_path": str(out), "output_video_name": "unused",
                           "return_frames": False},
            })
        except RuntimeError as exc:
            # VideoGenerator runs inference in a worker thread and wraps the
            # original exception when it rejoins. Accept only our sentinel.
            if not isinstance(exc.__cause__, CapturedProjection):
                raise
        else:
            raise RuntimeError("first attention boundary was not captured" if args.attention_inputs
                               else "first to_q projection was not captured")
        required = {"to_q_input", "to_q_output"}
        if args.all_qkv:
            required.update(("to_k_output", "to_v_output"))
            if manifest["gate_active"]:
                required.add("to_gate_compress_output")
        if args.attention_inputs:
            required.update(("q_norm_weight", "k_norm_weight", "rope_cos", "rope_sin",
                             "q_pre_tile", "k_pre_tile", "v_pre_tile", "gate_pre_tile",
                             "untile_combined_index", "variable_block_sizes"))
        if set(manifest["tensors"]) != required:
            raise RuntimeError("incomplete to_q tensor capture")
        manifest["verified"] = True
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        projection_module._shared_input_projections = source
        if args.attention_inputs:
            dit_module.fused_qknorm_rope = fused_qknorm_rope
            DistributedAttention_VSA.forward = distributed_forward
    print(json.dumps({"capture": str(out), "layer": manifest["layer_prefix"],
                      "tensors": manifest["tensors"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
