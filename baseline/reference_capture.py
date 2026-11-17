"""Capture the pinned FastH3 V2 Consumer / BF16-VSA reference in one run.

Run with fastvideo_modal.py::go and --script reference_capture.py. The only
prompt is read from prompts.json; this module never supplies a fallback prompt.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from fastvideo_capacity import ENV, MODEL, REVISION

FASTVIDEO_REVISION = "ce970eb715215c79a169658d7aa6c8a96e64891e"


def _git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--shape", default="1344x768")
    parser.add_argument("--frames", type=int, default=124)
    args = parser.parse_args()
    prompts_path = Path(__file__).with_name("prompts.json")
    prompts_bytes = prompts_path.read_bytes()
    prompts = json.loads(prompts_bytes)
    if not isinstance(prompts, list) or not 0 <= args.prompt_index < len(prompts):
        raise ValueError("--prompt-index must select a prompt from prompts.json")
    prompt = prompts[args.prompt_index]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("selected prompt is empty")
    width, height = (int(part) for part in args.shape.split("x"))
    out = Path(os.environ.get("OUT_DIR", "out/reference"))
    tensors = out / "tensors"
    tensors.mkdir(parents=True, exist_ok=True)

    env = {**ENV, "FASTVIDEO_H3_VSA_FP4": "0", "FASTVIDEO_VSA_PTX_BLACKWELL": "0"}
    os.environ.update(env)
    import torch
    from fastvideo import VideoGenerator
    from fastvideo.hooks.activation_trace import current_step_idx
    from fastvideo.models.dits import minimax_h3 as dit_module
    from fastvideo.attention.backends import video_sparse_attn_h3 as vsa_module
    from fastvideo.pipelines.stages.base import PipelineStage
    from huggingface_hub import snapshot_download

    manifest = {
        "checkpoint": MODEL,
        "checkpoint_revision": REVISION,
        "fastvideo_revision": _git_revision("/FastVideo") if Path("/FastVideo/.git").exists() else FASTVIDEO_REVISION,
        "prompt_file_sha256": hashlib.sha256(prompts_bytes).hexdigest(),
        "prompt_index": args.prompt_index,
        "prompt": prompt,
        "seed": args.seed,
        "shape": args.shape,
        "frames": args.frames,
        "env": env,
        "tensors": {},
        "counts": {},
    }

    def save(name, value):
        if not isinstance(value, torch.Tensor):
            return
        path = tensors / f"{name}.pt"
        if path.exists():
            raise RuntimeError(f"duplicate capture {name}")
        cpu = value.detach().contiguous().cpu()
        torch.save(cpu, path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["tensors"][name] = {
            "file": str(path.relative_to(out)), "sha256": digest,
            "shape": list(value.shape), "dtype": str(value.dtype),
            "stride": list(value.stride()), "numel": value.numel(),
        }

    selected_steps = {0, 7}
    selected_blocks = {0, 49}
    calls = defaultdict(int)
    stage_call = PipelineStage.__call__

    def traced_stage(self, batch, fastvideo_args):
        name = type(self).__name__
        result = stage_call(self, batch, fastvideo_args)
        if name == "MiniMaxH3ConditioningStage" and result.prompt_embeds:
            save("prompt_embeddings", result.prompt_embeds[0])
        elif name == "MiniMaxH3LatentPreparationStage":
            save("initial_video_latents", result.latents)
            save("initial_audio_latents", result.audio_latents)
        elif name == "MiniMaxH3DenoisingStage":
            save("final_video_latents", result.latents)
            save("final_audio_latents", result.audio_latents)
        elif name == "MiniMaxH3VideoDecodingStage":
            save("decoded_video", result.output)
        elif name == "MiniMaxH3AudioDecodingStage":
            save("decoded_audio", result.extra.get("audio"))
        return result

    block_forward = dit_module.MiniMaxH3TransformerBlock.forward

    def traced_block(self, hidden_states, *a, **kw):
        step = current_step_idx()
        index = calls[("block", step)]
        calls[("block", step)] += 1
        if step in selected_steps and index in selected_blocks:
            save(f"step{step}_block{index}_input", hidden_states)
        output = block_forward(self, hidden_states, *a, **kw)
        if step in selected_steps and index in selected_blocks:
            save(f"step{step}_block{index}_output", output)
        return output

    vsa_forward = vsa_module.MiniMaxH3VSAImpl.forward

    def traced_vsa(self, query, key, value, gate_compress, metadata):
        step = current_step_idx()
        layer = self.layer_idx
        calls[("vsa", step)] += 1
        if step in selected_steps and layer in selected_blocks:
            for label, tensor in (("q", query), ("k", key), ("v", value), ("gate", gate_compress)):
                save(f"step{step}_block{layer}_{label}", tensor)
            save(f"step{step}_block{layer}_valid_lengths", metadata.variable_block_sizes)
        if step == 0 and layer == 0:
            logical_rows = metadata.variable_block_sizes.numel() * metadata.tile_elems
            q_pooled = vsa_module._pool_tiles(
                query[:, :logical_rows], metadata.variable_block_sizes, metadata.tile_elems)
            k_pooled = vsa_module._pool_tiles(
                key[:, :logical_rows], metadata.variable_block_sizes, metadata.tile_elems)
            scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (query.shape[-1] ** 0.5)
            save("step0_block0_q_pooled", q_pooled)
            save("step0_block0_k_pooled", k_pooled)
            save("step0_block0_vsa_scores", scores)
            manifest["step0_block0_vsa_policy"] = {
                "tile_elems": metadata.tile_elems,
                "num_prefix_tiles": metadata.num_prefix_tiles,
                "video_tile_spans": [list(span) for span in metadata.video_tile_spans],
                "span_sparsities": list(metadata.span_sparsities),
                "VSA_sparsity": metadata.VSA_sparsity,
                "exempt": metadata.exempt,
                "dense_layers": list(metadata.dense_layers),
            }
        output = vsa_forward(self, query, key, value, gate_compress, metadata)
        if step in selected_steps and layer in selected_blocks:
            save(f"step{step}_block{layer}_attention", output)
        return output

    build_mask = vsa_module._build_block_mask

    def traced_mask(*a, **kw):
        mask = build_mask(*a, **kw)
        step = current_step_idx()
        index = calls[("mask", step)]
        calls[("mask", step)] += 1
        if step in selected_steps and index in selected_blocks:
            save(f"step{step}_block{index}_vsa_mask", mask)
        return mask

    def forbidden_fp4(*a, **kw):
        raise RuntimeError("FP4 attention ran in the BF16-VSA reference")

    PipelineStage.__call__ = traced_stage
    dit_module.MiniMaxH3TransformerBlock.forward = traced_block
    dit_module.vsa_fp4_attention = forbidden_fp4
    dit_module.vsa_fp4_attention_sp = forbidden_fp4
    vsa_module.MiniMaxH3VSAImpl.forward = traced_vsa
    vsa_module._build_block_mask = traced_mask

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
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("reference requires SM120")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = generator.generate({
            "prompt": prompt,
            "sampling": {"height": height, "width": width, "num_frames": args.frames,
                         "num_inference_steps": 9, "guidance_scale": 1.0, "seed": args.seed, "fps": 24},
            "output": {"output_path": str(out), "output_video_name": "reference", "return_frames": False},
        })
        torch.cuda.synchronize()
        manifest["wall_s"] = time.perf_counter() - start
        manifest["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        manifest["video_path"] = result.video_path
        manifest["stages"] = getattr(result.logging_info, "stages", {})
        manifest["gpu"] = torch.cuda.get_device_name()
        manifest["counts"] = {f"{name}_step{step}": count for (name, step), count in calls.items()}
        for step in range(8):
            if calls[("block", step)] != 50 or calls[("vsa", step)] != 50 or calls[("mask", step)] != 50:
                raise RuntimeError(f"reference call counts wrong at step {step}: {manifest['counts']}")
        required = {"prompt_embeddings", "initial_video_latents", "initial_audio_latents",
                    "final_video_latents", "final_audio_latents", "decoded_video", "decoded_audio",
                    "step0_block0_vsa_mask", "step7_block49_vsa_mask"}
        if missing := required - manifest["tensors"].keys():
            raise RuntimeError(f"reference tensors missing: {sorted(missing)}")
        manifest["verified"] = True
    finally:
        (out / "reference_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        PipelineStage.__call__ = stage_call
        dit_module.MiniMaxH3TransformerBlock.forward = block_forward
        vsa_module.MiniMaxH3VSAImpl.forward = vsa_forward
        vsa_module._build_block_mask = build_mask


if __name__ == "__main__":
    main()
