"""Print the loaded FastH3 DiT facts the Hazy DiT recipe depends on, then stop.

Run with fastvideo_modal.py::go and --script dit_introspect.py. It loads the
pinned BF16-VSA recipe, intercepts the denoising stage before the first
transformer forward, writes introspection.json, and exits without sampling.
"""
import json
import os
from pathlib import Path

from fastvideo_capacity import ENV, MODEL, REVISION


class Done(Exception):
    pass


def main():
    out = Path(os.environ.get("OUT_DIR", "out/introspect"))
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update({**ENV, "FASTVIDEO_H3_VSA_FP4": "0", "FASTVIDEO_VSA_PTX_BLACKWELL": "0"})
    import torch
    from fastvideo import VideoGenerator
    from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG, build_row_timesteps
    from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_denoising as stage_module
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
    from huggingface_hub import snapshot_download

    info = {}

    def describe(t):
        if t is None:
            return None
        return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}

    def linear(mod):
        if mod is None:
            return None
        row = {"type": type(mod).__name__,
               "quant": type(getattr(mod, "quant_method", None)).__name__,
               "weight": describe(getattr(mod, "weight", None)),
               "bias": describe(getattr(mod, "bias", None))}
        for name in ("_nvfp4_weight", "_nvfp4_weight_scale", "_nvfp4_alpha", "_weight_global_sf",
                     "_nvfp4_input_global_sf"):
            value = getattr(mod, name, None)
            if value is not None:
                row[name] = describe(value)
                if value.numel() == 1:
                    row[name]["value"] = float(value.float())
        method = getattr(mod, "quant_method", None)
        if method is not None and hasattr(method, "x_global_sf"):
            row["x_global_sf"] = float(method.x_global_sf)
            row["static_sf"] = (None if method._static_activation_global_sf() is None
                                else float(method._static_activation_global_sf()))
            row["dynamic"] = bool(method._dynamic_activation_scale())
        return row

    def forward(self, batch, fastvideo_args):
        t = self.transformer
        if hasattr(t, "materialize"):
            t.materialize()
        layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
        device = batch.latents.device if batch.latents.is_cuda else torch.device("cuda")
        self._configure_schedules(batch, fastvideo_args, device)
        info["transformer_type"] = type(t).__name__
        info["enabled_fusions"] = sorted(t.enabled_fusions)
        info["exact_kernels"] = sorted(t.enabled_exact_kernels)
        info["adaln_host_cache"] = bool(getattr(t, "_adaln_host_cache", False))
        info["adaln_rank"] = t.adaln_rank
        info["pdd_steps"] = t.pdd_steps
        info["splice"] = "_splice_late" in t.__dict__
        info["top"] = {name: linear(getattr(t, name)) for name in
                       ("proj_in", "audio_proj_in", "context_embedder", "proj_out", "audio_proj_out")}
        info["time_embedder"] = {"fc_in": linear(t.time_embedder.fc_in), "fc_out": linear(t.time_embedder.fc_out),
                                 "repr": repr(t.time_embedder)}
        info["time_proj"] = repr(t.time_proj) + " " + json.dumps(
            {k: str(v) for k, v in vars(t.time_proj).items() if not k.startswith("_")})
        info["norm_out"] = {"linear": linear(t.norm_out.linear), "norm_weight": describe(t.norm_out.norm.weight),
                            "eps": t.norm_out.norm.eps, "apply_silu": t.norm_out.apply_silu}
        blocks = []
        for index, block in enumerate(t.transformer_blocks):
            a = block.attn
            row = {
                "index": index,
                "gate_active": bool(a._gate_active()) if a.to_gate_compress is not None else None,
                "attn_impl": type(a.distributed_attention.attn_impl).__name__,
                "vsa_fp4": bool(a._vsa_fp4), "tile_first": bool(a._vsa_tile_first),
                "fuse": [block.fuse_modulate, a.fuse_qknorm_rope, block.ff.fuse_swiglu, block.mk_tier],
                "ffn_chunk": block.ff.chunk_tokens, "mxfp8": block.ff.use_mxfp8,
                "adaln_silu": block.adaln_proj.apply_silu,
                "adaln_cache": hasattr(block.adaln_proj, "_modulation_cache"),
            }
            if index in (0, 1, 49):
                row.update({
                    "norm1": describe(block.norm1.weight), "norm2": describe(block.norm2.weight),
                    "norm_eps": [block.norm1.eps, block.norm2.eps, a.norm_q.eps, a.norm_k.eps],
                    "norm_q": describe(a.norm_q.weight),
                    "adaln": linear(block.adaln_proj.linear),
                    **{name: linear(getattr(a, name)) for name in ("to_q", "to_k", "to_v", "to_out",
                                                                     "to_gate_compress")},
                    "fc_in": linear(block.ff.fc_in), "fc_out": linear(block.ff.fc_out),
                })
            else:
                row["input_sf"] = {name: (float(getattr(mod, "_nvfp4_input_global_sf").float())
                                          if getattr(mod, "_nvfp4_input_global_sf", None) is not None else None)
                                   for name, mod in (("to_q", a.to_q), ("to_k", a.to_k), ("to_v", a.to_v),
                                                     ("to_out", a.to_out), ("gate", a.to_gate_compress),
                                                     ("fc_in", block.ff.fc_in), ("fc_out", block.ff.fc_out))}
            blocks.append(row)
        info["blocks"] = blocks
        refiner = []
        for block in t.token_refiner.refiner_blocks:
            a = block.attn
            refiner.append({
                "attn_impl": type(a.distributed_attention.attn_impl).__name__,
                "attn_cls": type(a.distributed_attention).__name__,
                "fuse_qknorm_rope": a.fuse_qknorm_rope, "gate": a.to_gate_compress is not None,
                "norm1": describe(block.norm1.weight), "eps": [block.norm1.eps, block.norm2.eps, a.norm_q.eps],
                **{name: linear(getattr(a, name)) for name in ("to_q", "to_k", "to_v", "to_out")},
                "fc_in": linear(block.ff.fc_in), "fc_out": linear(block.ff.fc_out),
                "ff": [block.ff.fuse_swiglu, block.ff.exact_swiglu, block.ff.chunk_tokens],
            })
        info["refiner"] = refiner
        info["refiner_final_norm"] = {"weight": describe(t.token_refiner.final_norm.weight),
                                      "eps": t.token_refiner.final_norm.eps}
        info["layout"] = {
            "sequence_length": layout.sequence_length, "pad_tokens": layout.pad_tokens,
            "text": int(layout.text_indices.numel()), "video": int(layout.video_indices.numel()),
            "audio": int(layout.audio_indices.numel()),
            "text_range": [int(layout.text_indices.min()), int(layout.text_indices.max())],
            "video_range": [int(layout.video_indices.min()), int(layout.video_indices.max())],
            "audio_range": [int(layout.audio_indices.min()), int(layout.audio_indices.max())],
            "cond_video": layout.num_condition_video_rows, "cond_audio": layout.num_condition_audio_rows,
            "latent": [layout.num_video_latent_frames, layout.latent_height, layout.latent_width],
            "token_tags_unique": torch.unique(layout.token_tags, return_counts=True)[0].tolist(),
            "token_tags_counts": torch.unique(layout.token_tags, return_counts=True)[1].tolist(),
            "position_ids": describe(layout.position_ids),
        }
        info["inputs"] = {"latents": describe(batch.latents), "audio_latents": describe(batch.audio_latents),
                          "prompt_embeds": [describe(p) for p in batch.prompt_embeds],
                          "num_inference_steps": batch.num_inference_steps,
                          "VSA_sparsity": float(batch.VSA_sparsity),
                          "extra_keys": sorted(str(k) for k in batch.extra.keys()),
                          "vsa_mode": batch.extra.get("vsa_mode", "exempt"),
                          "vsa_dense_layers": list(batch.extra.get("vsa_dense_layers", ())),
                          "vsa_dense_first_n_steps": int(batch.extra.get("vsa_dense_first_n_steps", 0)),
                          "VSA_tile_size": int(fastvideo_args.VSA_tile_size),
                          "token_reduction": repr(getattr(fastvideo_args, "h3_token_reduction", {})),
                          "dmd_steps": fastvideo_args.pipeline_config.dmd_denoising_steps,
                          "vsa_ref_keep_rate": getattr(fastvideo_args.pipeline_config, "vsa_ref_keep_rate", None)}
        sched = {}
        for name, scheduler in (("video", self.scheduler), ("audio", self.audio_scheduler)):
            sched[name] = {"type": type(scheduler).__name__, "shift": float(scheduler.shift),
                           "sigmas": [float(x) for x in scheduler.sigmas],
                           "timesteps": [float(x) for x in scheduler.timesteps],
                           "config": {k: repr(v) for k, v in dict(scheduler.config).items()}}
        info["schedulers"] = sched
        rows = []
        for vt, at in zip(self.scheduler.timesteps, self.audio_scheduler.timesteps, strict=True):
            unique, inverse = build_row_timesteps(
                layout, video_timestep=float(vt), audio_timestep=float(at),
                condition_video_timestep=max(float(vt), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                condition_audio_timestep=1.0)
            rows.append({"unique": [float(x) for x in unique],
                         "counts": torch.bincount(inverse).tolist()})
        info["row_timesteps"] = rows
        builder = stage_module._h3_vsa_metadata_builder(t, fastvideo_args)
        patch = fastvideo_args.pipeline_config.dit_config.patch_size
        segments = stage_module._h3_vsa_single_region_segments(layout, patch)
        meta = builder.build(current_timestep=0, patch_size=patch, VSA_sparsity=float(batch.VSA_sparsity),
                             packed_segments=segments, device=torch.device("cuda"), exempt=True,
                             dense_layers=tuple(batch.extra.get("vsa_dense_layers", ())),
                             tile_size=int(fastvideo_args.VSA_tile_size), ref_keep_rate=None)
        info["vsa"] = {"segments": [list(s) if isinstance(s, tuple) else s for s in segments],
                       "total_seq_length": meta.total_seq_length, "num_prefix_tiles": meta.num_prefix_tiles,
                       "num_video_tiles": meta.num_video_tiles, "exempt": meta.exempt,
                       "spans": [list(s) for s in meta.video_tile_spans],
                       "span_sparsities": list(meta.span_sparsities), "pad_tokens": meta.pad_tokens,
                       "tile_elems": meta.tile_elems, "dense_layers": list(meta.dense_layers),
                       "valid": describe(meta.variable_block_sizes),
                       "valid_hist": torch.unique(meta.variable_block_sizes, return_counts=True)[0].tolist(),
                       "valid_first16": meta.variable_block_sizes[:16].tolist(),
                       "untile": describe(meta.untile_combined_index),
                       "untile_first8": meta.untile_combined_index[:8].tolist(),
                       "untile_is_perm": int(torch.unique(meta.untile_combined_index).numel())}
        import inspect as pyinspect
        from fastvideo.attention.backends import video_sparse_attn_h3 as vsa_module
        info["tile_source"] = pyinspect.getsource(vsa_module.MiniMaxH3VSAImpl.tile)
        info["builder_source"] = pyinspect.getsource(vsa_module.MiniMaxH3VSAMetadataBuilder.build)
        info["env"] = {k: v for k, v in os.environ.items() if k.startswith(("FASTVIDEO", "TORCH"))}
        info["memory_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
        raise Done

    original = stage_module.MiniMaxH3DenoisingStage.forward
    stage_module.MiniMaxH3DenoisingStage.forward = forward
    prompt = json.loads(Path(__file__).with_name("prompts.json").read_text())[0]
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
                         "num_inference_steps": 9, "guidance_scale": 1.0, "seed": 1101, "fps": 24},
            "output": {"output_path": str(out), "output_video_name": "unused", "return_frames": False},
        })
    except Done:
        pass
    finally:
        stage_module.MiniMaxH3DenoisingStage.forward = original
        (out / "introspection.json").write_text(json.dumps(info, indent=2, default=str))
    if "vsa" not in info:
        raise RuntimeError("introspection did not reach the denoising stage")
    print(json.dumps(info, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
