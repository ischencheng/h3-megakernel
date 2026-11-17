"""FastH3 DiT gates and end-to-end generation with the DiT in one Hazy launch.

The pinned FastVideo pipeline runs unmodified up to denoising. Its own
denoising loop is executed first and traced as the reference; the same
initial latents are then denoised by the Hazy megakernel in the gates
selected with --gates:

  block   step 0, block 0 only            -> block output
  step    step 0, all blocks and heads    -> block 49 output, velocity, latents
  full    all steps in ONE launch         -> final video and audio latents

With --pipeline megakernel the remaining FastVideo stages (video and audio
decoders, muxing) continue from the megakernel latents, so the saved clip is
produced by the one-launch DiT. Text encoding and decoding remain FastVideo
stages and are reported as such.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ATOL, RTOL = 0.03125, 0.015625


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="new run directory under /cache/runs")
    parser.add_argument("--gates", default="block,step,full")
    parser.add_argument("--pipeline", choices=("reference", "megakernel"), default="megakernel")
    parser.add_argument("--slow-attention", action="store_true",
                        help="use the warp-per-query fine attention instead of tensor cores")
    parser.add_argument("--frozen", default="/cache/runs/reference-bf16-vsa-20261115")
    parser.add_argument("--prompts-file", default="prompts.json",
                        help="JSON prompt list in baseline/; prompts are never written in code")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--profile-range", choices=("none", "denoising", "megakernel", "kernels"),
                        default="none",
                        help="cudaProfilerStart/Stop around FastVideo's loop plus the megakernel "
                             "(for nsys), around the megakernel launch only (for ncu), or around the "
                             "megakernel launch and FastVideo's step 1 block 1 attention (for ncu)")
    parser.add_argument("--record-timings", action="store_true",
                        help="decode per-instruction runtime clocks (module built with TIMINGS=1)")
    parser.add_argument("--seed", type=int, default=1101)
    args = parser.parse_args()
    gates = [g for g in args.gates.split(",") if g]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    from fastvideo_capacity import ENV, MODEL, REVISION
    os.environ.update({**ENV, "FASTVIDEO_H3_VSA_FP4": "0", "FASTVIDEO_VSA_PTX_BLACKWELL": "0"})
    if args.profile_range == "denoising":
        os.environ["FASTVIDEO_NVTX_PROFILE"] = "1"  # FastVideo stage, step and block ranges
    import torch
    from fastvideo import VideoGenerator
    from fastvideo.models.dits import minimax_h3 as dit_module
    from fastvideo.pipelines.basic.minimax_h3.packing import (MINIMAX_H3_KEYFRAME_NOISE_AUG,
                                                              build_row_timesteps)
    from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_denoising as stage_module
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import (
        MINIMAX_H3_LAYOUT_KEY)
    from fastvideo.pipelines.stages.base import PipelineStage
    from huggingface_hub import snapshot_download

    import h3_dit as hd
    import mk_h3_dit

    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("RTX PRO 6000 SM120 required")
    if "/" in args.prompts_file:
        raise ValueError("--prompts-file names a file in baseline/")
    prompts = json.loads((Path("/root/baseline") / args.prompts_file).read_text())
    prompt = prompts[args.prompt_index]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("selected prompt is empty")
    frozen_case = args.prompts_file == "prompts.json" and args.prompt_index == 0 and args.seed == 1101
    report = {"checkpoint": MODEL, "checkpoint_revision": REVISION, "prompt_index": args.prompt_index,
              "prompts_file": args.prompts_file, "prompt": prompt, "profile_range": args.profile_range,
              "seed": args.seed, "gates": {}, "tolerance": {"atol": ATOL, "rtol": RTOL},
              "fast_attention": not args.slow_attention, "pipeline_latents": args.pipeline,
              "gpu": torch.cuda.get_device_name()}

    def compare(name, actual, expected):
        a, e = actual.detach().float().reshape(-1), expected.detach().float().to(actual.device).reshape(-1)
        if a.numel() != e.numel():
            raise ValueError(f"{name}: {a.numel()} elements vs {e.numel()}")
        diff = (a - e).abs()
        row = {
            "elements": a.numel(),
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "reference_rms": float(e.square().mean().sqrt()),
            "relative_l2_error": float(torch.linalg.vector_norm(a - e) /
                                       torch.linalg.vector_norm(e).clamp_min(1e-30)),
            "cosine": float(torch.nn.functional.cosine_similarity(a, e, dim=0)),
            "outside_tolerance": int((~torch.isclose(a, e, atol=ATOL, rtol=RTOL)).count_nonzero()),
            "finite": bool(torch.isfinite(a).all()),
        }
        print(f"COMPARE {name} {json.dumps(row)}", flush=True)
        return row

    trace = {"block": {}, "velocity": [], "latents_in": [], "adaln": {}, "masks": {}}
    state = {"step": -1}
    block_forward = dit_module.MiniMaxH3TransformerBlock.forward
    transformer_forward = dit_module.MiniMaxH3Transformer3DModel.forward
    refined_text = dit_module.MiniMaxH3Transformer3DModel._refined_text
    stage_forward = stage_module.MiniMaxH3DenoisingStage.forward
    stage_call = PipelineStage.__call__
    decoded = {}

    def traced_block(self, hidden_states, temb, adaln_indices, *a, **kw):
        index = state["block"]
        state["block"] += 1
        first = state["step"] == 0 and index == 0
        hooks = []
        if args.profile_range == "kernels" and state["step"] == 1 and index == 1:
            # One warm FastVideo attention call as the Nsight Compute baseline.
            def begin(module, inputs):
                torch.cuda.synchronize()
                torch.cuda.profiler.start()

            def finish(module, inputs, output):
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()

            hooks += [self.attn.distributed_attention.register_forward_pre_hook(begin),
                      self.attn.distributed_attention.register_forward_hook(finish)]
        if first:
            def keep(name, pick=lambda inputs, output: output):
                def hook(module, inputs, output):
                    value = pick(inputs, output)
                    value = value[0] if isinstance(value, tuple) else value
                    trace[name] = value.detach().clone()
                return hook
            hooks = [
                self.attn.distributed_attention.register_forward_hook(keep("ref_attention")),
                self.attn.register_forward_hook(keep("ref_attn_out")),
                self.attn.register_forward_hook(keep("ref_normed1", lambda i, o: i[0])),
                self.ff.register_forward_hook(keep("ref_ff_out")),
                self.ff.register_forward_hook(keep("ref_normed2", lambda i, o: i[0])),
            ]
            trace["block_input"] = hidden_states.detach().clone()
            trace["adaln_indices"] = adaln_indices.detach().clone()
        try:
            output = block_forward(self, hidden_states, temb, adaln_indices, *a, **kw)
        finally:
            for hook in hooks:
                hook.remove()
        if state["step"] == 0 and index in (0, len(state["blocks"]) - 1):
            trace["block"][index] = output.detach().clone()
            if index == 0:
                trace["adaln"][0] = torch.cat([t.detach() for t in self.adaln_proj(temb)], dim=-1)
        return output

    def traced_transformer(self, hidden_states, audio_hidden_states, *a, **kw):
        state["step"] += 1
        state["block"] = 0
        state["blocks"] = self.transformer_blocks
        trace["latents_in"].append((hidden_states[0].detach().clone(),
                                    audio_hidden_states[0].detach().clone()))
        video, audio = transformer_forward(self, hidden_states, audio_hidden_states, *a, **kw)
        trace["velocity"].append((video[0].detach().float().clone(), audio[0].detach().float().clone()))
        return video, audio

    def traced_text(self, *a, **kw):
        text = refined_text(self, *a, **kw)
        trace["text"] = text.detach().clone()
        return text

    def run_gate(name, recipe, prompt_embeds, video0, audio0, **options):
        torch.cuda.synchronize()
        start = time.perf_counter()
        compiled = hd.compile_dit(mk_h3_dit, recipe, prompt_embeds, video0, audio0,
                                  fast_attention=not args.slow_attention, **options)
        compile_s = time.perf_counter() - start
        compiled.program.discover_sms()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.nvtx.range_push(f"hazy_megakernel.{name}")
        if args.profile_range in ("megakernel", "kernels"):
            torch.cuda.profiler.start()
        gpu_ms = compiled.program.run()
        if args.profile_range in ("megakernel", "kernels"):
            torch.cuda.profiler.stop()
        torch.cuda.nvtx.range_pop()
        row = {"instructions": compiled.stage_counts["instructions"],
               "projections": compiled.stage_counts.get("projection", 0),
               "attention_calls": compiled.stage_counts.get("attention", 0),
               "host_compile_s": compile_s, "launches": 1, "gpu_ms": gpu_ms,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "compare": {}}
        print(f"GATE {name}: {row['instructions']} instructions, one launch, {gpu_ms:.1f} ms GPU; "
              f"FastVideo denoising wall {report['fastvideo_denoising_wall_s']:.1f} s", flush=True)
        report["gates"][name] = row
        if args.record_timings:
            summary = compiled.program.timing_summary(gpu_ms)
            (out / f"timings-{name}.json").write_text(json.dumps(summary, indent=2))
            torch.save(compiled.program.last_timings, out / f"timings-{name}.pt")
            row["timings_by_op"] = summary["by_op"]
            row["timings_sum_of_stage_ms"] = summary["sum_of_stage_ms"]
            for op, entry in list(summary["by_op"].items())[:25]:
                print(f"TIMING {name} {op}: {entry['ms']:.1f} ms over {entry['count']}", flush=True)
            print(f"TIMING {name} sum of stages {summary['sum_of_stage_ms']:.1f} ms; launch {gpu_ms:.1f} ms",
                  flush=True)
        return compiled, row

    def megakernel_stage(self, batch, fastvideo_args):
        video0, audio0 = batch.latents.detach().clone(), batch.audio_latents.detach().clone()
        layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
        if args.profile_range == "denoising":
            torch.cuda.profiler.start()
        start = time.perf_counter()
        torch.cuda.nvtx.range_push("fastvideo_denoising_reference")
        result = stage_forward(self, batch, fastvideo_args)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        report["fastvideo_denoising_wall_s"] = time.perf_counter() - start
        reference = (result.latents.detach().clone(), result.audio_latents.detach().clone())
        steps = len(trace["velocity"])
        transformer = self.transformer
        device = video0.device

        # Rebuild the request description exactly as the stage did.
        row_plan = []
        for vt, at in zip(self.scheduler.timesteps, self.audio_scheduler.timesteps, strict=True):
            unique, inverse = build_row_timesteps(
                layout, video_timestep=float(vt.item()), audio_timestep=float(at.item()),
                condition_video_timestep=max(float(vt.item()), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                condition_audio_timestep=1.0)
            row_plan.append((unique, inverse))
        if batch.extra.get("vsa_mode", "exempt") != "exempt" or int(batch.extra.get(
                "vsa_dense_first_n_steps", 0)) or stage_module.MiniMaxH3TokenReductionSchedule.from_options(
                    getattr(fastvideo_args, "h3_token_reduction", {})) is not None:
            raise ValueError("request uses a VSA or token-reduction policy outside this recipe")
        builder = stage_module._h3_vsa_metadata_builder(transformer, fastvideo_args)
        patch = fastvideo_args.pipeline_config.dit_config.patch_size
        metadata = builder.build(
            current_timestep=0, patch_size=patch, VSA_sparsity=float(batch.VSA_sparsity),
            packed_segments=stage_module._h3_vsa_single_region_segments(layout, patch),
            device=device, exempt=True, dense_layers=tuple(batch.extra.get("vsa_dense_layers", ())),
            tile_size=int(fastvideo_args.VSA_tile_size), ref_keep_rate=None)
        recipe = hd.recipe_from_fastvideo(transformer, layout, metadata, self.scheduler,
                                          self.audio_scheduler, row_plan)
        prompt_embeds = batch.prompt_embeds[0][0]
        report["request"] = {"sequence": recipe.sequence, "text": int(recipe.text_index.numel()),
                             "video": int(recipe.video_index.numel()),
                             "audio": int(recipe.audio_index.numel()), "steps": steps,
                             "blocks": len(recipe.blocks), "tiles": int(recipe.valid.numel()),
                             "keep": recipe.steps[0].keep}
        H = recipe.hidden
        final = None
        for gate in gates:
            gc.collect()
            torch.cuda.empty_cache()
            (out / "report.json").write_text(json.dumps(report, indent=2))
            if gate == "block":
                compiled, row = run_gate("block", recipe, prompt_embeds, video0, audio0,
                                         steps=1, blocks=1)
                b = compiled.buffers
                T = recipe.text_index.numel()
                row["compare"]["refined_text"] = compare("refined text", b["text_refined"][:T],
                                                         trace["text"][0])
                row["compare"]["adaln_block0"] = compare(
                    "block 0 AdaLN table", b["adaln_tables"][0], trace["adaln"][0])
                row["compare"]["block0_output"] = compare("step 0 block 0 output", compiled.hidden,
                                                          trace["block"][0][0])
                # Localize: free-running stages against the FastVideo trace, and
                # FastVideo's own layers re-run on the megakernel's intermediates.
                from fastvideo.models.dits.minimax_h3_vsa_fp4 import _shared_input_projections
                c, S = row["compare"], recipe.sequence
                b0 = transformer.transformer_blocks[0]
                for label, rows_ in (("text", recipe.text_index), ("audio", recipe.audio_index),
                                     ("video", recipe.video_index)):
                    rows_ = rows_.to(device)
                    c[f"block0_output_{label}_rows"] = compare(
                        f"block 0 output, {label} rows", compiled.hidden[rows_],
                        trace["block"][0][0][rows_])
                c["normed1"] = compare("norm1 output", b["normed1"], trace["ref_normed1"][0])
                c["normed2"] = compare("norm2 output", b["normed2"], trace["ref_normed2"][0])
                untiled = b["attention"].view(-1, recipe.heads * hd.HEAD_DIM)[
                    recipe.tile_index.to(device)].contiguous()
                c["attention"] = compare("attention before to_out", untiled, trace["ref_attention"][0])
                c["attn_out"] = compare("attention branch", b["branch_attn"], trace["ref_attn_out"][0])
                c["ff_out"] = compare("feed-forward branch", b["branch_mlp"], trace["ref_ff_out"][0])
                # The linears are row independent, so a text/audio/video row
                # sample re-run through FastVideo's layers checks them exactly.
                sample = torch.cat([torch.arange(0, 1024, device=device),
                                    torch.arange(S - 1024, S, device=device)])
                with torch.no_grad():
                    a0 = b0.attn
                    forced = _shared_input_projections(
                        (a0.to_q, a0.to_k, a0.to_v, a0.to_gate_compress),
                        b["normed1"][sample][None].contiguous())
                    for label, tensor in zip(("q", "k", "v", "gate"), forced, strict=True):
                        c[f"forced_to_{label}"] = compare(f"FastVideo to_{label} on megakernel input",
                                                          b[f"proj_{label}"][sample], tensor[0])
                    del forced
                    c["forced_to_out"] = compare(
                        "FastVideo to_out on megakernel input", b["branch_attn"][sample],
                        a0.to_out(untiled[sample][None].contiguous())[0][0])
                    del untiled
                    ff_in = b["normed2"][sample][None].contiguous()
                    c["forced_fc_in"] = compare("FastVideo fc_in on megakernel input",
                                                b["ffn_mid"][sample], b0.ff.fc_in(ff_in)[0][0])
                    c["forced_ff"] = compare("FastVideo feed-forward on megakernel input",
                                             b["branch_mlp"][sample], b0.ff(ff_in)[0])
                    del ff_in
                timings = compiled.program.profile()
                row["profile_ms"] = {label: ms for label, ms in timings}
                for label, ms in sorted(timings, key=lambda item: -item[1])[:24]:
                    print(f"PROFILE {label} {ms:.2f} ms", flush=True)
            elif gate == "step":
                compiled, row = run_gate("step", recipe, prompt_embeds, video0, audio0, steps=1)
                row["compare"]["block49_output"] = compare(
                    "step 0 last block output", compiled.hidden,
                    trace["block"][len(recipe.blocks) - 1][0])
                b = compiled.buffers
                row["compare"]["video_velocity"] = compare("step 0 video velocity",
                                                           b["velocity_video"], trace["velocity"][0][0])
                row["compare"]["audio_velocity"] = compare("step 0 audio velocity",
                                                           b["velocity_audio"], trace["velocity"][0][1])
                row["compare"]["video_latents"] = compare("latents after step 0",
                                                          compiled.video_latents, trace["latents_in"][1][0])
                row["compare"]["audio_latents"] = compare("audio latents after step 0",
                                                          compiled.audio_latents, trace["latents_in"][1][1])
            elif gate == "full":
                compiled, row = run_gate("full", recipe, prompt_embeds, video0, audio0)
                row["compare"]["final_video_latents"] = compare("final video latents",
                                                                compiled.video_latents, reference[0])
                row["compare"]["final_audio_latents"] = compare("final audio latents",
                                                                compiled.audio_latents, reference[1])
                row["fastvideo_denoising_wall_s"] = report["fastvideo_denoising_wall_s"]
                final = (compiled.video_latents.clone(), compiled.audio_latents.clone())
                torch.save({"video": final[0].cpu(), "audio": final[1].cpu(),
                            "reference_video": reference[0].cpu(), "reference_audio": reference[1].cpu()},
                           out / "final_latents.pt")
            else:
                raise ValueError(f"unknown gate {gate}")
            del compiled
            if args.profile_range == "denoising":
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
            for key in [k for k in trace if k.startswith("ref_")] + ["block_input"]:
                trace.pop(key, None)
        frozen = Path(args.frozen)
        if frozen_case and (frozen / "reference_manifest.json").exists():
            for name, tensor in (("final_video_latents", reference[0]), ("final_audio_latents", reference[1]),
                                 ("initial_video_latents", video0)):
                stored = torch.load(frozen / "tensors" / f"{name}.pt")
                report.setdefault("frozen_reference", {})[f"fastvideo_rerun_{name}"] = compare(
                    f"FastVideo rerun vs frozen {name}", tensor, stored)
            if final is not None:
                report["frozen_reference"]["megakernel_final_video_latents"] = compare(
                    "megakernel vs frozen final video latents", final[0],
                    torch.load(frozen / "tensors" / "final_video_latents.pt"))
        if args.pipeline == "megakernel":
            if final is None:
                raise ValueError("--pipeline megakernel needs the full gate")
            result.latents = final[0].to(result.latents.dtype)
            result.audio_latents = final[1].to(result.audio_latents.dtype)
        (out / "report.json").write_text(json.dumps(report, indent=2))
        return result

    def traced_stage(self, batch, fastvideo_args):
        result = stage_call(self, batch, fastvideo_args)
        name = type(self).__name__
        if name == "MiniMaxH3VideoDecodingStage" and isinstance(result.output, torch.Tensor):
            decoded["video"] = result.output.detach().float().cpu()
        elif name == "MiniMaxH3AudioDecodingStage" and isinstance(result.extra.get("audio"), torch.Tensor):
            decoded["audio"] = result.extra["audio"].detach().float().cpu()
        return result

    dit_module.MiniMaxH3TransformerBlock.forward = traced_block
    dit_module.MiniMaxH3Transformer3DModel.forward = traced_transformer
    dit_module.MiniMaxH3Transformer3DModel._refined_text = traced_text
    stage_module.MiniMaxH3DenoisingStage.forward = megakernel_stage
    PipelineStage.__call__ = traced_stage
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
        result = generator.generate({
            "prompt": prompt,
            "sampling": {"height": 768, "width": 1344, "num_frames": 124,
                         "num_inference_steps": 9, "guidance_scale": 1.0, "seed": args.seed, "fps": 24},
            "output": {"output_path": str(out), "output_video_name": "hazy-dit", "return_frames": False},
        })
        report["video_path"] = getattr(result, "video_path", None)
        frozen = Path(args.frozen)
        for name, key in (("video", "decoded_video"), ("audio", "decoded_audio")):
            path = frozen / "tensors" / f"{key}.pt"
            if frozen_case and name in decoded and path.exists():
                reference = torch.load(path).float()
                row = compare(f"decoded {name} vs frozen FastVideo", decoded[name], reference)
                mse = float((decoded[name].reshape(-1) - reference.reshape(-1)).square().mean())
                peak = float(reference.abs().max())
                row["psnr_db"] = float("inf") if mse == 0 else 10 * torch.log10(
                    torch.tensor(peak * peak / mse)).item()
                report.setdefault("decoded", {})[name] = row
                print(f"PSNR decoded {name}: {row['psnr_db']:.2f} dB", flush=True)
    finally:
        dit_module.MiniMaxH3TransformerBlock.forward = block_forward
        dit_module.MiniMaxH3Transformer3DModel.forward = transformer_forward
        dit_module.MiniMaxH3Transformer3DModel._refined_text = refined_text
        stage_module.MiniMaxH3DenoisingStage.forward = stage_forward
        PipelineStage.__call__ = stage_call
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str), flush=True)
    print("DONE: Hazy DiT gates " + ",".join(gates), flush=True)


if __name__ == "__main__":
    main()
