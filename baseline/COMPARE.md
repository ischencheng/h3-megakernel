# Three-way H3 quality comparison

From the project root, with Modal installed and authenticated:

```bash
export H3MK_FV_SRC=/Users/ryanmorris/projects/FastVideo
modal run baseline/compare_modal.py --run-id quality-01
modal volume get h3-megakernel-cache runs/quality-01 ./quality-01
```

Keep the command running until all three jobs finish. It downloads weights on CPU,
then runs one RTX PRO 6000 at a time. FastVideo must be at `ce970eb` with its kernel
submodules initialized, as described in `docs/profiling.md`.

| Output folder | Recipe |
|---|---|
| `fast-fp4` | Existing FastH3 Consumer checkpoint, FP4 sparse attention, eight forwards |
| `fast-no-fp4` | Same configuration with FP4 attention disabled; native VSA fallback |
| `convrot` | Full unpruned FL2VA INT8 ConvRot, no LoRA, 20 steps, `res_multistep` / `simple` |

Edit `baseline/prompts.json` before launching. Defaults are three prompts, seeds
1101 and 1102, 1344×768, 124 frames at 24 fps: six clips per variant. Every variant
keeps its models loaded between requests. The first clip is marked `cold`; there
is no extra paid warmup. Run a subset or change seeds/resolution/base-model steps:

```bash
modal run baseline/compare_modal.py --run-id quick-01 --seeds 1101 --shape 832x480
modal run baseline/compare_modal.py --run-id convrot-50 --variants convrot --steps 50
```

Each folder contains videos, `driver.log`, `summary.json`, installed package
versions and subprocess wall time. ConvRot also saves each submitted API graph.
Use Modal's billing for total cost: process time excludes container startup and
CPU downloads, and GPU price excludes CPU/RAM/storage charges. Runs stop on failure;
use a new run ID and `--variants` to retry only the failed recipe.

The ConvRot graph follows Comfy-Org's `video_minimax_h3_t2v.json`, with the full
transformer substituted for its pruned default and Turbo omitted. It uses the
template's NVFP4 AWQ text encoder, INT8 ConvRot video VAE, and FP32 audio VAE.
ComfyUI manages offloading automatically. This compares complete generation
recipes; the ConvRot result does not isolate transformer precision and is not
proof that every component can remain resident for a megakernel. Equal seeds
across runtimes do not guarantee identical initial noise.

Model and ComfyUI revisions are pinned in `convrot.py`. ComfyUI source/template:
https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json

Local syntax, graph wiring, and model filenames have been checked. CUDA execution,
VRAM fit, and the resulting video/audio quality require the first GPU run.
