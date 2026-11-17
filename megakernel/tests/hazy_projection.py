"""Full-shape Hazy all-warp NVFP4 projection gate on RTX PRO 6000."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compiler import ProjectionTaskParams, compile_projection

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from flashinfer import SfLayout

from mk_h3_projection import prepare, params_words, resources, run
from projection_parity import CHECKPOINT, REVISION, captured_tensors, compare
from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize


def check_scheduler(m, n, ctas):
    tiles = ((m + 127) // 128) * ((n + 127) // 128)
    # CUTLASS SM120's persistent scheduler starts at the linear CTA ID and
    # advances by grid size. This includes the final 11-row M tile.
    seen = [0] * tiles
    for cta in range(ctas):
        for tile in range(cta, tiles, ctas):
            seen[tile] += 1
    assert all(count == 1 for count in seen)
    return tiles


def main():
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("RTX PRO 6000 SM120 required")
    source, captured = captured_tensors()
    path = snapshot_download(CHECKPOINT, revision=REVISION, local_files_only=True)
    with safe_open(f"{path}/transformer/nvfp4_weights.safetensors", framework="pt", device="cpu") as reader:
        prefix = "transformer_blocks.0.attn.to_q"
        weight = reader.get_tensor(f"{prefix}::_nvfp4_weight").cuda()
        weight_sf = reader.get_tensor(f"{prefix}::_nvfp4_weight_scale").cuda()
        alpha_raw = reader.get_tensor(f"{prefix}::_nvfp4_alpha").cuda()
        sf_key = f"{prefix}::_nvfp4_input_global_sf"
        input_sf = (reader.get_tensor(sf_key).cuda() if sf_key in reader.keys()
                    else torch.tensor(1.0, device="cuda", dtype=torch.float32))
    alpha = float((alpha_raw.float() / input_sf.float()).reshape(()).item())
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    assert sm_count == 188, f"measured SM count changed: {sm_count}"
    slots = 1024

    for m in (128, 37771):
        n, k = 7168, 5376
        tile_count = check_scheduler(m, n, sm_count)
        x = source.reshape(-1, k)[:m].contiguous().cuda()
        packed, packed_sf = _nvfp4_quantize(x, input_sf,
                                             sfLayout=SfLayout.layout_128x4,
                                             do_shuffle=False)
        expected = captured.reshape(-1, n)[:m].contiguous().cuda()
        metadata = dict(resources(m, n, k))
        assert metadata["threads_per_cta"] == 384
        assert metadata["shared_bytes"] <= 88480
        assert metadata["tile_count"] == tile_count
        print("Hazy projection resources", json.dumps(metadata), "CTAs", sm_count,
              flush=True)
        params = torch.empty(params_words(), device="cuda", dtype=torch.int32)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        prepare(params.data_ptr(), packed.data_ptr(), weight.data_ptr(),
                packed_sf.data_ptr(), weight_sf.data_ptr(), out.data_ptr(),
                m, n, k, alpha, sm_count)
        # First measure physical IDs using the same Hazy binding. The compiler
        # then emits only to addressable, resident SM slots.
        probe = torch.zeros((slots, 1, 32), device="cuda", dtype=torch.int32)
        probe[:, 0, 0] = 1
        probe_times = torch.zeros((slots, 1, 128), device="cuda", dtype=torch.int32)
        counts = torch.zeros(sm_count, device="cuda", dtype=torch.int32)
        ids = torch.full((sm_count,), -1, device="cuda", dtype=torch.int32)
        rendezvous = torch.zeros(1, device="cuda", dtype=torch.int32)
        dummy_float = torch.zeros(1, device="cuda", dtype=torch.float32)
        dummy_int = torch.zeros(1, device="cuda", dtype=torch.int32)
        run(probe, probe_times, counts, ids, rendezvous, dummy_float,
            dummy_int, dummy_int, params, sm_count,
            stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        measured_ids = ids.cpu().tolist()
        assert len(set(measured_ids)) == sm_count and min(measured_ids) >= 0
        assert max(measured_ids) < slots
        record = ProjectionTaskParams(m, n, k, sm_count)
        instructions = torch.tensor(compile_projection(record, measured_ids, slots),
                                    device="cuda", dtype=torch.int32)
        timings = torch.zeros((slots, 3, 128), device="cuda", dtype=torch.int32)
        for rep in range(3):
            out.fill_(float("nan"))
            counts.zero_()
            ids.fill_(-1)
            run(instructions, timings, counts, ids, rendezvous, dummy_float,
                dummy_int, dummy_int, params, sm_count,
                stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            assert torch.equal(counts, torch.full_like(counts, 2))
            measured_ids = ids.cpu().tolist()
            assert len(set(measured_ids)) == sm_count and min(measured_ids) >= 0
            assert max(measured_ids) < slots
            compare(f"Hazy vs captured to_q, M={m}, rep={rep}", out, expected)
        del x, packed, packed_sf, expected, params, out
    print("PASS: exact Hazy all-warp to_q parity, 128 and 37771 rows, 3 repeats",
          flush=True)


if __name__ == "__main__":
    main()
