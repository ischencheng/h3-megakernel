# swaps the video decoder's blocks for stages/video_decoder; fastvideo compiles the decoder forward,
# so the fused arm runs a copy of it with the fused blocks, eagerly, for each _decode_clip call
import os
import time

import torch
from fastvideo.models.vaes import minimax_h3_video as vae

from harness import check
from stages.video_decoder import fused_block

ARMS = ("fastvideo", "fused")
_DECODE_CLIP = vae.AutoencoderKLMiniMaxH3._decode_clip


# MiniMaxH3VideoViTDecoder3d.forward with the fused blocks
def decoder_forward(dec, packed, hidden_states, checker=None):
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch_size, num_frames * height * width, num_channels)
    hidden_states = dec.proj_in(hidden_states)
    num_patches = hidden_states.shape[1]
    register_tokens = dec.register_tokens.expand(batch_size, -1, -1)
    cls_token = torch.zeros_like(hidden_states[:, :1, :])
    hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)  # promotes to fp32
    grids = [
        2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
        for size in (num_frames, height, width)
    ]
    position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
    suffix_ids = position_ids.new_zeros((batch_size, dec.num_register_tokens + 1, 3))
    rotary_emb = dec.rope(torch.cat([position_ids, suffix_ids], dim=1))
    for index, block in enumerate(dec.transformer_blocks):
        checked = checker is not None and index in checker.check_blocks
        block_in = hidden_states.clone() if checked else None
        # in place is safe, the concatenation above made a new fp32 tensor
        hidden_states = fused_block.forward(block, packed[index], hidden_states, rotary_emb, inplace=True)
        if checked:
            checker.check(index, block, block_in, rotary_emb, hidden_states)
    hidden_states = dec.proj_out(dec.norm_out(hidden_states))[:, :num_patches, :]
    patch_size, patch_size_t = dec.patch_size, dec.patch_size_t
    hidden_states = hidden_states.view(batch_size, num_frames, height, width, dec.out_channels, patch_size_t,
                                       patch_size, patch_size)
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return hidden_states.reshape(batch_size, dec.out_channels, num_frames * patch_size_t, height * patch_size,
                                 width * patch_size)


class VaeSwap:

    def __init__(self, check_blocks=(0, 1, 13, 25)):
        # H3MK_VAE_CHECK=0 skips the checks, for memory runs
        self.check_blocks = tuple(check_blocks) if os.environ.get("H3MK_VAE_CHECK", "1") != "0" else ()
        self.packed = None
        self.checked = False  # only the first fused request runs the checks
        self.set_arm("fastvideo")
        swap = self
        vae.AutoencoderKLMiniMaxH3._decode_clip = lambda module, z: swap.decode_clip(module, z)

    def set_arm(self, arm):
        assert arm in ARMS, arm
        self.arm, self.checks, self.check_s, self.calls, self.clip_events = arm, [], 0.0, 0, []

    def decode_clip(self, module, z):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        dec = module.decoder
        if self.packed is None:  # during the warmup, so neither the build nor the packing is timed
            self.packed = [fused_block.Packed(b) for b in dec.transformer_blocks]
            fused_block.ext()
        start.record()
        if self.arm == "fastvideo":
            out = _DECODE_CLIP(module, z)
        else:
            # the first clip is one latent frame, so check on the second
            checker = self if self.calls == 1 and not self.checked and self.check_blocks else None
            compiled = dec.__dict__.get("forward")
            dec.forward = lambda x: decoder_forward(dec, self.packed, x, checker)
            try:
                out = _DECODE_CLIP(module, z)
            finally:
                if compiled is not None:
                    dec.forward = compiled
                else:
                    del dec.forward
            self.checked = self.checked or checker is not None
        end.record()
        self.clip_events.append((start, end))
        self.calls += 1
        return out

    # ours and fastvideo's eager block on the same input, both against fp32
    @torch.no_grad()
    def check(self, index, block, block_in, rotary_emb, ours):
        torch.cuda.synchronize()
        t0 = time.time()
        fastvideo = block(block_in, rotary_emb).float()
        ref = fused_block.reference_fp32(block, block_in, rotary_emb)
        x = block_in.float()
        row = {"block": index, "rows": x.shape[0] * x.shape[1], "input_dtype": str(block_in.dtype),
               "branch_vs_fp32": check.error_stats(ours.float() - x, ref - x),
               "fastvideo_branch_vs_fp32": check.error_stats(fastvideo - x, ref - x),
               "branch_vs_fastvideo": check.error_stats(ours.float() - x, fastvideo - x),
               "out_vs_fp32": check.error_stats(ours.float(), ref),
               "fastvideo_out_vs_fp32": check.error_stats(fastvideo, ref)}
        row["passes"] = check.passes(row["branch_vs_fp32"], row["fastvideo_branch_vs_fp32"])
        self.checks.append(row)
        print("vae check", row, flush=True)
        del fastvideo, ref
        torch.cuda.synchronize()
        self.check_s += time.time() - t0

    def report(self):
        torch.cuda.synchronize()
        clips = [s.elapsed_time(e) for s, e in self.clip_events]
        return {"arm": self.arm, "decode_clip_calls": len(clips), "decode_clip_ms_total": sum(clips),
                "check_s": self.check_s, "checks": self.checks}
