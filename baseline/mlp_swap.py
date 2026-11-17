# runs the mlp half of each dit block on flashinfer's fused sm120 mlp, as three launches (flashinfer_mlp)
# or as one persistent launch (mlp_one_launch, stages/dit/fi_mlp.py)
import time

import torch
from fastvideo.layers.quantization.nvfp4_config import H3_NVFP4_DIT_INPUT_SF_NAME
from fastvideo.models.dits import minimax_h3

from harness import check, reference, weights
from stages.dit import fi_mlp

ARMS = ("fastvideo", "flashinfer_mlp", "mlp_one_launch")
_FUSED = minimax_h3.fused_residual_gate_rmsnorm_modulate
_GATED = minimax_h3._gated_residual
_FF_FORWARD = minimax_h3.MiniMaxH3FeedForward.forward


class _Pending:  # the feed-forward output the fused block will produce
    pass


def _scalar(t):
    return t.reshape(1).float().contiguous() if torch.is_tensor(t) else torch.tensor([float(t)], device="cuda")


class MlpSwap:

    def __init__(self, check_calls=3):
        self.check_calls = check_calls
        self.packed = {}
        self.stash = None
        self.compiled = False
        self.set_arm("fastvideo")
        swap = self
        minimax_h3.fused_residual_gate_rmsnorm_modulate = lambda *a: swap.fused(*a)
        minimax_h3._gated_residual = lambda *a: swap.gated(*a)
        minimax_h3.MiniMaxH3FeedForward.forward = lambda module, x: swap.ff(module, x)

    def set_arm(self, arm):
        assert arm in ARMS, arm
        self.arm, self.events, self.checks, self.check_s, self.calls = arm, [], [], 0.0, 0

    # the checkpoint's nvfp4 bytes in the layout of minimax_h3_mlp_nvfp4_sm120, packed once per block
    def pack(self, module):
        key = id(module)
        if key in self.packed:
            return self.packed[key]
        fc_in, fc_out = module.fc_in, module.fc_out
        if not all(hasattr(l, "_nvfp4_weight") and hasattr(l, H3_NVFP4_DIT_INPUT_SF_NAME) for l in (fc_in, fc_out)):
            self.packed[key] = None
            return None
        lin = lambda l: weights.Linear(l._nvfp4_weight, l._nvfp4_weight_scale, l._nvfp4_alpha,
                                       getattr(l, H3_NVFP4_DIT_INPUT_SF_NAME))
        self.packed[key] = fi_mlp.pack(lin(fc_in), lin(fc_out))
        return self.packed[key]

    def run_block(self, s, gate_table, one_launch):
        x = s["hidden"].reshape(-1, s["hidden"].shape[-1])
        fn = fi_mlp.run_one_launch if one_launch else fi_mlp.run
        out = fn(x, s["norm_w"], s["scale"], s["shift"], s["idx"], gate_table, x, s["packed"], s["eps"])
        return out.view(s["hidden"].shape)

    def fused(self, residual, branch, gate, norm_w, scale, shift, idx, eps):
        start, mid = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        hidden, norm = _FUSED(residual, branch, gate, norm_w, scale, shift, idx, eps)
        mid.record()
        self.stash = {"start": start, "mid": mid, "hidden": hidden, "norm": norm, "norm_w": norm_w, "scale": scale,
                      "shift": shift, "idx": idx, "eps": eps, "packed": None}
        return hidden, norm

    def ff(self, module, x):
        s = self.stash
        if s is not None and s["norm"] is x:
            s["packed"] = self.pack(module)
            s["module"] = module
            if self.arm != "fastvideo" and s["packed"] is not None:
                return _Pending()
        return _FF_FORWARD(module, x)

    def gated(self, hidden, gate_table, idx, branch):
        s, self.stash = self.stash, None
        if s is None or s["hidden"] is not hidden:
            return _GATED(hidden, gate_table, idx, branch)
        if not self.compiled and s["packed"] is not None:  # jit-compile both builds during the warmup
            self.run_block(s, gate_table, one_launch=False)
            self.run_block(s, gate_table, one_launch=True)
            self.compiled = True
        if isinstance(branch, _Pending):
            out = self.run_block(s, gate_table, one_launch=self.arm == "mlp_one_launch")
        else:
            out = _GATED(hidden, gate_table, idx, branch)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((s["start"], s["mid"], end))
        if isinstance(branch, _Pending) and len(self.checks) < self.check_calls:
            self.check(s, gate_table, out)
        self.calls += 1
        return out

    # the output against FastVideo's, the three-launch block and an fp32 reference, on the same live inputs
    @torch.no_grad()
    def check(self, s, gate_table, out):
        torch.cuda.synchronize()
        t0 = time.time()
        module, x = s["module"], s["hidden"].reshape(-1, s["hidden"].shape[-1])
        fastvideo = _GATED(s["hidden"], gate_table, s["idx"], _FF_FORWARD(module, s["norm"])).reshape(x.shape)
        three = self.run_block(s, gate_table, one_launch=False).reshape(x.shape)
        xf = x.float()
        n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + s["eps"]) * s["norm_w"].float()
        a = s["shift"].index_select(0, s["idx"]).float() + n * (1 + s["scale"].index_select(0, s["idx"]).float())
        lin = lambda l: weights.Linear(l._nvfp4_weight.view(torch.uint8), l._nvfp4_weight_scale.view(torch.uint8),
                                       _scalar(l._nvfp4_alpha), None)
        o = reference.mlp_fp32(a, weights.Weights(lin(module.fc_in), lin(module.fc_out)))
        branch = gate_table.index_select(0, s["idx"]).float() * o
        ours, fv = out.reshape(x.shape), fastvideo
        row = {"call": self.calls, "tokens": x.shape[0], "table_rows": gate_table.shape[0],
               "branch_vs_fp32": check.error_stats(ours.float() - xf, branch),
               "fastvideo_branch_vs_fp32": check.error_stats(fv.float() - xf, branch),
               "branch_vs_fastvideo": check.error_stats(ours.float() - xf, fv.float() - xf),
               "equals_three_launches": torch.equal(ours, three)}
        row["passes"] = check.passes(row["branch_vs_fp32"], row["fastvideo_branch_vs_fp32"])
        self.checks.append(row)
        del a, n, o, branch, fastvideo, three
        torch.cuda.synchronize()
        self.check_s += time.time() - t0

    def report(self):
        torch.cuda.synchronize()
        section = [s.elapsed_time(e) for s, _, e in self.events]
        residual = [s.elapsed_time(m) for s, m, _ in self.events]
        return {"arm": self.arm, "mlp_calls": len(section), "mlp_section_ms_total": sum(section),
                "residual_norm_ms_total": sum(residual), "mlp_only_ms_total": sum(section) - sum(residual),
                "check_s": self.check_s, "checks": self.checks}
