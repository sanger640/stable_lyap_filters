"""Task-agnostic sensitivity monitor on the full-token DINO-WM (docs/FILTER_DEEP_DIVE.md steps 3-4).

For a state (current close-camera frame -> DINOv2 tokens, proprio) and candidate plans (C, 16, 4):
imagine each plan's horizon (cbf_filter.horizon: plan steps 1..15, then hold to 38 actions = 19 model steps) under n
execution-noise draws (the noise snippets on the 15 look-ahead actions, scales 0.5 / 1 / 2) - ACTIONS ONLY, no object
or pose sampler. Readouts per plan:
  h     fork scale as cbf.score: log2 of the smallest scale whose futures split (split test at model steps 8 and 18),
        log2 3 if none;
  g     sensitivity gain: RMS spread of the imagined end tokens (step 18, outside the hand) per unit noise RMS, scale 1;
  o     disturbance: max over futures of the mean squared outside-hand token change from the start (step 18, scale 1).
All differentiable in the plan (grad=True) except h (discrete).
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
SCALES = (0.5, 1.0, 2.0)
GRID, IMG = 14, 196


class TokMon:
    def __init__(self, model_path, snippets=None):
        import torch
        import tokwm
        import tok_data as td
        import uncertainty_monitor_dev as um
        self.torch, self.m = torch, tokwm.TokWM.load(model_path)
        self.tok = td.Tokens(); self.snippets = snippets if snippets is not None else np.load(um.BENCH)["snippets"]

    def start(self, frame384, prop):
        q, s = self.tok(frame384[None]); torch = self.torch
        vis = torch.as_tensor(q[0].astype(np.float32) * s[0].astype(np.float32)[:, None], device="cuda")
        return vis, torch.as_tensor(np.asarray(prop, np.float32), device="cuda")

    def hand_keep(self, site, plan):
        """(196,) bool: patches outside the robot's own region (cbf_barrier.hand_mask, projected onto the 14x14 grid)."""
        import cbf_barrier as cb
        m28 = cb.hand_mask(np.vstack([site[None], np.asarray(plan)[1:16, :3]]))           # (28, 28) on the 384 px camera
        m14 = m28.reshape(14, 2, 14, 2).any((1, 3))
        return ~m14.reshape(-1)

    def actions(self, plans, n, s, offset=0):
        """plans torch (C, 16, 4) -> (C * n, 19, 8) with noise draw offset + j at scale s on the 15 look-ahead actions."""
        torch = self.torch
        a = plans[:, 1:16]; a = torch.cat([a, a[:, -1:].expand(-1, 23, -1)], 1)                 # (C, 38, 4)
        nz = np.stack([np.concatenate([self.snippets[(offset + j) % 64], self.snippets[(offset + j + 1) % 64]])[:15]
                       for j in range(n)]) * s
        nz = torch.as_tensor(nz, device="cuda", dtype=a.dtype)                                    # (n, 15, 3)
        A = a[:, None].expand(-1, n, -1, -1).clone()
        A[:, :, :15, :3] = A[:, :, :15, :3] - nz[None]
        return A.reshape(len(plans) * n, 19, 8), float(nz.pow(2).mean().sqrt())

    def score(self, vis0, prop0, plans, keep, n=32, grad=False, scales=SCALES, offset=0):
        """-> dict h (C,), g (C,), o (C,) (g, o tensors if grad)."""
        import cem_filter as cf
        torch = self.torch; plans = torch.as_tensor(plans, device="cuda", dtype=torch.float32) if not torch.is_tensor(plans) else plans
        C = len(plans); split = np.zeros((C, len(scales)), bool); g = o = None
        kp = torch.as_tensor(keep, device="cuda")
        for si, s in enumerate(scales):
            A, nrms = self.actions(plans, n, s, offset)
            B = len(A)
            ends = self.m.rollout(vis0[None].expand(B, -1, -1), prop0[None].expand(B, -1), A, keep=(8, 18), grad=grad and s == 1.0)
            e8, e18 = ends[8].reshape(C, n, -1), ends[18].reshape(C, n, 196, 384)
            with torch.no_grad():
                ok8, _, _ = cf.split_test_gpu(e8.detach().float()); ok18, _, _ = cf.split_test_gpu(e18.detach().float().reshape(C, n, -1))
            split[:, si] = (ok8 & ok18).cpu().numpy()
            if s == 1.0:
                out = e18[:, :, kp]                                                              # (C, n, K, 384)
                g = (out - out.mean(1, keepdim=True)).pow(2).mean((1, 2, 3)).sqrt() / nrms
                o = (out - vis0[kp][None, None]).pow(2).mean((2, 3)).max(1).values
        h = np.array([np.log2(scales[np.flatnonzero(r)[0]]) if r.any() else np.log2(3.0) for r in split])
        if not grad:
            g, o = g.detach().cpu().numpy(), o.detach().cpu().numpy()
        return {"h": h, "g": g, "o": o}
