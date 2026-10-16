"""Hybrid world-model scoring for plans (shared by the offline checks and the closed-loop filter).

HYB_CF = deterministic closed-loop/counterfactual model (cf_v1/model/model_CF.pt) + residual flow-matching head
(model_HYB_CF.pt). Futures are sampled from the TRUE start latent: the head supplies the variability that the
deterministic monitor had to inject by hand (perception-error starts). For each plan: 64 futures with the 64 recorded
execution-noise snippets (x1) over the H15 horizon -> endpoints at 1 s and 3 s ->
  fork = the BOTH split test (1 s and 3 s),
  q    = imagined topple fraction: share of futures in the fork group with the larger change away from the hand (0 if
         no fork) -- ranks real risk 0.56 (Spearman) vs 0.34 for the deterministic model on 1,575 dev plans.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
MODEL_DIR = ROOT / "results/program_b/cf_v1/model"


class HybMonitor:
    def __init__(self, model_dir=MODEL_DIR, head_name="model_HYB_CF.pt", det_name="model_CF.pt"):
        import torch
        import phase2_hybrid as hy
        import step3_world_model as wm
        self.torch, self.hy, self.wm = torch, hy, wm
        s = torch.load(Path(model_dir) / head_name, map_location="cuda")
        self.head = hy.build_head(s["action_dim"]).cuda(); self.head.load_state_dict(s["state_dict"]); self.head.eval()
        self.rscale = s["residual_scale"]
        self.det = wm.WorldModel.load(Path(model_dir) / det_name); self.det.net.eval()
        self.snip = np.load(ROOT / "results/jenga/bench/jenga_bench.npz")["snippets"]

    def actions(self, plan, n):
        import lookahead_check as la
        return np.stack([la.horizon_actions(plan, "H15", la.noise_for(self.snip, j)) for j in range(n)])

    def score(self, z0, plans, site, seed, n=64, chunk=4):
        """z0 (28,28,8) torch cuda; plans (C,16,4) numpy -> fork (C,) bool, q (C,), obar (C,) mean change away from hand."""
        import cbf_barrier as cb
        import cem_filter as cf
        torch, hy, wm = self.torch, self.hy, self.wm
        C = len(plans); fork = np.zeros(C, bool); q = np.zeros(C); obar = np.zeros(C)
        for c0 in range(0, C, chunk):
            idx = list(range(c0, min(C, c0 + chunk)))
            acts = np.concatenate([self.actions(plans[c], n) for c in idx])
            a = self.det._act(torch.as_tensor(acts, device="cuda").reshape(len(idx) * n, wm.STEPS, -1))
            g = torch.Generator(device="cuda").manual_seed(int(seed) + c0)
            e = hy.sample(self.head, self.det, z0[None].expand(len(idx) * n, -1, -1, -1).contiguous(), a, self.rscale, g)
            e = e.reshape(len(idx), n, 2, 28, 28, 8)
            ok10, _, _ = cf.split_test_gpu(e[:, :, 0].reshape(len(idx), n, -1))
            ok30, lab, _ = cf.split_test_gpu(e[:, :, 1].reshape(len(idx), n, -1))
            for k, c in enumerate(idx):
                pts = np.vstack([np.asarray(site)[None], np.asarray(plans[c])[1:16, :3]])
                m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
                o_j = (((e[k, :, 1] - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum())
                obar[c] = float(o_j.mean())
                f = bool(ok10[k] & ok30[k]); fork[c] = f
                L = lab[k].bool()
                if f and not (L.all() or (~L).all()):
                    bad = L if o_j[L].mean() > o_j[~L].mean() else ~L
                    q[c] = float(bad.float().mean())
            torch.cuda.empty_cache()
        return fork, q, obar
