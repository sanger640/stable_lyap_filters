"""Fork-boundary CBF safety filter, purely in the world model's latent space (overnight plan 2026-10-15, step 3).

Safe set: the robot's current regime (the scene now, assumed safe). Boundary: the fork, where imagined futures split.
For a plan p (the policy's remaining 15 steps, then hold) the world model imagines 64 futures under the robot's
perception and execution uncertainty scaled by s in SCALES:
  h(p)    barrier = log2 of the smallest uncertainty scale at which the futures split (capped when they never do)
  o(p)    regime change = largest latent change, over the futures, outside the robot's own hand path
Discrete-time CBF condition, relative to doing nothing (h_now = barrier of holding still):
  accept p  iff  h(p) >= min(h_now, H_SAFE) - GAMMA  and  o(p) <= O_MAX
Filter: execute the policy's plan if it satisfies the condition. Otherwise
  minimise  || p - p_policy ||     over candidates = 15 more plans sampled from the policy + 3 simple moves
  subject to the condition;        none feasible -> the candidate with the largest barrier (lowest o as tie-break).
No object state, no failure labels, no task goal. O_MAX is calibrated label-free on recorded decisions (99th percentile
of o for the policy's own plans).
"""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
CFG = {"H_SAFE": 1.0, "GAMMA": 0.0, "O_MAX": None, "N_SAMPLES": 15, "FUT_CAND": 32, "PLAN_START": 1, "N_LOOK": 15}
CALIB = ROOT / "results/jenga/cbf_calibration.json"


def horizon(plan, n_look):
    a = np.asarray(plan[CFG["PLAN_START"]:CFG["PLAN_START"] + n_look], np.float32)
    return np.concatenate([a, np.repeat(a[-1:], 38 - n_look, 0)])


class CBF:
    def __init__(self, mon):
        import cbf_barrier as cb
        self.mon, self.cb = mon, cb
        if CFG["O_MAX"] is None and CALIB.exists():
            CFG["O_MAX"] = json.loads(CALIB.read_text())["O_MAX"]

    def score(self, sim, frame_z0, plans, n_fut):
        """plans (C, 16, 4) -> h (C,), o (C,) under SCALES, n_fut futures each."""
        import cem_filter as cf
        import step3_world_model as wm
        mon, cb, torch = self.mon, self.cb, self.mon.torch
        z0 = frame_z0
        C = len(plans)
        split = np.zeros((C, len(cb.SCALES)), bool); o = np.zeros(C)
        seed = int(mon.rng.integers(2 ** 31))
        for si, s in enumerate(cb.SCALES):
            d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                           for p, r in (mon.errors.sample(seed, 0, 0, j) for j in range(n_fut))])
            st = mon.g(z0[None].expand(n_fut, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
            acts = []
            for p in plans:
                base = horizon(p, CFG["N_LOOK"])
                for j in range(n_fut):
                    a = base.copy()
                    noise = np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:CFG["N_LOOK"]] * s
                    a[:CFG["N_LOOK"], :3] -= noise
                    acts.append(a)
            pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(C * n_fut, wm.STEPS, -1)
            z = st.float().repeat(C, 1, 1, 1); ends = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(wm.STEPS):
                    z = mon.model.step(z, pairs[:, t]).float()
                    if t in (wm.HOLD10, wm.HOLD30):
                        ends[t] = z.reshape(C, n_fut, -1)
            ok10, _, _ = cf.split_test_gpu(ends[wm.HOLD10]); ok30, _, _ = cf.split_test_gpu(ends[wm.HOLD30])
            split[:, si] = (ok10 & ok30).cpu().numpy()
            if s == 1.0:
                for c in range(C):
                    pts = np.vstack([sim.data.site_xpos[sim.site][None], np.asarray(plans[c])[CFG["PLAN_START"]:CFG["PLAN_START"] + CFG["N_LOOK"], :3]])
                    m = torch.as_tensor(~cb.hand_mask(pts), device="cuda")
                    diff = (ends[wm.HOLD30][c].reshape(n_fut, 28, 28, 8) - z0[None]).pow(2).mean(-1)
                    o[c] = float(((diff * m).sum((1, 2)) / m.sum().clamp_min(1)).max())
        h = np.full(C, np.log2(cb.SCALES[-1]))
        for c in range(C):
            hit = np.flatnonzero(split[c])
            if len(hit):
                h[c] = np.log2(cb.SCALES[hit[0]])
        return h, o

    def ok(self, h, o, h_now):
        return (h >= min(h_now, CFG["H_SAFE"]) - CFG["GAMMA"]) & (o <= (CFG["O_MAX"] if CFG["O_MAX"] is not None else np.inf))

    def primitives(self, sim, plan):
        """Simple alternative 16-step plans: half speed, vertical only, back off (up 3 cm), as absolute targets."""
        p = np.asarray(plan, np.float32); cur = np.r_[sim.target, 0.0].astype(np.float32)
        slow = p.copy(); w = np.linspace(0, 1, 16)[:, None] * 0.5
        slow[:, :3] = cur[:3] + (p[:, :3] - cur[:3]) * (0.5 + 0 * w); slow[:, 3] = p[:, 3]
        vert = p.copy(); vert[:, :2] = cur[:2]
        back = np.repeat(cur[None], 16, 0); back[:, 2] += np.linspace(0, 0.03, 16); back[:, 3] = 0.0
        return [slow, vert, back]

    def filter(self, sim, chunk, k, plan=None, hist=None, pol=None):
        import policy_rollout as pr
        torch = self.mon.torch
        t0 = time.time()
        frame = self.mon.frame(sim)
        z0 = torch.as_tensor(self.mon.latent(frame[None])[0], device="cuda").float()
        hold = np.repeat(np.r_[sim.target, 0.0][None].astype(np.float32), 16, 0)
        h, o = self.score(sim, z0, np.stack([plan, hold]), 64)
        h_nom, o_nom, h_now = h[0], o[0], h[1]
        info = {"h_nom": float(h_nom), "o_nom": float(o_nom), "h_now": float(h_now)}
        if self.ok(np.array([h_nom]), np.array([o_nom]), h_now)[0]:
            return chunk, {**info, "intervened": False, "seconds": round(time.time() - t0, 2)}
        rs = (torch.get_rng_state(), torch.cuda.get_rng_state())          # sampling candidates must not shift the policy's RNG
        try:
            samples = pr.policy_chunk(pol, hist, n=CFG["N_SAMPLES"])
        finally:
            torch.set_rng_state(rs[0]); torch.cuda.set_rng_state(rs[1])
        cands = list(samples) + self.primitives(sim, plan)
        hc, oc = self.score(sim, z0, np.stack(cands), CFG["FUT_CAND"])
        feas = self.ok(hc, oc, h_now)
        dist = np.array([np.linalg.norm(np.asarray(c)[1:9, :3] - np.asarray(plan)[1:9, :3]) for c in cands])
        if feas.any():
            j = int(np.flatnonzero(feas)[np.argmin(dist[feas])]); kind = "feasible"
        else:
            j = int(np.lexsort((oc, -hc))[0]); kind = "best_effort"
        out = np.asarray(cands[j], np.float32)[CFG["PLAN_START"]:CFG["PLAN_START"] + 8].copy()
        return out, {**info, "intervened": True, "abstained": False, "choice": j, "kind": kind,
                     "h_choice": float(hc[j]), "o_choice": float(oc[j]), "change_mm": float(dist[j] * 1000),
                     "seconds": round(time.time() - t0, 2)}
