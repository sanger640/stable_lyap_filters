"""Irreversibility-barrier CBF filter in closed loop (results/jenga/irr_protocol.json). The fork CBF of eval/cbf_filter.py
with its barrier replaced by CF2 / CF3's worst-case time irreversibility of the intuitive physics engine's futures
(eval/cf_rev.py, eval/cf_conf.py): for a plan p, 64 imagined futures in model_OIL at 2x the perception + execution errors,
  r_j(p) = q98 over the latent patches OUTSIDE p's hand path of mean_ch |f_rev_j - f_hold_j|
(reverse branch: p's 15 look-ahead steps with the draw's execution noise, then the same targets back to the current
target, then hold; hold branch: the current target held, same noise), h(p) = -max_j r_j(p).
  safe set:  h(p) >= -THR,  THR = 1.3356 (label-free: q98 of REAL latent change before the first grasp, training runs)
Holding still is always in the safe set (its reverse is itself: r = 0 exactly), so the filter is always feasible.
Trigger: the policy's plan violates the condition. Fix: G-CEM (eval/opt3.py), the cheapest transformation of this chunk
(speed, pause, amplitude, back-off; irs adds a 3-D offset) that satisfies the condition on fresh draws, else hold.
Certificate on model_OIL only (the model CF3 tested). Task-agnostic, label-free, no reference to the first frame.
This file only adds the barrier (patching cbf_filter.CBF.score / ok at start-up); policy_rollout / cbf_filter unchanged.
usage: <the paired env of policy_rollout> python3 eval/irr_rollout.py <irg|irs> <n>
"""
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
THR, S = 1.3356, float(os.environ.get("IRR_S", 2.0))     # S: the gate's imagined noise scale (N1M: 1x)


def score_irr(self, sim, frame_z0, plans, n_fut):
    """plans (C, 16, 4) -> h = -r_max (C,), o = r_max (C,), one seed per call (as CBF.score)."""
    import cbf_filter as cbff
    import cf_rev
    import step3_world_model as wm
    mon, cb, torch = self.mon, self.cb, self.mon.torch
    z0 = frame_z0
    S0, NL = cbff.CFG["PLAN_START"], cbff.CFG["N_LOOK"]
    total = 2 * wm.STEPS
    seed = int(mon.rng.integers(2 ** 31))
    d_ = np.stack([np.concatenate([(p * [1, 1, 0] * S).reshape(9), (r * [0, 0, 1] * S).reshape(9)])
                   for p, r in (mon.errors.sample(seed, 0, 0, j) for j in range(n_fut))])
    st = mon.g(z0[None].expand(n_fut, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
    noise = [np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:NL] * S for j in range(n_fut)]
    site = sim.data.site_xpos[sim.site].copy()
    plans = [np.asarray(p, np.float32) for p in plans]
    grips = sorted({float(p[0, 3]) for p in plans})
    bases = []
    for p in plans:                                           # reverse branches, one per plan
        cur = np.r_[sim.target, p[0, 3]].astype(np.float32)
        rev = cf_rev.reverse_plan(p, cur, NL, S0)
        bases.append(np.concatenate([rev[S0:], np.repeat(rev[-1:], total, 0)])[:total])
    for g in grips:                                           # hold branches, one per current gripper command
        bases.append(np.repeat(np.r_[sim.target, g][None].astype(np.float32), total, 0))
    acts = []
    for base in bases:
        for j in range(n_fut):
            a = base.copy(); a[:NL, :3] -= noise[j]; acts.append(a)
    B = len(bases)
    pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(B * n_fut, wm.STEPS, -1)
    z = st.repeat(B, 1, 1, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(wm.STEPS):
            z = mon.model.step(z, pairs[:, t]).float()
    z = z.float().reshape(B, n_fut, 28, 28, -1)
    r = np.zeros(len(plans))
    for c, p in enumerate(plans):
        m = torch.as_tensor(~cb.hand_mask(np.vstack([site[None], p[S0:S0 + NL, :3]])), device="cuda")
        zh = z[len(plans) + grips.index(float(p[0, 3]))]
        r[c] = float(torch.quantile((z[c] - zh).abs().mean(-1)[:, m], 0.98, dim=1).max())
    return -r, r


def ok_irr(self, h, o, h_now):
    return np.asarray(h) >= -THR


def setup(mode):
    os.environ["CBF_WM"] = str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); os.environ["OPT2_CERT"] = "OIL"
    os.environ["OPT3_GRIP"] = "fixed"
    if mode == "irs":
        os.environ["OPT3_STEER"] = "1"                          # must be set before opt3 is imported
    import cbf_filter as cbff
    import opt3                                                 # noqa: F401  (imported now: its env defaults apply from episode 1)
    assert opt3.CFG["STEER"] == (mode == "irs")
    cbff.CALIB = ROOT / "results/jenga/cbf_calibration_OIL.json"
    cbff.CFG["FIX"] = "g3"; cbff.CFG["TRIGGER"] = "both"; cbff.CFG["BARRIER"] = "irr"
    cbff.CBF.score, cbff.CBF.ok = score_irr, ok_irr
    init = cbff.CBF.__init__

    def init_oil(self, mon):                                    # certificate on model_OIL only (no ensemble members)
        init(self, mon); self._certs = [None]
    cbff.CBF.__init__ = init_oil


def setup_gated(value_or=False):
    """lfci000 (irr_protocol amendment_1): LF4's lfc000 (label-free value value_lf_alea.pt, eps 0.00, gradient edit of the
    policy's xyz under the latent CBF condition, gamma 0.5, every model step) acting ONLY in chunks whose plan the
    irreversibility barrier flags (r_max > THR, computed once per chunk before it runs); other chunks: the policy's commands.
    value_or (lfcj000, results/jenga/gate3_protocol.json): the gate also opens when the gripper is open AND the label-free
    value on the real latent is below its failure level, V(z, [target, plan's first gripper command]) < 0."""
    os.environ["CBF_WM"] = str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); os.environ["CBF_FIX"] = "lsf_step"
    os.environ.update({"LSF_EPS": "0.0", "LSF_VALUE_NAME": "value_lf_alea.pt", "LSF_ALEA": "8", "LSF_REL": "0",
                       "LSF_EDIT": "grad", "LSF_CBF_GAMMA": "0.5"})          # = policy_rollout's lfc000 settings
    import time
    import cbf_filter as cbff
    cbff.CALIB = ROOT / "results/jenga/cbf_calibration_OIL.json"
    cbff.CFG["FIX"] = "lsf_step"
    step = cbff.CBF.lsf_step

    def filter_gate(self, sim, chunk, k, plan=None, hist=None, pol=None):
        t0 = time.time(); torch = self.mon.torch
        z0 = torch.as_tensor(self.mon.latent(self.mon.frame(sim)[None])[0], device="cuda").float()
        r = float(score_irr(self, sim, z0, np.asarray(plan, np.float32)[None], 64)[1][0])
        self._gate = r > THR
        info = {"intervened": False, "abstained": False, "kind": "lsf_step", "irr_r": round(r, 4)}
        if value_or:
            import lsf as L
            if getattr(self, "_vgate", None) is None:
                self._vgate = L.ValueFn("value_lf_alea.pt")
            g = torch.as_tensor(np.r_[sim.target, plan[0][3]][None], dtype=torch.float32, device="cuda")
            with torch.no_grad():
                v = float(self._vgate.V(z0[None], g))
            opened = float(sim.gripper) >= 50                         # actuator: 110 open, 0 closed (as hj_data2's proprio)
            info.update({"gate_v": round(v, 4), "gate_open": opened, "irr_only": bool(self._gate)})
            self._gate = self._gate or (opened and v < 0.0)
        info.update({"irr_gate": bool(self._gate), "seconds": round(time.time() - t0, 2)})
        return chunk, info

    def step_gate(self, sim, cmds, k, j, info):
        if not getattr(self, "_gate", False):
            info["lsf_steps"] = info.get("lsf_steps", 0) + 1
            return np.asarray(cmds, np.float32)
        return step(self, sim, cmds, k, j, info)
    cbff.CBF.filter, cbff.CBF.lsf_step = filter_gate, step_gate


if __name__ == "__main__":
    mode, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100
    assert mode in ("irg", "irs", "lfci000", "lfcj000", "lfck000")
    if mode == "lfck000":                                     # gate3 amendment_2: lfcj000 with the pure edit (no actor fallback)
        os.environ["LSF_NO_FALLBACK"] = "1"
    setup_gated(mode in ("lfcj000", "lfck000")) if mode.startswith("lfc") else setup(mode)
    import policy_rollout as pr
    pr._paired(mode, "cbf", n)
