"""Exploratory (docs/FILTER_DEEP_DIVE.md section 5): is the residual a RESOLUTION limit of the criterion (too few imagined
futures), and which disturbance readout ranks chunks best? Same chunks and pair rule as eval/rank_check.py (model_OIL,
chunk + hold tail). Each decision is scored ONCE with NMAX futures (common draws for all its chunks); readouts are then
computed on nested subsets of n = 32 / 64 / 128 / NMAX futures:
  o_max  (current barrier: max over futures of the outside-hand change), o_mean, o_q90, o_exc = P(o_j > O_MAX)
  h      (fork scale from the split test on the subset).
usage: python3 eval/res_check.py <set> [analyse]   -> results/jenga/deep_dive/res_<set>{,_rows}.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
D = ROOT / "results/jenga/deep_dive"
NMAX = int(os.environ.get("RES_NMAX", 256))
NS = (32, 64, 128, NMAX)


def score_all(cbf, sim, z0, plans, n_fut):
    """cbf.score with per-future outputs: o (C, n) at scale 1, ends at HOLD10/HOLD30 per scale (scales 0.5, 1)."""
    import cbf_filter as cbff
    import step3_world_model as wm
    mon, cb, torch = cbf.mon, cbf.cb, cbf.mon.torch
    C = len(plans); seed = int(mon.rng.integers(2 ** 31)); ends_s, o = [], None
    for s in cb.SCALES[:-1]:
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                       for p, r in (mon.errors.sample(seed, 0, 0, j) for j in range(n_fut))])
        st = mon.g(z0[None].expand(n_fut, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
        acts = []
        for p in plans:
            base = cbff.horizon(p, cbff.CFG["N_LOOK"])
            for j in range(n_fut):
                a = base.copy()
                a[:cbff.CFG["N_LOOK"], :3] -= np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:cbff.CFG["N_LOOK"]] * s
                acts.append(a)
        pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(C * n_fut, wm.STEPS, -1)
        z = st.float().repeat(C, 1, 1, 1); ends = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.STEPS):
                z = mon.model.step(z, pairs[:, t]).float()
                if t in (wm.HOLD10, wm.HOLD30):
                    ends[t] = z.reshape(C, n_fut, -1)
        ends_s.append(ends)
        if s == 1.0:
            o = np.zeros((C, n_fut))
            for c in range(C):
                pts = np.vstack([sim.data.site_xpos[sim.site][None],
                                 np.asarray(plans[c])[cbff.CFG["PLAN_START"]:cbff.CFG["PLAN_START"] + cbff.CFG["N_LOOK"], :3]])
                m = torch.as_tensor(~cb.hand_mask(pts), device="cuda")
                diff = (ends[wm.HOLD30][c].reshape(n_fut, 28, 28, 8) - z0[None]).pow(2).mean(-1)
                o[c] = ((diff * m).sum((1, 2)) / m.sum().clamp_min(1)).cpu().numpy()
    return o, ends_s


def h_subset(ends_s, n, cb):
    import cem_filter as cf
    import step3_world_model as wm
    C = ends_s[0][wm.HOLD10].shape[0]; split = np.zeros((C, len(ends_s)), bool)
    for si, ends in enumerate(ends_s):
        ok10, _, _ = cf.split_test_gpu(ends[wm.HOLD10][:, :n]); ok30, _, _ = cf.split_test_gpu(ends[wm.HOLD30][:, :n])
        split[:, si] = (ok10 & ok30).cpu().numpy()
    return np.array([np.log2(cb.SCALES[np.flatnonzero(r)[0]]) if r.any() else np.log2(cb.SCALES[-1]) for r in split])


def main(name):
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    import rank_check as rc
    cbff.CFG["H_SAFE"] = float(np.log2(3.0))
    C, cand = rc.chunks_for(name); mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim(); rows = []
    O_MAX = cbff.CFG["O_MAX"]
    for i, d in cand.items():
        rk = [v[1] for v in d.values()]
        if len(d) < 2 or max(rk) - min(rk) < 2 / 32 - 1e-9:                 # no informative pair at this decision
            continue
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        names = list(d); tgt = np.r_[C["target"][i], 0.0]
        plans = np.stack([np.concatenate([tgt[None], d[a][0], np.repeat(d[a][0][-1:], 7, 0)]).astype(np.float32) for a in names])
        plans[:, 0, 3] = plans[:, 1, 3]
        mon.rng = np.random.default_rng([20261090, i])
        o, ends_s = score_all(cbf, sim, z0, plans, NMAX)
        hs = {n: h_subset(ends_s, n, cbf.cb) for n in NS}
        for j, a in enumerate(names):
            r = {"i": i, "arm": a, "risk": d[a][1]}
            for n in NS:
                x = o[j, :n]
                xs = np.sort(x)[::-1]
                r.update({f"o_cvar{a}_{n}": float(xs[:max(1, int(round(a / 100 * n)))].mean()) for a in (2, 5, 10)})
                r.update({f"o_max_{n}": float(x.max()), f"o_mean_{n}": float(x.mean()), f"o_q90_{n}": float(np.quantile(x, 0.9)),
                          f"o_exc_{n}": float((x > O_MAX).mean()), f"h_{n}": float(hs[n][j])})
            rows.append(r)
        print(name, i, len(rows), flush=True)
    (D / f"res_{name}_rows.json").write_text(json.dumps(rows)); analyse(name)


def analyse(name):
    import tok_rank as tr
    R = json.loads((D / f"res_{name}_rows.json").read_text())
    out = {"set": name, "chunks": len(R), "decisions": len({r["i"] for r in R}), "exploratory": True}
    for k in ("o_max", "o_cvar2", "o_cvar5", "o_cvar10", "o_mean", "o_q90", "o_exc", "h"):
        out[k] = {n: tr.conc(R, f"{k}_{n}", -1 if k == "h" else 1) for n in NS if f"{k}_{n}" in R[0]}
    (D / f"res_{name}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    analyse(sys.argv[1]) if len(sys.argv) > 2 else main(sys.argv[1])
