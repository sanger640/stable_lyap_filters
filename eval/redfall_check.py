"""Red-block-fall blind spot (2026-10-16, exploratory). The red (target) block falling over is the largest remaining
failure for all arms (dev10: 13-15 / 75) and the filters barely reduce it. The disturbance signal o masks the hand
region (which contains the red block); the fork signal h uses the whole latent. Question: does any task-agnostic latent
signal warn BEFORE a red-block fall, at the policy's own chunk?

Data: policy-alone episodes (dev10 record 12075-12149, final test 1 record 1000-1299). Each saved decision is replayed
(run's noise); label = the red block's tilt crosses 45 deg during this chunk or the next (and was < 45 at the start);
neighbour-topple label likewise. Positives: all such decisions; negatives: 4 random others per episode.
Scores on the policy's 16-step plan (model_OIL, 64 futures, scale 1 only + the full fork barrier h from cbf.score):
  h       fork barrier (whole latent)            o_out   max over futures of mean change OUTSIDE the hand mask (= o)
  o_in    mean over futures of change INSIDE      s_in    std over futures of the inside change (spread)
  fork_in fork test on the inside-hand latent only (split at scale 1)
-> results/jenga/redfall_check.json (AUC of each for red falls and for neighbour topples)
usage: CBF_WM=.../model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json python3 eval/redfall_check.py
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RUNS = ROOT / "results/program_b/policy_rollouts"


def labels(sim, S, noises):
    """per decision: (red falls this or next chunk, neighbour topples this or next chunk)."""
    tt, nt = [], []
    for d in range(len(S["chunk_index"])):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        k = int(S["chunk_index"][d]); t = sim.block_diagnostics()["tilt"]; a0 = (float(t[0]), float(t[1:].max()))
        mx = list(a0)
        for j, a in enumerate(S["executed"][d]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]; sim.execute(a)
            t = sim.block_diagnostics()["tilt"]; mx = [max(mx[0], float(t[0])), max(mx[1], float(t[1:].max()))]
        tt.append((a0[0], mx[0])); nt.append((a0[1], mx[1]))
    n = len(tt); red, nb = np.zeros(n, bool), np.zeros(n, bool)
    for d in range(n):
        nxt = range(d, min(d + 2, n))
        red[d] = tt[d][0] < 45 and any(tt[e][1] >= 45 for e in nxt)
        nb[d] = nt[d][0] < 45 and any(nt[e][1] >= 45 for e in nxt)
    return red, nb


def main():
    import torch
    import cbf_barrier as cb
    import cbf_filter as cbff
    import cem_filter as cf
    import inspector_data as insp
    import plausibility as pl
    import step3_world_model as wm
    from sklearn.metrics import roc_auc_score
    cbff.CFG["H_SAFE"] = float(np.log2(3.0))
    mon = cf.Monitor(seed=4242); cbf = cbff.CBF(mon); sim = insp._sim(); rng = np.random.default_rng(0)
    rows = []
    for run in ("paired_record_dev10", "paired_record_test2x"):
        for p in sorted((RUNS / run).glob("ep*.npz")):
            ep = int(p.stem[2:])
            with np.load(p) as f:
                S = {k: f[k] for k in f.files}
            noises = pl.episode_noises(ep, int(S["chunk_index"][-1]) + 1)
            red, nb = labels(sim, S, noises)
            pos = set(np.flatnonzero(red | nb).tolist())
            neg = [d for d in range(len(red)) if d not in pos]
            pick = sorted(pos | set(rng.choice(neg, min(4, len(neg)), replace=False).tolist()))
            for d in pick:
                sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
                z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
                plan = S["plan16"][d][None]
                h, o = cbf.score(sim, z0, plan, 64)
                # scale-1 futures again for the inside-hand variants
                d_ = np.stack([np.concatenate([(p_ * [1, 1, 0]).reshape(9), (r_ * [0, 0, 1]).reshape(9)])
                               for p_, r_ in (mon.errors.sample(ep * 100 + d, 0, 0, j) for j in range(64))])
                st = mon.g(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
                base = cbff.horizon(plan[0], cbff.CFG["N_LOOK"]); acts = []
                for j in range(64):
                    a = base.copy(); a[:cbff.CFG["N_LOOK"], :3] -= np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:cbff.CFG["N_LOOK"]]
                    acts.append(a)
                z = st.float(); pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(64, wm.STEPS, -1)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for t in range(wm.STEPS):
                        z = mon.model.step(z, pairs[:, t]).float()
                        if t == wm.HOLD30:
                            e30 = z.clone()
                pts = np.vstack([sim.data.site_xpos[sim.site][None], plan[0][1:16, :3]])
                m_in = torch.as_tensor(cb.hand_mask(pts), device="cuda").float()
                diff = (e30.reshape(64, 28, 28, 8) - z0[None]).pow(2).mean(-1)
                ins = (diff * m_in).sum((1, 2)) / m_in.sum().clamp_min(1)
                masked = (e30.reshape(64, 28, 28, 8) * m_in[None, :, :, None]).reshape(1, 64, -1)
                ok_in, _, _ = cf.split_test_gpu(masked)
                rows.append({"run": run, "ep": ep, "d": int(d), "red_fall": bool(red[d]), "neighbour_topple": bool(nb[d]),
                             "h": float(h[0]), "o_out": float(o[0]), "o_in": float(ins.mean()), "s_in": float(ins.std()),
                             "fork_in": bool(ok_in[0])})
                torch.cuda.empty_cache()
            print(run, ep, sum(r["red_fall"] for r in rows), "red-fall decisions so far", flush=True)
    res = {"decisions": len(rows), "red_fall": sum(r["red_fall"] for r in rows), "neighbour_topple": sum(r["neighbour_topple"] for r in rows)}
    for lab_ in ("red_fall", "neighbour_topple"):
        y = np.array([r[lab_] for r in rows]); keep = np.array([not (r["red_fall"] or r["neighbour_topple"]) or r[lab_] for r in rows])
        res[lab_ + "_auc (vs decisions with neither)"] = {
            k: round(float(roc_auc_score(y[keep], sgn * np.array([float(r[k]) for r in rows])[keep])), 3)
            for k, sgn in (("h", -1), ("o_out", 1), ("o_in", 1), ("s_in", 1), ("fork_in", 1))}
    (ROOT / "results/jenga/redfall_check.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
