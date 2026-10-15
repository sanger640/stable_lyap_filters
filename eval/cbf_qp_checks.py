"""Two quick checks that decide between the next CBF-optimisation directions (offline, dev decisions from cbf_qp prep).

opt1  (optimise in the policy's output space): for each really risky decision, 12 plans sampled from the diffusion
      policy (history = current frame twice, as at an episode start). Does the model's smooth barrier B rank them by
      real risk? -> within-decision Spearman, and the real risk of the plan B would pick vs the policy's plan vs the best.
opt2  (teach the model local sensitivity): the policy's plan nudged by +-5 / +-10 mm in x and y (steps 1-15). Is the
      real risk landscape smooth at the mm scale (5 mm and 10 mm nudges in the same direction agree), and does the
      model's barrier change predict the real change?
Real risk: exact simulator, TRUTH_N futures, the same draws for every plan of a decision (paired).

usage: python3 eval/cbf_qp_checks.py make | truth | report
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("EXEC_NOISE_SCALE", "2"); os.environ.setdefault("HARD_MASS", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import cbf_qp as q  # noqa: E402

OUT = q.OUT / "checks"
N_SAMP, NUDGES = 12, [(ax, mm) for ax in (0, 1) for mm in (-10, -5, 5, 10)]


def risky_ids():
    C, K = q.load_cache()
    T = json.loads((q.OUT / "truth_32.json").read_text())
    return [i for i, k in enumerate(K) if k["acted"] and T[q._hash(C["nominal"][i])] >= 2 / 32]


def make():
    import torch
    import cem_filter as cf
    import inspector_data as insp
    import policy_rollout as pr
    OUT.mkdir(parents=True, exist_ok=True)
    C, K = q.load_cache()
    mon, sim, pol = cf.Monitor(), insp._sim(), pr.load_policy()
    cams = pr.Cameras(sim)
    rows = []
    for n_, i in enumerate(risky_ids()):
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        o = cams.obs(); hist = [o, o]
        torch.manual_seed(5000 + i)
        samples = pr.policy_chunk(pol, hist, n=N_SAMP)                                      # (12, 16, 4)
        base = C["plan16"][i].astype(np.float32)
        nudged = []
        for ax, mm in NUDGES:
            p = base.copy(); p[1:16, ax] += mm / 1000; nudged.append(p)
        plans = np.concatenate([base[None], samples.astype(np.float32), np.stack(nudged)])     # 0 = policy plan
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        hreq = min(K[i]["h_now"], 1.0)
        scales = [s for s in (0.5, 1.0) if np.log2(s) < hreq - 1e-9] or [1.0]
        with torch.no_grad():
            P = torch.as_tensor(plans, device="cuda")
            B = torch.stack([q.Barrier(mon, z0, 1000 + i, q.QP["N"], s)(P) for s in scales]).min(0).values.cpu().numpy()
        rows.append({"i": i, "plans": plans.round(6).tolist(), "B": B.round(4).tolist()})
        print(f"make {n_ + 1}", flush=True)
    (OUT / "plans.json").write_text(json.dumps(rows))
    print("made", len(rows), "decisions", flush=True)


def truth():
    C, K = q.load_cache()
    rows = json.loads((OUT / "plans.json").read_text())
    n_truth = int(os.environ.get("TRUTH_N", 32)); os.environ["TRUTH_N"] = str(n_truth)
    cache_p = q.OUT / f"truth_{n_truth}.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    jobs = []
    for r in rows:
        i = r["i"]
        base = (C["state"][i], C["ctrl"][i], C["target"][i], C["gripper"][i], C["tilt"][i])
        for p in r["plans"]:
            ch = np.asarray(p, np.float32)[1:9]
            h = q._hash(ch)
            if h not in cache:
                jobs.append((h, ((i, 7, 0), *base, ch))); cache[h] = None
    print(len(jobs), "new truth jobs", flush=True)
    with mp.get_context("spawn").Pool(int(os.environ.get("TRUTH_WORKERS", 30))) as pool:
        for n_, (h, flags) in enumerate(pool.imap_unordered(q._truth_one, jobs), 1):
            cache[h] = sum(flags) / len(flags)
            if n_ % 100 == 0 or n_ == len(jobs):
                cache_p.write_text(json.dumps(cache)); print(f"truth {n_}/{len(jobs)}", flush=True)
    cache_p.write_text(json.dumps(cache))


def report():
    from scipy.stats import spearmanr
    T = json.loads((q.OUT / "truth_32.json").read_text())
    rows = json.loads((OUT / "plans.json").read_text())
    s1, pick, pol_r, best, sel_ = [], [], [], [], []
    d5, d10, dB, dR = [], [], [], []
    for r in rows:
        risk = np.array([T[q._hash(np.asarray(p, np.float32)[1:9])] for p in r["plans"]]); B = np.array(r["B"])
        S = slice(0, 1 + N_SAMP)                                      # policy plan + samples
        if np.ptp(risk[S]) > 0 and np.ptp(B[S]) > 0:
            s1.append(spearmanr(B[S], -risk[S]).correlation)
        pol_r.append(risk[0]); pick.append(risk[S][np.argmax(B[S])]); best.append(risk[S].min())
        sel_.append(risk[S][1:].mean())
        nr = risk[1 + N_SAMP:]; nb = B[1 + N_SAMP:]                  # nudges in NUDGES order: per axis -10, -5, +5, +10
        for ax in range(2):
            m10, m5, p5, p10 = nr[4 * ax:4 * ax + 4] - risk[0]
            d5 += [m5, p5]; d10 += [m10, p10]
        dB += list(nb - B[0]); dR += list(nr - risk[0])
    d5, d10, dB, dR = map(np.array, (d5, d10, dB, dR))
    res = {"decisions": len(rows),
           "opt1_policy_space": {"spearman_B_vs_minus_risk_mean": round(float(np.nanmean(s1)), 3), "n_with_variation": len(s1),
                                 "risk_policy_plan": round(float(np.mean(pol_r)), 4), "risk_random_sample": round(float(np.mean(sel_)), 4),
                                 "risk_pick_max_B": round(float(np.mean(pick)), 4), "risk_best_possible": round(float(np.mean(best)), 4)},
           "opt2_local_nudges": {"mean_abs_risk_change_5mm": round(float(np.abs(d5).mean()), 4), "mean_abs_risk_change_10mm": round(float(np.abs(d10).mean()), 4),
                                 "same_direction_5_vs_10mm_corr": round(float(np.corrcoef(d5, d10)[0, 1]), 3),
                                 "model_dB_vs_real_minus_dR_corr": round(float(np.corrcoef(dB, -dR)[0, 1]), 3),
                                 "share_nudges_changing_risk": round(float(np.mean(np.abs(dR) >= 2 / 32)), 3)}}
    (OUT / "report.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"make": make, "truth": truth, "report": report}[sys.argv[1]]()
