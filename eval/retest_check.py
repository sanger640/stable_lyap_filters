"""Test-retest of the fork trigger, and does pooling more imagined futures rank real risk better? (exploratory)
Benchmark v2 dev + holdout (300 decisions). Plans: policy chunk and hold. Scored R=4 times with independent seeds at
N=64 futures (the closed-loop setting) and once at N=256. Real one-chunk risk of the policy chunk from the cached
truth (bench2.truth_chunks). Adopted 3x trigger: h_plan < min(h_hold, log2 4) or o_plan > O_MAX.
usage: WM_COMPILE=1 python3 eval/retest_check.py [run|analyse] -> results/jenga/deep_dive/retest_rows.json, retest_check.json
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
import bench2 as b2  # noqa: E402

ROWS = ROOT / "results/jenga/deep_dive/retest_rows.json"
R, N, NBIG = 4, 64, 256


def run():
    import torch
    import cbf_barrier
    import cbf_filter as cbff
    import cem_filter as cf
    import inspector_data as insp
    cbf_barrier.SCALES = (0.5, 1.0, 2.0, 3.0, 4.0)
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    rows = json.loads(ROWS.read_text()) if ROWS.exists() else []
    done = {(r["set"], r["i"]) for r in rows}
    for name in ("dev", "holdout"):
        C, Rr = b2.load(name)
        for i in range(len(Rr)):
            if (name, i) in done:
                continue
            sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
            z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
            plans = np.stack([C["plan16"][i], np.repeat(np.r_[C["target"][i], C["plan16"][i][0, 3]][None], 16, 0).astype(np.float32)])
            r = {"set": name, "i": i, "gripper": float(C["gripper"][i]), "seeds": []}
            for s in range(R + 1):
                mon.rng = np.random.default_rng([20261024, s, i]); torch.manual_seed(20261024 + 1000 * s + i)
                h, o = cbf.score(sim, z0, plans, NBIG if s == R else N)
                r["seeds"].append({"h": [float(x) for x in h], "o": [float(x) for x in o]})
            rows.append(r); print("rt", name, i, [x["h"][0] for x in r["seeds"]], flush=True)
            if len(rows) % 10 == 0:
                ROWS.write_text(json.dumps(rows))
    ROWS.write_text(json.dumps(rows))


def auc(score, y):
    """P(score of a positive > score of a negative), ties 1/2."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    p, n = score[y], score[~y]
    if not len(p) or not len(n):
        return None
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))


def analyse():
    import cbf_filter as cbff
    OMAX = json.loads((ROOT / os.environ["CBF_CALIB"]).read_text())["O_MAX"]; HS = 2.0
    rows = json.loads(ROWS.read_text())
    truth = {}
    for name in ("dev", "holdout"):
        C, _ = b2.load(name); idx = sorted(r["i"] for r in rows if r["set"] == name)
        for i, t in zip(idx, b2.truth_chunks(name, [(i, C["nominal"][i]) for i in idx])):
            truth[(name, i)] = t
    risk = np.array([truth[(r["set"], r["i"])][0] for r in rows]); red = np.array([truth[(r["set"], r["i"])][1] for r in rows])
    trig = lambda x: x["h"][0] < min(x["h"][1], HS) - 1e-9 or x["o"][0] > OMAX
    T = np.array([[trig(x) for x in r["seeds"][:R]] for r in rows])                     # (n, R)
    hp = np.array([[x["h"][0] for x in r["seeds"][:R]] for r in rows]); hbig = np.array([r["seeds"][R]["h"][0] for r in rows])
    op = np.array([[x["o"][0] for x in r["seeds"][:R]] for r in rows]); obig = np.array([r["seeds"][R]["o"][0] for r in rows])
    tbig = np.array([trig(r["seeds"][R]) for r in rows])
    pairs = [(a, b) for a in range(R) for b in range(a + 1, R)]
    y = risk > 0
    out = {"decisions": len(rows), "risky_share": round(float(y.mean()), 3), "OMAX": OMAX,
           "trigger_rate_single": round(float(T.mean()), 3), "trigger_rate_N256": round(float(tbig.mean()), 3),
           "retest_agreement": round(float(np.mean([(T[:, a] == T[:, b]).mean() for a, b in pairs])), 3),
           "retest_kappa": None,
           "fires_in_k_of_4": [int(((T.sum(1)) == k).sum()) for k in range(R + 1)],
           "h_retest_exact": round(float(np.mean([(hp[:, a] == hp[:, b]).mean() for a, b in pairs])), 3),
           "real_risk_by_k_of_4": [round(float(risk[T.sum(1) == k].mean()), 4) if (T.sum(1) == k).any() else None for k in range(R + 1)],
           "auc_any_risk": {"neg_h_single": round(np.mean([auc(-hp[:, s], y) for s in range(R)]), 3),
                            "neg_h_mean4": round(auc(-hp.mean(1), y), 3), "neg_h_N256": round(auc(-hbig, y), 3),
                            "o_single": round(np.mean([auc(op[:, s], y) for s in range(R)]), 3),
                            "o_mean4": round(auc(op.mean(1), y), 3), "o_N256": round(auc(obig, y), 3),
                            "trigger_single": round(np.mean([auc(T[:, s], y) for s in range(R)]), 3),
                            "trigger_k_of_4": round(auc(T.sum(1), y), 3), "trigger_N256": round(auc(tbig, y), 3)},
           "real_risk_captured": {}}
    p0 = T.mean(); po = T[:, 0].mean()
    k = (p0 * po + (1 - p0) * (1 - po))
    out["retest_kappa"] = round(float((out["retest_agreement"] - k) / (1 - k)), 3)
    # share of the total real risk on the decisions a rule fires on, and its firing rate
    tot = risk.sum()
    for lab, f in [("single", T[:, 0]), ("N256", tbig), ("2_of_4", T.sum(1) >= 2), ("any_of_4", T.sum(1) >= 1), ("all_4", T.sum(1) == R)]:
        out["real_risk_captured"][lab] = {"fire_rate": round(float(f.mean()), 3), "risk_share": round(float(risk[f].sum() / tot), 3),
                                          "mean_risk_fired": round(float(risk[f].mean()), 4) if f.any() else None,
                                          "mean_risk_not_fired": round(float(risk[~f].mean()), 4) if (~f).any() else None,
                                          "red_mean_fired": round(float(red[f].mean()), 4) if f.any() else None}
    (ROOT / "results/jenga/deep_dive/retest_check.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "analyse": analyse}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
