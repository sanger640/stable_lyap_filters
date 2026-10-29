"""TAU1 (results/jenga/tau1_protocol.json, offline, CPU): sensitivity of the irreversibility threshold and two label-free
calibrations of it, on the stored 0.8 s undo-test residuals of CF3's fresh decisions (results/jenga/irrm1_rows_*.npz).
usage: python3 eval/tau_calib.py -> results/jenga/tau1_result.json"""
import json
from math import ceil
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RUNS = ("paired_record_s4", "paired_record_h2h2")
TAU, ALPHA, SEED = 1.3356, 0.02, 20261029


def conformal(x, alpha):
    x = np.sort(x); k = min(len(x), ceil((len(x) + 1) * (1 - alpha)))
    return float(x[k - 1])


def grade(r, pre, safe, cl, tau):
    f = r > tau
    return {"tau": round(tau, 4), "recall_pre": round(float(f[pre].mean()), 4), "safe_closed": round(float(f[safe & cl].mean()), 4),
            "safe_open": round(float(f[safe & ~cl].mean()), 4)}


def main():
    fs = sorted((ROOT / "results/jenga").glob("irrm1_rows_*.npz"))
    R = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("run", "ep", "i", "kt", "k", "closed", "r8")}
    r = R["r8"].max(1); kt, k, cl = R["kt"], R["k"], R["closed"] > 0
    pre = (kt >= 0) & (kt - k >= 1) & (kt - k <= 3); safe = kt < 0
    fc = {}                                                                     # first gripper-closed decision per episode (proprio)
    for run_, ep in sorted(set(zip(R["run"].tolist(), R["ep"].tolist()))):
        with np.load(ROOT / "results/jenga/hj2/data" / RUNS[run_] / f"ep{ep}.npz") as d:
            c = np.flatnonzero(d["proprio"][:, 3] > 0); fc[(run_, ep)] = int(c[0]) if len(c) else 10 ** 9
    pg = np.array([i < fc[(a, b)] for a, b, i in zip(R["run"].tolist(), R["ep"].tolist(), R["i"].tolist())])
    keys = sorted(fc); rng = np.random.default_rng(SEED); cal_keys = {keys[j] for j in rng.permutation(len(keys))[:len(keys) // 2]}
    cal = np.array([(a, b) in cal_keys for a, b in zip(R["run"].tolist(), R["ep"].tolist())]); ev = ~cal
    tau_b, tau_c = conformal(r[cal & pg], ALPHA), conformal(r[cal], ALPHA)
    g = lambda m, t: grade(r[m], pre[m], safe[m], cl[m], t)
    ok = pre | safe
    out = {"decisions": int(len(r)), "calibration": {"decisions": int(cal.sum()), "pregrasp": int((cal & pg).sum())},
           "evaluation": {"decisions": int(ev.sum()), "pre": int((ev & pre).sum()), "safe": int((ev & safe).sum())},
           "auc_r_all": round(float(roc_auc_score(pre[ok], r[ok])), 4),
           "tau": {"current": TAU, "B_pregrasp": round(tau_b, 4), "C_all": round(tau_c, 4)},
           "eval_half": {"current": g(ev, TAU), "B_pregrasp": g(ev, tau_b), "C_all": g(ev, tau_c)},
           "pregrasp_flag_rate_eval_at_current": round(float((r[ev & pg] > TAU).mean()), 4),
           "sweep_all": [g(np.ones_like(ev), TAU * m) | {"mult": m} for m in (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)]}
    E = out["eval_half"]
    near = lambda v: abs(v / TAU - 1) <= 0.15
    close = lambda a: abs(E[a]["recall_pre"] - E["current"]["recall_pre"]) <= 0.10 and abs(E[a]["safe_closed"] - E["current"]["safe_closed"]) <= 0.02
    out["robust"] = bool(near(tau_b) and near(tau_c) and close("B_pregrasp") and close("C_all"))
    out["verdict"] = "ROBUST" if out["robust"] else "RECALIBRATE recommended"
    (ROOT / "results/jenga/tau1_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
