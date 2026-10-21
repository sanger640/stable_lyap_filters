"""Head-to-head analysis as pre-registered (results/jenga/lsf_h2h_protocol.json, amendment 1: exploratory, LSF-hold eps 0):
same statistics as eval/stage4_analysis.py (paired by episode, normal approximation, one-sided 95%, exact McNemar).
usage: python3 eval/h2h_analysis.py  -> results/jenga/lsf_h2h_result.json"""
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STAGE = "h2h"
A = json.loads((ROOT / "results/jenga/outcome_audit_h2h.json").read_text())
ARMS = {"policy": "record", "G_CEM": "cbfg", "hold_replan": "cbfh", "LSF_hold": "lsfh000"}
V = {a: {r["ep"]: r for r in A[f"paired_{run}_{STAGE}"]["rows"]} for a, run in ARMS.items()}
EPS = sorted(set.intersection(*(set(v) for v in V.values())))
KEYS = ("topple", "v4_success", "v4b_success")


def vec(a, key):
    if key == "topple":
        return np.array([V[a][e]["logged_neighbour_toppled"] for e in EPS], float)
    return np.array([V[a][e][key] for e in EPS], float)


def paired(a, b, key):
    d = vec(a, key) - vec(b, key); se = d.std(ddof=1) / np.sqrt(len(d))
    n10, n01 = int(((vec(a, key) == 1) & (vec(b, key) == 0)).sum()), int(((vec(a, key) == 0) & (vec(b, key) == 1)).sum())
    n = n10 + n01; k = min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0          # exact McNemar
    return {"diff_pp": round(100 * d.mean(), 2), "lower95_pp": round(100 * (d.mean() - 1.645 * se), 2),
            "upper95_pp": round(100 * (d.mean() + 1.645 * se), 2), "a_only": n10, "b_only": n01, "mcnemar_p": float(f"{p:.3g}")}


def noninf(a, b):
    r = {k: paired(a, b, k) for k in KEYS}
    r["NONINF"] = {f"success {s} lower > -5pp": bool(r[f"{s}_success"]["lower95_pp"] > -5) for s in ("v4", "v4b")}
    r["NONINF"]["topple upper < +3pp"] = bool(r["topple"]["upper95_pp"] < 3)
    return r


out = {"stage": STAGE, "exploratory": True, "episodes": len(EPS),
       "counts": {a: {"topples": int(vec(a, "topple").sum()), "v4": int(vec(a, "v4_success").sum()),
                      "v4b": int(vec(a, "v4b_success").sum())} for a in ARMS}}
out["PRIMARY_G_vs_LSF_hold"] = noninf("G_CEM", "LSF_hold")          # (1) non-inferiority; (2) topple McNemar inside
out["trigger_hold_replan_vs_LSF_hold"] = noninf("hold_replan", "LSF_hold")
for a in ("G_CEM", "hold_replan", "LSF_hold"):
    out[f"{a}_vs_policy"] = {k: paired(a, "policy", k) for k in KEYS}
(ROOT / "results/jenga/lsf_h2h_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
