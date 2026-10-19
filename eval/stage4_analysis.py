"""Stage 4 / 4b analysis, exactly as pre-registered (results/jenga/stage4_protocol.json; Stage 4b in
docs/OPT_FILTER_REDESIGN.md, G2 section): paired by episode, normal approximation, one-sided 95%, exact McNemar.
usage: python3 eval/stage4_analysis.py [s4|s4b]  -> results/jenga/stage4_result.json | stage4b_result.json"""
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STAGE = sys.argv[1] if len(sys.argv) > 1 else "s4"
A = json.loads((ROOT / f"results/jenga/outcome_audit_{STAGE}.json").read_text())
ARMS = ({"policy": "record", "selection": "cbfmsoil", "hold_replan": "cbfh", "G_CEM": "cbfg"} if STAGE == "s4" else
        {"policy": "record", "selection": "cbfmsoil", "hold_replan2": "cbfh2", "G_CEM": "cbfg", "G2_CEM": "cbfg2"})
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


out = {"stage": STAGE, "episodes": len(EPS),
       "counts": {a: {"topples": int(vec(a, "topple").sum()), "v4": int(vec(a, "v4_success").sum()),
                      "v4b": int(vec(a, "v4b_success").sum()),
                      "red_fell_after_extraction": int(vec(a, "red_fell_after_extraction").sum())} for a in ARMS}}
if STAGE == "s4":
    for name, b in (("G_vs_selection", "selection"), ("G_vs_hold_replan", "hold_replan")):
        out[name] = noninf("G_CEM", b)
    for a in ("selection", "hold_replan", "G_CEM"):
        out[f"{a}_vs_policy"] = {k: paired(a, "policy", k) for k in KEYS}
else:
    out["PRIMARY_G2_vs_selection"] = noninf("G2_CEM", "selection")
    out["G2_vs_G"] = {k: paired("G2_CEM", "G_CEM", k) for k in KEYS + ("red_fell_after_extraction",)}
    out["G2_vs_hold_replan2"] = noninf("G2_CEM", "hold_replan2")
    out["G_vs_selection_replication"] = noninf("G_CEM", "selection")
    for a in ("selection", "hold_replan2", "G_CEM", "G2_CEM"):
        out[f"{a}_vs_policy"] = {k: paired(a, "policy", k) for k in KEYS}
dst = ROOT / f"results/jenga/{'stage4_result' if STAGE == 's4' else 'stage4b_result'}.json"
dst.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
