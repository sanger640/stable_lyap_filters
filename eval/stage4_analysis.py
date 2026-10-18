"""Stage 4 analysis, exactly as results/jenga/stage4_protocol.json: paired by episode, normal approximation, one-sided 95%.
-> results/jenga/stage4_result.json. usage: python3 eval/stage4_analysis.py"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
A = json.loads((ROOT / "results/jenga/outcome_audit_s4.json").read_text())
ARMS = {"policy": "paired_record_s4", "selection": "paired_cbfmsoil_s4", "hold_replan": "paired_cbfh_s4", "G_CEM": "paired_cbfg_s4"}
V = {a: {r["ep"]: r for r in A[run]["rows"]} for a, run in ARMS.items()}
EPS = sorted(set.intersection(*(set(v) for v in V.values())))


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


out = {"episodes": len(EPS), "counts": {a: {"topples": int(vec(a, "topple").sum()), "v4": int(vec(a, "v4_success").sum()),
                                            "v4b": int(vec(a, "v4b_success").sum())} for a in ARMS}}
for name, b in (("G_vs_selection", "selection"), ("G_vs_hold_replan", "hold_replan")):
    r = {k: paired("G_CEM", b, k) for k in ("topple", "v4_success", "v4b_success")}
    r["NONINF"] = {f"success {s} lower > -5pp": bool(r[f"{s}_success"]["lower95_pp"] > -5) for s in ("v4", "v4b")}
    r["NONINF"]["topple upper < +3pp"] = bool(r["topple"]["upper95_pp"] < 3)
    out[name] = r
for a in ("selection", "hold_replan", "G_CEM"):
    out[f"{a}_vs_policy"] = {k: paired(a, "policy", k) for k in ("topple", "v4_success", "v4b_success")}
(ROOT / "results/jenga/stage4_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
