"""Head-to-head 4 analysis as pre-registered (results/jenga/lsf_h2h4_protocol.json): the label-free arms run locally
(outcome_audit_h2h4.json: record, lfci000, lfc000) against head-to-head 3's Fir arms on the same episodes
(outcome_audit_h2h3.json: record, cbfps3m = psel, lsfa030 = labelled LSF stage 2). Same statistics as eval/h2h_analysis.py
(paired by episode, normal approximation, one-sided 95%, exact McNemar).
usage: python3 eval/h2h4_analysis.py  -> results/jenga/lsf_h2h4_result.json"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = {"policy": ("h2h4", "record"), "lfci000": ("h2h4", "lfci000"), "lfc000": ("h2h4", "lfc000"),
       "policy_fir": ("h2h3", "record"), "psel": ("h2h3", "cbfps3m"), "LSF": ("h2h3", "lsfa030")}
A = {s: json.loads((ROOT / f"results/jenga/outcome_audit_{s}.json").read_text()) for s in ("h2h3", "h2h4")}
V = {a: {r["ep"]: r for r in A[s][f"paired_{run}_{s}"]["rows"]} for a, (s, run) in SRC.items()}
EPS = sorted(set.intersection(*(set(v) for v in V.values())))
KEYS = ("topple", "v4_success", "v4b_success", "timeout")


def vec(a, key):
    if key == "topple":
        return np.array([V[a][e]["logged_neighbour_toppled"] for e in EPS], float)
    if key == "timeout":
        return np.array([not r["extracted"] and not r["logged_neighbour_toppled"] and not r["red_fell_before_extraction"]
                         for r in (V[a][e] for e in EPS)], float)
    return np.array([V[a][e][key] for e in EPS], float)


def paired(a, b, key):
    d = vec(a, key) - vec(b, key); se = d.std(ddof=1) / np.sqrt(len(d))
    n10, n01 = int(((vec(a, key) == 1) & (vec(b, key) == 0)).sum()), int(((vec(a, key) == 0) & (vec(b, key) == 1)).sum())
    n = n10 + n01; k = min(n10, n01)
    p2 = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0           # exact McNemar, two-sided
    p1 = sum(comb(n, i) for i in range(n10 + 1)) / 2 ** n if n else 1.0                       # one-sided: a has fewer
    return {"diff_pp": round(100 * d.mean(), 2), "lower95_pp": round(100 * (d.mean() - 1.645 * se), 2),
            "upper95_pp": round(100 * (d.mean() + 1.645 * se), 2), "a_only": n10, "b_only": n01,
            "mcnemar_p": float(f"{p2:.3g}"), "p_one_sided_a_fewer": float(f"{p1:.3g}")}


def noninf(a, b):
    r = {k: paired(a, b, k) for k in KEYS}
    r["NONINF"] = {f"success {s} lower > -5pp": bool(r[f"{s}_success"]["lower95_pp"] > -5) for s in ("v4", "v4b")}
    r["NONINF"]["topple upper < +3pp"] = bool(r["topple"]["upper95_pp"] < 3)
    r["NONINF"]["all"] = all(r["NONINF"].values())
    return r


def vs_policy(a, pol="policy"):
    r = {k: paired(a, pol, k) for k in KEYS}
    r["PASS"] = bool(r["topple"]["p_one_sided_a_fewer"] < 0.05 and r["v4_success"]["lower95_pp"] > 0)
    return r


out = {"stage": "h2h4", "episodes": len(EPS),
       "counts": {a: {k: int(vec(a, k).sum()) for k in KEYS} for a in SRC},
       "PRIMARY_P1_lfci000_vs_psel": noninf("lfci000", "psel"),
       "PRIMARY_P2_lfci000_vs_policy": vs_policy("lfci000"),
       "secondary": {"lfci000_vs_LSF": noninf("lfci000", "LSF"), "lfc000_vs_psel": noninf("lfc000", "psel"),
                     "lfc000_vs_LSF": noninf("lfc000", "LSF"), "lfc000_vs_policy": vs_policy("lfc000"),
                     "lfci000_vs_lfc000": {k: paired("lfci000", "lfc000", k) for k in KEYS},
                     "lfci000_vs_policy_fir": vs_policy("lfci000", "policy_fir"),
                     "lfc000_vs_policy_fir": vs_policy("lfc000", "policy_fir"),
                     "cross_machine_policy": {k: {"agree": int((vec("policy", k) == vec("policy_fir", k)).sum()),
                                                  "local_only": int(((vec("policy", k) == 1) & (vec("policy_fir", k) == 0)).sum()),
                                                  "fir_only": int(((vec("policy", k) == 0) & (vec("policy_fir", k) == 1)).sum())}
                                              for k in KEYS},
                     "red_falls": {a: {q: int(sum(bool(V[a][e][q]) for e in EPS)) for q in
                                       ("red_fell_before_extraction", "red_fell_after_extraction")} for a in SRC}}}
(ROOT / "results/jenga/lsf_h2h4_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
