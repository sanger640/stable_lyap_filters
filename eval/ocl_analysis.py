"""O-CEM closed-loop analysis as pre-registered (results/jenga/ocem_cl_protocol.json), written before any episode was seen.
Primary: cbfoa vs cbfg paired by episode: (1) fewer neighbour topples, exact McNemar one-sided, alpha 0.05; (2) success v4
one-sided 95% lower bound of (cbfoa - cbfg) > -5 pp (paired bootstrap). Adopt O-CEM only if both pass.
usage: python3 eval/ocl_analysis.py  (after outcome_audit.py on paired_{record,cbfg,cbfoa,cbfo}_ocl -> outcome_audit_ocl.json)"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
A = json.loads((ROOT / "results/jenga/outcome_audit_ocl.json").read_text())
ARMS = ("record", "cbfg", "cbfoa", "cbfo")
V = {a: {r["ep"]: r for r in A[f"paired_{a}_ocl"]["rows"]} for a in ARMS}
EPS = sorted(set.intersection(*(set(v) for v in V.values())))
rng = np.random.default_rng(20261080)


def vec(a, key):
    return np.array([V[a][e]["logged_neighbour_toppled" if key == "topple" else key] for e in EPS], float)


def mcnemar_less(a, b):                                   # one-sided: a topples less than b
    x, y = vec(a, "topple"), vec(b, "topple"); n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = n10 + n01
    return {"a_only": n10, "b_only": n01, "p_one_sided": float(f"{sum(comb(n, i) for i in range(n10 + 1)) / 2 ** n:.3g}") if n else 1.0}


def boot(a, b, key):
    d = vec(a, key) - vec(b, key); bs = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(10000)])
    return {"diff_pp": round(100 * d.mean(), 2), "lower95_one_sided_pp": round(100 * float(np.percentile(bs, 5)), 2),
            "ci95_pp": [round(100 * float(np.percentile(bs, 2.5)), 2), round(100 * float(np.percentile(bs, 97.5)), 2)]}


def ivstats(a):
    n_iv, n_dec, sec, unch = 0, 0, [], []
    for e in EPS:
        L = json.loads((ROOT / f"results/program_b/policy_rollouts/paired_{a}_ocl/ep{e}.json").read_text())["log"]
        n_dec += len(L); iv = [c for c in L if c.get("intervened")]; n_iv += len(iv); sec += [c["seconds"] for c in iv]
        unch += [bool(c.get("unchanged")) for c in iv if c.get("kind") == "o4"]
    return {"interventions_per_episode": round(n_iv / len(EPS), 2), "intervention_rate": round(n_iv / max(n_dec, 1), 4),
            "seconds_per_intervention_median": float(np.median(sec)) if sec else None,
            "o4_unchanged_share": round(float(np.mean(unch)), 3) if unch else None}


out = {"episodes": len(EPS), "counts": {a: {"topples": int(vec(a, "topple").sum()), "v4": int(vec(a, "v4_success").sum()),
                                            "v4b": int(vec(a, "v4b_success").sum())} for a in ARMS}}
out["PRIMARY_1_cbfoa_fewer_topples_than_cbfg"] = mcnemar_less("cbfoa", "cbfg")
out["PRIMARY_2_v4_cbfoa_minus_cbfg"] = boot("cbfoa", "cbfg", "v4_success")
out["PRIMARY_pass"] = bool(out["PRIMARY_1_cbfoa_fewer_topples_than_cbfg"]["p_one_sided"] < 0.05
                           and out["PRIMARY_2_v4_cbfoa_minus_cbfg"]["lower95_one_sided_pp"] > -5)
out["secondary"] = {
    "cbfo_fewer_topples_than_cbfg": mcnemar_less("cbfo", "cbfg"), "cbfoa_fewer_topples_than_cbfo": mcnemar_less("cbfoa", "cbfo"),
    **{f"{k}_{a}_minus_{b}": boot(a, b, k) for k in ("v4_success", "v4b_success") for a, b in
       (("cbfo", "cbfg"), ("cbfoa", "cbfo"), ("cbfoa", "cbfg"))},
    **{f"{a}_fewer_topples_than_policy": mcnemar_less(a, "record") for a in ARMS[1:]},
    **{f"v4_{a}_minus_policy": boot(a, "record", "v4_success") for a in ARMS[1:]},
    "interventions": {a: ivstats(a) for a in ARMS[1:]}}
(ROOT / "results/jenga/ocl_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
