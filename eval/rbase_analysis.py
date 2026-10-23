"""Random-hold baseline analysis as pre-registered (results/jenga/rbase_protocol.json).
usage: python3 eval/rbase_analysis.py  (after outcome_audit.py runs paired_{record,cbfh,cbfrh}_rb1 -> outcome_audit_rb1.json)"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
A = json.loads((ROOT / "results/jenga/outcome_audit_rb1.json").read_text())
ARMS = ("record", "cbfh", "cbfrh")
V = {a: {r["ep"]: r for r in A[f"paired_{a}_rb1"]["rows"]} for a in ARMS}
EPS = sorted(set.intersection(*(set(v) for v in V.values())))


def vec(a, key):
    return np.array([V[a][e]["logged_neighbour_toppled" if key == "topple" else key] for e in EPS], float)


def mcnemar_less(a, b):
    """One-sided exact McNemar: a has fewer topples than b."""
    x, y = vec(a, "topple"), vec(b, "topple"); n10 = int(((x == 1) & (y == 0)).sum()); n01 = int(((x == 0) & (y == 1)).sum())
    n = n10 + n01; p = sum(comb(n, i) for i in range(n10 + 1)) / 2 ** n if n else 1.0
    return {"a_only": n10, "b_only": n01, "p_one_sided": float(f"{p:.3g}")}


def boot(a, b, key, rng=np.random.default_rng(20261050)):
    d = vec(a, key) - vec(b, key); bs = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(10000)])
    return {"diff_pp": round(100 * d.mean(), 2), "ci95_pp": [round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2)]}


def rate(a):
    n = k = 0
    for e in EPS:
        log = json.loads((ROOT / f"results/program_b/policy_rollouts/paired_{a}_rb1/ep{e}.json").read_text())["log"]
        n += len(log); k += sum(bool(c.get("intervened")) for c in log)
    return round(k / max(n, 1), 4)


out = {"episodes": len(EPS), "counts": {a: {"topples": int(vec(a, "topple").sum()), "v4": int(vec(a, "v4_success").sum()),
                                            "v4b": int(vec(a, "v4b_success").sum())} for a in ARMS},
       "PRIMARY_cbfh_fewer_topples_than_cbfrh": mcnemar_less("cbfh", "cbfrh"),
       "cbfrh_fewer_topples_than_record": mcnemar_less("cbfrh", "record"),
       "cbfh_fewer_topples_than_record": mcnemar_less("cbfh", "record"),
       "v4_cbfh_minus_cbfrh": boot("cbfh", "cbfrh", "v4_success"), "v4b_cbfh_minus_cbfrh": boot("cbfh", "cbfrh", "v4b_success"),
       "intervention_rate": {a: rate(a) for a in ("cbfh", "cbfrh")}}
out["PRIMARY_pass"] = out["PRIMARY_cbfh_fewer_topples_than_cbfrh"]["p_one_sided"] < 0.05
(ROOT / "results/jenga/rbase_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
