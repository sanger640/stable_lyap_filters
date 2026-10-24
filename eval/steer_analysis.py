"""G-steer closed-loop analysis as pre-registered (results/jenga/stl_protocol.json), written before any episode was seen.
Primary: cbfgst3m vs cbfg3m paired by episode: (1) fewer time-outs (clean non-extraction), exact McNemar one-sided,
alpha 0.05; (2) neighbour topples non-inferior: one-sided 95% upper bound of the paired topple-rate difference <= +2 pp.
usage: [CL_TAG=stl] python3 eval/steer_analysis.py  (after outcome_audit.py on paired_{record,cbfg3m,cbfgst3m}_stl)
Also the gated-trigger test (results/jenga/gtv_protocol.json): CL_TAG=gtv CL_ARMS="cbfg3m cbfgtv3m" (no policy arm;
the new arm is the last one, the reference is cbfg3m; same primary rule)."""
import json
from math import comb
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAG = os.environ.get("CL_TAG", "stl")
ARMS = tuple(os.environ.get("CL_ARMS", "record cbfg3m cbfgst3m").split()); NEW = ARMS[-1]
A = json.loads((ROOT / f"results/jenga/outcome_audit_{TAG}.json").read_text())
V = {a: {r["ep"]: r for r in A[f"paired_{a}_{TAG}"]["rows"]} for a in ARMS}
EPS = sorted(set.intersection(*(set(v) for v in V.values())))
rng = np.random.default_rng(20261024)


def vec(a, key):
    def f(r):
        if key == "topple":
            return r["logged_neighbour_toppled"]
        if key == "timeout":
            return not r["logged_neighbour_toppled"] and not r["extracted"] and not r["red_fell_before_extraction"]
        if key == "failure":
            return r["logged_neighbour_toppled"] or r["red_fell_before_extraction"] or r["red_fell_after_extraction"]
        return r[key]
    return np.array([f(V[a][e]) for e in EPS], float)


def mcnemar_less(a, b, key):                               # one-sided: a has fewer events than b
    x, y = vec(a, key), vec(b, key); n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = n10 + n01
    return {"a_only": n10, "b_only": n01, "p_one_sided": float(f"{sum(comb(n, i) for i in range(n10 + 1)) / 2 ** n:.3g}") if n else 1.0}


def boot(a, b, key):
    d = vec(a, key) - vec(b, key); bs = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(10000)])
    return {"diff_pp": round(100 * d.mean(), 2), "upper95_one_sided_pp": round(100 * float(np.percentile(bs, 95)), 2),
            "lower95_one_sided_pp": round(100 * float(np.percentile(bs, 5)), 2)}


def ivstats(a):
    n_iv, sec = 0, []
    for e in EPS:
        L = json.loads((ROOT / f"results/program_b/policy_rollouts/paired_{a}_{TAG}/ep{e}.json").read_text())["log"]
        iv = [c for c in L if c.get("intervened")]; n_iv += len(iv); sec += [c["seconds"] for c in iv]
    return {"interventions_per_episode": round(n_iv / len(EPS), 2), "seconds_per_intervention_median": float(np.median(sec)) if sec else None}


KEYS = ("topple", "timeout", "failure", "red_fell_before_extraction", "red_fell_after_extraction", "v4_success", "v4b_success")
out = {"episodes": len(EPS), "counts": {a: {k: int(vec(a, k).sum()) for k in KEYS} for a in ARMS}}
p1 = out["PRIMARY_1_fewer_timeouts"] = mcnemar_less(NEW, "cbfg3m", "timeout")
p2 = out["PRIMARY_2_topples_noninferior"] = boot(NEW, "cbfg3m", "topple")
out["PRIMARY_pass"] = bool(p1["p_one_sided"] < 0.05 and p2["upper95_one_sided_pp"] <= 2.0)
out["secondary"] = {
    ("v4_steer_minus_g3m" if NEW == "cbfgst3m" else "v4_new_minus_g3m"): boot(NEW, "cbfg3m", "v4_success"),
    ("v4b_steer_minus_g3m" if NEW == "cbfgst3m" else "v4b_new_minus_g3m"): boot(NEW, "cbfg3m", "v4b_success"),
    ("failures_steer_fewer" if NEW == "cbfgst3m" else "failures_new_fewer"): mcnemar_less(NEW, "cbfg3m", "failure"),
    **{f"{a}_fewer_topples_than_policy": mcnemar_less(a, "record", "topple") for a in ARMS[1:] if "record" in ARMS},
    **{f"{a}_interventions": ivstats(a) for a in ARMS if a != "record"}}
(ROOT / f"results/jenga/{TAG}_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
