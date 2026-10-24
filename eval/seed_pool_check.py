"""Exploratory: is the fork trigger noise-limited, and does pooling independent fork tests (seeds) warn earlier at the
same alarm rate? Rows: eval/margin_check.py on policy-alone episodes (seed 0 = results/jenga/deep_dive/margin_rows_*,
seeds 1-3 = deep_dive/seed{1,2,3}/, MC_SEED 20261101-3, compiled WM). h = log2 of the first splitting scale (3 = none).
Rules (each also fires on o > O_MAX of seed 0's draw / the mean o): relative 3x as the adopted filter
(h_plan < min(h_hold, 2)); pooled: mean_K h_plan < min(mean_K h_hold, tau), tau chosen on safe episodes to match the
single-seed alarm rate. Metric as margin_check / HJ stage B: topples warned within 8 chunks, share warned >= 3 ahead.
usage: python3 eval/seed_pool_check.py [dir1 dir2 ...]  (default seed1 seed2 seed3; MC_RUN / MC_AUDIT for other sets)
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "results/jenga/deep_dive"
SC = (0.5, 1.0, 2.0, 3.0, 4.0)
RUN = ROOT / "results/program_b/policy_rollouts" / os.environ.get("MC_RUN", "paired_record_ocl")
AUDIT = ROOT / os.environ.get("MC_AUDIT", "results/jenga/outcome_audit_ocl.json")
BASE = Path(os.environ.get("MC_BASE", str(D)))
OUT = Path(os.environ.get("MC_RESULT", str(D / "seed_pool_check.json")))


def h_of(flags):
    i = next((j for j, f in enumerate(flags) if f), None)
    return 3.0 if i is None else float(np.log2(SC[i]))


def load(folder):
    return {(r["ep"], r["k"]): r for f in sorted(folder.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}


def main():
    O_MAX = json.loads((ROOT / "results/jenga/cbf_calibration_OIL.json").read_text())["O_MAX"]
    dirs = [BASE] + [D / d for d in (sys.argv[1:] or ["seed1", "seed2", "seed3"])]
    S = [load(d) for d in dirs]
    keys = sorted(set.intersection(*[set(s) for s in S]))
    A = json.loads(AUDIT.read_text())[RUN.name]["rows"]
    topple = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    kt = {e: next((c["chunk"] for c in json.loads((RUN / f"ep{e}.json").read_text())["log"] if c.get("topple_started")), None)
          for e in {k[0] for k in keys}}
    hp = np.array([[h_of(s[k]["plan"]) for s in S] for k in keys]); hh = np.array([[h_of(s[k]["hold"]) for s in S] for k in keys])
    o = np.array([[s[k]["o"] for s in S] for k in keys])
    safe = np.array([not topple.get(k[0]) for k in keys])
    K = len(S)

    def evaluate(fire):
        fire = np.asarray(fire, bool); by = {}
        for (e, k), f in zip(keys, fire):
            by.setdefault(e, {})[k] = f
        leads, n_t = [], 0
        for e, ks in by.items():
            if topple.get(e) and kt.get(e) is not None:
                n_t += 1
                hits = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and ks[k]]
                if hits:
                    leads.append(kt[e] - hits[0])
        return {"alarm_rate_safe": round(float(fire[safe].mean()), 4), "topple_episodes": n_t, "warned_within_8": len(leads),
                "lead_ge3_share_of_topples": round(sum(x >= 3 for x in leads) / max(n_t, 1), 3),
                "lead_median": float(np.median(leads)) if leads else None}

    single = lambda s: (hp[:, s] < np.minimum(hh[:, s], 2.0) - 1e-9) | (o[:, s] > O_MAX)
    out = {"decisions": len(keys), "seeds": K, "safe_decisions": int(safe.sum()), "exploratory": True,
           "note": "seed 0 eager WM, seeds 1+ compiled (equivalence gate: h identical in 99.6%)"}
    T = np.stack([single(s) for s in range(K)], 1)
    pa = [(a, b) for a in range(K) for b in range(a + 1, K)]
    agree = np.mean([(T[:, a] == T[:, b]).mean() for a, b in pa]); p = T.mean()
    out["trigger_retest"] = {"agreement": round(float(agree), 3), "kappa": round(float((agree - (p * p + (1 - p) ** 2)) / (1 - (p * p + (1 - p) ** 2))), 3),
                             "rate": round(float(p), 4), "fires_k_of_K": [int((T.sum(1) == j).sum()) for j in range(K + 1)]}
    out["single_seed_3x"] = [evaluate(T[:, s]) for s in range(K)]
    rate = float(np.mean([T[safe, s].mean() for s in range(K)]))
    out["single_seed_mean"] = {k: round(float(np.mean([r[k] for r in out["single_seed_3x"]])), 3) for k in ("alarm_rate_safe", "lead_ge3_share_of_topples")}
    mh, mhh, mo = hp.mean(1), hh.mean(1), o.mean(1)
    best = None
    for tau in np.round(np.arange(1.0, 3.001, 0.0625), 4):
        fire = (mh < np.minimum(mhh, tau) - 1e-9) | (mo > O_MAX)
        r = fire[safe].mean()
        if best is None or abs(r - rate) < abs(best[1] - rate):
            best = (tau, r, fire)
    out["pooled_mean_h_matched"] = {"tau": float(best[0]), **evaluate(best[2])}
    out["pooled_mean_h_tau2"] = evaluate((mh < np.minimum(mhh, 2.0) - 1e-9) | (mo > O_MAX))
    for j in range(1, K + 1):
        out[f"vote_{j}_of_{K}"] = evaluate(T.sum(1) >= j)
    OUT.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
