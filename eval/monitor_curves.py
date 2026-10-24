"""Held-out monitor comparison curves (paired_record_m3 policy-alone episodes): share of topples warned >= 3 chunks ahead
vs alarm rate on safe episodes, for the fork test (relative, 2x / 3x / 4x margins, + o barrier), relative spread (x3,
HOLD30), relative total variance (x4, HOLD30) and ensemble disagreement (d_all_plan). Exploratory summary of the
pre-registered gates (results/jenga/spread_protocol.json, disagree_protocol.json). -> results/jenga/deep_dive/monitor_curves_m3.json"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import seed_pool_check as sp  # noqa: E402

D = ROOT / "results/jenga/deep_dive"
RUN = ROOT / "results/program_b/policy_rollouts/paired_record_m3"


def main():
    M = {(r["ep"], r["k"]): r for f in sorted((D / "m3").glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    S = {(r["ep"], r["k"]): r for r in map(json.loads, (D / "spread_rows_m3.jsonl").read_text().splitlines())}
    G = {(r["ep"], r["k"]): r for r in map(json.loads, (D / "disagree_rows_m3.jsonl").read_text().splitlines())}
    keys = sorted(set(M) & set(S) & set(G))
    A = json.loads((ROOT / "results/jenga/outcome_audit_m3.json").read_text())["paired_record_m3"]["rows"]
    top = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    kt = {e: next((c["chunk"] for c in json.loads((RUN / f"ep{e}.json").read_text())["log"] if c.get("topple_started")), None) for e in {k[0] for k in keys}}
    safe = np.array([not top.get(k[0]) for k in keys]); OM = json.loads((ROOT / "results/jenga/cbf_calibration_OIL.json").read_text())["O_MAX"]

    def ev(f):
        by = {}
        for (e, k), x in zip(keys, f):
            by.setdefault(e, {})[k] = bool(x)
        L = []
        for e, ks in by.items():
            if top.get(e) and kt.get(e) is not None:
                h = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and ks[k]]
                L.append(bool(h) and kt[e] - h[0] >= 3)
        return [round(float(np.mean(f[safe])), 4), round(float(np.mean(L)), 3)]

    hp = np.array([sp.h_of(M[k]["plan"]) for k in keys]); hh = np.array([sp.h_of(M[k]["hold"]) for k in keys]); o = np.array([M[k]["o"] for k in keys])
    out = {"set": "paired_record_m3 (held-out)", "decisions": len(keys), "topple_episodes": int(sum(1 for e in kt if top.get(e) and kt[e] is not None)),
           "fork": {m: ev((hp < np.minimum(hh, t) - 1e-9) | (o > OM)) for m, t in (("2x", np.log2(3.0)), ("3x", 2.0), ("4x", 3.0))}}
    rel = lambda s, st: np.log(np.array([S[k]["stats"][str(s)][st][0] for k in keys]) + 1e-12) - np.log(np.array([S[k]["stats"][str(s)][st][1] for k in keys]) + 1e-12)
    scores = {"relative_spread_x3": rel(3.0, "spread30"), "relative_tvar_x4": rel(4.0, "tvar30"),
              "ensemble_disagreement": np.array([G[k]["d_all"][0] for k in keys])}
    rates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.25, 0.30]
    for name, sc in scores.items():
        out[name] = [ev(sc > np.quantile(sc[safe], 1 - r)) for r in rates]
    (D / "monitor_curves_m3.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))


if __name__ == "__main__":
    main()
