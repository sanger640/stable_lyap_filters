"""Primary analysis of results/jenga/spread_protocol.json: relative spread (HOLD30, x3) vs the fork test at matched alarm
rates (fork_2x / 3x / 4x relative rules, each with the o barrier as in the filter). usage: MC_RUN=... MC_TAG=... python3 eval/spread_gate.py"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
import margin_check as mc  # noqa: E402
import seed_pool_check as sp  # noqa: E402

RULES = {"spread30_rel_s3.0": (3.0, "spread30", True), "tvar30_rel_s4.0": (4.0, "tvar30", True), "spread10_plan_s0.5": (0.5, "spread10", False)}


def main():
    D = ROOT / "results/jenga/deep_dive"
    rows = {(r["ep"], r["k"]): r for r in map(json.loads, (D / f"spread_rows{mc.TAG}.jsonl").read_text().splitlines())}
    M = {(r["ep"], r["k"]): r for f in sorted(mc.DT.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    keys = sorted(set(rows) & set(M))
    A = json.loads(mc.AUDIT[0].read_text())[mc.AUDIT[1]]["rows"]
    top = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    kt = {e: next((c["chunk"] for c in json.loads((mc.RUN / f"ep{e}.json").read_text())["log"] if c.get("topple_started")), None) for e in {k[0] for k in keys}}
    safe = np.array([not top.get(k[0]) for k in keys]); OM = json.loads((ROOT / os.environ["CBF_CALIB"]).read_text())["O_MAX"]

    def ev(f):
        by = {}
        for (e, k), x in zip(keys, f):
            by.setdefault(e, {})[k] = bool(x)
        per = {}
        for e, ks in by.items():
            if top.get(e) and kt.get(e) is not None:
                h = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and ks[k]]
                per[e] = (bool(h) and kt[e] - h[0] >= 3, bool(h))
        return float(np.mean(np.asarray(f)[safe])), per

    hp = np.array([sp.h_of(M[k]["plan"]) for k in keys]); hh = np.array([sp.h_of(M[k]["hold"]) for k in keys]); o = np.array([M[k]["o"] for k in keys])
    fork = {"fork_2x": (hp < np.minimum(hh, np.log2(3.0)) - 1e-9) | (o > OM), "fork_3x": (hp < np.minimum(hh, 2.0) - 1e-9) | (o > OM),
            "fork_4x": (hp < np.minimum(hh, 3.0) - 1e-9) | (o > OM)}
    out = {"decisions": len(keys), "topple_episodes": None, "exploratory": mc.TAG != "_m3"}
    rng = np.random.default_rng(20261124)
    for fname, ff in fork.items():
        rate, pf = ev(ff); E = sorted(pf); out["topple_episodes"] = len(E)
        res = {"alarm_rate_safe": round(rate, 4), "fork_lead_ge3": round(float(np.mean([pf[e][0] for e in E])), 3)}
        for rname, (s, stat, rel) in RULES.items():
            P = np.log(np.array([rows[k]["stats"][str(s)][stat][0] for k in keys]) + 1e-12)
            H = np.log(np.array([rows[k]["stats"][str(s)][stat][1] for k in keys]) + 1e-12)
            sc = P - H if rel else P
            r2, ps = ev(sc > np.quantile(sc[safe], 1 - rate))
            x = np.array([ps[e][0] for e in E], float) - np.array([pf[e][0] for e in E], float)
            bs = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)])
            res[rname] = {"alarm_rate_safe": round(r2, 4), "lead_ge3": round(float(np.mean([ps[e][0] for e in E])), 3),
                          "warned_within_8": int(sum(ps[e][1] for e in E)), "minus_fork_pp": round(100 * float(x.mean()), 1),
                          "lower95_one_sided_pp": round(100 * float(np.percentile(bs, 5)), 1), "spread_only": int((x > 0).sum()), "fork_only": int((x < 0).sum())}
        out[fname + "_rate"] = res
    p = out["fork_3x_rate"]["spread30_rel_s3.0"]
    out["PRIMARY_pass"] = bool(p["lower95_one_sided_pp"] > 0)
    (D / f"spread_gate{mc.TAG}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
