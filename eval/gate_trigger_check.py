"""Exploratory (2026-10-24, after fixcf: time-out cost scales with the NUMBER of fixes): can the better monitor gate the
filter's trigger? Rule AND_q: fix only if the fork_3x trigger fires AND the relative total variance (tvar30_rel_s4.0)
exceeds its q-quantile on safe-episode decisions (deployment would use all decisions; topple decisions are rare).
With the fork certificate unchanged this equals "trigger = tvar, certificate = fork". Reports safe alarm rate (~ interventions)
and topple episodes warned >= 3 chunks ahead, vs fork_3x alone.
usage: MC_RUN=paired_record_<x> MC_TAG=_<x> python3 eval/gate_trigger_check.py  -> deep_dive/gate_trigger<TAG>.json"""
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


def main():
    D = ROOT / "results/jenga/deep_dive"
    rows = {(r["ep"], r["k"]): r for r in map(json.loads, (D / f"spread_rows{mc.TAG}.jsonl").read_text().splitlines())}
    M = {(r["ep"], r["k"]): r for f in sorted(mc.DT.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    keys = sorted(set(rows) & set(M))
    A = json.loads(mc.AUDIT[0].read_text())[mc.AUDIT[1]]["rows"]; top = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    kt = {e: next((c["chunk"] for c in json.loads((mc.RUN / f"ep{e}.json").read_text())["log"] if c.get("topple_started")), None) for e in {k[0] for k in keys}}
    safe = np.array([not top.get(k[0]) for k in keys]); OM = json.loads((ROOT / os.environ["CBF_CALIB"]).read_text())["O_MAX"]
    hp = np.array([sp.h_of(M[k]["plan"]) for k in keys]); hh = np.array([sp.h_of(M[k]["hold"]) for k in keys]); o = np.array([M[k]["o"] for k in keys])
    fork = (hp < np.minimum(hh, 2.0) - 1e-9) | (o > OM)
    tv = np.log(np.array([rows[k]["stats"]["4.0"]["tvar30"][0] for k in keys]) + 1e-12) - np.log(np.array([rows[k]["stats"]["4.0"]["tvar30"][1] for k in keys]) + 1e-12)

    def ev(f):
        by = {}
        for (e, k), x in zip(keys, f):
            by.setdefault(e, {})[k] = bool(x)
        lead, w8, last3 = [], [], []
        for e, ks in by.items():
            if top.get(e) and kt.get(e) is not None:
                h = [k for k in sorted(ks) if kt[e] - 8 <= k <= kt[e] and ks[k]]
                lead.append(bool(h) and kt[e] - h[0] >= 3); w8.append(bool(h))
                last3 += [ks[k] for k in ks if kt[e] - 3 <= k <= kt[e]]
        return {"alarm_rate_safe": round(float(f[safe].mean()), 4), "lead_ge3": round(float(np.mean(lead)), 3),
                "warned_within_8": round(float(np.mean(w8)), 3), "alarm_rate_last3": round(float(np.mean(last3)), 3), "topple_episodes": len(lead)}
    out = {"set": mc.RUN.name, "decisions": len(keys), "fork_3x": ev(fork)}
    for q in (0.5, 0.6, 0.7, 0.8, 0.9):
        out[f"AND_q{q}"] = ev(fork & (tv > np.quantile(tv[safe], q)))
    if os.environ.get("GT_TAU"):                                      # a fixed threshold (the pre-registered one)
        out["AND_tau"] = ev(fork & (tv > float(os.environ["GT_TAU"])))
    out["tv_safe_quantiles"] = {str(q): round(float(np.quantile(tv[safe], q)), 4) for q in (0.5, 0.7, 0.8, 0.9)}
    np.save(D / f"gate_tv_safe{mc.TAG}.npy", tv[safe])
    (D / f"gate_trigger{mc.TAG}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))


if __name__ == "__main__":
    main()
