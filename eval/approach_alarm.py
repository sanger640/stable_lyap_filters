"""Exploratory (docs/FILTER_DEEP_DIVE.md 5j): do the triggers fire on the grasp approach, where the 3x filter stalls?
Policy-alone episodes without a topple; approach window = the APPROACH decisions just before the policy first closes the
gripper. Fire rate inside vs outside the window for the fork 2x / 3x triggers and the deployed spread rule
(log spread_plan - log spread_hold > TAU_S, HOLD30 x3, or o > O_MAX). Fewer approach alarms = fewer stalls expected.
usage: MC_RUN=paired_record_m3 MC_TAG=_m3 python3 eval/approach_alarm.py
"""
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

APPROACH = 4


def main():
    D = ROOT / "results/jenga/deep_dive"
    rows = {(r["ep"], r["k"]): r for r in map(json.loads, (D / f"spread_rows{mc.TAG}.jsonl").read_text().splitlines())}
    M = {(r["ep"], r["k"]): r for f in sorted(mc.DT.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    keys = sorted(set(rows) & set(M))
    A = json.loads(mc.AUDIT[0].read_text())[mc.AUDIT[1]]["rows"]
    top = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    OM = json.loads((ROOT / os.environ["CBF_CALIB"]).read_text())["O_MAX"]
    TAU = json.loads((ROOT / "results/jenga/spread_calibration.json").read_text())["TAU_S"]
    kc = {}
    for e in {k[0] for k in keys}:
        with np.load(mc.RUN / f"ep{e}.npz") as z:
            g, ci = z["gripper"], z["chunk_index"]
        c = np.flatnonzero(g < 55)
        kc[e] = int(ci[c[0]]) if len(c) else None
    hp = np.array([sp.h_of(M[k]["plan"]) for k in keys]); hh = np.array([sp.h_of(M[k]["hold"]) for k in keys])
    o = np.array([M[k]["o"] for k in keys])
    r = np.log(np.array([rows[k]["stats"]["3.0"]["spread30"][0] for k in keys]) + 1e-12) - \
        np.log(np.array([rows[k]["stats"]["3.0"]["spread30"][1] for k in keys]) + 1e-12)
    rules = {"fork_2x": (hp < np.minimum(hh, np.log2(3.0)) - 1e-9) | (o > OM), "fork_3x": (hp < np.minimum(hh, 2.0) - 1e-9) | (o > OM),
             "spread_deployed": (r > TAU) | (o > OM)}
    safe = np.array([not top.get(k[0]) for k in keys])
    win = np.array([kc[k[0]] is not None and kc[k[0]] - APPROACH <= k[1] < kc[k[0]] for k in keys]) & safe
    pre = np.array([kc[k[0]] is not None and k[1] < kc[k[0]] - APPROACH for k in keys]) & safe
    out = {"set": mc.TAG or "_dev", "decisions_safe": int(safe.sum()), "approach_decisions": int(win.sum()),
           "earlier_decisions": int(pre.sum()), "approach_window_chunks": APPROACH, "TAU_S": TAU}
    for n, f in rules.items():
        out[n] = {"rate_all_safe": round(float(f[safe].mean()), 4), "rate_approach": round(float(f[win].mean()), 4),
                  "rate_earlier": round(float(f[pre].mean()), 4)}
    (D / f"approach_alarm{mc.TAG}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
