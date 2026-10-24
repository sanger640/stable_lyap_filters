"""Gate of results/jenga/hmf_protocol.json: hand-masked fork trigger vs the deployed fork_3x trigger, same futures.
usage: python3 eval/hmf_gate.py [scl|ocl]   (rows from eval/margin_check.py with MC_MASK=1 in deep_dive/hmf_<set>)"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import seed_pool_check as sp  # noqa: E402

RUN = ROOT / "results/program_b/policy_rollouts"


def main(name):
    D = ROOT / f"results/jenga/deep_dive/hmf_{name}"; run = f"paired_record_{name}"
    M = {(r["ep"], r["k"]): r for f in sorted(D.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    A = json.loads((ROOT / f"results/jenga/outcome_audit_{name}.json").read_text())[run]["rows"]
    top = {r["ep"]: r["logged_neighbour_toppled"] for r in A}
    om = json.loads((ROOT / "results/jenga/cbf_calibration_OIL.json").read_text())["O_MAX"]
    kt, kg = {}, {}
    for e in {k[0] for k in M}:
        kt[e] = next((c["chunk"] for c in json.loads((RUN / run / f"ep{e}.json").read_text())["log"] if c.get("topple_started")), None)
        with np.load(RUN / run / f"ep{e}.npz") as z:
            g = z["gripper"] < 55; kg[e] = int(z["chunk_index"][int(np.argmax(g))]) if g.any() else None
    def fire(r, m):
        hp, hh = sp.h_of(r["plan_m" if m else "plan"]), sp.h_of(r["hold_m" if m else "hold"])
        return hp < min(hh, 2.0) - 1e-9 or r["o"] > om
    out = {"decisions": len(M), "set": name}
    per = {}
    for m, key in ((False, "fork_3x"), (True, "hmf_3x")):
        safe = [(k, fire(r, m)) for k, r in M.items() if not top[k[0]]]
        near = [f for (e, k), f in safe if kg[e] is not None and kg[e] - 3 <= k < kg[e]]
        away = [f for (e, k), f in safe if not (kg[e] is not None and kg[e] - 3 <= k < kg[e])]
        hit = {}
        for e in {k[0] for k in M if top[k[0]] and kt.get(k[0]) is not None}:
            ks = [k for (ee, k) in M if ee == e and kt[e] - 8 <= k <= kt[e] and fire(M[(e, k)], m)]
            hit[e] = bool(ks) and kt[e] - min(ks) >= 3
        per[key] = hit
        out[key] = {"safe_alarm_rate": round(float(np.mean([f for _, f in safe])), 4), "near_grasp_safe_alarm_rate": round(float(np.mean(near)), 4),
                    "near_grasp_decisions": len(near), "away_safe_alarm_rate": round(float(np.mean(away)), 4),
                    "topple_episodes": len(hit), "lead_ge3_share": round(float(np.mean(list(hit.values()))), 3)}
    E = sorted(per["fork_3x"]); x = np.array([per["hmf_3x"][e] for e in E], float) - np.array([per["fork_3x"][e] for e in E], float)
    rng = np.random.default_rng(20261028); bs = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)])
    out["lead_diff_pp"] = round(100 * float(x.mean()), 1); out["lead_diff_lower95_pp"] = round(100 * float(np.percentile(bs, 5)), 1)
    out["near_grasp_ratio"] = round(out["hmf_3x"]["near_grasp_safe_alarm_rate"] / max(out["fork_3x"]["near_grasp_safe_alarm_rate"], 1e-9), 3)
    out["PRIMARY_pass"] = bool(out["near_grasp_ratio"] <= 0.7 and out["lead_diff_lower95_pp"] > -5)
    (ROOT / f"results/jenga/hmf_result{'' if name == 'scl' else '_' + name}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scl")
