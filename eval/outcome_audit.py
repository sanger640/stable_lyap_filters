"""Outcome audit (2026-10-16, user report: the red block toppled in both filter videos but only one was counted a
failure). Replays every saved dev10 episode step by step (restoring each decision's saved state, executing the logged
chunk with the run's noise) and tracks the RED (target) block too: its lift above rest and its tilt, plus the neighbours'
tilt. Checks the replay reproduces the logged max lift / neighbour topple, and recounts success under stricter rules.

usage: python3 eval/outcome_audit.py [tag ...]   (default dev10) -> results/jenga/outcome_audit_<tag>.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RUNS = ROOT / "results/program_b/policy_rollouts"
ARMS = ("record", "cbfmoil", "cbfmsoil")


def replay(sim, S, noises):
    """-> per-step arrays: target lift (m, vs rest height at the first saved decision), target tilt, max neighbour tilt."""
    lift, ttilt, ntilt = [], [], []
    for d in range(len(S["chunk_index"])):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        if d == 0:
            z0 = float(sim.data.xpos[sim.tracked_block_ids[0]][2])
        k = int(S["chunk_index"][d])
        for j, a in enumerate(S["executed"][d]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]; sim.execute(a)
            t = sim.block_diagnostics()["tilt"]
            lift.append(float(sim.data.xpos[sim.tracked_block_ids[0]][2] - z0)); ttilt.append(float(t[0])); ntilt.append(float(np.max(t[1:])))
    return np.array(lift), np.array(ttilt), np.array(ntilt)


def main(tag="dev10"):
    import inspector_data as insp
    import plausibility as pl
    sim = insp._sim(); out = {}
    for arm in ARMS:
        res = {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/paired_{arm}_{tag}.jsonl").read_text().splitlines()}
        rows = []
        for ep, r in sorted(res.items()):
            with np.load(RUNS / f"paired_{arm}_{tag}" / f"ep{ep:03d}.npz") as f:
                S = {k: f[k] for k in f.files}
            L, TT, NT = replay(sim, S, pl.episode_noises(ep, int(S["chunk_index"][-1]) + 1))
            i3 = int(np.argmax(L >= 0.03)) if (L >= 0.03).any() else None
            rows.append({"ep": ep, "logged_success_3cm": r.get("seconds_to_3cm") is not None and not r["neighbour_toppled"],
                         "logged_neighbour_toppled": r["neighbour_toppled"], "logged_max_lift_mm": r["max_lift_mm"],
                         "replay_max_lift_mm": round(float(L.max() * 1000), 1), "replay_neighbour_toppled": bool(NT.max() >= 45),
                         "target_max_tilt": round(float(TT.max()), 1), "target_tilt_at_3cm": round(float(TT[i3]), 1) if i3 is not None else None,
                         "target_toppled_before_3cm": bool((TT[:i3 if i3 is not None else len(TT)] >= 45).any()),
                         "end_lift_mm": round(float(L[-1] * 1000), 1), "end_target_tilt": round(float(TT[-1]), 1)})
            print(arm, ep, flush=True)
        n = len(rows); R = rows
        top = lambda x: x["logged_neighbour_toppled"]
        clean = lambda x: (not top(x)) and x["target_tilt_at_3cm"] is not None and x["target_tilt_at_3cm"] < 45 and not x["target_toppled_before_3cm"]
        out[arm] = {"episodes": n,
                    "replay_matches_log": sum(abs(x["replay_max_lift_mm"] - x["logged_max_lift_mm"]) < 2 and x["replay_neighbour_toppled"] == top(x) for x in R),
                    "neighbour_topples": sum(top(x) for x in R),
                    "success_logged (3 cm, no neighbour topple)": sum(x["logged_success_3cm"] for x in R),
                    "of which red block toppled before reaching 3 cm": sum(x["logged_success_3cm"] and x["target_toppled_before_3cm"] for x in R),
                    "of which red block tilted >= 45 deg at 3 cm": sum(x["logged_success_3cm"] and (x["target_tilt_at_3cm"] or 0) >= 45 for x in R),
                    "clean_success (3 cm while red block < 45 deg, never toppled before, no neighbour topple)": sum(clean(x) for x in R),
                    "red block toppled at any time": sum(x["target_max_tilt"] >= 45 for x in R),
                    "red block toppled, no neighbour topple, never lifted 3 cm": sum(x["target_max_tilt"] >= 45 and not top(x) and x["target_tilt_at_3cm"] is None for x in R),
                    "rows": R}
    summ = {a: {k: v for k, v in out[a].items() if k != "rows"} for a in out}
    (ROOT / f"results/jenga/outcome_audit_{tag}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
