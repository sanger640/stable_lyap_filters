"""Exploratory diagnosis: red-block falls caused under the always-on edit at 1x (episodes the policy extracts, the filter
knocks the red block over). Replays each episode from its saved decision states with the run's execution noise and finds
the first step the red block tilts > 45 deg; reports the gripper state then, time from the first gripper close, and the
edits (|executed - proposed|) in the 2 s before."""
import json, os, sys
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ["EXEC_NOISE_SCALE"] = "1"
ROOT = "/home/corey/wksp/stable_lyap_filters"; sys.path[:0] = [ROOT + "/eval", ROOT + "/src"]
import inspector_data as insp, uncertainty_monitor_dev as um

def rows(f, k): return {r["ep"]: r for r in json.load(open(f"{ROOT}/results/jenga/{f}"))[k]["rows"]}

def replay(run, ep, scale):
    S = dict(np.load(f"{ROOT}/results/program_b/policy_rollouts/{run}/ep{ep}.npz"))
    snip = np.load(um.BENCH)["snippets"]; nrng = np.random.default_rng([20261502, ep])
    noises = [snip[nrng.integers(len(snip))] * scale for _ in range(int(S["chunk_index"][-1]) + 1)]
    sim = insp._sim(); T = []
    for d in range(len(S["chunk_index"])):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d]))); k = int(S["chunk_index"][d])
        nom, exe = S["nominal"][d], S["executed"][d]
        for j, a in enumerate(exe):
            a = np.array(a, np.float32); a[:3] -= noises[k][j]; sim.execute(a)
            T.append(((k * 8 + j) * 0.1, float(sim.block_diagnostics()["tilt"][0]), bool(exe[j, 3] > 0), float(np.linalg.norm(exe[j, :3] - nom[j, :3]) * 1000),
                      float(sim.data.xpos[sim.tracked_block_ids[0]][2])))
    sim.close(); return T

P = rows("outcome_audit_n1dev.json", "paired_record_n1dev")
out = {}
for name, f, k, run in (("B", "outcome_audit_v1xdev.json", "paired_lfc000_v1xdev", "paired_lfc000_v1xdev"),
                        ("N1M", "outcome_audit_n1m.json", "paired_lfc000_n1mdev", "paired_lfc000_n1mdev")):
    V = rows(f, k); eps = [e for e in P if P[e]["v4_success"] and V[e]["red_fell_before_extraction"]]
    res = []
    for e in eps:
        T = replay(run, e, 1.0)
        i = next((n for n, t in enumerate(T) if t[1] > 45), None)
        if i is None:
            res.append({"ep": e, "fall": None}); continue
        closes = [n for n, t in enumerate(T) if t[2]]
        first_close = T[closes[0]][0] if closes else None
        win = [t for t in T if T[i][0] - 2.0 <= t[0] <= T[i][0]]
        res.append({"ep": e, "fall_t": round(T[i][0], 1), "grip_closed_at_fall": T[i][2], "first_close_t": first_close,
                    "edit_steps_2s_before": sum(t[3] > 0.5 for t in win), "max_edit_mm_2s_before": round(max(t[3] for t in win), 1),
                    "red_lift_mm_max_before": round((max(t[4] for t in T[:i + 1]) - T[0][4]) * 1000, 1)})
    out[name] = res; print(name, json.dumps(res), flush=True)
json.dump(out, open(ROOT + "/results/jenga/deep_dive/redfall.json", "w"), indent=1)
