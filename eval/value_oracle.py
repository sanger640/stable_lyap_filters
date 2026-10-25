"""6a stage A0 (docs/FILTER_DEEP_DIVE.md 6a; results/jenga/vo_protocol.json): the label-free ORACLE of the relative
outcome spread, from the exact simulator. For a saved decision state: N noisy futures (the monitor's perception error
model on the block poses + the measured execution-noise snippets x EXEC_NOISE_SCALE, paired draws) of the policy's
8-step chunk then 30 holds, and of holding still (target, gripper 0, as the trigger's hold); total variance of the final
block poses (positions in cm, rotation-matrix entries x 10); y = log tvar(plan) - log tvar(hold). No failure labels.
Decisions = those of eval/margin_check.py rows of a set (MC_RUN / VO_ROWS), so it pairs with the world model's rules.
usage: VO_SET=scl TASK=i/n python3 eval/value_oracle.py run -> results/jenga/deep_dive/vo_<set>/rows_<i>.jsonl
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
N_FUT, SEED, EPS = int(os.environ.get("VO_N", 16)), 20261029, 1e-4
SET = os.environ.get("VO_SET", "scl")
RUN = ROOT / "results/program_b/policy_rollouts" / f"paired_record_{SET}"
ROWS = ROOT / "results/jenga/deep_dive" / os.environ.get("VO_ROWS", f"hmf_{SET}")
LOOK, VSCALE = int(os.environ.get("VO_LOOK", 8)), float(os.environ.get("VO_SCALE", 0))   # matched variant: 15 steps, 4x
OUT = ROOT / "results/jenga/deep_dive" / (f"vo_{SET}" if LOOK == 8 and not VSCALE else f"vo_{SET}_l{LOOK}_s{VSCALE:g}")


def feats(state):
    s = np.asarray(state, np.float64)
    return np.concatenate([s[:9] * 100.0, s[9:27] * 10.0])


def tvar(F):
    F = np.asarray(F); return float(((F - F.mean(0)) ** 2).sum(1).mean())


def run():
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import inspector_data as insp
    si, sn = (int(x) for x in os.environ.get("TASK", "0/1").split("/"))
    keys = sorted({(r["ep"], r["k"]) for f in sorted(ROWS.glob("margin_rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())})
    OUT.mkdir(parents=True, exist_ok=True); fout = OUT / f"rows_{si}.jsonl"
    done = {(r["ep"], r["k"]) for r in map(json.loads, fout.read_text().splitlines())} if fout.exists() else set()
    sim = insp._sim(); errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    lim = int(os.environ.get("LIMIT", 10 ** 9)); cache = {}
    for n, (ep, k) in enumerate(keys):
        if n % sn != si or (ep, k) in done:
            continue
        if lim <= 0:
            break
        lim -= 1
        if ep not in cache:
            with np.load(RUN / f"ep{ep}.npz") as z:
                cache = {ep: {q: z[q] for q in z.files}}
        S = cache[ep]; d = int(np.flatnonzero(S["chunk_index"] == k)[0])
        snap = (S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d]))
        plan = np.asarray(S["plan16"][d][1:1 + LOOK], np.float32).copy()          # the policy's next LOOK commands
        hold = np.repeat(np.r_[S["target"][d], 0.0][None], LOOK, 0).astype(np.float32)
        sc_e = VSCALE or insp.EXEC_SCALE; sc_p = VSCALE or 1.0                       # matched: both errors x VSCALE (as the WM rule)
        out = {}
        for name, ch in (("plan", plan), ("hold", hold)):
            F = []
            for j in range(N_FUT):
                nz = np.concatenate([snippets[j % 64], snippets[(j + 1) % 64]])[:LOOK] * sc_e
                pos, rot = errors.sample(SEED, ep, k, j)
                F.append(feats(um.rollout(sim, snap, ch, nz, [(pos * sc_p, rot * sc_p)])[-1]))
            out[name] = tvar(F)
        row = {"ep": ep, "k": k, "tvar_plan": out["plan"], "tvar_hold": out["hold"],
               "y": float(np.log(out["plan"] + EPS) - np.log(out["hold"] + EPS))}
        with open(fout, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    {"run": run}[sys.argv[1]]()
