"""(2026-10-29, user spotted a red block dropping then reappearing in a deck video.) For every episode the H2H5 replay audit flags as 'red block fell before extraction': is the fall visible in the REAL run's
saved states (red tilt >= 45 at some saved decision state), or only inside a replayed chunk that diverged from the run?"""
import json, os, sys, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"
sys.path[:0] = ["eval", "src"]
import inspector_data as insp, plausibility as pl
nz = sys.argv[1]
A = json.load(open(f"results/jenga/outcome_audit_h5_{nz}.json"))
sim = insp._sim(); bid = sim.tracked_block_ids[0]; out = {}
for run, blk in A.items():
    flagged = [r["ep"] for r in blk["rows"] if r["red_fell_before_extraction"]]
    real, diverged = 0, 0
    for ep in flagged:
        with np.load(f"results/program_b/policy_rollouts/{run}/ep{ep}.npz") as f: S = {k: f[k] for k in f.files}
        st = []
        for d in range(len(S["chunk_index"])):
            sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d]))); st.append(float(sim.block_diagnostics()["tilt"][0]))
        if max(st) >= 45: real += 1
        else: diverged += 1
    out[run] = {"flagged_red_fall_before_extraction": len(flagged), "fall_seen_in_saved_states": real, "only_in_replay": diverged}
    print(run, out[run], flush=True)
json.dump(out, open(f"{os.environ.get('OUTDIR', '.')}/redfall_check_{nz}.json", "w"), indent=1)
