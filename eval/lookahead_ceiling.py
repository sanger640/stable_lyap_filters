"""Ceiling check for closed-loop decisions: is the camera latent able to show the forks at all?

For decisions from the development recordings (paired_record_dev2x) with known 15-step-look-ahead ground truth, render
the REAL 64 futures (same perception error, 2x execution noise, H15 commands) at hold 10 / hold 30 in the monitor's
close camera, encode them with the same DINOv2/PCA latent, and run the monitor's split test on the real latent futures.
Compare with the world model's imagined futures (lookahead_check latent stage) on the same decisions.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import lookahead_check as la  # noqa: E402

OUT = ROOT / "results/jenga/lookahead_ceiling.json"


def job(args):
    key, state, ctrl, target, gripper, plan = args
    import mujoco
    import torch
    import program_b_reference as pr
    import step3_data as sd
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from jenga_render_frames import close_camera
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    g = globals()
    if "_W" not in g:
        sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="ce_"))))
        g["_W"] = (sim, mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent())
    sim, r, cam, latent = g["_W"]
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    snap = (state, ctrl, target, float(gripper))
    frames = []
    for j in range(64):
        sim.restore(snap); pr.perturb_blocks(sim, *errors.sample(la.SEED, key[0], key[1], j))
        cmds = la.horizon_actions(plan, "H15", la.noise_for(snippets, j) * la.EXEC_SCALE)
        for t, a in enumerate(cmds):
            sim.execute(a)
            if t in sd.D0_INDEX:
                r.update_scene(sim.data, camera=cam); frames.append(r.render().copy())
    z = latent(np.stack(frames)).astype(np.float32).reshape(64, 2, -1)
    torch.cuda.empty_cache()
    from phase2_amend4 import alarm
    return key, bool(alarm(z[:, 0].astype(np.float64), z[:, 1].astype(np.float64), "BOTH"))


def main(workers=int(os.environ.get("CE_WORKERS", 6))):
    data = json.loads(la.OUT.read_text())
    T, L = data["truth"], data["latent"]
    keys = [k for k in T if k in L]
    risk = {k: T[k]["H15"]["topples"] / 64 for k in keys}
    rng = np.random.default_rng(20261590)
    forks = [k for k in keys if 0.1 <= risk[k] <= 0.9]; safe = [k for k in keys if risk[k] == 0]
    forks = [forks[i] for i in rng.permutation(len(forks))[:80]]; safe = [safe[i] for i in rng.permutation(len(safe))[:80]]
    jobs = []
    for k in forks + safe:
        e, c = map(int, k.split(":"))
        S, i = la.load_state((e, c))
        jobs.append(((e, c), S["state"][i], S["ctrl"][i], S["target"][i], S["gripper"][i], S["plan16"][i]))
    real = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (key, a) in enumerate(pool.imap_unordered(job, jobs), 1):
            real[f"{key[0]}:{key[1]}"] = a
            if n % 20 == 0:
                print(f"ceiling {n}/{len(jobs)}", flush=True)
    res = {"forks": len(forks), "safe": len(safe)}
    for name, f in (("real_latent_split", lambda k: real[k]), ("imagined_split_1x", lambda k: L[k]["H15"]["split"][1]),
                    ("imagined_barrier_le_0", lambda k: L[k]["H15"]["h"] <= 0)):
        c = sum(f(k) for k in forks) / len(forks); fa = sum(f(k) for k in safe) / len(safe)
        res[name] = {"caught": round(c, 3), "false_alarm": round(fa, 3), "score": round(c - fa, 3)}
    OUT.write_text(json.dumps({"summary": res, "real": real, "forks": forks, "safe": safe}, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
