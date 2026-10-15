"""World-model training data from the policy's own closed-loop states (overnight 2026-10-15).

The ceiling check showed the camera latent sees closed-loop forks (97.5% / 3.7%) but the world model, trained on
replayed demonstrations, predicts only 35%. This renders training futures from the saved decision states of the
development recordings (paired_record_dev2x; test episodes untouched), in the same format as the three-block training
data (z0, z0k, z (K, 19, ...), d_pos, d_rot = applied perception error, actions (K, 38, 4) executed commands, traces).

Selection (by episode: e % 5 != 0 -> train, e % 5 == 0 -> val): every decision in the 4 chunks before a real topple
(16 futures), plus random decisions (8 futures) up to N_TRAIN / N_VAL states. Futures alternate the monitor's horizon
(8 commands + 30 hold) and the filter's look-ahead (15 + 23), with the policy's plan, 2x execution noise (the run's
level) and the corrected perception error. Robot-still check: same perturbation code as all other corrected data.

usage: CL_SHARD=k/N python3 eval/closedloop_data.py  (episodes with e % 7 in the shard's set are processed locally)
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

TAG = "paired_record_dev2x"
RUN = ROOT / "results/program_b/policy_rollouts" / TAG
OUT = ROOT / "results/program_b/closedloop_v1"
SEED, N_TRAIN, N_VAL = 20261600, 2200, 300


def selection():
    eps = {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{TAG}.jsonl").read_text().splitlines()}
    near = set()
    for e, r in eps.items():
        for c in r["log"]:
            if c.get("topple_started"):
                near |= {(e, c["chunk"] - d) for d in range(4)}
    allk = [(e, c["chunk"]) for e, r in eps.items() for c in r["log"]]
    have = set()
    for p in RUN.glob("ep*.npz"):
        with np.load(p) as f:
            have |= {(int(p.stem[2:]), int(c)) for c in f["chunk_index"]}
    allk = [k for k in allk if k in have]
    rng = np.random.default_rng(SEED)
    sel = {}
    for split, cond, n in (("train", lambda e: e % 5 != 0, N_TRAIN), ("val", lambda e: e % 5 == 0, N_VAL)):
        nk = [k for k in allk if cond(k[0]) and k in near]
        rest = [k for k in allk if cond(k[0]) and k not in near]
        rest = [rest[i] for i in rng.permutation(len(rest))[:max(0, n - len(nk))]]
        for k in nk:
            sel[k] = (split, 16)
        for k in rest:
            sel[k] = (split, 8)
    return sel


def job(args):
    import traceback
    try:
        return args[0], _job(args), None
    except Exception:
        return args[0], None, traceback.format_exc()


def _job(args):
    (e, c), split, K, state, ctrl, target, gripper, tilt, plan = args
    import mujoco
    import torch
    import program_b_reference as pr
    import step3_data as sd
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from jenga_render_frames import close_camera
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    from jenga_state_data import step_state
    path = OUT / split / f"ep{e}_s{c}.npz"
    if path.exists():
        return "exists"
    g = globals()
    if "_W" not in g:
        sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="cl_"))))
        g["_W"] = (sim, mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent())
    sim, r, cam, latent = g["_W"]
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    snap = (state, ctrl, target, float(gripper))
    sim.restore(snap); start = step_state(sim).copy()
    r.update_scene(sim.data, camera=cam); true_start = r.render().copy()
    starts, frames, traces, acts, pos, rot = [], [], [], [], [], []
    for k in range(K):
        hz = "H8" if k % 2 == 0 else "H15"
        j = int(np.random.default_rng([SEED, e, c, k]).integers(64))
        sim.restore(snap); pr.perturb_blocks(sim, *errors.sample(SEED, e, c, k))
        pos.append(sim.last_perturbation[0]); rot.append(sim.last_perturbation[1])
        r.update_scene(sim.data, camera=cam); starts.append(r.render().copy())
        cmds = la.horizon_actions(plan, hz, la.noise_for(snippets, j) * la.EXEC_SCALE)
        tr = []
        for t, a in enumerate(cmds):
            sim.execute(a); tr.append(step_state(sim))
            if t in sd.FRAME_INDEX:
                r.update_scene(sim.data, camera=cam); frames.append(r.render().copy())
        traces.append(np.stack(tr)); acts.append(cmds)
    z = latent(np.concatenate([true_start[None], np.stack(starts), np.stack(frames)]))
    torch.cuda.empty_cache()
    np.save(path.with_name(path.stem + ".start.npy"), start)
    np.savez(path.with_suffix(".tmp.npz"), z0=z[0], z0k=z[1:1 + K], z=z[1 + K:].reshape(K, len(sd.FRAME_INDEX), 28, 28, 8),
             d_pos=np.stack(pos), d_rot=np.stack(rot), traces=np.stack(traces).astype(np.float32), actions=np.stack(acts),
             start_tilt=tilt, frame_index=np.array(sd.FRAME_INDEX))
    path.with_suffix(".tmp.npz").replace(path)
    return "ok"


def main(workers=int(os.environ.get("CL_WORKERS", 6))):
    for d in ("train", "val"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    shard = [int(x) for x in os.environ.get("CL_SHARD", "0,1,2,3,4,5,6").split(",")]
    sf = OUT / "selection.json"
    if sf.exists():                                   # one shared selection for both machines
        sel = {tuple(map(int, k.split(":"))): tuple(v) for k, v in json.loads(sf.read_text()).items()}
    else:
        sel = selection()
        sf.write_text(json.dumps({f"{e}:{c}": v for (e, c), v in sel.items()}) + "\n")
    jobs, cache = [], {}
    for (e, c), (split, K) in sorted(sel.items()):
        if e % 7 not in shard:
            continue
        if e not in cache:
            cache.clear(); cache[e] = dict(np.load(RUN / f"ep{e:03d}.npz"))
        S = cache[e]; i = int(np.flatnonzero(S["chunk_index"] == c)[0])
        jobs.append(((e, c), split, K, S["state"][i], S["ctrl"][i], S["target"][i], S["gripper"][i], S["start_tilt"][i], S["plan16"][i]))
    fails = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (key, res, err) in enumerate(pool.imap_unordered(job, jobs), 1):
            if err:
                fails.append(key); print(err, flush=True)
            if n % 100 == 0 or n == len(jobs):
                print(f"data {n}/{len(jobs)}", flush=True)
    print("done", len(jobs), "failures", len(fails), flush=True)


if __name__ == "__main__":
    main()
