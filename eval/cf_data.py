"""Counterfactual world-model training data: the policy's plan AND nudged versions of it, from the same starts.

Why (2026-10-15 checks, results/program_b/cbf_qp/checks/report.json): real topple risk changes smoothly and strongly
with mm-scale plan changes (5 mm nudge: 23 points on average; 5 vs 10 mm in one direction r = 0.91), but the world
model's barrier barely sees it (r = 0.15). It was only ever trained on the plans the policy executed. Here every state
gets P perturbation draws x V plans (the policy plan + V-1 smooth random nudges), PAIRED: plan v under perturbation /
execution noise p uses exactly the same draws as every other plan, so the data shows what the nudge alone does.

States: dev recordings paired_record_dev2x (episodes 0-299; the offline CBF test uses episodes 2000-2149, untouched),
the shared closed-loop selection (near-topple states first); train e % 5 != 0, val e % 5 == 0. Horizon H15 (the
filter's look-ahead), 2x execution noise, corrected perception error. Same file format as closedloop_v1.

usage: CF_SHARD=0,1,2,3 CF_WORKERS=6 python3 eval/cf_data.py
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
import closedloop_data as cl  # noqa: E402
import lookahead_check as la  # noqa: E402

OUT = ROOT / "results/program_b/cf_v1"
SEED, N_TRAIN, N_VAL, P, V = 20261700, 1700, 200, 4, 3
KNOTS, SIGMA_MM, CLIP_MM = (1, 5, 10, 15), (3.0, 10.0), 15.0


def nudge(plan, rng):
    """Smooth random xyz offset: knot offsets ~ N(0, sigma) (z at half), linearly interpolated over steps 1..15."""
    sig = rng.uniform(*SIGMA_MM)
    k = np.clip(rng.normal(0, sig, (len(KNOTS), 3)) * [1, 1, 0.5], -CLIP_MM, CLIP_MM) / 1000
    p = np.asarray(plan, np.float32).copy()
    for t in range(1, 16):
        j = np.searchsorted(KNOTS, t)
        if j == 0 or KNOTS[min(j, len(KNOTS) - 1)] == t:
            off = k[min(j, len(KNOTS) - 1)]
        else:
            w = (t - KNOTS[j - 1]) / (KNOTS[j] - KNOTS[j - 1]); off = (1 - w) * k[j - 1] + w * k[j]
        p[t, :3] += off
    return p


def job(args):
    import traceback
    try:
        return args[0], _job(args), None
    except Exception:
        return args[0], None, traceback.format_exc()


def _job(args):
    (e, c), split, state, ctrl, target, gripper, tilt, plan = args
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
        sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="cf_"))))
        g["_W"] = (sim, mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent())
    sim, r, cam, latent = g["_W"]
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    rng = np.random.default_rng([SEED, e, c])
    plans = [np.asarray(plan, np.float32)] + [nudge(plan, rng) for _ in range(V - 1)]
    snap = (state, ctrl, target, float(gripper))
    sim.restore(snap); start = step_state(sim).copy()
    r.update_scene(sim.data, camera=cam); true_start = r.render().copy()
    starts, frames, traces, acts, pos, rot, which = [], [], [], [], [], [], []
    for p_ in range(P):
        j = int(np.random.default_rng([SEED, e, c, p_]).integers(64))
        for v, pl in enumerate(plans):
            sim.restore(snap); pr.perturb_blocks(sim, *errors.sample(SEED, e, c, p_))          # same draw for every plan
            pos.append(sim.last_perturbation[0]); rot.append(sim.last_perturbation[1])
            r.update_scene(sim.data, camera=cam); starts.append(r.render().copy())
            cmds = la.horizon_actions(pl, "H15", la.noise_for(snippets, j) * la.EXEC_SCALE)
            tr = []
            for t, a in enumerate(cmds):
                sim.execute(a); tr.append(step_state(sim))
                if t in sd.FRAME_INDEX:
                    r.update_scene(sim.data, camera=cam); frames.append(r.render().copy())
            traces.append(np.stack(tr)); acts.append(cmds); which.append((p_, v))
    K = len(acts)
    z = latent(np.concatenate([true_start[None], np.stack(starts), np.stack(frames)]))
    torch.cuda.empty_cache()
    np.save(path.with_name(path.stem + ".start.npy"), start)
    np.savez(path.with_suffix(".tmp.npz"), z0=z[0], z0k=z[1:1 + K], z=z[1 + K:].reshape(K, len(sd.FRAME_INDEX), 28, 28, 8),
             d_pos=np.stack(pos), d_rot=np.stack(rot), traces=np.stack(traces).astype(np.float32), actions=np.stack(acts),
             start_tilt=tilt, frame_index=np.array(sd.FRAME_INDEX), pair=np.array(which), plans=np.stack(plans))
    path.with_suffix(".tmp.npz").replace(path)
    return "ok"


def selection():
    sel = {tuple(map(int, k.split(":"))): v for k, v in json.loads((cl.OUT / "selection.json").read_text()).items()}
    near = [k for k, v in sel.items() if v[1] == 16]                    # states within 4 chunks of a real topple
    rest = [k for k, v in sel.items() if v[1] != 16]
    rng = np.random.default_rng(SEED)
    out = {}
    for split, cond, n in (("train", lambda e: e % 5 != 0, N_TRAIN), ("val", lambda e: e % 5 == 0, N_VAL)):
        a = [k for k in near if cond(k[0])]; b = [k for k in rest if cond(k[0])]
        b = [b[i] for i in rng.permutation(len(b))]
        for k in (a + b)[:n]:
            out[k] = split
    return out


def main(workers=int(os.environ.get("CF_WORKERS", 6))):
    for d in ("train", "val"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    shard = [int(x) for x in os.environ.get("CF_SHARD", "0,1,2,3,4,5,6").split(",")]
    sf = OUT / "selection.json"
    if sf.exists():
        sel = {tuple(map(int, k.split(":"))): v for k, v in json.loads(sf.read_text()).items()}
    else:
        sel = selection(); sf.write_text(json.dumps({f"{e}:{c}": v for (e, c), v in sel.items()}) + "\n")
    jobs, cache = [], {}
    for (e, c), split in sorted(sel.items()):
        if e % 7 not in shard:
            continue
        if e not in cache:
            cache.clear(); cache[e] = dict(np.load(cl.RUN / f"ep{e:03d}.npz"))
        S = cache[e]; i = int(np.flatnonzero(S["chunk_index"] == c)[0])
        jobs.append(((e, c), split, S["state"][i], S["ctrl"][i], S["target"][i], S["gripper"][i], S["start_tilt"][i], S["plan16"][i]))
    fails = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (key, res, err) in enumerate(pool.imap_unordered(job, jobs), 1):
            if err:
                fails.append(key); print(err, flush=True)
            if n % 100 == 0 or n == len(jobs):
                print(f"cf {n}/{len(jobs)}", flush=True)
    print("done", len(jobs), "failures", len(fails), flush=True)


if __name__ == "__main__":
    main()
