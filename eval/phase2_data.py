"""Phase 2 training data (F1 rows, F2 stacks): screen, select, render.

Frozen protocol: results/jenga/phase2_generality_protocol.json (amendments 1-2). Phase 1f's image-split curation
failed its rule, so selection uses the pre-declared fallback: generic harm disagreement among 8 screen futures
(sim-only, used ONLY to choose training states). Rendering mirrors eval/phase1_data.py: encoded true start z0,
per-future perturbed start z0k, 19 latent frames per future, executed actions, object traces, per-object pose error.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_scenes as ps  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase2_generality_protocol.json"
OUT = ROOT / "results/program_b/phase2"
SEEDS = {"screen": 20261209, "select": 20261210, "futures": 20261211}
TRAIN_SEEDS = range(0, 1500)
PER_FAMILY, DISAGREE_CAP, K_SCREEN = 2400, 1200, 8
FUTURES = {"t16": 16, "t8": 8}
FRAME_INDEX = tuple(range(1, 38, 2))                      # as eval/step3_data.py (hold 10 = 17, hold 30 = 37)


def prepare(family, seed):
    """Built, reset and settled sim + actions + chunk starts, or None if the scene fails visibility/settling."""
    from jenga_short_held_tails import chunk_starts
    scene = ps.sample_scene(family, seed)
    sim = ps.make_sim(scene)
    sim.reset()
    ok, _, _ = ps.settle(sim, family)
    vis, alone = ps.visibility(sim)
    if not (ok and ps.visible_ok(sim, vis, alone)):
        return None
    acts = ps.primitive_actions(sim, scene)
    return scene, sim, acts, [s for s in chunk_starts(len(acts)) if s + 8 <= len(acts)]


def screen_job(job):
    import traceback
    family, seed = job
    done = OUT / "screen" / f"{family}_{seed}.json"
    try:
        if done.exists():
            return family, seed, json.loads(done.read_text()), None
        res = _screen(family, seed)
        tmp = done.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); tmp.replace(done)
        return family, seed, res, None
    except Exception:
        return family, seed, None, traceback.format_exc()


def _screen(family, seed):
    import uncertainty_monitor_dev as um
    prep = prepare(family, seed)
    if prep is None:
        return {"skipped": True}
    scene, sim, acts, starts = prep
    snippets = np.load(um.BENCH)["snippets"]
    pos_cov, rot_cov = ps.pooled_error_model()
    n = len(scene["objects"])
    out = {}
    for index, act in enumerate(acts):
        if index in starts:
            snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy()
            start = sim.object_state()
            harms = []
            for k in range(K_SCREEN):
                rng = np.random.default_rng([SEEDS["screen"], {"F1": 1, "F2": 2}[family], seed, index, k])
                harms.append(ps.harm(ps.future(sim, snap, acts[index:index + 8], snippets[rng.integers(len(snippets))],
                                               ps.draw_error(n, pos_cov, rot_cov, SEEDS["screen"], seed, index, k)), start))
            out[index] = int(sum(harms))
            sim.restore(snap); sim.data.qacc_warmstart[:] = warm
        sim.execute(act)
    return {"skipped": False, "n_objects": n, "harm_counts": out}


def select(screens):
    rng = np.random.default_rng(SEEDS["select"])
    roles, summary = {}, {}
    for family in ("F1", "F2"):
        rows = [(seed, int(i), c) for (f, seed), r in screens.items() if f == family and not r["skipped"]
                for i, c in r["harm_counts"].items()]
        dis = [r for r in rows if 0 < r[2] < K_SCREEN]
        oth = [r for r in rows if not 0 < r[2] < K_SCREEN]
        dis = [dis[i] for i in sorted(rng.permutation(len(dis))[:DISAGREE_CAP])]
        oth = [oth[i] for i in sorted(rng.permutation(len(oth))[:PER_FAMILY - len(dis)])]
        for group, role in ((dis, "t16"), (oth, "t8")):
            for seed, i, _ in group:
                roles.setdefault((family, seed), {})[i] = role
        summary[family] = {"scenes_screened": sum(f == family for f, _ in screens),
                           "scenes_skipped": sum(f == family and r["skipped"] for (f, _), r in screens.items()),
                           "chunk_starts": len(rows), "harm_disagree": sum(0 < r[2] < K_SCREEN for r in rows),
                           "selected_disagree": len(dis), "selected_other": len(oth)}
    return roles, summary


def render_job(job):
    import traceback
    try:
        return job[0], job[1], _render(job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _render(job):
    import mujoco
    import torch
    import step3_data as sd
    import uncertainty_monitor_dev as um
    from jenga_render_frames import close_camera
    family, seed, roles = job
    scene, sim, acts, starts = prepare(family, seed)
    snippets = np.load(um.BENCH)["snippets"]
    pos_cov, rot_cov = ps.pooled_error_model()
    n_obj = len(scene["objects"])
    renderer, camera, latent = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent()

    def render():
        renderer.update_scene(sim.data, camera=camera); return renderer.render().copy()
    done = {}
    try:
        for index, act in enumerate(acts):
            if index in roles:
                path = OUT / "train" / f"{family}_{seed}_s{index}.npz"
                if not path.exists():
                    snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy()
                    start = sim.object_state()
                    true_start = render()
                    chunk = acts[index:index + 8]
                    n = FUTURES[roles[index]]
                    traces, executed, starts_k, frames, pos, rot, harms = [], [], [], [], [], [], []
                    for k in range(n):
                        key = ({"F1": 1, "F2": 2}[family], seed, index, k)
                        noise = snippets[np.random.default_rng([SEEDS["futures"], *key]).integers(len(snippets))]
                        err = ps.draw_error(n_obj, pos_cov, rot_cov, SEEDS["futures"], *key)
                        sim.restore(snap); sim.perturb(*err)
                        starts_k.append(render())
                        a = np.asarray(chunk, np.float32).copy(); a[:, :3] -= noise
                        ex = list(a) + [chunk[-1]] * 30
                        tr = []
                        for t, c in enumerate(ex):
                            sim.execute(c); tr.append(sim.object_state())
                            if t in FRAME_INDEX:
                                frames.append(render())
                        traces.append(np.stack(tr)); executed.append(np.asarray(ex, np.float32))
                        pos.append(err[0]); rot.append(err[1]); harms.append(ps.harm(np.stack(tr), start))
                    z = latent(np.concatenate([true_start[None], np.stack(starts_k), np.stack(frames)]))
                    torch.cuda.empty_cache()
                    np.savez(path.with_suffix(".tmp.npz"), z0=z[0], z0k=z[1:1 + n],
                             z=z[1 + n:].reshape(n, len(FRAME_INDEX), 28, 28, 8), d_pos=np.stack(pos), d_rot=np.stack(rot),
                             traces=np.stack(traces).astype(np.float32), actions=np.stack(executed), harm=np.array(harms),
                             start_state=start, frame_index=np.array(FRAME_INDEX))
                    path.with_suffix(".tmp.npz").replace(path)
                    sim.restore(snap); sim.data.qacc_warmstart[:] = warm
                done[index] = roles[index]
                if index == max(roles):
                    break
            sim.execute(act)
    finally:
        renderer.close()
    return done


def pool_run(fn, jobs, workers, label):
    results, failures = {}, []
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (a, b, res, err) in enumerate(pool.imap_unordered(fn, jobs), 1):
            if err:
                failures.append((a, b)); print(err, flush=True)
            else:
                results[(a, b)] = res
            if n % 50 == 0 or n == len(jobs):
                print(f"{label} {n}/{len(jobs)}", flush=True)
    if failures:
        raise SystemExit(f"{label} failures ({len(failures)}): {failures[:10]}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-workers", type=int, default=14)
    parser.add_argument("--render-workers", type=int, default=5)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["amendment_2"]["status"].startswith("frozen"):
        raise SystemExit("amendment 2 not frozen")
    for d in ("screen", "train"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    screens = pool_run(screen_job, [(f, s) for f in ("F1", "F2") for s in TRAIN_SEEDS], args.screen_workers, "screen")
    roles, summary = select(screens)
    (OUT / "selection.json").write_text(json.dumps({"summary": summary, "roles": {f"{f}:{s}": v for (f, s), v in roles.items()}}) + "\n")
    print(json.dumps(summary), flush=True)
    jobs = [(f, s, r) for (f, s), r in sorted(roles.items())]
    pool_run(render_job, jobs, args.render_workers, "render")
    print("done", flush=True)


if __name__ == "__main__":
    main()
