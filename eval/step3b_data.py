"""Step 3 iteration 2 data: the iteration-1 states and future draws, now with every future's own
perturbed start frame (encoded) and its physical perturbation d_k.

Frozen protocol: results/jenga/step3b_uncertain_start_protocol.json. Fresh chain run; same selection and
seeds as eval/step3_data.py, so the physical futures are the same draws.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import step3_data as sd  # noqa: E402

PROTOCOL = ROOT / "results/jenga/step3b_uncertain_start_protocol.json"
OUT = ROOT / "results/program_b/step3b"


def future_with_start(sim, snapshot, chunk, noise, perturbation, render_at, renderer, camera):
    """As step3_data.future, plus the frame of the perturbed start scene (before the first action)."""
    import program_b_reference as pr
    from jenga_state_data import step_state
    sim.restore(snapshot)
    pr.perturb_blocks(sim, *perturbation)
    renderer.update_scene(sim.data, camera=camera)
    start = renderer.render().copy()
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise
    executed = list(actions) + [chunk[-1]] * 30
    states, frames = [], []
    for k, action in enumerate(executed):
        sim.execute(action)
        states.append(step_state(sim))
        if k in render_at:
            renderer.update_scene(sim.data, camera=camera)
            frames.append(renderer.render().copy())
    return np.stack(states), np.asarray(executed, np.float32), start, frames


def chain_job(job):
    import traceback
    try:
        return job[0], job[1], _chain_job(job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _chain_job(job):
    episode_id, seed, selected = job
    import mujoco
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    errors = p3.ErrorModel()
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="step3b_"))))
    renderer, camera, latent = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent()
    done = {}
    try:
        sim.reset(int(seed))
        for index, action in enumerate(actions):
            if index in selected:
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                warm0 = sim.data.qacc_warmstart.copy()
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                key = (int(episode_id), int(seed), int(index))
                for role in selected[index]:
                    sim.restore(snapshot)
                    renderer.update_scene(sim.data, camera=camera)
                    true_start = renderer.render().copy()
                    n, seed_name, at = ((sd.D0_FUTURES, "d0_futures", sd.D0_INDEX) if role == "d0"
                                        else (sd.K_FUTURES, "futures", sd.FRAME_INDEX))
                    traces, executed, starts, frames, pos, rot = [], [], [], [], [], []
                    for k in range(n):
                        noise = snippets[k] if role == "d0" else snippets[
                            np.random.default_rng([sd.SEEDS[seed_name], *key, k]).integers(len(snippets))]
                        d = errors.sample(sd.SEEDS[seed_name], *key, k)
                        t, a, s0, f = future_with_start(sim, snapshot, chunk, noise, d, at, renderer, camera)
                        traces.append(t); executed.append(a); starts.append(s0); frames.extend(f)
                        pos.append(d[0]); rot.append(d[1])
                    z = latent(np.concatenate([true_start[None], np.stack(starts), np.stack(frames)]))
                    path = OUT / role / f"ep{episode_id}_seed{seed}_s{index}.npz"
                    np.savez(path.with_suffix(".tmp.npz"), z0=z[0], z0k=z[1:1 + n],
                             z=z[1 + n:].reshape(n, len(at), 28, 28, 8), d_pos=np.stack(pos), d_rot=np.stack(rot),
                             traces=np.stack(traces).astype(np.float32), actions=np.stack(executed),
                             start_frame=true_start, start_tilt=start_tilt, frame_index=np.array(at))
                    path.with_suffix(".tmp.npz").replace(path)
                    done.setdefault(index, []).append(role)
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm0
                if index == max(selected):
                    break
            sim.execute(action)
    finally:
        renderer.close()
        sim.close()
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    for d in ("train", "val", "d0"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    roles = json.loads((sd.OUT / "selection.json").read_text())["roles"]
    jobs = []
    for chain, sel in sorted(roles.items(), key=lambda x: (int(x[0].split(":")[0]), int(x[0].split(":")[1]))):
        e, s = chain.split(":")
        todo = {int(i): [r for r in rs if not (OUT / r / f"ep{e}_seed{s}_s{i}.npz").exists()] for i, rs in sel.items()}
        todo = {i: rs for i, rs in todo.items() if rs}
        if todo:
            jobs.append((e, int(s), todo))
    jobs.sort(key=lambda j: not any("d0" in rs for rs in j[2].values()))
    ctx = mp.get_context("spawn")
    failures = []
    with ctx.Pool(args.workers) as pool:
        for n, (e, s, res, err) in enumerate(pool.imap_unordered(chain_job, jobs), 1):
            if err:
                failures.append((e, s)); print(err, flush=True)
            if n % 10 == 0 or n == len(jobs):
                print(f"render {n}/{len(jobs)}", flush=True)
    if failures:
        raise SystemExit(f"failures: {failures}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
