"""Branching-WM Phase 1 data: fork-enriched training futures and the held-out selection set H.

Frozen protocol: results/jenga/phase1_branching_protocol.json. Reuses the iteration-1 screen
(eval/step3_data.py) and the iteration-2 per-future start rendering (eval/step3b_data.py) with Phase-1
seeds; outputs go to results/program_b/phase1/{train,h}.
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

PROTOCOL = ROOT / "results/jenga/phase1_branching_protocol.json"
OUT = ROOT / "results/program_b/phase1"
NEW_SEEDS = range(110, 150)
SEEDS = {"screen": 20261080, "select": 20261081, "futures": 20261082, "h_select": 20261083, "h_futures": 20261084}
FUTURES = {"t16": 16, "t8": 8, "h": 64}
sd.SEEDS = dict(sd.SEEDS, screen=SEEDS["screen"])          # workers import this module, so the patch applies there


def screen_job(job):
    e, s = job
    return sd.chain_job((e, s, "screen", None, str(OUT)))


def render_job(job):
    import traceback
    try:
        return job[0], job[1], _render(job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _render(job):
    episode_id, seed, roles = job
    import mujoco
    import visual_pipeline_panel3 as p3
    import uncertainty_monitor_dev as um
    from step3b_data import future_with_start
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="phase1_"))))
    renderer, camera, latent = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent()
    done = {}
    try:
        sim.reset(int(seed))
        for index, action in enumerate(actions):
            if index in roles:
                role = roles[index]
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                warm0 = sim.data.qacc_warmstart.copy()
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                key = (int(episode_id), int(seed), int(index))
                sim.restore(snapshot)
                renderer.update_scene(sim.data, camera=camera)
                true_start = renderer.render().copy()
                n = FUTURES[role]
                at = sd.D0_INDEX if role == "h" else sd.FRAME_INDEX
                seed_name = "h_futures" if role == "h" else "futures"
                traces, executed, starts, frames, pos, rot = [], [], [], [], [], []
                for k in range(n):
                    noise = snippets[k] if role == "h" else snippets[
                        np.random.default_rng([SEEDS[seed_name], *key, k]).integers(len(snippets))]
                    d = errors.sample(SEEDS[seed_name], *key, k)
                    t, a, s0, f = future_with_start(sim, snapshot, chunk, noise, d, at, renderer, camera)
                    traces.append(t); executed.append(a); starts.append(s0); frames.extend(f)
                    pos.append(d[0]); rot.append(d[1])
                z = latent(np.concatenate([true_start[None], np.stack(starts), np.stack(frames)]))
                folder = "h" if role == "h" else "train"
                path = OUT / folder / f"ep{episode_id}_seed{seed}_s{index}.npz"
                np.savez(path.with_suffix(".tmp.npz"), z0=z[0], z0k=z[1:1 + n],
                         z=z[1 + n:].reshape(n, len(at), 28, 28, 8), d_pos=np.stack(pos), d_rot=np.stack(rot),
                         traces=np.stack(traces).astype(np.float32), actions=np.stack(executed),
                         start_tilt=start_tilt, frame_index=np.array(at))
                path.with_suffix(".tmp.npz").replace(path)
                done[index] = role
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm0
                if index == max(roles):
                    break
            sim.execute(action)
    finally:
        renderer.close()
        sim.close()
    return done


def pool_run(fn, jobs, workers, label):
    ctx = mp.get_context("spawn")
    out, failures = {}, []
    with ctx.Pool(workers) as pool:
        for n, res in enumerate(pool.imap_unordered(fn, jobs), 1):
            e, s, val, err = res
            if err:
                failures.append((e, s)); print(err, flush=True)
            else:
                out[f"{e}:{s}"] = val
            if n % 20 == 0 or n == len(jobs):
                print(f"{label} {n}/{len(jobs)}", flush=True)
    if failures:
        raise SystemExit(f"{label} failures: {failures}")
    return out


def select(screen_new, screen_old, held):
    rows = [(k.split(":")[0], int(k.split(":")[1]), int(i), v["topples"])
            for scr in (screen_new, screen_old) for k, r in scr.items() for i, v in r.items()]
    dis = lambda r: 0 < r[3] < sd.K_SCREEN
    new_train = [r for r in rows if r[0] not in held and r[1] in NEW_SEEDS]
    rng = np.random.default_rng(SEEDS["select"])
    t_dis = [r for r in new_train if dis(r)]
    t_oth = [r for r in new_train if not dis(r)]
    t_oth = [t_oth[i] for i in rng.permutation(len(t_oth))[:len(t_dis)]]
    held_rows = [r for r in rows if r[0] in held]
    hr = np.random.default_rng(SEEDS["h_select"])
    h_dis = [r for r in held_rows if dis(r)]
    h_oth = [r for r in held_rows if not dis(r)]
    h = [h_dis[i] for i in hr.permutation(len(h_dis))[:80]] + [h_oth[i] for i in hr.permutation(len(h_oth))[:80]]
    roles = {}
    for group, role in ((t_dis, "t16"), (t_oth, "t8"), (h, "h")):
        for e, s, i, _ in group:
            roles.setdefault((e, s), {})[i] = role
    summary = {"new_train_states": len(new_train), "train_disagree": len(t_dis), "train_other": len(t_oth),
               "held_disagree_available": len(h_dis), "h_states": len(h)}
    return roles, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    for d in ("train", "h"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    chains_all, held = sd.train_chains()
    episodes = sorted({e for e, _ in chains_all}, key=int)
    screen_file = OUT / "screen.json"
    if not screen_file.exists():
        jobs = [(e, s) for e in episodes for s in NEW_SEEDS]
        screen = pool_run(screen_job, jobs, args.workers, "screen")
        screen_file.write_text(json.dumps(screen) + "\n")
    screen_new = json.loads(screen_file.read_text())
    screen_old = json.loads((sd.OUT / "screen.json").read_text())
    roles, summary = select(screen_new, screen_old, held)
    (OUT / "selection.json").write_text(json.dumps({"summary": summary,
                                                    "roles": {f"{e}:{s}": v for (e, s), v in roles.items()}}) + "\n")
    print(json.dumps(summary), flush=True)
    jobs = []
    for (e, s), sel in sorted(roles.items(), key=lambda x: (x[0][1], int(x[0][0]))):
        todo = {i: r for i, r in sel.items()
                if not (OUT / ("h" if r == "h" else "train") / f"ep{e}_seed{s}_s{i}.npz").exists()}
        if todo:
            jobs.append((e, s, todo))
    jobs.sort(key=lambda j: "h" not in j[2].values())          # H first, so selection data exists early
    pool_run(render_job, jobs, args.workers, "render")
    print("done", flush=True)


if __name__ == "__main__":
    main()
