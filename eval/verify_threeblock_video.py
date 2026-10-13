"""Verification video for the three-block data correction (as eval/verify_perturbation_video.py for the varied scenes).

Scan: for every H state, the first 4 perception-error draws with the robot holding still, OLD vs NEW injection.
Video: the 5 states where the OLD injection moves blocks most with the robot still + 3 random states; per state four
panels: OLD still | NEW still (same draw) | NEW acting, a future with an event | NEW acting, a future without.
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

import threeblock_v2 as tb  # noqa: E402

OUTV = ROOT / "results/program_b/threeblock_fix_verification.mp4"
SUMMARY = ROOT / "results/jenga/threeblock_fix_video_summary.json"


def _setup(name):
    import uncertainty_monitor_dev as um
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    e, rest = name[2:].split("_seed"); seed, index = rest.split("_s"); e, seed, index = int(e), int(seed), int(index)
    replay = JengaReplay(DEFAULT_LMDB); actions = np.asarray(replay.episode(e).actions, np.float32); replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="vid3_"))))
    sim.reset(seed)
    for i, a in enumerate(actions):
        if i == index:
            break
        sim.execute(a)
    return sim, actions[index:index + 8], (e, seed, index)


def _run(sim, snap, d, valid, chunk, noise, render=None):
    import program_b_reference as pr
    pr.VALID_PERTURBATION = valid
    sim.restore(snap); pr.perturb_blocks(sim, *d)
    pr.VALID_PERTURBATION = True
    p0 = pr._pose(sim)[0]
    cmds = [np.r_[sim.target, 0.0].astype(np.float32)] * 38 if chunk is None else \
        list(np.asarray(chunk, np.float32) - np.c_[noise, np.zeros(8)]) + [chunk[-1]] * 30
    frames = [render()] if render else []
    for c in cmds:
        sim.execute(c)
        if render:
            frames.append(render())
    return frames, pr._pose(sim)


def scan_job(name):
    import program_b_reference as pr
    import visual_pipeline_panel3 as p3
    sim, chunk, key = _setup(name)
    snap = sim.snapshot(); p0, t0 = pr._pose(sim); errors = p3.ErrorModel(); worst = 0.0
    for k in range(4):
        _, (p1, t1) = _run(sim, snap, errors.sample(tb.H_FUTURES_SEED, *key, k), False, None, None)
        worst = max(worst, float(np.linalg.norm(p1 - p0, axis=1).max()))
    sim.close()
    return name, worst


def event(pose0, pose1):
    (p0, t0), (p1, t1) = pose0, pose1
    return bool(((np.linalg.norm(p1 - p0, axis=1) >= 0.03) | (np.abs(t1 - t0) >= 45)).any())


def main():
    import imageio_ffmpeg
    import mujoco
    import program_b_reference as pr
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from jenga_render_frames import close_camera
    from verify_perturbation_video import label
    with mp.get_context("spawn").Pool(16) as pool:
        scan = dict(pool.map(scan_job, tb.states()))
    at_rest = {n: bool(np.load(tb.FRAMES / f"{n}.npz")["at_rest"]) for n in scan}
    order = sorted(scan, key=lambda n: -scan[n])
    not_rest = [n for n in order if not at_rest[n]][:1]       # one excluded start, for illustration
    order = [n for n in order if at_rest[n]]
    rng = np.random.default_rng(3)
    rest = [n for n in order[5:]]
    chosen = not_rest + order[:5] + [rest[i] for i in rng.permutation(len(rest))[:3]]
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    writer = imageio_ffmpeg.write_frames(str(OUTV), (1024, 256), fps=8, codec="libx264", quality=8, macro_block_size=1)
    writer.send(None)
    blank = np.zeros((256, 256, 3), np.uint8); summary = []
    for name in chosen:
        sim, chunk, key = _setup(name)
        rest_ok = pr.blocks_at_rest(sim)
        snap = sim.snapshot(); pose0 = pr._pose(sim)
        r = mujoco.Renderer(sim.model, height=384, width=384); cam = close_camera(mujoco)
        render = lambda: (r.update_scene(sim.data, camera=cam), r.render().copy())[1]
        errs = [errors.sample(tb.H_FUTURES_SEED, *key, k) for k in range(64)]
        worst = int(np.argmax([np.linalg.norm(_run(sim, snap, errs[k], False, None, None)[1][0] - pose0[0], axis=1).max() for k in range(4)]))
        f_old, pose_old = _run(sim, snap, errs[worst], False, None, None, render)
        f_new, pose_new = _run(sim, snap, errs[worst], True, None, None, render)
        yes = no = None
        for k in range(64):
            ev = event(pose0, _run(sim, snap, errs[k], True, chunk, snippets[k])[1])
            yes = k if ev and yes is None else yes
            no = k if not ev and no is None else no
            if yes is not None and no is not None:
                break
        f_yes = _run(sim, snap, errs[yes], True, chunk, snippets[yes], render)[0] if yes is not None else None
        f_no = _run(sim, snap, errs[no], True, chunk, snippets[no], render)[0] if no is not None else None
        r.close(); sim.close()
        row = {"state": name, "at_rest": rest_ok, "old_still_moved_mm": round(scan[name] * 1000, 1),
               "old_still_event": event(pose0, pose_old), "new_still_event": event(pose0, pose_new),
               "has_event_future": yes is not None, "has_quiet_future": no is not None}
        summary.append(row); print(row, flush=True)
        title = label(blank, [name, f"at rest: {rest_ok}", f"old still moved {row['old_still_moved_mm']} mm"], (255, 255, 255))
        for _ in range(10):
            writer.send(np.concatenate([title, blank, blank, blank], 1))
        for t in range(len(f_old)):
            writer.send(np.concatenate([
                label(f_old[t], ["OLD perturbation", "robot holds still"], (255, 120, 120)),
                label(f_new[t], ["NEW perturbation", "robot holds still"], (140, 255, 140)),
                label(f_yes[t], ["NEW, robot acts", "future with event"], (255, 200, 120)) if f_yes else label(blank, ["NEW, robot acts", "(no future with event)"], (200, 200, 200)),
                label(f_no[t], ["NEW, robot acts", "future without event"], (140, 220, 255)) if f_no else label(blank, ["NEW, robot acts", "(no quiet future)"], (200, 200, 200))], 1))
    writer.close()
    SUMMARY.write_text(json.dumps({"scan_old_still_moved_mm": {n: round(v * 1000, 1) for n, v in scan.items()}, "video": summary}, indent=1) + "\n")
    print("video", OUTV, flush=True)


if __name__ == "__main__":
    main()
