"""Render the V1 chunk-start frames once, for comparing frozen visual encoders.

Same episodes (the persistent-branch-margin panel), reset seeds 100-109 and chunk starts (>= 2) as
``jenga_v1_encode.py``; a single frame at the chunk start, rendered square at ``--pixels`` from the
same default camera. Writes one npz per episode/seed with uint8 frames and the true 61-D state, so
any encoder can be run over the cache without re-simulating.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))


CLOSE_CAMERA = {"lookat": (0.5, 0.0, 0.46), "distance": 0.30, "azimuth": 180.0,
                "elevation": -63.43}   # cam_fixed's viewing direction, about 3x closer to the blocks


def close_camera(mujoco):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CLOSE_CAMERA["lookat"]
    cam.distance, cam.azimuth, cam.elevation = (CLOSE_CAMERA["distance"], CLOSE_CAMERA["azimuth"],
                                                CLOSE_CAMERA["elevation"])
    return cam


def render_file(job):
    episode_id, seed, pixels, out, archive = job[:5]
    camera = job[5] if len(job) > 5 else "default"
    target = Path(out) / f"ep{episode_id}_seed{seed}.npz"
    if target.exists():
        return target.name
    import mujoco
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim
    from jenga_state_data import step_state
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    starts = [s for s in chunk_starts(len(actions)) if s >= 2]
    xml = str(extract_sim(archive, tempfile.mkdtemp(prefix="jenga_frames_")))
    sim = DirectJengaSim(xml)
    sim.renderer.close()
    sim.renderer = mujoco.Renderer(sim.model, height=pixels, width=pixels)
    frames, states = [], []
    try:
        sim.reset(int(seed))
        for step, action in enumerate(actions):
            if step in starts:
                if camera == "close":
                    sim.renderer.update_scene(sim.data, camera=close_camera(mujoco))
                    frames.append(sim.renderer.render().copy())
                else:
                    frames.append(sim.render())
                states.append(step_state(sim))
            sim.execute(action)
    finally:
        sim.close()
    tmp = target.with_suffix(".tmp.npz")
    np.savez(tmp, frames=np.stack(frames).astype(np.uint8), state=np.stack(states).astype(np.float32),
             starts=np.asarray(starts), episode_id=episode_id, seed=int(seed), pixels=pixels)
    tmp.replace(target)
    return target.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(ROOT / "results/jenga/persistent_branch_margin.json"))
    ap.add_argument("--out", default=str(ROOT / "results/program_b/vjepa/frames_384"))
    ap.add_argument("--pixels", type=int, default=384)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--camera", choices=("default", "close"), default="default")
    args = ap.parse_args()
    if not 1 <= args.workers <= 4:
        ap.error("simulator work is capped at four workers")
    episodes = sorted({r["episode_id"] for r in json.loads(Path(args.panel).read_text())["rows"]},
                      key=int)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(e, s, args.pixels, args.out, str(ROOT / "vendor/panda_express_sim.tar"), args.camera)
            for s in range(args.first_seed, args.first_seed + args.seeds) for e in episodes]
    with ProcessPoolExecutor(args.workers) as pool:
        for number, name in enumerate(pool.map(render_file, jobs), 1):
            if number % 25 == 0 or number == len(jobs):
                print(f"{number}/{len(jobs)} {name}", flush=True)


if __name__ == "__main__":
    main()
