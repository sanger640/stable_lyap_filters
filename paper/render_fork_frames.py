"""Crisp close-camera renders of one held-out fork for Figure 1 (descriptive; no results depend on it).

Replays the D0 chain of STATE exactly as eval/step3_data.py ran it, then re-runs two of its 64 stored
futures (one where the neighbour stays, one where it topples) with a 512 px close camera and checks the
replayed trajectories against the stored ones.
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
EPISODE, SEED, INDEX = "63", 104, 74           # D0 state ep63_seed104_s74 (most balanced real split)
OUT = Path(__file__).with_name("fig_assets")
PIXELS = 512


def main():
    import mujoco
    from PIL import Image
    import step3_data as sd
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import topple_count
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    OUT.mkdir(exist_ok=True)
    stored = np.load(sd.OUT / "d0" / f"ep{EPISODE}_seed{SEED}_s{INDEX}.npz")
    tilt = stored["start_tilt"]
    topples = np.array([topple_count(stored["traces"][k:k + 1], tilt) > 0 for k in range(64)])
    final_tilt = np.array([max(fp_tilts(stored["traces"][k, -1])) for k in range(64)])
    k_stay = int(np.flatnonzero(~topples)[np.argmin(final_tilt[~topples])])
    k_top = int(np.flatnonzero(topples)[np.argmax(final_tilt[topples])])
    visited = {int(k): v for k, v in json.loads((sd.OUT / "selection.json").read_text())["roles"][f"{EPISODE}:{SEED}"].items()}
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(EPISODE).actions, np.float32)
    replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="figframes_"))))
    sim.model.vis.global_.offwidth = sim.model.vis.global_.offheight = PIXELS
    renderer, camera = mujoco.Renderer(sim.model, height=PIXELS, width=PIXELS), close_camera(mujoco)
    report = {}
    try:
        sim.reset(SEED)
        for index, action in enumerate(actions):
            if index in visited:
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                warm0 = sim.data.qacc_warmstart.copy()
                sim.block_diagnostics()
                key = (int(EPISODE), SEED, index)
                for role in visited[index]:
                    n, seed_name = (sd.D0_FUTURES, "d0_futures") if role == "d0" else (sd.K_FUTURES, "futures")
                    sim.restore(snapshot)
                    for k in range(n):
                        noise = snippets[k] if role == "d0" else snippets[
                            np.random.default_rng([sd.SEEDS[seed_name], *key, k]).integers(len(snippets))]
                        sd.future(sim, snapshot, chunk, noise, errors.sample(sd.SEEDS[seed_name], *key, k))
                if index == INDEX:
                    sim.restore(snapshot)
                    renderer.update_scene(sim.data, camera=camera)
                    Image.fromarray(renderer.render().copy()).save(OUT / "start.png")
                    for name, k in (("stays", k_stay), ("topples", k_top)):
                        traces, _, frames = sd.future(sim, snapshot, chunk, snippets[k],
                                                      errors.sample(sd.SEEDS["d0_futures"], *key, k),
                                                      (37,), renderer, camera)
                        Image.fromarray(frames[0]).save(OUT / f"end_{name}.png")
                        report[name] = {"future": k, "max_trace_diff": float(np.abs(traces - stored["traces"][k]).max())}
                    # ghosts: all 64 imagined starts (perception draws) and all 64 imagined endings, averaged
                    import program_b_reference as pr
                    starts, ends = [], []
                    for k in range(sd.D0_FUTURES):
                        d = errors.sample(sd.SEEDS["d0_futures"], *key, k)
                        sim.restore(snapshot)
                        pr.perturb_blocks(sim, *d)
                        renderer.update_scene(sim.data, camera=camera)
                        starts.append(renderer.render().astype(np.float32))
                        ends.append(sd.future(sim, snapshot, chunk, snippets[k], d, (37,), renderer, camera)[2][0]
                                    .astype(np.float32))
                    Image.fromarray(np.mean(starts, 0).astype(np.uint8)).save(OUT / "start_ghost.png")
                    Image.fromarray(np.mean(ends, 0).astype(np.uint8)).save(OUT / "end_ghost.png")
                    break
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm0
            sim.execute(action)
    finally:
        renderer.close()
        sim.close()
    report["topples_of_64"] = int(topples.sum())
    (OUT / "frames.json").write_text(json.dumps(report, indent=1) + "\n")
    print(report)


def fp_tilts(state):
    from fresh_fragility_panel import neighbour_tilts
    return neighbour_tilts(state)


if __name__ == "__main__":
    main()
