"""Exact replays of selected set-C (panel 4) states for the results video (descriptive; no results depend on it).

For each scenario: replay the panel-4 chain exactly as eval/visual_v2_panel4.py ran it, re-run every condition at
the target state in the original order (warm starts carry over between rollouts), and while doing so film the
64 true futures (truth_A) and the MuJoCo probe's 64 imagined futures (aware) at every control step. Replayed traces
are checked against the stored ones. Also encodes every filmed future into the world model's latent, and rolls
the deterministic token world model from the observed image exactly as in step3_world_model.evaluate.
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
HERE = Path(__file__).parent
OUT = HERE / "scenes"
DISPLAY = 480
WM_FRAMES = list(range(1, 38, 2))          # control step after each world-model step (2 actions per step)


def pick_scenarios():
    rows = json.loads((HERE / "scenarios_table.json").read_text())["rows"]

    def best(cond, key):
        return min((r for r in rows if cond(r)), key=key)
    balanced = lambda r: (abs(r["truth_topples"] - 32), r["n"])
    return {
        "1_safe": best(lambda r: not r["fork"] and not r["probe_alarm"] and not r["wm_alarm"] and r["probe_topples"] == 0,
                       lambda r: r["n"]),
        "2_fork_both": best(lambda r: r["fork"] and r["probe_alarm"] and r["wm_alarm"], balanced),
        "3_fork_wm_misses": best(lambda r: r["fork"] and r["probe_alarm"] and not r["wm_alarm"], balanced),
        "4_wm_false_alarm": best(lambda r: not r["fork"] and r["wm_alarm"] and not r["probe_alarm"], lambda r: r["n"]),
        "5_fork_both_miss": best(lambda r: r["fork"] and not r["probe_alarm"] and not r["wm_alarm"],
                                 lambda r: (-r["truth_topples"], r["n"])),
    }


def film(sim, snapshot, chunk, noise, perturbations, renderers, camera, latent):
    """visual_pipeline_panel3.rollout_render with every control step filmed (rendering never touches physics)."""
    import program_b_reference as pr
    from jenga_state_data import step_state
    sim.restore(snapshot)
    for position, rotation in perturbations:
        pr.perturb_blocks(sim, position, rotation)
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise
    states, show, enc = [], [], []
    for k, action in enumerate(list(actions) + [chunk[-1]] * 30):
        sim.execute(action)
        states.append(step_state(sim))
        for r in renderers.values():
            r.update_scene(sim.data, camera=camera)
        show.append(renderers["display"].render().copy())
        if k in WM_FRAMES:
            enc.append(renderers["encode"].render().copy())
    return np.stack(states), np.stack(show), latent(np.stack(enc))


def worker(job):
    name, row = job
    import mujoco
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import visual_v2_panel4 as v4
    from fresh_fragility_panel import topple_count
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim
    from step3_data import Latent
    episode_id, target_index = row["episode"], row["index"]
    stored = np.load(v4.OUT / f"ep{episode_id}_s{target_index}.npz")
    screen = json.loads((v4.OUT / "screen.json").read_text())
    tilt = np.array(screen[episode_id][str(target_index)]["start_tilt"])
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    latent = Latent()
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    starts = set(chunk_starts(len(actions)))
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="video_"))))
    sim.model.vis.global_.offwidth = sim.model.vis.global_.offheight = max(DISPLAY, 384)
    renderers = {"display": mujoco.Renderer(sim.model, height=DISPLAY, width=DISPLAY),
                 "encode": mujoco.Renderer(sim.model, height=384, width=384)}
    camera = close_camera(mujoco)
    plain = p3.rollout_render
    diffs, filmed = {}, {}
    try:
        sim.reset(int(episode_id) + v4.RESET_BASE)
        for index, action in enumerate(actions):
            if index in starts and index + 8 <= len(actions):
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                target = index == target_index
                exact = [plain(sim, snapshot, chunk, n, [], False)[0] for n in snippets]
                warm_end = sim.data.qacc_warmstart.copy()
                if target:
                    key = (int(episode_id) + v4.RESET_BASE, index)
                    sim.restore(snapshot)
                    hat = p3.delta_to(sim, stored["hat_pose"])
                    traces = {"exact": np.stack(exact)}
                    traces["naive"] = np.stack([plain(sim, snapshot, chunk, n, [hat], False)[0] for n in snippets])
                    shots = [film(sim, snapshot, chunk, n, [hat, errors.sample(v4.SEEDS["aware"], *key, j)],
                                  renderers, camera, latent) for j, n in enumerate(snippets)]
                    traces["aware"] = np.stack([s[0] for s in shots]); filmed["aware"] = shots
                    for m in range(5):
                        d_m = errors.sample(v4.SEEDS["v2_hyp"], *key, m)
                        traces[f"v2h{m}"] = np.stack([plain(sim, snapshot, chunk, n,
                                                            [hat, d_m, errors.sample(v4.SEEDS["v2_probe"], *key, m, j)],
                                                            False)[0] for j, n in enumerate(snippets)])
                    shots = [film(sim, snapshot, chunk, n, [errors.sample(v4.SEEDS["truth_A"], *key, j)],
                                  renderers, camera, latent) for j, n in enumerate(snippets)]
                    traces["truth_A"] = np.stack([s[0] for s in shots]); filmed["truth_A"] = shots
                    traces["truth_B"] = np.stack([plain(sim, snapshot, chunk, n, [errors.sample(v4.SEEDS["truth_B"], *key, j)],
                                                        False)[0] for j, n in enumerate(snippets)])
                    diffs = {k: float(np.abs(v - stored[k]).max()) for k, v in traces.items()}
                    break
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm_end
            sim.execute(action)
    finally:
        sim.close()
    out = {"diffs": diffs}
    arrays = {"close_frame": stored["close_frame"]}
    for cond, shots in filmed.items():
        topples = np.array([topple_count(s[0][None], tilt) > 0 for s in shots])
        def mean(idx):                                                            # uint8 mean over futures, per frame
            return np.stack([np.mean([shots[j][1][t] for j in idx], 0, dtype=np.float32)
                             for t in range(38)]).round().astype(np.uint8)
        arrays[f"{cond}_topples"] = topples
        arrays[f"{cond}_latents"] = np.stack([s[2] for s in shots])              # (64, 19, 28, 28, 8)
        arrays[f"{cond}_mean_all"] = mean(range(64))
        for grp, mask in (("stay", ~topples), ("topple", topples)):
            if mask.any():
                arrays[f"{cond}_mean_{grp}"] = mean(np.flatnonzero(mask))
                arrays[f"{cond}_one_{grp}"] = shots[int(np.flatnonzero(mask)[0])][1]
        arrays[f"{cond}_traces"] = traces[cond]
        out[f"{cond}_topples"] = int(topples.sum())
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    return name, out


def world_model(scenarios):
    import torch
    import uncertainty_monitor_dev as um
    import visual_v2_panel4 as v4
    import step3_world_model as wm
    from step3_data import Latent
    snippets = np.load(um.BENCH)["snippets"]
    latent, model = Latent(), wm.WorldModel.load(wm.DATA / "wm_deterministic.pt")
    for name, row in scenarios.items():
        with np.load(v4.OUT / f"ep{row['episode']}_s{row['index']}.npz") as f:
            z0 = latent(f["close_frame"][None])[0].astype(np.float32)
            actions = wm.panel_actions(f["chunk"], snippets)
        g = torch.Generator(device=wm.DEVICE).manual_seed(20261047 + row["n"])     # as wm_rollouts
        z = torch.as_tensor(z0, device=wm.DEVICE).float()[None].expand(64, -1, -1, -1).contiguous()
        pairs = torch.as_tensor(actions, device=wm.DEVICE).reshape(64, wm.STEPS, -1)
        steps = []
        with torch.no_grad():
            for t in range(wm.STEPS):
                z = model.step(z, pairs[:, t], generator=g)
                steps.append(z.cpu().numpy().astype(np.float16))
        np.savez_compressed(OUT / f"{name}_wm.npz", z0=z0.astype(np.float16), latents=np.stack(steps, 1))


def main():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    OUT.mkdir(exist_ok=True)
    scenarios = pick_scenarios()
    (OUT / "scenarios.json").write_text(json.dumps(scenarios, indent=1) + "\n")
    world_model(scenarios)
    with ProcessPoolExecutor(len(scenarios), mp_context=mp.get_context("spawn")) as pool:
        report = dict(pool.map(worker, scenarios.items()))
    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
