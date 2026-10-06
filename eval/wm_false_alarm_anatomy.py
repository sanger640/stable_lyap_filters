"""Where do the deterministic token world model's false alarms on set C come from? (descriptive diagnostic)

For every set-C state, re-runs the world model exactly as step3_world_model.evaluate (alarms reproduce the
stored 44), takes the split test's own two groups at hold 30, and measures:
  * where in the 28x28 latent grid the difference between the two groups lives (robot / middle red block /
    neighbour blocks / background, from a colour segmentation of the observed close-camera frame);
  * how wide the gap is compared with the spread of the real futures (scale);
  * whether the minority group is made of the same execution-error snippets across states (if so, the split
    is driven by how the arm moves, not by the scene).
Compared between false alarms (safe states) and true alarms (forks).
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/jenga/wm_false_alarm_anatomy.json"
FIG = ROOT / "results/jenga/wm_false_alarm_anatomy.png"
REGIONS = ("robot", "middle_block", "neighbours", "background")


SEG_REGIONS = ("robot", "middle_block", "neighbours", "floor_shadow", "floor_lit", "other")


def replay_segmentation(job):
    """Replay a panel-4 chain exactly (as visual_v2_panel4) to the target state; render RGB and segmentation
    with the close camera. Returns (key, rgb_max_abs_diff_vs_stored, (28,28,6) region fractions)."""
    episode_id, index = job
    import tempfile
    import mujoco
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import visual_v2_panel4 as v4
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import BLOCK_NAMES, DirectJengaSim, chunk_starts, extract_sim
    from PIL import Image
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    starts = set(chunk_starts(len(actions)))
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="seg_"))))
    rgb_r = mujoco.Renderer(sim.model, height=384, width=384)
    seg_r = mujoco.Renderer(sim.model, height=384, width=384)
    seg_r.enable_segmentation_rendering()
    cam = close_camera(mujoco)
    try:
        sim.reset(int(episode_id) + v4.RESET_BASE)
        for k, action in enumerate(actions):
            if k in starts and k + 8 <= len(actions):
                snapshot = sim.snapshot()
                if k == index:
                    sim.restore(snapshot)
                    rgb_r.update_scene(sim.data, camera=cam); rgb = rgb_r.render().copy()
                    seg_r.update_scene(sim.data, camera=cam); seg = seg_r.render()[..., 0].copy()
                    p3.rollout_render(sim, snapshot, actions[k:k + 8], snippets[0], [], False)   # one nominal future
                    rgb_r.update_scene(sim.data, camera=cam); rgb_end = rgb_r.render().copy()
                    seg_r.update_scene(sim.data, camera=cam); seg_end = seg_r.render()[..., 0].copy()
                    break
                for n in snippets:
                    p3.rollout_render(sim, snapshot, actions[k:k + 8], n, [], False)
                warm = sim.data.qacc_warmstart.copy()
                sim.restore(snapshot); sim.data.qacc_warmstart[:] = warm
            sim.execute(action)
        stored = np.load(v4.OUT / f"ep{episode_id}_s{index}.npz")["close_frame"]
        start_cells = regions(sim.model, seg, rgb)
        end_cells = regions(sim.model, seg_end, rgb_end)
        return f"{episode_id}:{index}", float(np.abs(rgb.astype(int) - stored.astype(int)).max()), start_cells, end_cells
    finally:
        sim.close()


def regions(m, seg, rgb):
    from PIL import Image
    from jenga_short_held_tails import BLOCK_NAMES
    if True:
        body = np.where(seg >= 0, m.geom_bodyid[np.clip(seg, 0, m.ngeom - 1)], -1)
        blocks = [m.body(nm).id for nm in BLOCK_NAMES]
        floor_ids = {g for g in range(m.ngeom) if m.geom_bodyid[g] in (0, m.body("table").id)}
        floor = np.isin(seg, list(floor_ids))
        robot = np.zeros_like(floor)
        for b in range(m.nbody):
            name = m.body(b).name.lower()
            if b != 0 and b not in blocks and any(t in name for t in ("link", "hand", "finger", "wrist")):
                robot |= body == b
        middle = body == m.body("block_middle").id
        neighbours = (body == m.body("block_left").id) | (body == m.body("block_right").id)
        lum = rgb.mean(-1)
        lit_level = np.median(lum[floor]) if floor.any() else 255
        shadow = floor & (lum < 0.85 * lit_level)
        lit = floor & ~shadow
        other = ~(robot | middle | neighbours | floor)
        masks = np.stack([robot, middle, neighbours, shadow, lit, other], -1).astype(float)
        return np.stack([np.asarray(Image.fromarray((masks[..., c] * 255).astype(np.uint8)).resize((28, 28), Image.BOX))
                         / 255.0 for c in range(masks.shape[-1])], -1)


def segment(frame):
    """(384,384,3) uint8 -> (28,28,4) fraction of each region per latent cell (colour heuristic)."""
    import matplotlib.colors as mc
    hsv = mc.rgb_to_hsv(frame / 255.0)
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    block = s > 0.25
    red = block & ((h < 15) | (h > 340)) & (s > 0.45)
    neighbour = block & ~red
    robot = ~block & (v < 0.86)
    background = ~block & ~robot
    masks = np.stack([robot, red, neighbour, background], -1).astype(float)
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray((masks[..., k] * 255).astype(np.uint8)).resize((28, 28), Image.BOX))
                     / 255.0 for k in range(4)], -1)


def main():
    import torch
    import uncertainty_monitor_dev as um
    import visual_v2_panel4 as v4
    import step3_world_model as wm
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from outcome_modes import two_mode_test
    from step3_data import Latent
    panel, screen, _ = v4.screen_panel(1)
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    snippets = np.load(um.BENCH)["snippets"]
    latent, model = Latent(), wm.WorldModel.load(wm.DATA / "wm_deterministic.pt")
    rows, maps = [], {}
    for n, (e, i) in enumerate(keys):
        tilt = np.array(screen[e][str(i)]["start_tilt"])
        with np.load(v4.OUT / f"ep{e}_s{i}.npz") as f:
            fork = 2 <= topple_count(f["truth_A"], tilt) <= 62
            frame = f["close_frame"]
            z0 = latent(frame[None])[0].astype(np.float32)
            early, late = wm.wm_rollouts(model, z0, wm.panel_actions(f["chunk"], snippets), seed=20261047 + n)
        alarm = bool(persistent_mode_alarm(early, late).alarm)
        if not alarm:
            continue
        test = two_mode_test(late, perturbation_mm=0.0)
        labels = test.labels
        if labels.sum() > 32:
            labels = 1 - labels
        diff = (late[labels == 1].mean(0) - late[labels == 0].mean(0)).reshape(28, 28, 8)
        energy = (diff ** 2).sum(-1)
        seg = segment(frame)
        share = {r: float((energy * seg[..., k]).sum() / energy.sum()) for k, r in enumerate(REGIONS)}
        area = {r: float(seg[..., k].mean()) for k, r in enumerate(REGIONS)}
        within = np.sqrt(np.mean([late[labels == g].var(0).sum() for g in (0, 1)]))
        rows.append({"state": f"{e}:{i}", "n": n, "fork": bool(fork), "minority": int(labels.sum()),
                     "minority_snippets": np.flatnonzero(labels).tolist(), "ashman_d": float(test.ashman_d),
                     "gap": float(np.linalg.norm(diff)), "within_group_spread": float(within),
                     "energy_share": share, "area_share": area})
        maps[f"{e}:{i}"] = (frame, energy, bool(fork))
        torch.cuda.empty_cache()
    assert len(rows) == 44, len(rows)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    jobs = [(r["state"].split(":")[0], int(r["state"].split(":")[1])) for r in rows]
    with ProcessPoolExecutor(16, mp_context=mp.get_context("spawn")) as pool:
        segs = {k: (d, c, ce) for k, d, c, ce in pool.map(replay_segmentation, jobs)}
    for r in rows:
        diff_rgb, cells, cells_end = segs[r["state"]]
        energy = maps[r["state"]][1]
        r["replay_rgb_max_diff"] = diff_rgb
        r["exact_energy_share"] = {g: float((energy * cells[..., k]).sum() / energy.sum()) for k, g in enumerate(SEG_REGIONS)}
        r["exact_area_share"] = {g: float(cells[..., k].mean()) for k, g in enumerate(SEG_REGIONS)}
        r["end_energy_share"] = {g: float((energy * cells_end[..., k]).sum() / energy.sum()) for k, g in enumerate(SEG_REGIONS)}
        r["end_area_share"] = {g: float(cells_end[..., k].mean()) for k, g in enumerate(SEG_REGIONS)}

    def summary(group):
        g = [r for r in rows if r["fork"] == group]
        return {"alarms": len(g),
                "median_energy_share": {r: float(np.median([x["energy_share"][r] for x in g])) for r in REGIONS},
                "median_area_share": {r: float(np.median([x["area_share"][r] for x in g])) for r in REGIONS},
                "median_exact_energy_share": {r: float(np.median([x["exact_energy_share"][r] for x in g])) for r in SEG_REGIONS},
                "median_exact_area_share": {r: float(np.median([x["exact_area_share"][r] for x in g])) for r in SEG_REGIONS},
                "median_end_energy_share": {r: float(np.median([x["end_energy_share"][r] for x in g])) for r in SEG_REGIONS},
                "median_end_area_share": {r: float(np.median([x["end_area_share"][r] for x in g])) for r in SEG_REGIONS},
                "max_replay_rgb_diff": float(max(x["replay_rgb_max_diff"] for x in g)),
                "median_minority": float(np.median([x["minority"] for x in g])),
                "median_gap": float(np.median([x["gap"] for x in g]))}
    false_rows = [r for r in rows if not r["fork"]]
    sets = [set(r["minority_snippets"]) for r in false_rows]
    overlap = [len(a & b) / len(a | b) for k, a in enumerate(sets) for b in sets[k + 1:]]
    counts = np.bincount([s for r in false_rows for s in r["minority_snippets"]], minlength=64)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        rs = [set(rng.choice(64, len(s), replace=False).tolist()) for s in sets]
        null.append(np.mean([len(a & b) / len(a | b) for k, a in enumerate(rs) for b in rs[k + 1:]]))
    result = {"note": __doc__.strip().splitlines()[0],
              "false_alarms": summary(False), "true_alarms": summary(True),
              "false_alarm_minority_overlap": {"mean_jaccard": float(np.mean(overlap)),
                                               "null_mean_jaccard_95pct": float(np.percentile(null, 95)),
                                               "snippets_in_minority_of_most_false_alarms":
                                                   [int(s) for s in np.flatnonzero(counts >= len(sets) / 2)]},
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fa = [r["state"] for r in false_rows]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, key in zip(axes.ravel(), fa):
        frame, energy, _ = maps[key]
        ax.imshow(frame.mean(-1), cmap="gray", extent=(0, 28, 28, 0))
        ax.imshow(energy / energy.max(), cmap="magma", alpha=np.clip(energy / energy.max(), 0, 1) * 0.9,
                  extent=(0, 28, 28, 0), interpolation="bilinear")
        r = next(x for x in rows if x["state"] == key)
        es = r["end_energy_share"]
        ax.set_title(f"{key}: robot {es['robot']:.0%}, shadow {es['floor_shadow']:.0%}, nbrs {es['neighbours']:.0%}",
                     fontsize=10)
        ax.axis("off")
    fig.suptitle("World-model false alarms on set C: where its two groups differ (heat)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG, dpi=110)
    print(json.dumps({k: result[k] for k in ("false_alarms", "true_alarms", "false_alarm_minority_overlap")},
                     indent=1))


if __name__ == "__main__":
    main()
