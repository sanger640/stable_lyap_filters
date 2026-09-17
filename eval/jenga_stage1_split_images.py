"""Render the non-topple outcome splits that Stage 1 flagged on unanimous-safe states.

For each flagged state (at one noise scale), pick the run nearest each group's median ending
and re-simulate it. Panels: start of the chunk; each group's settled ending (hold step 30),
from an oblique close-up and from above; and a top-down overlay (group A red, group B cyan).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from outcome_modes import corner_displacements_mm, two_mode_test  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import HOLD, RECORD_AT, SCALES  # noqa: E402
from jenga_stage1_outcome_modes import HALF_SIZE_M, NEIGHBORS  # noqa: E402

W, H = 400, 300


def camera(mj, lookat, distance, azimuth, elevation):
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def shot(sim, renderer, cam):
    renderer.update_scene(sim.data, camera=cam)
    return renderer.render().copy()


def label(image, text, sub=None):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 34 if sub else 18], fill=(0, 0, 0))
    draw.text((5, 3), text, fill=(255, 255, 255))
    if sub:
        draw.text((5, 18), sub, fill=(255, 220, 120))
    return img


def run_probe(sim, snapshot, chunk, noise):
    sim.restore(snapshot)
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise
    for action in actions:
        sim.execute(action)
    for _ in range(HOLD):
        sim.execute(chunk[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="1.0")
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--stage1", default=str(ROOT / "results/jenga/stage1_outcome_modes.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/stage0_noise_oracle_cache.npz"))
    ap.add_argument("--out", default=str(ROOT / "results/jenga/stage1_split_images"))
    args = ap.parse_args()
    stage1 = json.loads(Path(args.stage1).read_text())
    cache = np.load(args.cache, allow_pickle=False)
    s = SCALES.index(float(args.scale))
    t30 = RECORD_AT.index(30)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    flagged = [(i, r) for i, r in enumerate(stage1["rows"])
               if r["grade"][args.scale] == "unanimous_safe" and r["scores"][args.scale]["full"]]
    replay = JengaReplay(args.lmdb)
    manifest = []
    with tempfile.TemporaryDirectory(prefix="jenga_split_img_") as temp:
        sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
        import mujoco as mj
        renderer = mj.Renderer(sim.model, height=H, width=W)
        try:
            for i, row in flagged:
                ep, start = row["episode_id"], row["chunk_start"]
                endings = corner_displacements_mm(cache["start_pose"][i], cache["pose"][i, s, :, t30],
                                                  HALF_SIZE_M, NEIGHBORS)
                test = two_mode_test(endings, 0.0)
                groups = [np.flatnonzero(test.labels == g) for g in (0, 1)]
                means = [endings[g].mean(0) for g in groups]
                if len(groups[0]) < len(groups[1]):  # A = majority, B = minority
                    groups, means = groups[::-1], means[::-1]
                reps = [int(g[np.argmin(np.linalg.norm(endings[g] - m, axis=1))])
                        for g, m in zip(groups, means)]
                diff = (means[1] - means[0]).reshape(2, 8, 3)
                moved = int(np.argmax(np.linalg.norm(diff, axis=2).mean(1)))
                block_name = ("left", "right")[moved]
                shift_mm = diff[moved].mean(0)
                tilts = cache["tilt"][i, s, :, t30, moved]

                episode = replay.episode(ep)
                sim.reset(int(ep))
                for action in episode.actions[:start]:
                    sim.execute(action)
                snapshot = sim.snapshot()
                chunk = episode.actions[start:start + HORIZON]
                body = sim.tracked_block_ids[1 + moved]
                focus = sim.data.xpos[body].copy()
                middle = sim.data.xpos[sim.tracked_block_ids[0]].copy()
                lookat = (focus + middle) / 2
                oblique = camera(mj, lookat, .32, 135, -25)
                top = camera(mj, lookat, .30, 90, -89)
                start_img = shot(sim, renderer, oblique)
                noise = cache["snippets"] * SCALES[s]
                views = []
                for g, rep in enumerate(reps):
                    run_probe(sim, snapshot, chunk, noise[rep])
                    views.append((shot(sim, renderer, oblique), shot(sim, renderer, top),
                                  1000 * sim.data.xpos[body].copy()))
                sim.restore(snapshot)
                overlay = np.clip(.5 * views[0][1] + .5 * views[1][1], 0, 255).astype(np.uint8)
                a, b = views[0][1].astype(int), views[1][1].astype(int)
                changed = np.abs(a - b).sum(2) > 40
                overlay[changed & (a.sum(2) > b.sum(2))] = (255, 60, 60)
                overlay[changed & (a.sum(2) <= b.sum(2))] = (60, 220, 255)

                na, nb = len(groups[0]), len(groups[1])
                ta = f"tilt {tilts[groups[0]].min():.1f}-{tilts[groups[0]].max():.1f} deg"
                tb = f"tilt {tilts[groups[1]].min():.1f}-{tilts[groups[1]].max():.1f} deg"
                pos_shift = views[1][2] - views[0][2]
                panels = [
                    label(start_img, f"ep{ep} chunk {start}: before chunk",
                          f"{block_name} neighbor centred; {row['stratum']}"),
                    label(views[0][0], f"A: {na}/64 runs (oblique, settled)", ta),
                    label(views[1][0], f"B: {nb}/64 runs (oblique, settled)", tb),
                    label(views[0][1], "A from above", None),
                    label(views[1][1], "B from above", None),
                    label(overlay, "overlay: changed pixels",
                          f"B-A {block_name} centre: {np.linalg.norm(pos_shift):.1f} mm"),
                ]
                sheet = Image.new("RGB", (3 * W, 2 * H))
                for k, p in enumerate(panels):
                    sheet.paste(p, ((k % 3) * W, (k // 3) * H))
                name = f"split_ep{ep}_chunk{start}.png"
                sheet.save(out / name)
                manifest.append({"file": name, "episode_id": ep, "chunk_start": start,
                                 "stratum": row["stratum"], "moved_neighbor": block_name,
                                 "group_sizes": [na, nb], "representative_runs": reps,
                                 "group_mean_corner_shift_mm": shift_mm.round(2).tolist(),
                                 "representative_centre_shift_mm": pos_shift.round(2).tolist(),
                                 "moved_neighbor_tilt_deg_range": [
                                     [float(tilts[g].min()), float(tilts[g].max())] for g in groups]})
                print(manifest[-1], flush=True)
        finally:
            renderer.close()
            sim.close()
            replay.close()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
