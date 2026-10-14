"""Verification video for the 2026-10-13 data correction: old vs new perception-error perturbations.

Per scene, four synchronised panels (close camera, every control step, 3.8 s):
  1 OLD perturbation, robot holding still   2 NEW perturbation (same draw), robot holding still
  3 NEW perturbation, robot acting: a future with an event   4 NEW perturbation, robot acting: a future without
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_data as pdata  # noqa: E402
import phase2_models as pm  # noqa: E402
import phase2_scenes as ps  # noqa: E402

OUTV = ROOT / "results/program_b/perturbation_fix_verification.mp4"
SUMMARY = ROOT / "results/jenga/perturbation_fix_video_summary.json"
SCENES = ["F2_500022_s34", "F2_500148_s10", "F2_500010_s18", "F2_500076_s2", "F1_500095_s26", "F1_500041_s50", "T2_500003_s26", "T2_500010_s34"]


def label(img, lines, color):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).resize((256, 256)); d = ImageDraw.Draw(im)
    y = 4
    for ln in lines:
        d.rectangle([2, y - 1, 4 + 6 * len(ln), y + 11], fill=(0, 0, 0)); d.text((4, y), ln, fill=color); y += 14
    return np.asarray(im)


def events(tr, start):
    end = tr[-1]
    return bool(((np.linalg.norm(end[:, :3] - start[:, :3], axis=1) >= 0.03) |
                 (np.abs(ps.tilt_deg(end[:, 3:]) - ps.tilt_deg(start[:, 3:])) >= 45)).any())


def run(sim, snap, err, valid, chunk, noise, render):
    ps.VALID_PERTURBATION = valid
    sim.restore(snap); sim.perturb(*err)
    frames, tr = [render()], []
    cmds = [np.r_[sim.target, 0.0]] * 38 if chunk is None else list(np.asarray(chunk, np.float32) - np.c_[noise, np.zeros(8)]) + [chunk[-1]] * 30
    for c in cmds:
        sim.execute(c); tr.append(sim.object_state()); frames.append(render())
    ps.VALID_PERTURBATION = True
    return frames, np.stack(tr)


def main():
    import imageio_ffmpeg
    import mujoco
    import uncertainty_monitor_dev as um
    from jenga_render_frames import close_camera
    snippets = np.load(um.BENCH)["snippets"]; pc, rc = ps.pooled_error_model()
    writer = imageio_ffmpeg.write_frames(str(OUTV), (1024, 256), fps=8, codec="libx264", quality=8, macro_block_size=1)
    writer.send(None)
    summary = []
    for name in SCENES:
        fam, seed, s = name.split("_"); seed, index = int(seed), int(s[1:])
        prep = pdata.prepare(fam, seed)
        if prep is None:
            continue
        scene, sim, acts, starts = prep
        if index not in starts:
            index = starts[min(len(starts) - 1, 2)]
        r = mujoco.Renderer(sim.model, height=384, width=384); cam = close_camera(mujoco)
        render = lambda: (r.update_scene(sim.data, camera=cam), r.render().copy())[1]
        for i, a in enumerate(acts):
            if i == index:
                break
            sim.execute(a)
        rest = ps.at_rest(sim)
        snap = sim.snapshot(); start = sim.object_state(); chunk = acts[index:index + 8]; n = len(scene["objects"])
        key = (pdata.FAMILY_CODE[fam], seed, index)
        errs = [ps.draw_error(n, pc, rc, pm.SEEDS["h2_futures"], *key, j) for j in range(64)]
        # the draw whose OLD perturbation moves things most with the robot still (the bug's worst case)
        worst, worst_ev = 0, -1.0
        for j in range(16):
            _, tr = run(sim, snap, errs[j], False, None, None, lambda: None)
            m = float(np.linalg.norm(tr[-1][:, :3] - start[:, :3], axis=1).max())
            if m > worst_ev:
                worst, worst_ev = j, m
        f_old, tr_old = run(sim, snap, errs[worst], False, None, None, render)
        f_new, tr_new = run(sim, snap, errs[worst], True, None, None, render)
        yes = no = None
        for j in range(64):
            _, tr = run(sim, snap, errs[j], True, chunk, snippets[j], lambda: None)
            if events(tr, start) and yes is None:
                yes = j
            if not events(tr, start) and no is None:
                no = j
            if yes is not None and no is not None:
                break
        f_yes = run(sim, snap, errs[yes], True, chunk, snippets[yes], render)[0] if yes is not None else None
        f_no = run(sim, snap, errs[no], True, chunk, snippets[no], render)[0] if no is not None else None
        r.close()
        row = {"scene": name, "at_rest": rest, "old_still_moved_mm": round(worst_ev * 1000, 1),
               "new_still_event": events(tr_new, start), "has_event_future": yes is not None, "has_quiet_future": no is not None}
        summary.append(row); print(row, flush=True)
        title = np.zeros((256, 1024, 3), np.uint8)
        title = label(np.zeros((256, 256, 3), np.uint8), [name, f"scene at rest: {rest}"], (255, 255, 255))
        blank = np.zeros((256, 256, 3), np.uint8)
        for _ in range(10):
            writer.send(np.concatenate([title, blank, blank, blank], 1))
        for t in range(len(f_old)):
            panels = [label(f_old[t], ["OLD perturbation", "robot holds still"], (255, 120, 120)),
                      label(f_new[t], ["NEW perturbation", "robot holds still"], (140, 255, 140)),
                      label(f_yes[t], ["NEW, robot acts", "future with event"], (255, 200, 120)) if f_yes else label(blank, ["NEW, robot acts", "(no future with event)"], (200, 200, 200)),
                      label(f_no[t], ["NEW, robot acts", "future without event"], (140, 220, 255)) if f_no else label(blank, ["NEW, robot acts", "(no quiet future)"], (200, 200, 200))]
            writer.send(np.concatenate(panels, 1))
    writer.close()
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print("video", OUTV, flush=True)


if __name__ == "__main__":
    main()
