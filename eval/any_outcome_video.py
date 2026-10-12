"""Video: H2 scenes that are forks only under the any-outcome label (Path C) - two real futures side by side.

For each chosen state, the H2 truth futures are re-simulated in their original order (identical seeds) and two are
rendered every control step: one where the varying event happens and one where it does not. The events are
re-checked on the re-simulated traces.
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

CHOSEN = ["F1_500073_s18", "F1_500155_s26", "F1_500544_s42", "F2_500141_s10", "F2_500467_s34", "F2_500582_s18"]
OUTV = ROOT / "results/program_b/phase2/any_outcome_forks.mp4"


def events(trace, start, obj):
    end = trace[-1, obj]; s = start[obj]
    tilt = abs(ps.tilt_deg(end[3:]) - ps.tilt_deg(s[3:])) >= 45
    off = end[0] < 0.3 or end[0] > 0.7 or abs(end[1]) > 0.4 or end[2] < ps.TABLE_TOP - 0.01
    mov = np.linalg.norm(end[:3] - s[:3]) >= 0.03
    return {"tilts >=45 deg": bool(tilt), "leaves table": bool(off), "moves >=30 mm": bool(mov)}


def render_state(row):
    import mujoco
    import uncertainty_monitor_dev as um
    from jenga_render_frames import close_camera
    fam, seed, s = row["state"].split("_"); seed, index = int(seed), int(s[1:])
    scene, sim, acts, starts = pdata.prepare(fam, seed)
    snippets = np.load(um.BENCH)["snippets"]; pos_cov, rot_cov = ps.pooled_error_model()
    n_obj = len(scene["objects"])
    r = mujoco.Renderer(sim.model, height=384, width=384); cam = close_camera(mujoco)
    for i, a in enumerate(acts):
        if i == index:
            break
        sim.execute(a)
    snap = sim.snapshot(); start = sim.object_state()
    chunk = acts[index:index + 8]
    want = {row["future_yes"]: "yes", row["future_no"]: "no"}
    frames, traces = {}, {}
    for j in range(max(want) + 1):
        err = ps.draw_error(n_obj, pos_cov, rot_cov, pm.SEEDS["h2_futures"], {"F1": 1, "F2": 2}[fam], seed, index, j)
        sim.restore(snap); sim.perturb(*err)
        a = np.asarray(chunk, np.float32).copy(); a[:, :3] -= snippets[j]
        tr, fr = [], []
        if j in want:
            r.update_scene(sim.data, camera=cam); fr.append(r.render().copy())
        for c in list(a) + [chunk[-1]] * 30:
            sim.execute(c); tr.append(sim.object_state())
            if j in want:
                r.update_scene(sim.data, camera=cam); fr.append(r.render().copy())
        if j in want:
            frames[want[j]] = np.stack(fr); traces[want[j]] = np.stack(tr)
    r.close()
    check = {k: events(traces[k], start, row["object"])[row["event"]] for k in ("yes", "no")}
    return frames, check


def label(img, lines, color=(255, 255, 255)):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    y = 6
    for ln in lines:
        d.rectangle([4, y - 2, 4 + 7 * len(ln), y + 12], fill=(0, 0, 0)); d.text((6, y), ln, fill=color); y += 16
    return np.asarray(im)


def main():
    import imageio.v2 as imageio
    rows = {r["state"]: r for r in json.loads((pm.OUT / "any_outcome_only_states.json").read_text())}
    writer = imageio.get_writer(OUTV, fps=8, codec="libx264", quality=8)
    summary = []
    for name in CHOSEN:
        row = rows[name]
        frames, check = render_state(row)
        what = f"{'target' if row['object_is_target'] else 'object ' + str(row['object'])} {row['event']} in {row['count_of_64']}/64 futures"
        title = np.zeros((384, 768, 3), np.uint8)
        title = label(title, [f"H2 scene {name}  ({row['family']} {'rows' if row['family'] == 'F1' else 'stacks'}, {row['primitive']})",
                              "Harm label: SAFE (no other object falls or moves)",
                              f"Any-outcome label: FORK - {what}",
                              f"re-simulation check: event in left={check['yes']}, right={check['no']}"])
        for _ in range(16):
            writer.append_data(title)
        for rep in range(2):
            for t in range(len(frames["yes"])):
                L = label(frames["yes"][t], ["future A: event happens", f"t={t * 0.1:.1f}s"], (255, 120, 120))
                R = label(frames["no"][t], ["future B: event does not happen", f"t={t * 0.1:.1f}s"], (140, 255, 140))
                writer.append_data(np.concatenate([L, R], 1))
        summary.append({"state": name, "what": what, "check": check})
        print(name, what, check, flush=True)
    writer.close()
    (pm.OUT / "any_outcome_video_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("video", OUTV, flush=True)


if __name__ == "__main__":
    main()
