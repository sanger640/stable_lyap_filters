"""dev10 side-by-side videos: the same episode (same start, same execution-noise seeds) with the policy alone, the
optimisation-only filter and the selection filter, both filters on model_OIL. Replayed from the saved decision states
(as inspector_data.video_job). Under each view: when the filter fired (timeline, orange ticks) and the action difference
at the current decision: the policy's own chunk (grey) vs the executed chunk (orange), top view (x-y) and side view (x-z),
plus the largest per-step change in mm.

usage: python3 eval/dev10_videos.py 12101 12096 12144   -> results/program_b/dev10_videos/{triple_ep<e>.mp4, reel.mp4}
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ["EXEC_NOISE_SCALE"] = "2"
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/program_b/dev10_videos"
RUNS = ROOT / "results/program_b/policy_rollouts"
ARMS = {"policy": ("paired_record_dev10", "POLICY ALONE"), "opt": ("paired_cbfmoil_dev10", "OPTIMISATION FILTER"),
        "sel": ("paired_cbfmsoil_dev10", "SELECTION FILTER")}
PW, VH, SH = 560, 260, 120                         # panel width, video height, strip height
BG, FG, GREY, ORANGE, GREEN, RED = (20, 26, 24), (220, 230, 225), (150, 158, 154), (235, 140, 30), (90, 200, 130), (235, 95, 80)


def strip(S, d, k_total):
    """Action-difference strip for decision d: timeline + top / side views of policy vs executed chunk."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (PW, SH), BG); dr = ImageDraw.Draw(im)
    n = len(S["chunk_index"]); x0, x1 = 8, PW - 8
    dr.text((x0, 2), "filter fired:", fill=GREY)
    for j in range(n):
        x = x0 + 80 + (x1 - x0 - 80) * j / max(k_total - 1, 1)
        dr.line([(x, 4), (x, 14)], fill=ORANGE if S["intervened"][j] else (60, 70, 66), width=2)
    xc = x0 + 80 + (x1 - x0 - 80) * d / max(k_total - 1, 1)
    dr.polygon([(xc - 4, 22), (xc + 4, 22), (xc, 16)], fill=FG)
    nom, exe = S["nominal"][d][:, :3], S["executed"][d][:, :3]
    diff = float(np.abs(exe - nom).max() * 1000) if S["intervened"][d] else 0.0
    for v, (a, b, lab) in enumerate(((0, 1, "top view (x-y)"), (0, 2, "side view (x-z)"))):
        ox, oy, w, h = 8 + v * 200, 30, 180, 84
        dr.rectangle([ox, oy, ox + w, oy + h], outline=(60, 70, 66))
        dr.text((ox + 4, oy + 2), lab, fill=GREY)
        pts = np.vstack([nom, exe]); c = pts.mean(0); span = max(float(np.ptp(pts[:, [a, b]], 0).max()), 0.02)
        to = lambda p: (ox + w / 2 + (p[a] - c[a]) / span * (w - 30), oy + h / 2 + 6 - (p[b] - c[b]) / span * (h - 30))
        for path, col in ((nom, GREY), (exe, ORANGE)):
            dr.line([to(p) for p in path], fill=col, width=2); x, y = to(path[-1]); dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    tx = 8 + 2 * 200 + 4
    dr.text((tx, 32), "grey: policy's chunk", fill=GREY); dr.text((tx, 48), "orange: executed", fill=ORANGE)
    dr.text((tx, 72), f"max change {diff:5.1f} mm" if S["intervened"][d] else "unchanged", fill=ORANGE if S["intervened"][d] else GREEN)
    dr.text((tx, 92), f"fired {int(S['intervened'][:d + 1].sum())} / {d + 1} so far", fill=GREY)
    return im


def render(job):
    """One arm's episode -> list of panel frames (numpy), with the action strip."""
    arm, ep, path, k_total = job
    import mujoco
    import inspector_data as insp
    import uncertainty_monitor_dev as um
    from PIL import Image, ImageDraw
    from jenga_render_frames import close_camera
    with np.load(path) as f:
        S = {k: f[k] for k in f.files}
    snippets = np.load(um.BENCH)["snippets"]
    nrng = np.random.default_rng([20261502, ep])
    noises = [snippets[nrng.integers(len(snippets))] * insp.EXEC_SCALE for _ in range(int(S["chunk_index"][-1]) + 1)]
    sim = insp._sim()
    rf = mujoco.Renderer(sim.model, height=240, width=320); rc = mujoco.Renderer(sim.model, height=240, width=240)
    cam = close_camera(mujoco); frames = []
    for d in range(len(S["chunk_index"])):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        k = int(S["chunk_index"][d]); st = strip(S, d, k_total)
        txt, col = ("filter changed the action", (215, 130, 20)) if S["intervened"][d] else (("monitor flag", (194, 73, 29)) if S["alarm"][d] else ("ok", (47, 125, 79)))
        if arm == "policy":
            txt, col = "policy alone (no filter)", (70, 80, 76)
        for j, a in enumerate(S["executed"][d]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]
            sim.execute(a)
            rf.update_scene(sim.data, camera="cam_fixed"); rc.update_scene(sim.data, camera=cam)
            im = Image.new("RGB", (PW, 20 + 240 + SH), BG)
            im.paste(Image.fromarray(rf.render()), (0, 20)); im.paste(Image.fromarray(rc.render()), (320, 20))
            dr = ImageDraw.Draw(im); dr.rectangle([0, 0, PW, 19], fill=col)
            dr.text((6, 4), f"t = {(k * 8 + j) * 0.1:4.1f} s   decision {d}   {txt}", fill=(255, 255, 255))
            if arm != "policy":
                im.paste(st, (0, 260))
            frames.append(np.asarray(im))
    rf.close(); rc.close(); sim.close()
    np.save(OUT / "frames" / f"{arm}_{ep}.npy", np.stack(frames))
    return arm, ep


def outcome(r):
    if r["neighbour_toppled"]:
        return "neighbour toppled", RED
    if r.get("seconds_to_3cm") is not None:
        return f"success (3 cm at {r['seconds_to_3cm']} s)", GREEN
    return f"no topple, not lifted (max {r['max_lift_mm']:.0f} mm)", ORANGE


def compose(ep, res, writer):
    from PIL import Image, ImageDraw
    F = {a: np.load(OUT / "frames" / f"{a}_{ep}.npy", mmap_mode="r") for a in ARMS}
    n = max(len(v) for v in F.values()); W, H = 3 * PW + 16, 40 + 20 + 240 + SH + 8
    for k in range(n):
        im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
        dr.text((8, 4), f"dev10 episode {ep}: same start, same noise seeds; both filters use the round-1 world model (model_OIL)", fill=FG)
        for i, (a, (_, name)) in enumerate(ARMS.items()):
            t, c = outcome(res[a]); x = 8 + i * (PW + 4)
            dr.text((x, 22), f"{name}  ->  {t}", fill=c)
            im.paste(Image.fromarray(np.asarray(F[a][min(k, len(F[a]) - 1)])), (x, 40))
        writer.send(np.asarray(im))
    card = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(card)
    dr.text((40, 120), f"Episode {ep}", fill=FG)
    for i, (a, (_, name)) in enumerate(ARMS.items()):
        t, c = outcome(res[a]); nint = sum(bool(x.get("intervened")) for x in res[a]["log"])
        dr.text((40, 150 + 24 * i), f"{name}: {t}" + (f"; filter fired at {nint} of {len(res[a]['log'])} decisions" if a != "policy" else ""), fill=c)
    for _ in range(30):
        writer.send(np.asarray(card))


def main():
    import imageio_ffmpeg
    eps = [int(v) for v in sys.argv[1:]]
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    res = {a: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{t}.jsonl").read_text().splitlines()} for a, (t, _) in ARMS.items()}
    jobs = []
    for e in eps:
        k_total = max(len(res[a][e]["log"]) for a in ARMS)
        jobs += [(a, e, str(RUNS / t / f"ep{e:03d}.npz"), k_total) for a, (t, _) in ARMS.items()]
    with mp.get_context("spawn").Pool(min(9, len(jobs))) as pool:
        for a, e in pool.imap_unordered(render, jobs):
            print("rendered", a, e, flush=True)
    W, H = 3 * PW + 16, 40 + 20 + 240 + SH + 8
    mk = lambda p: imageio_ffmpeg.write_frames(str(p), (W, H), fps=10, codec="libx264", quality=7, macro_block_size=1)
    reel = mk(OUT / "reel.mp4"); reel.send(None)
    for e in eps:
        w = mk(OUT / f"triple_ep{e}.mp4"); w.send(None)
        compose(e, {a: res[a][e] for a in ARMS}, w); w.close()
        compose(e, {a: res[a][e] for a in ARMS}, reel); print("composed", e, flush=True)
    reel.close(); print("reel done", flush=True)


if __name__ == "__main__":
    main()
