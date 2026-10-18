"""Long comparison video (dev10, model_OIL): POLICY ALONE | SELECTION FILTER | OPTIMISATION FILTER on the same episodes
(same start, same noise seeds), replayed from the saved decision states.

Readable intervention view: the hand's paths are drawn INTO the close-up camera image (projected with MuJoCo's own
camera frustum): green = the executed path when the filter did not act (= the policy's own); when the filter acts, a
dashed white path = what the policy wanted and a thick orange path = what the filter executed instead, with a plain
sentence (how far, which way, slower or not). Timeline at the bottom: orange ticks = decisions where that filter acted.
Outcome per column uses the audited definition (results/jenga/outcome_audit_dev10.json): success = red block lifted
>= 2 cm and at the end still held or placed upright, no neighbour topple.

usage: python3 eval/dev10_long_video.py [episodes...]  -> results/program_b/dev10_videos/long_comparison.mp4
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
TAG = os.environ.get("VID_TAG", "dev10")                 # s4: Stage 4 runs (policy | hold and re-plan | G-CEM)
OUT = ROOT / ("results/program_b/dev10_videos" if TAG == "dev10" else f"results/program_b/{TAG}_videos")
RUNS = ROOT / "results/program_b/policy_rollouts"
ARMS = [("record", "POLICY ALONE"), ("cbfmsoil", "SELECTION FILTER"), ("cbfmoil", "OPTIMISATION FILTER")] if TAG == "dev10" else \
    [("record", "POLICY ALONE"), ("cbfh", "HOLD AND RE-PLAN"), ("cbfg", "G-CEM (optimises the one chunk)")]
EPS = [12101, 12111, 12115, 12144, 12102, 12135, 12075, 12103, 12110, 12084, 12099, 12097, 12096, 12080]
CW, CH, SH, HH, PAD = 520, 390, 74, 64, 8
BG, FG, GREY, ORANGE, GREEN, RED, WHITE = (18, 22, 21), (225, 232, 228), (140, 150, 146), (255, 150, 30), (80, 210, 120), (240, 90, 75), (250, 250, 250)
LABEL = {"S": ("success", GREEN), "T": ("neighbour toppled", RED), "D": ("lifted, then dropped it", RED),
         "F": ("knocked the red block over", RED), "M": ("never picked it up", ORANGE),
         "A": ("success; red block tipped over afterwards", ORANGE)}


def audit_labels():
    """v4 / v4b labels from the replay audit: S success (v4b), A v4 only (tipped after extraction), T neighbour toppled,
    F red block knocked over before extraction, M never extracted."""
    A = json.loads((ROOT / f"results/jenga/outcome_audit_{TAG}.json").read_text()); lab = {}
    for a, _ in ARMS:
        lab[a] = {}
        for r in A[f"paired_{a}_{TAG}"]["rows"]:
            lab[a][str(r["ep"])] = ("T" if r["logged_neighbour_toppled"] else "F" if r["red_fell_before_extraction"] else
                                    "M" if not r["extracted"] else "A" if r["red_fell_after_extraction"] else "S")
    return lab


def g_text(entry):
    """Plain description of a G-CEM / hold decision from its log entry."""
    if entry.get("kind") == "hold" or entry.get("fix_kind") == "fallback_hold":
        return "held still for this chunk"
    th = entry.get("theta")
    if not th:
        return "no safe version found: least risky one"
    v, k, s_, b = th; parts = []
    if v < 0.95:
        parts.append(f"speed {100 * v:.0f}%")
    if k >= 0.5:
        parts.append(f"paused {k * 0.1:.1f} s first")
    if s_ < 0.95:
        parts.append(f"did {100 * s_:.0f}% of the motion")
    if b >= 2:
        parts.append(f"backed off {b:.0f} mm")
    return ", ".join(parts) or "policy's chunk kept (passed the check)"


def project(scene, pts, W, H):
    """World points (N,3) -> pixel coords (N,2) and in-front mask, with MuJoCo's GL camera (pos/forward/up/frustum).
    scene.camera holds a stereo pair (+-ipd/2): use their mean; frustum_width 0 means 'from the aspect ratio'."""
    c = scene.camera[0]
    pos = (np.array(scene.camera[0].pos) + np.array(scene.camera[1].pos)) / 2
    fwd, up = np.array(c.forward), np.array(c.up)
    right = np.cross(fwd, up)
    v = pts - pos; z = v @ fwd; x = v @ right; y = v @ up
    xn, yn = x / z * c.frustum_near, y / z * c.frustum_near
    hw = c.frustum_width or (c.frustum_top - c.frustum_bottom) / 2 * W / H
    left, rgt = c.frustum_center - hw, c.frustum_center + hw
    u = (xn - left) / (rgt - left) * W; w = (c.frustum_top - yn) / (c.frustum_top - c.frustum_bottom) * H
    return np.stack([u, w], 1), z > 0


def describe(nom, exe):
    d = (exe - nom)[:, :3] * 1000; m = float(np.abs(d).max())
    dz = float(d[:, 2].mean()); dh = float(np.linalg.norm(d[:, :2].mean(0)))
    parts = []
    if abs(dz) >= 3:
        parts.append(f"{'lower' if dz < 0 else 'higher'} by {abs(dz):.0f} mm")
    if dh >= 3:
        parts.append(f"sideways by {dh:.0f} mm")
    ln = lambda p: float(np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1).sum())
    if ln(exe) < 0.8 * ln(nom):
        parts.append("slower")
    return m, (", ".join(parts) or "small adjustment")


def render(job):
    """One arm's episode -> its column video (CW x (20 + CH + SH)) on disk."""
    arm, ep, k_total = job
    log = {}
    if TAG != "dev10":
        for l in (ROOT / f"results/jenga/paired_{arm}_{TAG}.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r["episode"] == ep:
                log = {c["chunk"]: c for c in r["log"]}
    import mujoco
    import imageio_ffmpeg
    import inspector_data as insp
    import plausibility as pl
    from PIL import Image, ImageDraw
    from jenga_render_frames import close_camera
    with np.load(RUNS / f"paired_{arm}_{TAG}" / f"ep{ep:03d}.npz") as f:
        S = {k: f[k] for k in f.files}
    noises = pl.episode_noises(ep, int(S["chunk_index"][-1]) + 1)
    sim = insp._sim()
    rc = mujoco.Renderer(sim.model, height=CH, width=CW); rf = mujoco.Renderer(sim.model, height=135, width=180)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE          # front view, slightly above: heights visible
    cam.lookat[:] = [0.5, 0.0, 0.46]; cam.distance, cam.azimuth, cam.elevation = 0.27, 180.0, -18.0
    top = close_camera(mujoco)                                                     # top-down inset
    w = imageio_ffmpeg.write_frames(str(OUT / "cols" / f"{arm}_{ep}.mp4"), (CW, 20 + CH + SH), fps=10, codec="libx264",
                                    quality=8, macro_block_size=1); w.send(None)
    n = len(S["chunk_index"]); fired_so_far = 0
    for d in range(n):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        k = int(S["chunk_index"][d]); acted = bool(S["intervened"][d]) and arm != "record"
        fired_so_far += acted
        nom, exe = S["nominal"][d], S["executed"][d]
        start = sim.data.site_xpos[sim.site].copy()
        msg = describe(nom, exe) if acted else None
        if acted and TAG != "dev10":
            msg = (msg[0], g_text(log.get(k, {})))
        for j, a in enumerate(exe):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]; sim.execute(a)
            rc.update_scene(sim.data, camera=cam); img = Image.fromarray(rc.render())
            gl = rc.scene
            dr = ImageDraw.Draw(img)
            paths = [(np.vstack([start, exe[:, :3]]), ORANGE, 6, False), (np.vstack([start, nom[:, :3]]), WHITE, 3, True)] if acted \
                else [(np.vstack([start, exe[:, :3]]), GREEN, 4, False)]
            ends = []
            for P, col, wd, dashed in paths:
                uv, ok = project(gl, P, CW, CH)
                for q in range(len(uv) - 1):
                    if ok[q] and ok[q + 1] and (not dashed or q % 2 == 0):
                        dr.line([tuple(uv[q]), tuple(uv[q + 1])], fill=col, width=wd)
                x, y = uv[-1]; r = wd + 3
                (dr.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3) if dashed else dr.ellipse([x - r, y - r, x + r, y + r], fill=col))
                ends.append((x, y))
            if acted:                                                              # arrow: policy's target -> filter's target
                (xo, yo), (xw, yw) = ends
                if np.hypot(xo - xw, yo - yw) > 6:
                    dr.line([(xw, yw), (xo, yo)], fill=ORANGE, width=2)
            rf.update_scene(sim.data, camera=top); img.paste(Image.fromarray(rf.render()), (CW - 184, CH - 139))
            fr = Image.new("RGB", (CW, 20 + CH + SH), BG); fr.paste(img, (0, 20)); dr = ImageDraw.Draw(fr)
            if arm == "record":
                dr.rectangle([0, 0, CW, 19], fill=(60, 68, 65)); dr.text((6, 4), f"t = {(k * 8 + j) * 0.1:4.1f} s   no filter", fill=WHITE)
            elif acted:
                dr.rectangle([0, 0, CW, 19], fill=(200, 110, 10)); dr.text((6, 4), f"t = {(k * 8 + j) * 0.1:4.1f} s   FILTER ACTED", fill=WHITE)
            else:
                dr.rectangle([0, 0, CW, 19], fill=(40, 110, 70)); dr.text((6, 4), f"t = {(k * 8 + j) * 0.1:4.1f} s   filter checked: policy's action OK", fill=WHITE)
            y0 = 20 + CH
            if arm != "record":
                x0, x1 = 8, CW - 8
                for q in range(n):
                    x = x0 + (x1 - x0) * q / max(k_total - 1, 1)
                    dr.line([(x, y0 + 6), (x, y0 + 18)], fill=ORANGE if S["intervened"][q] else (55, 65, 61), width=2)
                xc = x0 + (x1 - x0) * d / max(k_total - 1, 1); dr.polygon([(xc - 5, y0 + 27), (xc + 5, y0 + 27), (xc, y0 + 20)], fill=FG)
                if acted:
                    m, txt = msg
                    dr.text((8, y0 + 32), f"white dashed = policy wanted;  orange = filter executed", fill=FG)
                    dr.text((8, y0 + 50), f"changed up to {m:.0f} mm: {txt}", fill=ORANGE)
                else:
                    dr.text((8, y0 + 32), "green = hand path (the policy's own action)", fill=GREY)
                    dr.text((8, y0 + 50), f"filter acted {fired_so_far} times so far", fill=GREY)
            else:
                dr.text((8, y0 + 32), "green = hand path (the policy's own action)", fill=GREY)
            w.send(np.asarray(fr))
    w.close(); rc.close(); rf.close(); sim.close()
    return arm, ep


def frames(path):
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(str(path)); meta = next(rd); w, h = meta["size"]
    return [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in rd]


def main():
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    eps = [int(v) for v in sys.argv[1:]] or EPS
    (OUT / "cols").mkdir(parents=True, exist_ok=True)
    lab = json.loads((ROOT / "results/program_b/dev10_videos/labels.json").read_text()) if TAG == "dev10" else audit_labels()
    res = {a: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/paired_{a}_{TAG}.jsonl").read_text().splitlines()} for a, _ in ARMS}
    jobs = [(a, e, max(len(res[b][e]["log"]) for b, _ in ARMS)) for e in eps for a, _ in ARMS]
    jobs = [j for j in jobs if not (OUT / "cols" / f"{j[0]}_{j[1]}.mp4").exists() or os.environ.get("RERENDER")]
    with mp.get_context("spawn").Pool(max(1, min(12, len(jobs)))) as pool:
        for a, e in pool.imap_unordered(render, jobs):
            print("rendered", a, e, flush=True)
    W = 3 * CW + 4 * PAD; H = HH + 20 + CH + SH + PAD
    wr = imageio_ffmpeg.write_frames(str(OUT / ("long_comparison.mp4" if TAG == "dev10" else f"{TAG}_long_comparison.mp4")), (W, H), fps=10, codec="libx264", quality=7, macro_block_size=1); wr.send(None)
    total = 0
    for n_e, e in enumerate(eps):
        F = [frames(OUT / "cols" / f"{a}_{e}.mp4") for a, _ in ARMS]; n = max(len(f) for f in F)
        for k in range(n):
            im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
            dr.text((PAD, 6), f"Episode {e}  ({n_e + 1} of {len(eps)}): same start and noise seeds in all three; both filters use the same trigger and world model (model_OIL)", fill=FG)
            for i, (a, name) in enumerate(ARMS):
                x = PAD + i * (CW + PAD); t, c = LABEL[lab[a][str(e)]]
                nint = sum(bool(z.get("intervened")) for z in res[a][e]["log"]) if a != "record" else None
                dr.text((x, 26), name, fill=FG); dr.text((x, 42), f"outcome: {t}" + (f"   (acted {nint}x)" if nint is not None else ""), fill=c)
                im.paste(Image.fromarray(F[i][min(k, len(F[i]) - 1)]), (x, HH))
                if k >= len(F[i]):                         # this arm's episode already ended (block 5 cm up, or time)
                    dr.rectangle([x, HH, x + CW, HH + 19], fill=(70, 70, 120)); dr.text((x + 6, HH + 4), "episode finished (frozen on its last frame)", fill=WHITE)
            wr.send(np.asarray(im)); total += 1
        card = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(card)
        dr.text((60, 180), f"Episode {e}", fill=FG)
        for i, (a, name) in enumerate(ARMS):
            t, c = LABEL[lab[a][str(e)]]; dr.text((60, 210 + 26 * i), f"{name}: {t}", fill=c)
        for _ in range(25):
            wr.send(np.asarray(card)); total += 1
        print("composed", e, f"{total / 10:.0f} s so far", flush=True)
    wr.close(); print("done", f"{total / 10:.0f} s", flush=True)


if __name__ == "__main__":
    main()
