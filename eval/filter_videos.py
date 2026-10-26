"""Annotated four-way filter videos (dashboard): the same episode (same start, same seeds) under four controllers, replayed
from each arm's saved decision states with the run's execution-noise sequence (as eval/h2h4_videos.py), plus what each
filter did to the policy's actions:
  camera view:  the policy's proposed 8 targets for this chunk (dashed white) vs the targets actually executed (solid,
                arm colour), the hand's recent path (thin), gripper command (open / closed)
  action view:  the same two paths top-down at the hand, 1 cm grid, block top and bottom faces, edit in mm
  signal strip: the filter's own safety number with its threshold (shaded where the filter acted), the edit size
                |executed - proposed| (mm, per control step), the TRUE neighbour tilt from the simulator (no filter sees it)
  right column: top-down hand paths of all four arms from the common start (X = neighbour topple), hand height vs time
  pauses:       1.2 s freeze at each arm's first edit and at each topple; end card with per-arm numbers
Sets: A = head-to-head 4 episodes (policy, lfci000, psel, labelled LSF), same episodes as results/program_b/h2h4_videos;
      B = dev episodes 15000-15099 (policy, lfci000, lfcj000, lfck000), episodes by a fixed rule (choose_b()).
Output: results/program_b/filter_videos/{A,B}/ep<i>.mp4
usage: python3 eval/filter_videos.py A|B [ep ...]      (COMPOSE_ONLY=1: re-compose from the saved raw renders)
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
OUT = ROOT / "results/program_b/filter_videos"
RUNS = ROOT / "results/program_b/policy_rollouts"
THR_IRR, TOPPLE = 1.3356, 45.0
GREY, TEAL, ORANGE, PURPLE, BLUE, GREEN = (160, 168, 164), (40, 185, 170), (235, 145, 45), (165, 135, 235), (90, 150, 240), (110, 205, 95)
SETS = {
    "A": {"policy": ("paired_record_h2h4", "POLICY ALONE", GREY), "lfci000": ("paired_lfci000_h2h4", "LABEL-FREE GATED EDIT (lfci000)", TEAL),
          "psel": ("paired_cbfps3m_h2h3", "PSEL (BASELINE)", ORANGE), "lsf": ("paired_lsfa030_h2h3", "LABELLED LSF (BASELINE)", PURPLE)},
    "B": {"policy": ("paired_record_lflsfdev", "POLICY ALONE", GREY), "lfci000": ("paired_lfci000_irrdev", "lfci000: IRR GATE, EDIT + ACTOR", TEAL),
          "lfcj000": ("paired_lfcj000_gate3dev", "lfcj000: WIDER GATE, EDIT + ACTOR", BLUE),
          "lfck000": ("paired_lfck000_gate3dev", "lfck000: WIDER GATE, PURE EDIT", GREEN)}}
EPS_A = [25630, 25765, 25880, 25882, 26043]          # results/program_b/h2h4_videos (h2h4_videos.choose())
FW, FH, AV = 240, 240, 320                            # close camera, action view
PW, STRIP = FW + AV, 96
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def signal(arm, c):
    """(name, value, threshold, unsafe_side, acted) of one chunk's log."""
    if arm.startswith("lfc"):
        return "undo residual r", c.get("irr_r"), THR_IRR, "above", bool(c.get("irr_gate"))
    if arm == "psel":
        return "fork distance h", c.get("h_nom"), 2.0, "below", bool(c.get("intervened"))
    if arm == "lsf":
        return "value V (labelled)", c.get("lsf_vmin"), 0.3, "below", bool(c.get("lsf_overrides"))
    return "no filter", None, None, None, False


def gl_project(cam, w, h, fovy, P):
    """world points (n, 3) -> pixels (n, 2) for an mjvGLCamera (pos / forward / up), vertical fov fovy (deg)."""
    c0, c1 = cam                                              # the scene's two eye cameras (mono render = their mean)
    pos, fw, up = (np.asarray(c0.pos) + c1.pos) / 2, np.asarray(c0.forward), np.asarray(c0.up)
    right = np.cross(fw, up); d = np.asarray(P) - pos
    x, y, z = d @ right, d @ up, np.maximum(d @ fw, 1e-6)
    f = 0.5 * h / np.tan(np.radians(fovy) / 2)
    return np.stack([w / 2 + f * x / z, h / 2 - f * y / z], 1)


def dashed(dr, pts, fill, width=1, on=5, off=4):
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.hypot(*(b - a)))
        for s in np.arange(0, L, on + off):
            e = min(s + on, L)
            dr.line([tuple(a + (b - a) * s / max(L, 1e-6)), tuple(a + (b - a) * e / max(L, 1e-6))], fill=fill, width=width)


def render(job):
    """One arm, one episode -> raw panel video (cameras + action overlay) and a per-frame trace."""
    st, arm, ep = job
    import mujoco
    import imageio_ffmpeg
    import inspector_data as insp
    import uncertainty_monitor_dev as um
    from PIL import Image, ImageDraw, ImageFont
    from jenga_render_frames import close_camera
    run, _, col = SETS[st][arm]
    with np.load(RUNS / run / f"ep{ep}.npz") as f:
        S = {k: f[k] for k in f.files}
    log = {c["chunk"]: c for c in json.loads((RUNS / run / f"ep{ep}.json").read_text())["log"]}
    snippets = np.load(um.BENCH)["snippets"]
    nrng = np.random.default_rng([20261502, ep])
    noises = [snippets[nrng.integers(len(snippets))] * insp.EXEC_SCALE for _ in range(int(S["chunk_index"][-1]) + 1)]
    sim = insp._sim()
    rc = mujoco.Renderer(sim.model, height=FH, width=FW)
    cam = close_camera(mujoco); fov = float(sim.model.vis.global_.fovy)
    font = ImageFont.truetype(FONT, 11)
    geoms = [next(i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == b) for b in sim.tracked_block_ids]
    out = OUT / st / "raw" / f"{arm}_ep{ep}.mp4"
    w = imageio_ffmpeg.write_frames(str(out), (PW, FH), fps=10, codec="libx264", quality=8, macro_block_size=1); w.send(None)
    T = {k: [] for k in ("t", "hand", "tilt", "edit", "chunk", "grip")}
    sim.restore((S["state"][0], S["ctrl"][0], S["target"][0], float(S["gripper"][0])))
    blocks = sim.block_diagnostics()["position"].copy()
    trail = []
    PX = AV / 0.10                                            # action view: 10 cm across, centred on the hand

    def footprint(i):                                         # block i: top and bottom face corners (world)
        b, gid = sim.tracked_block_ids[i], geoms[i]
        R, c, hs = sim.data.xmat[b].reshape(3, 3), sim.data.xpos[b], sim.model.geom_size[gid]
        sq = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * hs[:2]
        return [c + (np.c_[sq, np.full(4, z * hs[2])] @ R.T) for z in (-1, 1)]
    for d in range(len(S["chunk_index"])):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        k = int(S["chunk_index"][d])
        nom, exe = np.asarray(S["nominal"][d], np.float32), np.asarray(S["executed"][d], np.float32)
        plan = np.asarray(S["plan16"][d], np.float32)
        for j, a in enumerate(exe):
            a = a.copy(); a[:3] -= noises[k][j]
            sim.execute(a)
            hand = sim.data.site_xpos[sim.site].copy(); trail = (trail + [hand])[-40:]
            rc.update_scene(sim.data, camera=cam); imc = rc.render(); camc = (rc.scene.camera[0], rc.scene.camera[1])
            im = Image.new("RGB", (PW, FH), (24, 30, 28)); im.paste(Image.fromarray(imc), (0, 0))
            dr = ImageDraw.Draw(im)
            pr = lambda P: gl_project(camc, FW, FH, fov, P)
            tr = pr(np.array(trail))
            if len(tr) > 1:
                dr.line([tuple(p) for p in tr], fill=(235, 235, 235), width=1)
            pe, pn = pr(exe[:, :3]), pr(nom[:, :3])
            dr.line([tuple(p) for p in pe], fill=col, width=3)
            dashed(dr, pn, (255, 255, 255), 2)
            hp = pr(hand[None])[0]; dr.ellipse([hp[0] - 4, hp[1] - 4, hp[0] + 4, hp[1] + 4], outline=col, width=2)
            # action view: top-down, same orientation as the camera (screen right = +y, screen down = +x)
            cx, cy = FW + AV / 2, FH / 2
            av = lambda P: np.stack([cx + (np.asarray(P)[..., 1] - hand[1]) * PX, cy + (np.asarray(P)[..., 0] - hand[0]) * PX], -1)
            for gx in np.arange(-0.05, 0.051, 0.01):
                q = cx + gx * PX; dr.line([q, 0, q, FH], fill=(40, 50, 46)); q = cy + gx * PX; dr.line([FW, q, PW, q], fill=(40, 50, 46))
            for i in range(3):
                bot, top = footprint(i)
                fc = (200, 60, 55) if i == 0 else (185, 160, 120)
                dr.polygon([tuple(p) for p in av(bot)], outline=tuple(v // 2 for v in fc))
                dr.polygon([tuple(p) for p in av(top)], fill=tuple(v // 2 for v in fc), outline=fc)
            rest = av(plan[8:, :3])
            if len(rest) > 1:
                dr.line([tuple(p) for p in rest], fill=(90, 100, 96), width=1)
            qn, qe = av(nom[:, :3]), av(exe[:, :3])
            dashed(dr, np.vstack([av(hand[None]), qn]), (255, 255, 255), 2, 4, 3)
            dr.line([tuple(p) for p in np.vstack([av(hand[None]), qe])], fill=col, width=3)
            dr.ellipse([qe[-1][0] - 4, qe[-1][1] - 4, qe[-1][0] + 4, qe[-1][1] + 4], fill=col)
            dr.ellipse([qn[-1][0] - 4, qn[-1][1] - 4, qn[-1][0] + 4, qn[-1][1] + 4], outline=(255, 255, 255), width=2)
            dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=(235, 235, 235), width=2)
            dz = (exe[:, :3] - nom[:, :3]).mean(0) * 1000
            dr.rectangle([FW, 0, PW, 14], fill=(24, 30, 28)); dr.text((FW + 4, 1), "action view: top-down at the hand, 1 cm grid", fill=(200, 210, 205), font=font)
            dr.rectangle([FW, FH - 16, PW, FH], fill=(24, 30, 28))
            dr.text((FW + 4, FH - 15), f"edit x {dz[0]:+.1f}  y {dz[1]:+.1f}  z {dz[2]:+.1f} mm", fill=(230, 230, 230) if np.abs(dz).max() < 0.5 else col, font=font)
            g = "closed" if exe[j, 3] > 0 else "open"
            dr.rectangle([4, FH - 18, 120, FH - 3], fill=(20, 26, 24)); dr.text((8, FH - 17), f"gripper cmd: {g}", fill=(230, 230, 230), font=font)
            w.send(np.asarray(im))
            T["t"].append((k * 8 + j) * 0.1); T["hand"].append(hand); T["chunk"].append(k); T["grip"].append(exe[j, 3] > 0)
            T["tilt"].append(float(np.max(sim.block_diagnostics()["tilt"][1:])))
            T["edit"].append(float(np.abs(exe[j, :3] - nom[j, :3]).max() * 1000))
    w.close(); rc.close(); sim.close()
    sig = [signal(arm, log.get(int(k), {})) for k in T["chunk"]]
    np.savez(OUT / st / "raw" / f"{arm}_ep{ep}.npz", t=T["t"], hand=np.array(T["hand"]), tilt=T["tilt"], edit=T["edit"],
             chunk=T["chunk"], grip=T["grip"], blocks=blocks, sig=np.array([np.nan if s[1] is None else s[1] for s in sig], float),
             acted=np.array([s[4] for s in sig]))
    return arm, ep


def frames(path):
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(str(path)); meta = next(rd); w, h = meta["size"]
    return [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in rd]


AUDIT = {"A": {"policy": "h2h4", "lfci000": "h2h4", "psel": "h2h3", "lsf": "h2h3"},
         "B": {"policy": "lflsfdev", "lfci000": "irrdev", "lfcj000": "gate3dev", "lfck000": "gate3kdev"}}


def audited(st, arm, ep):
    """the replay-audited outcome row (eval/outcome_audit.py) of one arm's episode"""
    run = SETS[st][arm][0]
    rows = json.loads((ROOT / f"results/jenga/outcome_audit_{AUDIT[st][arm]}.json").read_text())[run]["rows"]
    return next(r for r in rows if r["ep"] == ep)


def outcome(r):
    """replay-audited outcome (v4: lifted or pulled >= 2 cm toward the arm, no topple), never the logged lift flag"""
    if r["logged_neighbour_toppled"]:
        return "neighbour toppled", (245, 140, 120)
    if r["red_fell_before_extraction"]:
        return "red block knocked over", (245, 175, 120)
    return ("block extracted (v4), no topple", (150, 225, 170)) if r["v4_success"] else ("time-out: not extracted", (225, 205, 140))


W, H, RX = 1500, 760, 1140


def compose(st, ep):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    A = SETS[st]
    f12, f11, f14 = ImageFont.truetype(FONT, 12), ImageFont.truetype(FONT, 10), ImageFont.truetype(FONT, 15)
    F = {a: frames(OUT / st / "raw" / f"{a}_ep{ep}.mp4") for a in A}
    T = {a: dict(np.load(OUT / st / "raw" / f"{a}_ep{ep}.npz")) for a in A}
    R = {a: audited(st, a, ep) for a in A}
    n = max(len(v) for v in F.values()); tmax = max(T[a]["t"][-1] for a in A) + 0.1
    pos = {a: (8 + (i % 2) * (PW + 12), 40 + (i // 2) * (FH + STRIP + 30)) for i, a in enumerate(A)}
    ev = {}                                                       # frame -> captions (first edit / topple per arm)
    for a in A:
        e = np.flatnonzero(T[a]["edit"] > 0.5); tp = np.flatnonzero(T[a]["tilt"] >= TOPPLE)
        if len(e):
            ev.setdefault(int(e[0]), []).append((a, f"{A[a][1].split(':')[0].split('(')[0].strip().lower()}: first edit of the policy's move"))
        if len(tp):
            ev.setdefault(int(tp[0]), []).append((a, f"{A[a][1].split(':')[0].split('(')[0].strip().lower()}: NEIGHBOUR TOPPLES"))
    allh = np.concatenate([T[a]["hand"][:, :2] for a in A] + [T[a]["blocks"][:, :2] for a in A])
    lo, hi = allh.min(0) - 0.03, allh.max(0) + 0.03; span = float((hi - lo).max())
    TD = (RX + 10, 60, 330)                                       # top-down box x, y, size

    def td(p):                                                    # world xy -> pixel (x up the screen = +y world)
        q = (np.asarray(p)[..., :2] - lo) / span                  # camera orientation: right = +y, down = +x
        return np.stack([TD[0] + q[..., 1] * TD[2], TD[1] + q[..., 0] * TD[2]], -1)
    ZB = (RX + 10, 450, 330, 150); z_lo = min(T[a]["hand"][:, 2].min() for a in A) - 0.01; z_hi = max(T[a]["hand"][:, 2].max() for a in A) + 0.01

    def strip(dr, a, x, y, k):
        _, lab, col = A[a]; Tt = T[a]; m = min(k, len(Tt["t"]) - 1)
        name, _, thr, side, _ = signal(a, {})
        X0, X1 = x + 104, x + PW - 6; tx = lambda t: X0 + (X1 - X0) * t / tmax
        rows = [(name, Tt["sig"], thr), ("edit (mm)", Tt["edit"], None), ("true tilt (sim)", Tt["tilt"], TOPPLE)]
        for i, (nm, v, th) in enumerate(rows):
            y0, y1 = y + 4 + i * 30, y + 4 + i * 30 + 26
            dr.rectangle([X0, y0, X1, y1], fill=(30, 38, 35)); dr.text((x + 4, y0 + 7), nm, fill=(200, 210, 205), font=f11)
            if i == 0:
                for jj in np.flatnonzero(Tt["acted"]):
                    dr.line([tx(Tt["t"][jj]), y0, tx(Tt["t"][jj]), y1], fill=(95, 70, 30))
            if i == 0 and name == "no filter":
                continue
            vv = np.asarray(v, float); ok = np.isfinite(vv)
            if not ok.any():
                continue
            if i == 1:
                vmax = 12.0
            elif i == 2:
                vmax = 60.0
            else:
                vmax = max(float(np.nanmax(vv)), th * 1.5) if side == "above" else max(float(np.nanmax(vv)), th * 1.2, 1.0)
            vmin = 0.0 if i else min(0.0, float(np.nanmin(vv)))
            ty = lambda u: y1 - 2 - (y1 - y0 - 4) * (np.clip(u, vmin, vmax) - vmin) / (vmax - vmin)
            if th is not None:
                dashed(dr, np.array([[X0, ty(th)], [X1, ty(th)]]), (245, 110, 90) if i == 2 else (230, 200, 120), 1, 3, 3)
            idx = np.flatnonzero(ok)
            pts = [(tx(Tt["t"][q]), ty(vv[q])) for q in idx]
            if len(pts) > 1:
                dr.line(pts, fill=col, width=2 if i != 1 else 1)
            cx = tx(Tt["t"][m]); dr.line([cx, y0, cx, y1], fill=(255, 255, 255))
        if thr is not None:
            dr.text((X1 - 150, y + 4), f"threshold {thr:g} ({'unsafe above' if side == 'above' else 'unsafe below'})", fill=(230, 200, 120), font=f11)

    def draw(k, captions=()):
        im = Image.new("RGB", (W, H), (18, 23, 21)); dr = ImageDraw.Draw(im)
        title = f"Set {st}, episode {ep}: same start, same seeds.  Dashed white = the policy's proposed move, solid = what was executed."
        dr.text((8, 8), title, fill=(225, 232, 228), font=f14)
        if captions:
            dr.rectangle([8, 26, RX - 8, 38], fill=(70, 50, 20)); dr.text((12, 26), "  |  ".join(c for _, c in captions) + "   (paused)", fill=(255, 225, 170), font=f11)
        for a, (x, y) in pos.items():
            _, lab, col = A[a]; t, c = outcome(R[a]); m = min(k, len(F[a]) - 1)
            hl = any(a == ca for ca, _ in captions)
            dr.rectangle([x - 3, y - 3, x + PW + 2, y + FH + STRIP + 18], outline=(255, 210, 120) if hl else (45, 55, 50), width=2 if hl else 1)
            dr.text((x, y + FH + STRIP + 2), f"{lab}", fill=col, font=f12)
            im.paste(Image.fromarray(F[a][m]), (x, y))
            done = k >= len(F[a]) - 1
            if done:
                dr.rectangle([x + 4, y + 4, x + 330, y + 22], fill=(20, 26, 24)); dr.text((x + 8, y + 6), f"ended: {t}", fill=c, font=f12)
            else:
                tt = T[a]["t"][m]; dr.rectangle([x + 4, y + 4, x + 70, y + 20], fill=(20, 26, 24)); dr.text((x + 8, y + 5), f"t {tt:4.1f} s", fill=(230, 230, 230), font=f11)
            strip(dr, a, x, y + FH, k)
        # right column: top-down paths, height vs time, legend
        dr.text((TD[0], TD[1] - 18), "hand paths from above (common start)", fill=(220, 228, 224), font=f12)
        dr.rectangle([TD[0], TD[1], TD[0] + TD[2], TD[1] + TD[2]], fill=(28, 35, 32))
        a0 = next(iter(A)); bl = T[a0]["blocks"]
        for i, b in enumerate(bl[:3]):
            p = td(b); s = 9 if i == 0 else 7
            dr.rectangle([p[0] - s, p[1] - s, p[0] + s, p[1] + s], fill=(200, 60, 55) if i == 0 else (185, 160, 120))
        for a in A:
            _, lab, col = A[a]; m = min(k, len(T[a]["t"]) - 1); P = td(T[a]["hand"][:m + 1])
            if len(P) > 1:
                dr.line([tuple(p) for p in P], fill=col, width=2)
            dr.ellipse([P[-1][0] - 4, P[-1][1] - 4, P[-1][0] + 4, P[-1][1] + 4], fill=col)
            tp = np.flatnonzero(T[a]["tilt"][:m + 1] >= TOPPLE)
            if len(tp):
                q = td(T[a]["hand"][tp[0]]); dr.line([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], fill=col, width=3); dr.line([q[0] - 6, q[1] + 6, q[0] + 6, q[1] - 6], fill=col, width=3)
        dr.text((TD[0] + 4, TD[1] + TD[2] - 16), "red = target block, tan = neighbours, X = topple", fill=(190, 198, 194), font=f11)
        dr.text((ZB[0], ZB[1] - 18), "hand height vs time", fill=(220, 228, 224), font=f12)
        dr.rectangle([ZB[0], ZB[1], ZB[0] + ZB[2], ZB[1] + ZB[3]], fill=(28, 35, 32))
        for a in A:
            _, lab, col = A[a]; m = min(k, len(T[a]["t"]) - 1); tt, zz = T[a]["t"][:m + 1], T[a]["hand"][:m + 1, 2]
            P = [(ZB[0] + ZB[2] * t / tmax, ZB[1] + ZB[3] - ZB[3] * (z - z_lo) / (z_hi - z_lo)) for t, z in zip(tt, zz)]
            if len(P) > 1:
                dr.line(P, fill=col, width=2)
        ly = ZB[1] + ZB[3] + 14
        for a in A:
            _, lab, col = A[a]; dr.line([ZB[0], ly + 6, ZB[0] + 18, ly + 6], fill=col, width=3); dr.text((ZB[0] + 24, ly), lab, fill=col, font=f11); ly += 16
        dr.line([ZB[0], ly + 6, ZB[0] + 18, ly + 6], fill=(255, 255, 255), width=1); dr.text((ZB[0] + 24, ly), "shaded strip = filter acted that chunk", fill=(200, 205, 200), font=f11)
        return np.asarray(im)

    def card():
        im = Image.new("RGB", (W, H), (18, 23, 21)); dr = ImageDraw.Draw(im)
        dr.text((60, 120), f"Set {st}, episode {ep}: what each controller did", fill=(225, 232, 228), font=f14)
        hdr = ["controller", "outcome", "chunks filter acted", "steps edited", "mean / max edit", "time"]
        xs = [60, 420, 720, 880, 1010, 1180]
        for x, hname in zip(xs, hdr):
            dr.text((x, 170), hname, fill=(170, 180, 175), font=f12)
        for i, a in enumerate(A):
            _, lab, col = A[a]; Tt = T[a]; t, c = outcome(R[a]); e = np.asarray(Tt["edit"]); ed = e[e > 0.5]
            nchunk = len(set(np.asarray(Tt["chunk"])[np.asarray(Tt["acted"])]))
            row = [lab, t, f"{nchunk} of {len(set(Tt['chunk']))}", f"{len(ed)} of {len(e)}",
                   f"{ed.mean():.1f} / {ed.max():.1f} mm" if len(ed) else "-", f"{Tt['t'][-1] + 0.1:.1f} s"]
            for x, v in zip(xs, row):
                dr.text((x, 200 + 28 * i), v, fill=c if v == t else (col if x == 60 else (225, 230, 228)), font=f12)
        return np.asarray(im)

    w = imageio_ffmpeg.write_frames(str(OUT / st / f"ep{ep}.mp4"), (W, H), fps=10, codec="libx264", quality=7, macro_block_size=1); w.send(None)
    for k in range(n):
        w.send(draw(k))
        if k in ev:
            fr = draw(k, ev[k])
            for _ in range(12):
                w.send(fr)
    for _ in range(10):
        w.send(draw(n - 1))
    c = card()
    for _ in range(40):
        w.send(c)
    w.close()


def choose_b():
    """fixed rule (seed 20261026, on the LOGGED flags; outcomes shown are the replay audit's): 2 random dev episodes where the
    policy alone topples and lfck000 lifts the block, 1 where lfci000 topples and lfck000 does not, 1 where lfck000 topples."""
    R = {a: {int(r["episode"]): r for r in map(json.loads, (ROOT / f"results/jenga/{run}.jsonl").read_text().splitlines())}
         for a, (run, _, _) in SETS["B"].items()}
    eps = sorted(set.intersection(*(set(v) for v in R.values())))
    rng = np.random.default_rng(20261026)
    pick = lambda c, m: [int(e) for e in rng.permutation([e for e in eps if c(e)])[:m]]
    return (pick(lambda e: R["policy"][e]["neighbour_toppled"] and R["lfck000"][e]["success"], 2)
            + pick(lambda e: R["lfci000"][e]["neighbour_toppled"] and not R["lfck000"][e]["neighbour_toppled"], 1)
            + pick(lambda e: R["lfck000"][e]["neighbour_toppled"], 1))


def main():
    st = sys.argv[1]
    (OUT / st / "raw").mkdir(parents=True, exist_ok=True)
    eps = [int(x) for x in sys.argv[2:]] or (EPS_A if st == "A" else choose_b())
    print("set", st, "episodes", eps, flush=True)
    jobs = [] if os.environ.get("COMPOSE_ONLY") == "1" else [(st, a, e) for e in eps for a in SETS[st]]
    with mp.get_context("spawn").Pool(max(1, min(int(os.environ.get("NPROC", 8)), len(jobs)))) as pool:
        for a, e in pool.imap_unordered(render, jobs):
            print("rendered", a, e, flush=True)
    for e in eps:
        compose(st, e); print("composed", e, flush=True)


if __name__ == "__main__":
    main()
