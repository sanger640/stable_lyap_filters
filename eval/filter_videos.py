"""Four-way filter videos for the dashboard, built to show ONE thing: how much each filter changes the policy's action.
The same episode (same start, same seeds) under four controllers, replayed from each arm's saved decision states with the
run's execution-noise sequence (as eval/h2h4_videos.py). Per controller:
  camera:    the policy's proposed next 8 targets (dashed white) and the targets actually executed (solid), true scale
  badge:     "policy's action, untouched" or "FILTER EDITING: <mm> (<direction words>)"; orange border while editing
  timeline:  one bar per 0.1 s step, height = |executed - proposed| target (mm); summary: share of steps edited, mean, max
  stamps:    neighbour topple when it happens; the replay-audited outcome (v4) at the end; end card with the same numbers
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
    "A": {"policy": ("paired_record_h2h4", "Policy alone", GREY), "lfci000": ("paired_lfci000_h2h4", "Label-free filter (gated edit)", TEAL),
          "psel": ("paired_cbfps3m_h2h3", "psel (baseline)", PURPLE), "lsf": ("paired_lsfa030_h2h3", "Labelled latent safety filter", BLUE)},
    "B": {"policy": ("paired_record_lflsfdev", "Policy alone", GREY), "lfci000": ("paired_lfci000_irrdev", "Narrow gate + actor fallback", TEAL),
          "lfcj000": ("paired_lfcj000_gate3dev", "Wider gate + actor fallback", BLUE),
          "lfck000": ("paired_lfck000_gate3dev", "Wider gate, pure edit (selected)", GREEN)}}
EPS_A = [25630, 25765, 25880, 25882, 26043]          # results/program_b/h2h4_videos (h2h4_videos.choose())
CAM = 400                                             # close camera, square
EDIT_MIN, EDIT_MAX = 0.5, 15.0                         # mm: below = untouched; timeline full scale
# safe-set slice (latent filters only): the filter's own one-step value check, re-run on a 9 x 9 grid of targets around the
# policy's (both commands of the model step shifted by the same xy offset, +-2 cm): green = passes, red = fails.
# label-free arms (cbf_filter.lsf_edit, eps 0, gamma 0.5): min_j V(z'_j) >= V(z, g) / 2; labelled LSF (lsf_step): min_j V > 0.3.
# Same world model, value, noise model and 8 draws as the filter, fresh draws (the run's own draws are not stored).
VALUE_ARMS = {"lfci000": ("value_lf_alea.pt", "cbf"), "lfcj000": ("value_lf_alea.pt", "cbf"), "lfck000": ("value_lf_alea.pt", "cbf"),
              "lsf": ("value_unisafe_alea.pt", "level")}
GRID, CELL = np.linspace(-0.02, 0.02, 9), 0.0025
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


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


def slicer(arm, ep):
    """-> f(sim, cmds (2, 4)) -> (9, 9) margins min_j V(z'_j) - threshold over the target grid (x rows, y columns)."""
    os.environ["CBF_WM"] = str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); os.environ["LSF_ALEA"] = "8"
    import cem_filter as cf
    import lsf as L
    import step3_world_model as wm
    mon = cf.Monitor(); torch = mon.torch; mdl = mon.model
    vf, noise = L.ValueFn(VALUE_ARMS[arm][0]), L.Noise(); noise.rng = np.random.default_rng([20261777, ep])
    n = 8
    D = torch.as_tensor(np.array([[gx, gy, 0, 0, gx, gy, 0, 0] for gx in GRID for gy in GRID], np.float32), device="cuda")

    def f(sim, cmds):
        cmds = np.asarray(cmds, np.float32)
        with torch.no_grad():
            z = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()[None]
            g = torch.as_tensor(np.r_[sim.target, cmds[0][3]][None], dtype=torch.float32, device="cuda")
            a_pol = torch.as_tensor(cmds.reshape(1, 8), device="cuda"); A = a_pol + D; c = len(A)
            zs = noise.starts(z.expand(n, *z.shape[1:]).contiguous())
            G2 = torch.cat([A[:, 4:7], A[:, 7:8]], 1).repeat_interleave(n, 0)
            Z = zs.repeat(c, *[1] * (zs.dim() - 1))
            off = noise.actions(a_pol, n)[:, 0] - a_pol         # execution noise, common to all candidates (as lsf_edit)
            if VALUE_ARMS[arm][1] == "cbf":                   # = cbf_filter.lsf_edit's vals(), threshold eps + (1 - gamma)(V(z,g) - eps)
                act = mdl._act((A[:, None] + off[None]).reshape(c * n, 8))
                if mdl.kind == "deterministic":
                    r = mdl.net(Z, torch.zeros_like(Z), act, torch.zeros(len(Z), device="cuda"))
                else:
                    r = torch.randn(Z.shape, device="cuda").float()
                    for kk in range(wm.EULER):
                        r = r + mdl.net(Z, r, act, torch.full((len(Z),), kk / wm.EULER, device="cuda")) / wm.EULER
                V = vf.V((Z + r * mdl.stats.res_std).float(), G2).view(c, n).min(1).values
                thr = 0.5 * float(vf.V(z, g))
            else:                                              # = cbf_filter.lsf_step (stage 2): model.step, level eps 0.3
                act = (A[:, None] + off[None]).reshape(c * n, 8)
                V = vf.V(mdl.step(Z, act).float(), G2).view(c, n).min(1).values; thr = 0.3
        return (V.cpu().numpy() - thr).reshape(len(GRID), len(GRID))
    return f


def render(job):
    """One arm, one episode -> raw camera video with the proposed (dashed) and executed (solid) paths, and a per-step trace."""
    st, arm, ep = job
    import mujoco
    import imageio_ffmpeg
    import inspector_data as insp
    import uncertainty_monitor_dev as um
    from PIL import Image, ImageDraw
    from jenga_render_frames import close_camera
    run, _, col = SETS[st][arm]
    with np.load(RUNS / run / f"ep{ep}.npz") as f:
        S = {k: f[k] for k in f.files}
    snippets = np.load(um.BENCH)["snippets"]
    nrng = np.random.default_rng([20261502, ep])
    noises = [snippets[nrng.integers(len(snippets))] * insp.EXEC_SCALE for _ in range(int(S["chunk_index"][-1]) + 1)]
    sim = insp._sim()
    rc = mujoco.Renderer(sim.model, height=CAM, width=CAM)
    cam = close_camera(mujoco); fov = float(sim.model.vis.global_.fovy)
    cam.lookat[:] = (0.49, 0.0, 0.49); cam.distance = 0.20          # closer than eval/jenga_render_frames: blocks + gripper fill the frame
    log = {c["chunk"]: c for c in json.loads((RUNS / run / f"ep{ep}.json").read_text())["log"]}
    smap = slicer(arm, ep) if arm in VALUE_ARMS and os.environ.get("NO_SLICE") != "1" else None
    w = imageio_ffmpeg.write_frames(str(OUT / st / "raw" / f"{arm}_ep{ep}.mp4"), (CAM, CAM), fps=10, codec="libx264", quality=8,
                                    macro_block_size=1); w.send(None)
    T = {k: [] for k in ("t", "tilt", "edit", "dvec", "grip_pol", "grip_exe", "slice_on", "slice_pol_ok")}
    for d in range(len(S["chunk_index"])):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        k = int(S["chunk_index"][d])
        nom, exe = np.asarray(S["nominal"][d], np.float32), np.asarray(S["executed"][d], np.float32)
        on = smap is not None and (VALUE_ARMS[arm][1] == "level" or bool(log.get(k, {}).get("irr_gate")))
        for j, a in enumerate(exe):
            if on and j % 2 == 0:                                 # the filter's check runs here, before the model step's 2 commands
                mg = smap(sim, nom[j:j + 2])
                ctr = nom[j + 1, :3].copy()
            a = a.copy(); a[:3] -= noises[k][j]
            sim.execute(a)
            hand = sim.data.site_xpos[sim.site].copy()
            rc.update_scene(sim.data, camera=cam); im = Image.fromarray(rc.render())
            camc = (rc.scene.camera[0], rc.scene.camera[1])
            pr = lambda P: gl_project(camc, CAM, CAM, fov, P)
            if on:
                ov = Image.new("RGBA", (CAM, CAM), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
                for ix, gx in enumerate(GRID):
                    for iy, gy in enumerate(GRID):
                        q = pr(ctr + np.array([[gx - CELL, gy - CELL, 0], [gx - CELL, gy + CELL, 0], [gx + CELL, gy + CELL, 0], [gx + CELL, gy - CELL, 0]]))
                        od.polygon([tuple(v) for v in q], fill=(60, 200, 90, 115) if mg[ix, iy] >= 0 else (200, 40, 190, 115))
                if VALUE_ARMS[arm][1] == "cbf":                   # the edit's bound: +-1 cm
                    q = pr(ctr + np.array([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, 0.01, 0], [0.01, -0.01, 0], [-0.01, -0.01, 0]]))
                    od.line([tuple(v) for v in q], fill=(255, 255, 255, 200), width=1)
                im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
            dr = ImageDraw.Draw(im)
            pn, pe = pr(np.vstack([hand[None], nom[j:, :3]])), pr(np.vstack([hand[None], exe[j:, :3]]))
            if arm != "policy":
                dashed(dr, pn, (255, 255, 255), 3, 6, 4)
            dr.line([tuple(q) for q in pe], fill=col if arm != "policy" else (250, 250, 250), width=3)
            if arm != "policy" and np.linalg.norm(exe[-1, :3] - nom[-1, :3]) > EDIT_MIN / 1000:
                a_, b_ = pn[-1], pe[-1]
                dr.ellipse([a_[0] - 6, a_[1] - 6, a_[0] + 6, a_[1] + 6], outline=(255, 255, 255), width=2)
                dr.ellipse([b_[0] - 6, b_[1] - 6, b_[0] + 6, b_[1] + 6], fill=col, outline=(20, 20, 20))
            w.send(np.asarray(im))
            dv = (exe[j, :3] - nom[j, :3]) * 1000
            T["t"].append((k * 8 + j) * 0.1); T["tilt"].append(float(np.max(sim.block_diagnostics()["tilt"][1:])))
            T["edit"].append(float(np.linalg.norm(dv))); T["dvec"].append(dv)
            T["grip_pol"].append(bool(nom[j, 3] > 0)); T["grip_exe"].append(bool(exe[j, 3] > 0))
            T["slice_on"].append(on); T["slice_pol_ok"].append(bool(on and mg[4, 4] >= 0))
    w.close(); rc.close(); sim.close()
    np.savez(OUT / st / "raw" / f"{arm}_ep{ep}.npz", **{k: np.asarray(v) for k, v in T.items()})
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


def words(dv):
    """edit vector (mm, world) -> direction words as seen in the camera (right = +y, up = +z, toward the camera = +x)"""
    parts = [(abs(dv[1]), "right" if dv[1] > 0 else "left"), (abs(dv[2]), "up" if dv[2] > 0 else "down"),
             (abs(dv[0]), "toward camera" if dv[0] > 0 else "away from camera")]
    return ", ".join(f"{n} {m:.0f}" for m, n in sorted(parts, reverse=True) if m >= 1.0) or "under 1 mm"


HEAD, TL, GAP = 30, 78, 14
PWID, PHEI = CAM, 30 + CAM + 78
W, H = 2 * CAM + GAP + 24, 60 + 2 * PHEI + GAP + 12


def compose(st, ep):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    A = SETS[st]
    fb, f13, f12, f11 = (ImageFont.truetype(FONT.replace("Sans.ttf", "Sans-Bold.ttf"), 13), ImageFont.truetype(FONT, 13),
                         ImageFont.truetype(FONT, 12), ImageFont.truetype(FONT, 11))
    F = {a: frames(OUT / st / "raw" / f"{a}_ep{ep}.mp4") for a in A}
    T = {a: dict(np.load(OUT / st / "raw" / f"{a}_ep{ep}.npz")) for a in A}
    R = {a: audited(st, a, ep) for a in A}
    n = max(len(v) for v in F.values()); tmax = max(T[a]["t"][-1] for a in A) + 0.1
    pos = {a: (12 + (i % 2) * (CAM + GAP), 60 + (i // 2) * (PHEI + GAP)) for i, a in enumerate(A)}

    def summary(a):
        e = T[a]["edit"]; on = e > EDIT_MIN
        if a == "policy":
            return "no filter"
        return (f"edits {100 * on.mean():.0f}% of steps · avg {e[on].mean():.1f} mm · max {e.max():.0f} mm"
                if on.any() else "never edited the policy's action")

    def draw(k):
        im = Image.new("RGB", (W, H), (18, 23, 21)); dr = ImageDraw.Draw(im)
        dr.text((12, 8), f"Episode {ep}: same start and same noise for all four.", fill=(230, 236, 232), font=fb)
        dr.text((12, 25), "Dashed white = the policy's move (o = its end).  Solid = what was executed.  Orange frame = filter is changing it.",
                fill=(170, 180, 175), font=f12)
        dr.text((12, 40), "Green / magenta patch (learned-value filters, while on) = nearby targets the filter's value accepts / rejects.",
                fill=(170, 180, 175), font=f12)
        for a, (x, y) in pos.items():
            _, lab, col = A[a]; Tt = T[a]; m = min(k, len(F[a]) - 1); ended = k >= len(F[a])
            e = float(Tt["edit"][m]); editing = a != "policy" and e > EDIT_MIN and not ended
            dr.text((x, y + 8), lab, fill=col, font=fb)
            im.paste(Image.fromarray(F[a][m]), (x, y + HEAD))
            dr.rectangle([x - 2, y + HEAD - 2, x + CAM + 1, y + HEAD + CAM + 1], outline=ORANGE if editing else (50, 60, 56), width=4 if editing else 1)
            if a == "policy":
                badge, bc, tc = "no filter: the policy's own action", (40, 48, 45), (220, 225, 222)
            elif editing:
                badge, bc, tc = f"FILTER EDITING: target moved {e:.0f} mm", ORANGE, (20, 20, 20)
                if Tt["grip_pol"][m] != Tt["grip_exe"][m]:
                    badge += f", gripper {'closed' if Tt['grip_exe'][m] else 'kept open'}"
            else:
                badge, bc, tc = "policy's action, untouched", (40, 48, 45), (200, 208, 204)
            if not ended:
                bw = dr.textlength(badge, font=fb) + 16
                dr.rectangle([x + 6, y + HEAD + 6, x + 6 + bw, y + HEAD + 28], fill=bc); dr.text((x + 14, y + HEAD + 10), badge, fill=tc, font=fb)
                dr.rectangle([x + CAM - 62, y + HEAD + CAM - 24, x + CAM - 6, y + HEAD + CAM - 6], fill=(20, 26, 24))
                dr.text((x + CAM - 56, y + HEAD + CAM - 22), f"{Tt['t'][m]:4.1f} s", fill=(225, 230, 228), font=f12)
            tp = np.flatnonzero(Tt["tilt"][:m + 1] >= TOPPLE)
            if len(tp):
                dr.rectangle([x + 6, y + HEAD + CAM - 30, x + 190, y + HEAD + CAM - 6], fill=(200, 60, 45))
                dr.text((x + 14, y + HEAD + CAM - 26), "NEIGHBOUR TOPPLED", fill=(255, 255, 255), font=fb)
            if ended:
                t, c = outcome(R[a]); bw = dr.textlength("ended: " + t, font=fb) + 16
                dr.rectangle([x + 6, y + HEAD + 6, x + 6 + bw, y + HEAD + 28], fill=(20, 26, 24)); dr.text((x + 14, y + HEAD + 10), "ended: " + t, fill=c, font=fb)
            # timeline: how far the filter moved the policy's target, every 0.1 s step
            ty0, ty1 = y + HEAD + CAM + 8, y + HEAD + CAM + 50
            dr.rectangle([x, ty0, x + CAM, ty1], fill=(28, 35, 32))
            tx = lambda t: x + 4 + (CAM - 8) * t / tmax
            if a != "policy":
                for q in range(len(Tt["t"])):
                    if Tt["edit"][q] > EDIT_MIN:
                        hgt = (ty1 - ty0 - 4) * min(Tt["edit"][q], EDIT_MAX) / EDIT_MAX
                        dr.rectangle([tx(Tt["t"][q]), ty1 - 2 - hgt, tx(Tt["t"][q]) + max(1, (CAM - 8) * 0.1 / tmax - 0.5), ty1 - 2], fill=ORANGE)
                dr.text((x + CAM - 44, ty0 + 2), f"{EDIT_MAX:.0f} mm", fill=(120, 130, 126), font=f11)
            dr.line([x + 4, ty1 - 2, tx(Tt["t"][-1] + 0.1), ty1 - 2], fill=(110, 120, 116))
            cx = tx(Tt["t"][m] if not ended else Tt["t"][-1]); dr.line([cx, ty0, cx, ty1], fill=(255, 255, 255), width=1)
            dr.text((x, ty1 + 6), summary(a), fill=(200, 208, 204), font=f12)
        return np.asarray(im)

    def card():
        im = Image.new("RGB", (W, H), (18, 23, 21)); dr = ImageDraw.Draw(im)
        dr.text((40, 160), f"Episode {ep}: how much each controller changed the policy's action", fill=(230, 236, 232), font=fb)
        for i, a in enumerate(A):
            _, lab, col = A[a]; t, c = outcome(R[a]); y0 = 210 + 70 * i
            dr.text((40, y0), lab, fill=col, font=fb); dr.text((40, y0 + 20), "outcome: " + t, fill=c, font=f13)
            dr.text((40, y0 + 38), summary(a), fill=(215, 222, 218), font=f13)
        return np.asarray(im)

    w = imageio_ffmpeg.write_frames(str(OUT / st / f"ep{ep}.mp4"), (W, H), fps=10, codec="libx264", quality=7, macro_block_size=1); w.send(None)
    for k in range(n + 10):
        w.send(draw(k))
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
