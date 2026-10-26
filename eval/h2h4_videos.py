"""Four-way videos for head-to-head 4 (results/jenga/lsf_h2h4_protocol.json): the same episode (same start, same seeds) with
the policy alone, the label-free irreversibility-gated edit (lfci000), psel and labelled LSF stage 2, replayed from each
arm's saved decision states with the run's execution-noise sequence (as eval/inspector_data.video_job). Banner per decision
from the arm's own log. Episodes are chosen by a fixed rule, not by eye (choose()).
Output: results/program_b/h2h4_videos/{quad_ep<i>.mp4, reel.mp4}
usage: python3 eval/h2h4_videos.py [ep ...]     (no args: choose() on the audited episodes)
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
OUT = ROOT / "results/program_b/h2h4_videos"
RUNS = ROOT / "results/program_b/policy_rollouts"
ARMS = {"policy": ("paired_record_h2h4", "POLICY ALONE"), "lfci000": ("paired_lfci000_h2h4", "LABEL-FREE, IRREVERSIBILITY-GATED"),
        "psel": ("paired_cbfps3m_h2h3", "PSEL (BASELINE)"), "lsf": ("paired_lsfa030_h2h3", "LATENT SAFETY FILTER (LABELLED)")}
GREEN, ORANGE, PURPLE, GREY = (47, 125, 79), (215, 130, 20), (107, 90, 166), (90, 100, 96)


def banner(arm, c):
    n = int(c.get("lsf_overrides", 0) or 0)
    if arm == "psel":
        return ("filter changed the move", ORANGE) if c.get("intervened") else ("ok", GREEN)
    if arm == "lsf":
        return (f"safety controller took over ({n} of 4 steps)", ORANGE) if n else ("ok", GREEN)
    if arm == "lfci000":
        if c.get("irr_gate"):
            return (f"irreversible: value edit ({n} of 4 steps)", ORANGE) if n else ("irreversible: edit not needed", PURPLE)
        return ("ok", GREEN)
    return ("policy only", GREY)


def render(job):
    arm, ep = job
    import mujoco
    import imageio_ffmpeg
    import inspector_data as insp
    import uncertainty_monitor_dev as um
    from PIL import Image, ImageDraw
    from jenga_render_frames import close_camera
    run = ARMS[arm][0]
    with np.load(RUNS / run / f"ep{ep}.npz") as f:
        S = {k: f[k] for k in f.files}
    log = {c["chunk"]: c for c in json.loads((RUNS / run / f"ep{ep}.json").read_text())["log"]}
    snippets = np.load(um.BENCH)["snippets"]
    nrng = np.random.default_rng([20261502, ep])
    noises = [snippets[nrng.integers(len(snippets))] * insp.EXEC_SCALE for _ in range(int(S["chunk_index"][-1]) + 1)]
    sim = insp._sim()
    rf = mujoco.Renderer(sim.model, height=240, width=320); rc = mujoco.Renderer(sim.model, height=240, width=240)
    cam = close_camera(mujoco)
    out = OUT / "video" / f"{arm}_ep{ep}.mp4"
    w = imageio_ffmpeg.write_frames(str(out), (560, 260), fps=10, codec="libx264", quality=7, macro_block_size=1); w.send(None)
    for d in range(len(S["chunk_index"])):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        k = int(S["chunk_index"][d]); txt, col = banner(arm, log.get(k, {}))
        for j, a in enumerate(S["executed"][d]):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]
            sim.execute(a)
            rf.update_scene(sim.data, camera="cam_fixed"); rc.update_scene(sim.data, camera=cam)
            im = Image.new("RGB", (560, 260), (20, 26, 24))
            im.paste(Image.fromarray(rf.render()), (0, 20)); im.paste(Image.fromarray(rc.render()), (320, 20))
            dr = ImageDraw.Draw(im); dr.rectangle([0, 0, 560, 19], fill=col)
            dr.text((6, 4), f"t = {(k * 8 + j) * 0.1:4.1f} s   {txt}", fill=(255, 255, 255))
            w.send(np.asarray(im))
    w.close(); rf.close(); rc.close(); sim.close()
    return arm, ep


def frames(path):
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(str(path)); meta = next(rd); w, h = meta["size"]
    return [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in rd]


def outcome(r):
    if r["neighbour_toppled"]:
        return "neighbour toppled", (240, 150, 120)
    return ("block extracted, no topple", (160, 220, 180)) if r.get("success") else ("no topple, not extracted in time", (220, 200, 140))


W, H = 1140, 620


def compose(ep, writer):
    from PIL import Image, ImageDraw
    F = {a: frames(OUT / "video" / f"{a}_ep{ep}.mp4") for a in ARMS}
    R = {a: json.loads((RUNS / ARMS[a][0] / f"ep{ep}.json").read_text()) for a in ARMS}
    n = max(len(v) for v in F.values())
    pos = {"policy": (8, 44), "lfci000": (578, 44), "psel": (8, 340), "lsf": (578, 340)}
    for k in range(n):
        im = Image.new("RGB", (W, H), (20, 26, 24)); dr = ImageDraw.Draw(im)
        dr.text((8, 6), f"Head-to-head 4, episode {ep}: same start, same seeds, four controllers", fill=(220, 230, 225))
        for a, (x, y) in pos.items():
            t, c = outcome(R[a]); dr.text((x, y - 18), f"{ARMS[a][1]}  ->  {t}", fill=c)
            im.paste(Image.fromarray(F[a][min(k, len(F[a]) - 1)]), (x, y))
        writer.send(np.asarray(im))
    card = Image.new("RGB", (W, H), (20, 26, 24)); dr = ImageDraw.Draw(card)
    dr.text((40, 200), f"Episode {ep}", fill=(220, 230, 225))
    for i, a in enumerate(ARMS):
        t, c = outcome(R[a]); dr.text((40, 235 + 25 * i), f"{ARMS[a][1].lower()}: {t}", fill=c)
    for _ in range(25):
        writer.send(np.asarray(card))


def choose():
    """fixed rule (seed 20261026): 3 random episodes where the policy alone topples and lfci000 extracts the block, 1 where
    lfci000 topples, 1 where psel extracts and lfci000 does not (the label-free filter's failures are shown too)."""
    R = {a: {int(r["episode"]): r for r in map(json.loads, (ROOT / f"results/jenga/{ARMS[a][0]}.jsonl").read_text().splitlines())} for a in ARMS}
    eps = sorted(set.intersection(*(set(v) for v in R.values())))
    rng = np.random.default_rng(20261026)
    pick = lambda c, m: [int(e) for e in rng.permutation([e for e in eps if c(e)])[:m]]
    return (pick(lambda e: R["policy"][e]["neighbour_toppled"] and R["lfci000"][e]["success"], 3)
            + pick(lambda e: R["lfci000"][e]["neighbour_toppled"], 1)
            + pick(lambda e: R["psel"][e]["success"] and not R["lfci000"][e]["success"], 1))


def main():
    import imageio_ffmpeg
    (OUT / "video").mkdir(parents=True, exist_ok=True)
    eps = [int(x) for x in sys.argv[1:]] or choose()
    print("episodes", eps, flush=True)
    jobs = [(a, e) for e in eps for a in ARMS]
    with mp.get_context("spawn").Pool(min(8, len(jobs))) as pool:
        for a, e in pool.imap_unordered(render, jobs):
            print("rendered", a, e, flush=True)
    reel = imageio_ffmpeg.write_frames(str(OUT / "reel.mp4"), (W, H), fps=10, codec="libx264", quality=7, macro_block_size=1); reel.send(None)
    for e in eps:
        w = imageio_ffmpeg.write_frames(str(OUT / f"quad_ep{e}.mp4"), (W, H), fps=10, codec="libx264", quality=7, macro_block_size=1); w.send(None)
        compose(e, w); w.close(); compose(e, reel); print("composed", e, flush=True)
    reel.close(); print("reel done", flush=True)


if __name__ == "__main__":
    main()
