"""Side-by-side videos for final test 2: the same episode with the policy alone (left) and with the final filter
(h_safe = log2 3, right), replayed from the saved decision states with the run's own execution-noise sequence
(eval/inspector_data.video_job). Banner per decision: ok / filter changed the action. Outcome card at the end.
Output: results/program_b/final2_videos/{pair_ep<i>.mp4, reel.mp4}

usage: HARD_MASS=0 EXEC_NOISE_SCALE=2 python3 eval/final2_videos.py
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
OUT = ROOT / "results/program_b/final2_videos"
os.environ["INSP_OUT"] = str(OUT)
RUNS = ROOT / "results/program_b/policy_rollouts"
ARMS = {"alone": "paired_record_final2", "filter": "paired_cbfm_final2"}


def render(job):
    import inspector_data as insp
    return insp.video_job(job)


def frames(path):
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(str(path)); meta = next(rd)
    w, h = meta["size"]
    return [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in rd]


def compose(ep, res, out_path, writer=None):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    A, F = frames(OUT / "video" / f"alone_ep{ep:03d}.mp4"), frames(OUT / "video" / f"filter_ep{ep:03d}.mp4")
    n = max(len(A), len(F)); W, H = 1140, 330
    pad = lambda L: L + [L[-1]] * (n - len(L))
    A, F = pad(A), pad(F)
    own = writer is None
    if own:
        writer = imageio_ffmpeg.write_frames(str(out_path), (W, H), fps=10, codec="libx264", quality=7, macro_block_size=1); writer.send(None)
    tA = "neighbour toppled" if res["alone"]["neighbour_toppled"] else "no topple"
    tF = ("success, no topple" if res["filter"].get("success") else "no topple") if not res["filter"]["neighbour_toppled"] else "neighbour toppled"
    for k in range(n):
        im = Image.new("RGB", (W, H), (20, 26, 24)); dr = ImageDraw.Draw(im)
        dr.text((8, 6), f"Final test 2, episode {ep}: same start, same seeds", fill=(220, 230, 225))
        dr.text((8, 24), f"POLICY ALONE  ->  {tA}", fill=(240, 150, 120) if res["alone"]["neighbour_toppled"] else (160, 220, 180))
        dr.text((578, 24), f"POLICY + SAFETY FILTER  ->  {tF}", fill=(160, 220, 180))
        im.paste(Image.fromarray(A[k]), (8, 44)); im.paste(Image.fromarray(F[k]), (578, 44))
        dr.line([(573, 40), (573, H - 8)], fill=(90, 100, 96), width=2)
        writer.send(np.asarray(im))
    card = Image.new("RGB", (W, H), (20, 26, 24)); dr = ImageDraw.Draw(card)
    dr.text((40, 110), f"Episode {ep}", fill=(220, 230, 225))
    dr.text((40, 140), f"Policy alone: {tA}", fill=(240, 150, 120))
    dr.text((40, 165), f"With the filter: {tF} (lifted 3 cm after {res['filter'].get('seconds_to_3cm')} s)", fill=(160, 220, 180))
    nint = sum(bool(c.get("intervened")) for c in res["filter"]["log"])
    dr.text((40, 190), f"The filter changed the action at {nint} of {len(res['filter']['log'])} decisions (orange banner).", fill=(200, 205, 200))
    for _ in range(25):
        writer.send(np.asarray(card))
    if own:
        writer.close()


def main():
    import imageio_ffmpeg
    (OUT / "video").mkdir(parents=True, exist_ok=True)
    eps = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else []
    res = {a: {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/{t}.jsonl").read_text().splitlines()} for a, t in ARMS.items()}
    jobs = [(a, e, str(RUNS / t / f"ep{e:03d}.npz")) for e in eps for a, t in ARMS.items()]
    with mp.get_context("spawn").Pool(min(16, len(jobs))) as pool:
        for a, e, _ in pool.imap_unordered(render, jobs):
            print("rendered", a, e, flush=True)
    for e in eps:
        compose(e, {a: res[a][e] for a in ARMS}, OUT / f"pair_ep{e}.mp4"); print("composed", e, flush=True)
    w = imageio_ffmpeg.write_frames(str(OUT / "reel.mp4"), (1140, 330), fps=10, codec="libx264", quality=7, macro_block_size=1); w.send(None)
    for e in eps[:5]:
        compose(e, {a: res[a][e] for a in ARMS}, None, writer=w)
    w.close(); print("reel done", flush=True)


if __name__ == "__main__":
    main()
