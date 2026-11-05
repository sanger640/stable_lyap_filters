"""KNOCK stage 5 videos (exploratory, for viewing): the same knock-down episode under the scripted pusher alone, irreversible
fragility zero-shot (z_fx, stage 4), our filter with the labelled UNISafe-r value (lcbf, LatentCBF-style) and irreversible
fragility without the safety-actor fallback (z_fxnf), all imagining with model_KNK. Replayed from each arm's saved decision
states with the run's execution noise, drawn by eval/filter_videos.py (row layout, four panels). Rendered one at a time
(a single process, while the HZN1 pool runs).
Episodes by a fixed rule (seed 20261106), one each: (1) z_fx refuses, lcbf and z_fxnf knock the red block down cleanly;
(2) z_fxnf topples a neighbour where z_fx succeeds; (3) the pusher alone topples, lcbf succeeds; (4) all four succeed.
Output: results/program_b/filter_videos/K5/row/ep<i>.mp4
usage: python3 eval/knock5_videos.py [ep ...]"""
import json
import os
import sys

import numpy as np

os.environ["VID_LAYOUT"] = "row"; os.environ["NO_SLICE"] = "1"
os.environ["VID_CAM"] = "0.53,0.0,0.46,0.34,135,-22"            # oblique side view, as eval/knock_videos.py
import filter_videos as FV  # noqa: E402

FV.SETS["K5"] = {"policy": ("paired_record_knock2", "Scripted pusher alone", FV.GREY),
                 "z_fx": ("paired_lfc000_knock4_zfx", "Irreversible fragility (ours)", FV.GREEN),
                 "lcbf": ("paired_lfc000_knock5_lcbf", "Our filter + labelled value", FV.BLUE),
                 "z_fxnf": ("paired_lfc000_knock5_zfxnf", "Irr. fragility, no fallback", FV.ORANGE)}
FV.W = 4 * FV.CAM + 3 * FV.GAP + 24


def row(arm, ep):
    return json.loads((FV.RUNS / FV.SETS["K5"][arm][0] / f"ep{ep}.json").read_text())


def audited(st, arm, ep):
    return row(arm, ep)


def outcome(r):
    if r["neighbour_toppled"]:
        return "neighbour toppled", (245, 140, 120)
    if r["red_down"]:
        return "success: red block down, neighbours standing", (150, 225, 170)
    return "red block not pushed over (time-out)", (225, 205, 140)


FV.audited, FV.outcome = audited, outcome


def choose():
    eps = range(45000, 45200)
    R = {a: {e: row(a, e) for e in eps} for a in FV.SETS["K5"]}
    rng = np.random.default_rng(20261106)
    pick = lambda c: int(rng.choice(c))
    s = lambda a, e: R[a][e]["success"]
    c1 = [e for e in eps if not R["z_fx"][e]["red_down"] and not R["z_fx"][e]["neighbour_toppled"] and s("lcbf", e) and s("z_fxnf", e)]
    c2 = [e for e in eps if R["z_fxnf"][e]["neighbour_toppled"] and s("z_fx", e)]
    c3 = [e for e in eps if R["policy"][e]["neighbour_toppled"] and s("lcbf", e)]
    c4 = [e for e in eps if all(s(a, e) for a in R)]
    return [pick(c) for c in (c1, c2, c3, c4)]


if __name__ == "__main__":
    eps = [int(x) for x in sys.argv[1:]] or choose()
    (FV.OUT / "K5" / "raw").mkdir(parents=True, exist_ok=True); print("episodes", eps, flush=True)
    for e in eps:
        for a in FV.SETS["K5"]:
            FV.render(("K5", a, e)); print("rendered", a, e, flush=True)
        FV.compose("K5", e); print("composed", e, flush=True)
