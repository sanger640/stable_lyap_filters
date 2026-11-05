"""KNOCK stage 3 videos (exploratory, for viewing): the same knock-down episode under the scripted pusher alone, ours 90%
zero-shot (z_o90) and the fork-signal value zero-shot (z_frag), both imagining with model_KNK. Replayed from each arm's
saved decision states with the run's execution noise, drawn by eval/filter_videos.py (row layout, three panels).
Episodes by a fixed rule (seed 20261105): 2 where the pusher alone topples a neighbour and ours 90% knocks the red block
down cleanly, 1 where the pusher succeeds and ours 90% refuses the push, 1 where the pusher topples and ours 90% refuses.
Output: results/program_b/filter_videos/K3/row/ep<i>.mp4
usage: python3 eval/knock_videos.py [ep ...]"""
import json
import os
import sys

import numpy as np

os.environ["VID_LAYOUT"] = "row"; os.environ["NO_SLICE"] = "1"
os.environ["VID_CAM"] = "0.53,0.0,0.46,0.34,135,-22"            # oblique side view: pusher, red block and both neighbours
import filter_videos as FV  # noqa: E402

FV.SETS["K3"] = {"policy": ("paired_record_knock2", "Scripted pusher alone", FV.GREY),
                 "ours90": ("paired_lfc000_knock3_zo90", "Ours 90%, Jenga-trained", FV.GREEN),
                 "frag": ("paired_lfc000_knock3_zfrag", "Fork-signal value, Jenga-trained", FV.ORANGE)}
FV.W = 3 * FV.CAM + 2 * FV.GAP + 24


def row(arm, ep):
    return json.loads((FV.RUNS / FV.SETS["K3"][arm][0] / f"ep{ep}.json").read_text())


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
    P, Z = {e: row("policy", e) for e in eps}, {e: row("ours90", e) for e in eps}
    rng = np.random.default_rng(20261105)
    pick = lambda c, n: sorted(int(x) for x in rng.choice(c, n, replace=False))
    rescue = [e for e in eps if P[e]["neighbour_toppled"] and Z[e]["success"]]
    cost = [e for e in eps if P[e]["success"] and not Z[e]["red_down"]]
    refuse = [e for e in eps if P[e]["neighbour_toppled"] and not Z[e]["red_down"]]
    return pick(rescue, 2) + pick(cost, 1) + pick(refuse, 1)


def render(job):                                              # module-level so spawn workers import the K3 set
    return FV.render(job)


if __name__ == "__main__":
    import multiprocessing as mp
    eps = [int(x) for x in sys.argv[1:]] or choose()
    (FV.OUT / "K3" / "raw").mkdir(parents=True, exist_ok=True); print("episodes", eps, flush=True)
    jobs = [("K3", a, e) for e in eps for a in FV.SETS["K3"]]
    with mp.get_context("spawn").Pool(min(6, len(jobs))) as pool:
        for a, e in pool.imap_unordered(render, jobs):
            print("rendered", a, e, flush=True)
    for e in eps:
        FV.compose("K3", e); print("composed", e, flush=True)
