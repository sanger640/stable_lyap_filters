"""The physical mechanism behind the systematically missed forks (PLAN_NEXT.md Phase 2, step 6).

Reads the dense re-simulations (`jenga_blind_resim.py`) and the 10 frozen `gnn_n5` seeds, and writes
the three results `blind_fork_analysis.md` is built on:

  tilt course   the toppling neighbour's tilt over time, real vs predicted (median over the probes
                that truly topple, and over seeds): start tilt, tilt at the end of the perturbed
                chunk, predicted peak and final tilt
  handoff       the model is handed the TRUE simulator state at control step k and rolls out the
                rest: the fraction of truly-toppling probes it then topples. Separates "the push is
                under-transmitted" (fails before contact completes, succeeds after) from "the
                tipping point is misplaced" (fails even from the true tipped state)
  classes       a reproducible class for every fork, from its topple count and the toppling
                neighbour's start tilt and push timing:
                  A1  upright neighbour (start < 8 deg) pushed past 15 deg by control step 10
                  A2  upright neighbour pushed past 15 deg after step 10, i.e. in the hold
                  B   near-unanimous: >= 60 of 64 probes topple
                  C   fewer than 2 probes topple from the clean start state (weak by the
                      benchmark's own mixed-fork rule)
                  D   pre-leaning neighbour (start >= 8 deg)

It also counts, in the privileged training data, how often a toppling neighbour started upright.
Physics quantities are used for analysis only (Track A).
"""
import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

UPRIGHT_DEG, CRITICAL_DEG, TOPPLE = 8.0, 15.0, 45.0
HANDOFF = (0, 2, 4, 6, 8, 10, 12, 16, 20)
PER_STEP = 5                                   # readings per control step in the re-simulation


def classify(topples, start_tilt, cross15_step):
    if topples < 2:
        return "C"
    if topples >= 60:
        return "B"
    if start_tilt >= UPRIGHT_DEG:
        return "D"
    return "A1" if cross15_step is not None and cross15_step <= 10 else "A2"


def fork_record(meta, data, models, device):
    from jenga_w5_eval import predict
    from state_dynamics import neighbour_tilt_deg, rollout
    start, traces, windows = data["start"], data["traces"], data["windows"]
    start_tilt = neighbour_tilt_deg(start[None])[0]
    eligible = start_tilt < TOPPLE
    tilt = neighbour_tilt_deg(traces)                                     # (64, T, 2)
    topple = ((tilt[:, -1] >= TOPPLE) & eligible).any(-1)
    n = int(topple.sum())
    rec = {"group": meta["group"], "episode_id": meta["episode_id"],
           "chunk_start": meta["chunk_start"], "topples": n, "q": meta["q"],
           "q_matched": meta["q_matched"]}
    if n in (0, len(topple)):
        rec["class"] = "none"
        return rec
    which = int(np.argmax(((tilt[topple, -1] >= TOPPLE) & eligible).sum(0)))
    course = np.median(tilt[topple][:, :, which], axis=0)                  # (T,)
    crossing = np.flatnonzero(course > CRITICAL_DEG)
    cross15 = float(crossing[0] + 1) / PER_STEP if len(crossing) else None
    rec.update({"toppling_neighbour": ("left", "right")[which],
                "start_tilt_deg": float(start_tilt[which]),
                "real_tilt_chunk_end_deg": float(course[8 * PER_STEP - 1]),
                "real_cross_15deg_step": cross15})
    rec["class"] = classify(n, rec["start_tilt_deg"], cross15)

    at_step_end = traces[:, PER_STEP - 1::PER_STEP]                          # (64, 38, 61)
    actions = windows[:, 2:]
    peaks, ends, handoff = [], [], {k: [] for k in HANDOFF}
    for model, scale in models.values():
        predicted = neighbour_tilt_deg(predict(model, scale, start, windows, device))
        path = np.median(predicted[topple][:, :, which], axis=0)
        peaks.append(float(path.max())); ends.append(float(path[-1]))
        for k in HANDOFF:
            state = np.tile(start, (len(windows), 1)) if k == 0 else at_step_end[:, k - 1]
            with torch.no_grad():
                final = rollout(model, torch.as_tensor(state, device=device),
                                torch.as_tensor(actions[:, k:], device=device), scale)[:, -1]
            tilt_k = neighbour_tilt_deg(final.cpu().numpy())[:, which]
            handoff[k].append(float(np.mean(tilt_k[topple] >= TOPPLE)))
    rec["pred_tilt_peak_deg"] = float(np.median(peaks))
    rec["pred_tilt_end_deg"] = float(np.median(ends))

    def true_tilt(k):
        if k == 0:
            return float(start_tilt[which])
        return float(np.median(tilt[topple][:, PER_STEP * k - 1, which]))
    rec["handoff"] = {str(k): {"true_tilt_deg": true_tilt(k),
                               "model_completes_topple": float(np.median(v))}
                      for k, v in handoff.items()}
    return rec


def training_coverage():
    from jenga_w5_train import start_from_full_state
    from state_dynamics import neighbour_tilt_deg
    starts, rollouts = [], 0
    for path in sorted(glob.glob(str(ROOT / "results/jenga/trace_data/ep*_seed*.npz"))):
        d = np.load(path, allow_pickle=False)
        begin = neighbour_tilt_deg(start_from_full_state(d["start_state"]))  # (S, 2)
        final = neighbour_tilt_deg(d["traces"][:, :, -1])                     # (S, K, 2)
        rollouts += final.shape[0] * final.shape[1]
        new = (final >= TOPPLE) & (begin < TOPPLE)[:, None]
        for s, k in zip(*np.nonzero(new.any(-1))):
            starts.append(float(begin[s][np.argmax(new[s, k])]))
    starts = np.asarray(starts)
    return {"training_rollouts": rollouts, "toppling_rollouts": int(len(starts)),
            "toppling_neighbour_start_tilt_median_deg": float(np.median(starts)),
            "upright_start_fraction": float(np.mean(starts < 5.0)),
            "upright_start_count": int(np.sum(starts < 5.0)),
            "bins_deg": {f"{lo}-{hi}": int(np.sum((starts >= lo) & (starts < hi)))
                         for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 45))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resim", default=str(ROOT / "results/jenga/blind_resim"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/blind_mechanism.json"))
    args = ap.parse_args()

    from jenga_w5_eval import load_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {s: load_model(str(ROOT / f"results/jenga/w6_gnn_n5_s{s}.pt"), device)
              for s in range(1, 11)}
    index = json.loads((Path(args.resim) / "index.json").read_text())
    forks = []
    for meta in index["forks"]:
        data = np.load(Path(args.resim) / meta["file"], allow_pickle=False)
        forks.append(fork_record(meta, {k: data[k] for k in ("start", "traces", "windows")},
                                 models, device))
        f = forks[-1]
        print(f"  {f['group']:7s} ep{f['episode_id']:>3s}:{f['chunk_start']:<4d} class "
              f"{f['class']:4s} topples {f['topples']:2d}", flush=True)

    summary = {}
    for group in ("blind", "control"):
        rows = [f for f in forks if f["group"] == group and f["class"] != "none"]
        summary[group] = {
            "classes": dict(sorted(Counter(f["class"] for f in rows).items())),
            "start_tilt_median_deg": float(np.median([f["start_tilt_deg"] for f in rows])),
            "real_tilt_chunk_end_median_deg": float(np.median(
                [f["real_tilt_chunk_end_deg"] for f in rows])),
            "pred_tilt_peak_median_deg": float(np.median([f["pred_tilt_peak_deg"] for f in rows])),
            "pred_tilt_end_median_deg": float(np.median([f["pred_tilt_end_deg"] for f in rows])),
            "handoff": {str(k): {
                "true_tilt_deg": float(np.median([f["handoff"][str(k)]["true_tilt_deg"]
                                                  for f in rows])),
                "model_completes_topple": float(np.median(
                    [f["handoff"][str(k)]["model_completes_topple"] for f in rows]))}
                for k in HANDOFF}}
    result = {"protocol": {"upright_deg": UPRIGHT_DEG, "critical_deg": CRITICAL_DEG,
                           "topple_deg": TOPPLE, "handoff_steps": list(HANDOFF),
                           "seeds": list(range(1, 11))},
              "groups": summary, "training_coverage": training_coverage(), "forks": forks}
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"groups": {g: {k: v for k, v in s.items() if k != "handoff"}
                                 for g, s in summary.items()},
                      "training_coverage": result["training_coverage"]}, indent=1))


if __name__ == "__main__":
    main()
