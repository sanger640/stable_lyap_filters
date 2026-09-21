"""Why are the systematically missed forks invisible to the dynamics? (PLAN_NEXT.md Phase 2, step 6)

Reads the dense re-simulations from `jenga_blind_resim.py` (blind forks AND matched usually-detected
controls) and measures, for every fork, the same quantities on both sides:

  real       per-probe topple labels; the real spread D_real(t) over the 64 probes at every reading;
             when it first exceeds 1 mm; tipping onset; which contact flags separate the toppling
             probes from the stable ones and WHEN they first do; neighbour sliding before tipping;
             start geometry (contacts, gaps, tilt)
  predicted  for each of the 10 gnn_n5 seeds: D_pred(t) at control rate; when it first exceeds
             1 mm; its peak and final value (responds / never responds / responds then reconverges);
             predicted topples; whether the probes the model moves most are the ones that topple
  action     whether the toppling probes share a perturbation direction (AUC of the best 1-D
             projection of the mean chunk offset) and the direction itself

A mechanism explains blindness only if it separates BLIND from CONTROL, so every feature is
reported per group. Topple labels and physics quantities are used only to analyse (Track A).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

CONTACT_LABELS = ("middle-left", "middle-right", "left-right", "middle-table", "left-table",
                  "right-table", "middle-floor", "left-floor", "right-floor", "middle-robot",
                  "left-robot", "right-robot")
ONSET_MM = 1.0          # a divergence "exists" once the probes' RMS spread passes 1 mm
TIP_DEG = 5.0           # tipping onset: a toppling neighbour's tilt passes 5 degrees
CHUNK_STEPS = 8         # control steps of perturbed chunk before the 30-step hold


def spread_series(positions_mm):
    """(probes, T, 9) -> (T,) RMS spread over probes."""
    centred = positions_mm - positions_mm.mean(0, keepdims=True)
    return np.sqrt((centred ** 2).sum(-1).mean(0))


def first_above(series, level, per_step=1):
    hit = np.flatnonzero(series > level)
    return None if not len(hit) else float(hit[0] + 1) / per_step


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum())
                 / (len(pos) * len(neg)))


def analyse_fork(data, meta, models, device, tilt_fn):
    from jenga_w5_eval import predict
    start, traces, windows = data["start"], data["traces"], data["windows"]
    per_step = traces.shape[1] // (windows.shape[0] - 2)          # readings per control step
    start_tilt = tilt_fn(start[None])[0]
    eligible = start_tilt < 45.0
    tilts = tilt_fn(traces)                                         # (probes, T, 2)
    topple = ((tilts[:, -1] >= 45.0) & eligible).any(-1)
    n_top = int(topple.sum())
    out = {"episode_id": meta["episode_id"], "chunk_start": meta["chunk_start"],
           "group": meta["group"], "q": meta["q"], "q_matched": meta["q_matched"],
           "topples": n_top, "topples_frozen": meta["topple_count"]}

    # ---- real divergence over time
    pos = 1000 * (traces[..., 0:9] - start[None, None, 0:9])
    d_real = spread_series(pos)
    out["real_onset_step"] = first_above(d_real, ONSET_MM, per_step)
    out["real_final_mm"] = float(d_real[-1])
    out["real_peak_mm"] = float(d_real.max())

    # ---- which neighbour topples, tipping onset, sliding before tipping
    if n_top and n_top < len(topple):
        which = int(np.argmax(((tilts[topple, -1] >= 45.0) & eligible).sum(0)))
        tip = np.median(tilts[topple, :, which], axis=0)
        out["toppling_neighbour"] = ("left", "right")[which]
        out["tip_onset_step"] = first_above(tip, TIP_DEG, per_step)
        block = 1 + which
        xy = pos[:, :, 3 * block: 3 * block + 2]
        onset = int(round((out["tip_onset_step"] or traces.shape[1] / per_step) * per_step)) - 1
        slide_top = np.linalg.norm(xy[topple, max(onset, 0)], axis=-1)
        slide_stable = np.linalg.norm(xy[~topple, max(onset, 0)], axis=-1)
        out["neighbour_slide_before_tip_mm"] = {"topple": float(np.median(slide_top)),
                                                "stable": float(np.median(slide_stable))}
        # ---- contact flags that separate toppling from stable probes, and when they first do
        flags = traces[..., 45:57] > 0.5                           # (probes, T, 12)
        gap = flags[topple].mean(0) - flags[~topple].mean(0)       # (T, 12)
        separating = {}
        for c, label in enumerate(CONTACT_LABELS):
            hit = np.flatnonzero(np.abs(gap[:, c]) > 0.5)
            if len(hit):
                separating[label] = {"first_step": float(hit[0] + 1) / per_step,
                                     "direction": "on_in_topple" if gap[hit[0], c] > 0
                                     else "off_in_topple"}
        out["separating_contacts"] = dict(sorted(separating.items(),
                                                 key=lambda kv: kv[1]["first_step"]))
        out["first_separating_contact"] = (next(iter(out["separating_contacts"]))
                                           if separating else None)
        # ---- perturbation direction: do the toppling probes share one?
        offset = (windows[:, 2:2 + CHUNK_STEPS, :3] - windows[:, 2:2 + CHUNK_STEPS, :3]
                  .mean(0, keepdims=True)).mean(1) * 1000              # mean chunk offset, mm
        direction = offset[topple].mean(0) - offset[~topple].mean(0)
        unit = direction / max(np.linalg.norm(direction), 1e-12)
        proj = offset @ unit
        out["perturbation"] = {"direction_xyz": unit.round(3).tolist(),
                               "separability_auc": auc(proj[topple], proj[~topple]),
                               "group_gap_mm": float(np.linalg.norm(direction))}

    # ---- start geometry
    c0 = start[45:57] > 0.5
    p = start[0:9].reshape(3, 3)
    out["start"] = {"contacts_on": [CONTACT_LABELS[i] for i in np.flatnonzero(c0)],
                    "gap_middle_left_mm": float(1000 * np.linalg.norm(p[0] - p[1])),
                    "gap_middle_right_mm": float(1000 * np.linalg.norm(p[0] - p[2])),
                    "neighbour_tilt_deg": start_tilt.round(2).tolist(),
                    "gripper_to_left_mm": float(1000 * np.linalg.norm(start[57:60] - p[1])),
                    "gripper_to_right_mm": float(1000 * np.linalg.norm(start[57:60] - p[2]))}

    # ---- predicted, every seed
    seeds = []
    for seed, (model, scale) in models.items():
        pred = predict(model, scale, start, windows, device)                  # (probes, 38, 61)
        ppos = 1000 * (pred[..., 0:9] - start[None, None, 0:9])
        d_pred = spread_series(ppos)
        ptilt = tilt_fn(pred)
        ptop = ((ptilt[:, -1] >= 45.0) & eligible).any(-1)
        entry = {"seed": seed, "onset_step": first_above(d_pred, ONSET_MM),
                 "peak_mm": float(d_pred.max()), "final_mm": float(d_pred[-1]),
                 "predicted_topples": int(ptop.sum())}
        entry["regime"] = ("never_responds" if entry["peak_mm"] < ONSET_MM else
                           "reconverges" if entry["final_mm"] < 0.5 * entry["peak_mm"] else
                           "responds")
        if n_top and n_top < len(topple):
            moved = np.linalg.norm(ppos[:, -1] - ppos[:, -1].mean(0), axis=-1)
            entry["moves_the_toppling_probes_auc"] = auc(moved[topple], moved[~topple])
        # The real series is at 5 readings per control step; compare at control-step ends.
        entry["pred_over_real_final"] = entry["final_mm"] / max(out["real_final_mm"], 1e-9)
        seeds.append(entry)
    out["predicted"] = seeds
    finals = [s["final_mm"] for s in seeds]
    out["pred_final_mm_median"] = float(np.median(finals))
    out["pred_regime_counts"] = {r: sum(s["regime"] == r for s in seeds)
                                 for r in ("never_responds", "reconverges", "responds")}
    aucs = [s["moves_the_toppling_probes_auc"] for s in seeds
            if s.get("moves_the_toppling_probes_auc") is not None]
    out["moves_the_toppling_probes_auc_median"] = float(np.median(aucs)) if aucs else None
    return out


def group_summary(rows):
    """Per group: medians and fractions of every feature, so blind and control line up."""
    def med(key, sub=None):
        vals = [r[key] if sub is None else r.get(key, {}).get(sub) for r in rows]
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None
    first = {}
    for r in rows:
        first[r.get("first_separating_contact")] = first.get(r.get("first_separating_contact"),
                                                             0) + 1
    return {
        "forks": len(rows),
        "topples_median": med("topples"),
        "real_onset_step_median": med("real_onset_step"),
        "onset_during_chunk": sum(1 for r in rows if r["real_onset_step"] is not None
                                  and r["real_onset_step"] <= CHUNK_STEPS),
        "tip_onset_step_median": med("tip_onset_step"),
        "real_final_mm_median": med("real_final_mm"),
        "pred_final_mm_median": med("pred_final_mm_median"),
        "moves_toppling_probes_auc_median": med("moves_the_toppling_probes_auc_median"),
        "perturbation_separability_auc_median": med("perturbation", "separability_auc"),
        "first_separating_contact": first,
        "regime_totals": {k: sum(r["pred_regime_counts"][k] for r in rows)
                          for k in ("never_responds", "reconverges", "responds")},
        "gripper_to_toppling_neighbour_mm_median": float(np.median([
            r["start"][f"gripper_to_{r['toppling_neighbour']}_mm"] for r in rows
            if r.get("toppling_neighbour")])) if any(r.get("toppling_neighbour")
                                                     for r in rows) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resim", default=str(ROOT / "results/jenga/blind_resim"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/blind_analysis.json"))
    args = ap.parse_args()

    from jenga_w5_eval import load_model
    from state_dynamics import neighbour_tilt_deg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {s: load_model(str(ROOT / f"results/jenga/w6_gnn_n5_s{s}.pt"), device)
              for s in args.seeds}
    index = json.loads((Path(args.resim) / "index.json").read_text())
    rows = []
    for meta in index["forks"]:
        data = np.load(Path(args.resim) / meta["file"], allow_pickle=False)
        rows.append(analyse_fork({k: data[k] for k in ("start", "traces", "windows")}, meta,
                                 models, device, neighbour_tilt_deg))
        print(f"  {meta['group']:7s} ep{meta['episode_id']:>3s} step{meta['chunk_start']:>4d} "
              f"topples {rows[-1]['topples']:2d}/64 (frozen {meta['topple_count']:2d})  "
              f"real onset {rows[-1]['real_onset_step']}  first contact "
              f"{rows[-1].get('first_separating_contact')}  pred final "
              f"{rows[-1]['pred_final_mm_median']:.2f} mm  "
              f"regimes {rows[-1]['pred_regime_counts']}",
              flush=True)
    result = {"protocol": {"onset_mm": ONSET_MM, "tip_deg": TIP_DEG, "chunk_steps": CHUNK_STEPS,
                           "seeds": args.seeds,
                           "readings_per_control_step": index["readings_per_control_step"]},
              "groups": {g: group_summary([r for r in rows if r["group"] == g])
                         for g in ("blind", "control")},
              "forks": rows}
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["groups"], indent=1))


if __name__ == "__main__":
    main()
