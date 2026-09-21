"""Why does a model's alarm fire? Per-probe anatomy of the monitor score.

m2 recovered fork recall without recovering fork separation, which means the score crossed its
threshold for some reason other than the branch structure it was supposed to capture. This opens
up the score on the same test states and asks:

  distributions   the score for true positives, false positives, false negatives, true negatives
  outlier share   how far the score falls when the single most extreme probe is dropped. A state
                  whose alarm rests on one probe is not detecting a branch, it is detecting a
                  rollout that ran away
  direction       whether the probes that deviate most share a displacement direction, which would
                  mean the model responds to one axis rather than to proximity to a fork
  components      the score recomputed from position only, rotation only and the contact channel,
                  to see which part of the state carries it

Threshold and classes come from the model's own Gate 3 run, so the labels here are the ones the
monitor actually produced. Topple labels are used only to group results.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402
from jenga_stage3_predicted_forks import action_windows  # noqa: E402
from jenga_state_data import step_state  # noqa: E402
from jenga_w5_eval import hold_index, load_model, predict  # noqa: E402

AXES = ("x", "y", "z")


def rms_spread(values):
    """values (probes, d) -> RMS distance from the mean over probes."""
    return float(np.sqrt(np.mean(np.sum((values - values.mean(0)) ** 2, 1))))


def anatomy(endings, snippets, scale):
    """Break one state's predicted endings down. endings (probes, 61), snippets (probes, 8, 3)."""
    positions = 1000 * endings[:, 0:9]
    score = rms_spread(positions)
    deviation = np.linalg.norm(positions - positions.mean(0), axis=1)
    worst = int(np.argmax(deviation))
    without = rms_spread(np.delete(positions, worst, axis=0))
    # Net commanded displacement of each probe, and how the deviations line up with each axis.
    net = -snippets.sum(1) * scale                      # (probes, 3), metres
    centred = deviation - deviation.mean()
    alignment = {}
    for a, axis in enumerate(AXES):
        column = net[:, a] - net[:, a].mean()
        denominator = np.linalg.norm(column) * np.linalg.norm(centred)
        alignment[axis] = float(column @ centred / denominator) if denominator > 1e-12 else 0.0
    return {"score": score, "score_without_worst_probe": without,
            "outlier_share": float(1 - without / score) if score > 1e-12 else 0.0,
            "top_probe_deviation_mm": float(deviation[worst]),
            "median_probe_deviation_mm": float(np.median(deviation)),
            "rotation_only": rms_spread(endings[:, 9:27]),
            "contact_only": rms_spread(endings[:, 45:57]),
            "velocity_only": rms_spread(endings[:, 27:45]),
            "direction_alignment": alignment}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate3", required=True, help="that model's Gate 3 run, for the threshold")
    ap.add_argument("--test", default=str(ROOT / "results/jenga/holdout3_stage2_shared.json"))
    ap.add_argument("--stage0-cache", default=str(ROOT / "results/jenga/holdout_stage0_cache.npz"))
    ap.add_argument("--reset-seed-base", type=int, default=1000)
    ap.add_argument("--scale", default="1.0")
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scale = load_model(args.model, device)
    snippets = np.load(args.stage0_cache, allow_pickle=False)["snippets"]
    rows = json.loads(Path(args.test).read_text())["rows"]
    threshold = json.loads(Path(args.gate3).read_text())["model"][args.scale]["threshold"]
    scale_value = float(args.scale)

    wanted = {}
    for r in rows:
        wanted.setdefault(r["episode_id"], {})[r["chunk_start"]] = r
    replay = JengaReplay(args.lmdb)
    records = []
    with tempfile.TemporaryDirectory(prefix="jenga_diag_") as temp:
        sim = DirectJengaSim(str(extract_sim(args.sim_archive, temp)))
        try:
            for episode_id in sorted(wanted, key=int):
                episode = replay.episode(episode_id)
                sim.reset(int(episode_id) + args.reset_seed_base)
                for step, action in enumerate(episode.actions):
                    if step in wanted[episode_id]:
                        row = wanted[episode_id][step]
                        windows = action_windows(episode.actions, step, snippets,
                                                 scale_value, own_hold=False)
                        endings = predict(model, scale, step_state(sim), windows,
                                          device)[:, hold_index(model, 30)]
                        entry = anatomy(endings, snippets, scale_value)
                        entry["episode_id"] = episode_id
                        entry["chunk_start"] = step
                        entry["class"] = row["by_scale"][args.scale]["class"]
                        entry["alarm"] = bool(entry["score"] > threshold)
                        records.append(entry)
                    sim.execute(action)
        finally:
            sim.close()
            replay.close()

    def outcome(r):
        if r["class"] == "topple_fork":
            return "true_positive" if r["alarm"] else "false_negative"
        if r["class"] == "quiet":
            return "false_positive" if r["alarm"] else "true_negative"
        return "other"

    summary = {}
    for name in ("true_positive", "false_positive", "false_negative", "true_negative"):
        part = [r for r in records if outcome(r) == name]
        if not part:
            summary[name] = {"states": 0}
            continue
        summary[name] = {
            "states": len(part),
            "score_mm": {q: float(np.quantile([r["score"] for r in part], v))
                         for q, v in (("p25", .25), ("median", .5), ("p75", .75))},
            "outlier_share_median": float(np.median([r["outlier_share"] for r in part])),
            "states_where_worst_probe_carries_the_alarm": sum(
                1 for r in part if r["alarm"] and r["score_without_worst_probe"] <= threshold),
            "top_over_median_probe_deviation": float(np.median(
                [r["top_probe_deviation_mm"] / max(r["median_probe_deviation_mm"], 1e-9)
                 for r in part])),
            "rotation_only_median": float(np.median([r["rotation_only"] for r in part])),
            "contact_only_median": float(np.median([r["contact_only"] for r in part])),
            "velocity_only_median": float(np.median([r["velocity_only"] for r in part])),
            "direction_alignment_median": {
                a: float(np.median([r["direction_alignment"][a] for r in part])) for a in AXES}}

    result = {"model": args.model, "scale": args.scale, "threshold_mm": threshold,
              "summary": summary, "rows": records}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
