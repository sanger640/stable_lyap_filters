"""Stage 1: does a label-free two-mode test on noisy endings recover the Stage 0 answer key?

Reads the Stage 0 cache (true simulator endings; this is a physics upper bound, not a deployable
image monitor). Positives: MIXED states. Negatives: UNANIMOUS-safe states. Weak states (one
dissenter) are reported but not scored. Topple labels are used only to grade.

Pre-declared variants: full rule; no amplification floor; no persistence; all three blocks
(including the grasped one). Baseline: ending spread alone (AUC, and false alarms at the
detector's recall).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from outcome_modes import corner_displacements_mm, same_partition, two_mode_test  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_stage0_noise_oracle import RECORD_AT, SCALES  # noqa: E402

HALF_SIZE_M = (0.0125, 0.01, 0.0375)  # every block's box geom in panda_jenga_setup.xml
NEIGHBORS, ALL_BLOCKS = [1, 2], [0, 1, 2]
T_FINAL, T_EARLY = RECORD_AT.index(30), RECORD_AT.index(10)


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def score_state(start_pose, pose, perturbation_mm, blocks):
    final = corner_displacements_mm(start_pose, pose[:, T_FINAL], HALF_SIZE_M, blocks)
    early = corner_displacements_mm(start_pose, pose[:, T_EARLY], HALF_SIZE_M, blocks)
    late, first = two_mode_test(final, perturbation_mm), two_mode_test(early, perturbation_mm)
    no_floor = (late.bic_prefers_two and late.ashman_d > 2 and late.minority >= 2)
    no_floor_early = (first.bic_prefers_two and first.ashman_d > 2 and first.minority >= 2)
    persistent = same_partition(late.labels, first.labels)
    return {"full": bool(late.alarm and first.alarm and persistent),
            "no_floor": bool(no_floor and no_floor_early and persistent),
            "no_persistence": bool(late.alarm),
            "final": late.summary(), "early": first.summary(),
            "partition_persistent": bool(persistent)}


def confusion(rows, scale, variant):
    out = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "weak_alarms": 0, "weak_states": 0}
    for r in rows:
        grade, alarm = r["grade"][scale], r["scores"][scale][variant]
        if grade == "mixed":
            out["tp" if alarm else "fn"] += 1
        elif grade == "unanimous_safe":
            out["fp" if alarm else "tn"] += 1
        else:
            out["weak_states"] += 1
            out["weak_alarms"] += int(alarm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", default=str(ROOT / "results/jenga/stage0_noise_oracle.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/stage0_noise_oracle_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/stage1_outcome_modes.json"))
    args = ap.parse_args()
    stage0 = json.loads(Path(args.stage0).read_text())
    npz = np.load(args.cache, allow_pickle=False)
    # Load each array ONCE: indexing an NpzFile decompresses the whole array on every access,
    # which made this loop quadratic in the number of states (~40 min at 2,308 states).
    cache = {k: npz[k] for k in ("snippets", "pose", "start_pose")}
    snippet_mm = 1000 * np.linalg.norm(cache["snippets"], axis=2)
    base_perturbation = float(np.median(snippet_mm.max(1)))

    rows = []
    for i, meta in enumerate(stage0["rows"]):
        row = {"episode_id": meta["episode_id"], "chunk_start": meta["chunk_start"],
               "stratum": meta["stratum"], "grade": {}, "topple_count": {}, "scores": {}}
        for s, scale in enumerate(SCALES):
            key = str(scale)
            perturbation = base_perturbation * scale
            pose = cache["pose"][i, s]
            neighbor = score_state(cache["start_pose"][i], pose, perturbation, NEIGHBORS)
            every = score_state(cache["start_pose"][i], pose, perturbation, ALL_BLOCKS)
            neighbor["all_blocks"] = every["full"]
            neighbor["all_blocks_final"] = every["final"]
            row["scores"][key] = neighbor
            row["grade"][key] = meta["by_scale"][key]["grade"]
            row["topple_count"][key] = meta["by_scale"][key]["topple_count"]
        rows.append(row)

    variants = ("full", "no_floor", "no_persistence", "all_blocks")
    results = {}
    for scale in map(str, SCALES):
        scored = [r for r in rows if r["grade"][scale] in ("mixed", "unanimous_safe")]
        labels = [r["grade"][scale] == "mixed" for r in scored]
        spread = [r["scores"][scale]["final"]["spread_rms_mm"] for r in scored]
        entry = {v: confusion(rows, scale, v) for v in variants}
        full = entry["full"]
        # Spread baseline at the same number of true positives as the full rule.
        pos = sorted((s for s, l in zip(spread, labels) if l), reverse=True)
        if full["tp"]:
            cut = pos[full["tp"] - 1]
            fp = sum(s >= cut for s, l in zip(spread, labels) if not l)
        else:
            fp = 0
        entry["spread_baseline"] = {"auc": auc(spread, labels),
                                    "fp_at_full_rule_tp": int(fp)}
        entry["by_stratum_full"] = {}
        for stratum in sorted({r["stratum"] for r in rows}):
            entry["by_stratum_full"][stratum] = confusion(
                [r for r in rows if r["stratum"] == stratum], scale, "full")
        results[scale] = entry

    result = {"protocol": {
                  "input": "Stage 0 true simulator endings (physics upper bound)",
                  "features": "neighbor box-corner displacement in mm (16 corners)",
                  "rule": "PC1 best split; BIC 2>1; Ashman D>2; minority>=2; group-mean "
                          "RMS corner separation > median injected per-chunk max perturbation "
                          "x scale; same split (<=1 probe) passes at hold step 10 and 30",
                  "perturbation_mm_at_1x": base_perturbation,
                  "positives": "mixed", "negatives": "unanimous_safe",
                  "weak_states": "reported, not scored",
                  "constants_fixed_before_results": True},
              "results": results, "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False, default=float) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
