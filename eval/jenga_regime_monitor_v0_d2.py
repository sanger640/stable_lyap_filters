"""Apply frozen Regime Monitor v0 to learned Jenga state-GNN world models.

This is the learned-rollout counterpart of ``jenga_regime_monitor_v0_test.py``.  The detector,
execution probes, selected benchmark split, labels, and all structural constants stay frozen. The
default remains TEST; DEV is available for prospective model selection. The only changed
variable is the trajectory source: every original probe, adaptive midpoint, and final refined pair
is rolled out by one learned GNN instead of MuJoCo.

Physical classes are joined after each decision for grading.  They never enter candidate discovery,
adaptive refinement, or the final alarm.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (action_branch_partition, boundary_refinement_alarm,  # noqa: E402
                                   nearest_cross_branch_pairs,
                                   result_dict as branch_result_dict,
                                   smooth_vs_branch_alarm)
from consequence_monitor import (result_dict as consequence_result_dict,  # noqa: E402
                                 whole_trajectory_consequence_alarm)
from jenga_action_boundary_refine import PAIRS, REFINEMENTS, _projector  # noqa: E402
from jenga_action_branch import (action_errors, pose_features, sampled_predicted_pose,  # noqa: E402
                                 summarise)
from jenga_bench import (BENCH_FILE, EVAL_CONFIG, checkpoint_identity,  # noqa: E402
                         verify as verify_benchmark)
from jenga_generic_regime_audit import reconstruct_endpoints  # noqa: E402
from jenga_regime_monitor_v0_freeze import (MANIFEST_FILE as V0_MANIFEST,  # noqa: E402
                                            sha256_file, verify as verify_v0)
from jenga_stage0_noise_oracle import RECORD_AT  # noqa: E402
from jenga_w5_eval import load_model, predict  # noqa: E402

SPLIT = "test"
SCALE = 1.0
OUTPUT_DIR = ROOT / "results/jenga/regime_monitor_v0/d2"
AGGREGATE = OUTPUT_DIR / "aggregate.json"


def probe_windows(reference_windows, snippets, noises):
    """Build exact D2 rollout windows for arbitrary H=8 execution-noise snippets.

    The frozen benchmark uses a common nominal hold, not each noisy probe's last target.  Recover
    the nominal H=8 chunk from the frozen first probe and its known snippet, then preserve the
    frozen two-step history, gripper commands, and 30-step common hold exactly.
    """
    reference = np.asarray(reference_windows, np.float32)
    snippets = np.asarray(snippets, np.float32)
    noises = np.asarray(noises, np.float32)
    nominal = reference[0, 2:10].copy()
    nominal[:, :3] += snippets[0]
    out = np.repeat(reference[0:1], len(noises), axis=0)
    out[:, 2:10] = nominal
    out[:, 2:10, :3] -= noises
    out[:, 10:] = nominal[-1]
    return out


def verify_original_windows(reference_windows, snippets):
    """Refuse evaluation if reconstructed frozen probes differ beyond float32 roundoff."""
    rebuilt = probe_windows(reference_windows, snippets, snippets)
    if not np.allclose(rebuilt, np.asarray(reference_windows, np.float32), rtol=1e-6, atol=1e-7):
        maximum = float(np.max(np.abs(rebuilt - reference_windows)))
        raise RuntimeError(f"frozen D2 windows cannot be reconstructed (max difference {maximum})")


def _sample_features(predicted, model):
    return sampled_predicted_pose(predicted, model)


def refine_with_model(model, model_scale, start, reference_windows, snippets, features,
                      labels, pairs, device):
    """Run the frozen five-level adaptive bisection entirely through one D2 model."""
    early_stop = RECORD_AT.index(10) + 1
    early_project = _projector(features[:, :early_stop])
    full_project = _projector(features)
    original_early = early_project(features[:, :early_stop])
    original_full = full_project(features)
    centroids = np.stack([original_full[labels == group].mean(0) for group in (0, 1)])

    endpoint_noise = np.stack([
        np.stack([snippets[left], snippets[right]]) for left, right in pairs])
    endpoint_features = np.stack([
        np.stack([features[left], features[right]]) for left, right in pairs])
    early_gaps = [[float(np.linalg.norm(original_early[left] - original_early[right]))]
                  for left, right in pairs]
    full_gaps = [[float(np.linalg.norm(original_full[left] - original_full[right]))]
                 for left, right in pairs]
    midpoint_sides = [[] for _ in pairs]

    for _ in range(REFINEMENTS):
        midpoints = endpoint_noise.mean(axis=1)
        windows = probe_windows(reference_windows, snippets, midpoints)
        predicted = predict(model, model_scale, start, windows, device)
        midpoint_features = _sample_features(predicted, model)
        midpoint_full = full_project(midpoint_features)
        for pair_index in range(len(pairs)):
            side = int(np.argmin(np.linalg.norm(
                centroids - midpoint_full[pair_index][None], axis=1)))
            endpoint_noise[pair_index, side] = midpoints[pair_index]
            endpoint_features[pair_index, side] = midpoint_features[pair_index]
            projected_early = early_project(endpoint_features[pair_index, :, :early_stop])
            projected_full = full_project(endpoint_features[pair_index])
            early_gaps[pair_index].append(float(np.linalg.norm(
                projected_early[0] - projected_early[1])))
            full_gaps[pair_index].append(float(np.linalg.norm(
                projected_full[0] - projected_full[1])))
            midpoint_sides[pair_index].append(side)

    evidence = boundary_refinement_alarm(early_gaps, full_gaps)
    details = [{"probe_indices": [int(left), int(right)],
                "midpoint_sides": midpoint_sides[index],
                "early_gaps": early_gaps[index], "full_gaps": full_gaps[index]}
               for index, (left, right) in enumerate(pairs)]
    return evidence, endpoint_noise, details


def evaluate_checkpoint(path, device):
    verify_benchmark()
    verify_v0(V0_MANIFEST)
    bench = np.load(BENCH_FILE, allow_pickle=False)
    snippets = bench["snippets"]
    scale_index = EVAL_CONFIG["scales"].index(SCALE)
    model, model_scale = load_model(path, device)
    rows = []

    for index, start in enumerate(bench[f"{SPLIT}_start"]):
        windows = bench[f"{SPLIT}_windows"][index, scale_index]
        verify_original_windows(windows, snippets)
        predicted = predict(model, model_scale, start, windows, device)
        features = _sample_features(predicted, model)
        early_stop = RECORD_AT.index(10) + 1
        initial = smooth_vs_branch_alarm(
            action_errors(windows), features[:, :early_stop], features)
        row = {
            "split": SPLIT,
            "episode_id": str(bench[f"{SPLIT}_episode"][index]),
            "chunk_start": int(bench[f"{SPLIT}_chunk"][index]),
            "scale": str(SCALE),
            "class": str(bench[f"{SPLIT}_classes"][index, scale_index]),
            "topple_count": int(bench[f"{SPLIT}_topples"][index, scale_index]),
            "initial_alarm": bool(initial.alarm),
            "boundary_refined_alarm": False,
            "alarm": False,
            "initial_evidence": branch_result_dict(initial),
        }
        if initial.alarm:
            coordinates, labels = action_branch_partition(
                action_errors(windows), features[:, :early_stop], features)
            pairs = nearest_cross_branch_pairs(coordinates, labels, PAIRS)
            boundary, endpoints, details = refine_with_model(
                model, model_scale, start, windows, snippets, features, labels, pairs, device)
            row["boundary_refined_alarm"] = bool(boundary.alarm)
            row["boundary_evidence"] = branch_result_dict(boundary)
            row["pair_details"] = details
            # Match the frozen physical evaluator exactly: record commitment and persistence for
            # every initial candidate, even when boundary refinement already makes the final
            # conjunction impossible.  This does not change alarms but makes stage attrition exact.
            endpoint_windows = probe_windows(
                windows, snippets, endpoints.reshape(-1, *snippets.shape[1:]))
            traces = predict(model, model_scale, start, endpoint_windows, device)
            traces = traces.reshape(PAIRS, 2, traces.shape[1], traces.shape[2])
            final = whole_trajectory_consequence_alarm(
                traces[:, :, :, :45], endpoints[:, 0] - endpoints[:, 1],
                boundary.early_delta_bic, boundary.full_delta_bic)
            row.update(consequence_result_dict(final))
        rows.append(row)
        if (index + 1) % 10 == 0 or index + 1 == len(bench[f"{SPLIT}_start"]):
            print(f"{Path(path).stem}: {index + 1}/{len(bench[f'{SPLIT}_start'])}", flush=True)
    return model, rows


def candidate_summary(rows):
    classes = ("topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    return {name: {
        "states": sum(row["class"] == name for row in rows),
        "initial_candidates": sum(row["class"] == name and row["initial_alarm"] for row in rows),
        "boundary_candidates": sum(row["class"] == name and row["boundary_refined_alarm"]
                                   for row in rows),
        "commitment_majority": sum(row["class"] == name
                                   and row.get("commitment_pairs", 0) >= 2 for row in rows),
        "persistence_majority": sum(row["class"] == name
                                    and row.get("persistence_pairs", 0) >= 2 for row in rows),
        "v0_alarms": sum(row["class"] == name and row["alarm"] for row in rows),
    } for name in classes}


def _decision_map(rows):
    return {(row["episode_id"], row["chunk_start"]): bool(row["alarm"]) for row in rows}


def physical_agreement(rows):
    physical = json.loads((ROOT / "results/jenga/regime_monitor_v0/test_ground_truth.json").read_text())
    reference = _decision_map(physical["rows"])
    predicted = _decision_map(rows)
    if reference.keys() != predicted.keys():
        raise RuntimeError("D2 and physical v0 rows do not identify the same frozen TEST states")
    keys = list(reference)
    true_positive = sum(reference[key] and predicted[key] for key in keys)
    false_negative = sum(reference[key] and not predicted[key] for key in keys)
    false_positive = sum(not reference[key] and predicted[key] for key in keys)
    true_negative = sum(not reference[key] and not predicted[key] for key in keys)
    return {
        "states": len(keys), "agreement": (true_positive + true_negative) / len(keys),
        "physical_alarm_recall": true_positive / max(true_positive + false_negative, 1),
        "added_alarm_rate": false_positive / max(false_positive + true_negative, 1),
        "tp": true_positive, "fn": false_negative, "fp": false_positive, "tn": true_negative,
    }


def checkpoint_result(path, device):
    model, rows = evaluate_checkpoint(path, device)
    v0 = verify_v0(V0_MANIFEST)
    result = {
        "protocol": {
            "id": "regime-monitor-v0-on-learned-gnn",
            "split": f"Jenga {SPLIT.upper()}",
            "calibration_free": True,
            "trajectory_source": "learned state GNN; no MuJoCo future rollouts",
            "changed_from_ground_truth_v0": "trajectory source only",
            "task_information": "physical classes joined for grading only",
        },
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regime_monitor_v0_sha256": v0["protocol_sha256"],
        "benchmark_sha256": sha256_file(BENCH_FILE),
        "model": checkpoint_identity(path),
        "model_substeps": int(getattr(model, "substeps", 1)),
        "summary": summarise(rows, [SCALE]),
        "candidate_summary": candidate_summary(rows),
        "rows": rows,
    }
    if SPLIT == "test":
        result["physical_v0_agreement"] = physical_agreement(rows)
    return result


def complete_reused_result(result, path, device):
    """Backfill full frozen-stage evidence in early result files without repeating refinement."""
    missing = [row for row in result["rows"]
               if row["initial_alarm"] and "commitment_pairs" not in row]
    if not missing:
        return result, False
    verify_benchmark()
    verify_v0(V0_MANIFEST)
    bench = np.load(BENCH_FILE, allow_pickle=False)
    snippets = bench["snippets"]
    scale_index = EVAL_CONFIG["scales"].index(SCALE)
    index = {(str(episode), int(chunk)): i for i, (episode, chunk) in enumerate(zip(
        bench[f"{SPLIT}_episode"], bench[f"{SPLIT}_chunk"]))}
    model, model_scale = load_model(path, device)
    for number, row in enumerate(missing, 1):
        state_index = index[(row["episode_id"], row["chunk_start"])]
        start = bench[f"{SPLIT}_start"][state_index]
        windows = bench[f"{SPLIT}_windows"][state_index, scale_index]
        endpoints = reconstruct_endpoints(snippets, row["pair_details"])
        endpoint_windows = probe_windows(
            windows, snippets, endpoints.reshape(-1, *snippets.shape[1:]))
        traces = predict(model, model_scale, start, endpoint_windows, device)
        traces = traces.reshape(PAIRS, 2, traces.shape[1], traces.shape[2])
        boundary = row["boundary_evidence"]
        final = whole_trajectory_consequence_alarm(
            traces[:, :, :, :45], endpoints[:, 0] - endpoints[:, 1],
            boundary["early_delta_bic"], boundary["full_delta_bic"])
        row.update(consequence_result_dict(final))
        if number % 20 == 0 or number == len(missing):
            print(f"{Path(path).stem}: backfilled {number}/{len(missing)}", flush=True)
    result["summary"] = summarise(result["rows"], [SCALE])
    result["candidate_summary"] = candidate_summary(result["rows"])
    if SPLIT == "test":
        result["physical_v0_agreement"] = physical_agreement(result["rows"])
    return result, True


def _mean_range(values):
    values = np.asarray([value for value in values if value is not None], float)
    if not len(values):
        return {"mean": None, "min": None, "max": None}
    return {"mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max())}


def aggregate(results):
    classes = ("topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    class_rates = {}
    for name in classes:
        rates = [result["summary"][SPLIT][str(SCALE)][name]["rate"] for result in results]
        class_rates[name] = _mean_range(rates)
    stages = {}
    for name in classes:
        stages[name] = {}
        for field in ("initial_candidates", "boundary_candidates", "commitment_majority",
                      "persistence_majority", "v0_alarms"):
            stages[name][field] = _mean_range([
                result["candidate_summary"][name][field] for result in results])
        stages[name]["states"] = results[0]["candidate_summary"][name]["states"]
    agreement = None
    if all("physical_v0_agreement" in result for result in results):
        agreement = {field: _mean_range([result["physical_v0_agreement"][field]
                                         for result in results])
                     for field in ("agreement", "physical_alarm_recall", "added_alarm_rate")}

    keys = [(row["episode_id"], row["chunk_start"])
            for row in results[0]["rows"]]
    robust_alarm = []
    robust_miss = []
    for key in keys:
        decisions = []
        cls = None
        for result in results:
            row = next(row for row in result["rows"]
                       if (row["episode_id"], row["chunk_start"]) == key)
            decisions.append(bool(row["alarm"])); cls = row["class"]
        count = sum(decisions)
        if count >= int(np.ceil(.8 * len(results))):
            robust_alarm.append({"episode_id": key[0], "chunk_start": key[1], "class": cls,
                                 "alarm_seeds": count})
        if cls == "topple_fork" and len(results) - count >= int(np.ceil(.8 * len(results))):
            robust_miss.append({"episode_id": key[0], "chunk_start": key[1],
                                "alarm_seeds": count})
    combined = {
        "protocol": results[0]["protocol"],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regime_monitor_v0_sha256": results[0]["regime_monitor_v0_sha256"],
        "seeds": [result["model"].get("seed") for result in results],
        "class_alarm_rates": class_rates,
        "stage_counts": stages,
        "robust_alarm_count": len(robust_alarm),
        "robust_alarms": robust_alarm,
        "robust_blind_topple_count": len(robust_miss),
        "robust_blind_topples": robust_miss,
        "per_seed": [{
            "model": result["model"],
            "topple_recall": result["summary"][SPLIT][str(SCALE)]["topple_fork"]["rate"],
            "quiet_alarm_rate": result["summary"][SPLIT][str(SCALE)]["quiet"]["rate"],
            **result.get("physical_v0_agreement", {}),
        } for result in results],
    }
    if agreement is not None:
        combined["physical_v0_agreement"] = agreement
    return combined


def main():
    global SPLIT
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        str(ROOT / f"results/jenga/w6_cw_d2_s{seed}.pt") for seed in range(1, 11)])
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split", choices=("dev", "test"), default="test",
                        help="DEV is for prospective model selection; TEST remains the frozen gate")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse complete per-checkpoint JSON files after identity checks")
    args = parser.parse_args()
    SPLIT = args.split
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for model_path in map(Path, args.models):
        output = output_dir / f"{model_path.stem}.json"
        expected = checkpoint_identity(model_path)
        if args.reuse and output.is_file():
            result = json.loads(output.read_text())
            if result.get("model") != expected:
                raise SystemExit(f"refusing stale result with different checkpoint identity: {output}")
            if result.get("regime_monitor_v0_sha256") != verify_v0(V0_MANIFEST)["protocol_sha256"]:
                raise SystemExit(f"refusing stale result with different v0 protocol: {output}")
            print(f"reusing {output}", flush=True)
            result, changed = complete_reused_result(result, model_path, device)
            if changed:
                output.write_text(json.dumps(result, indent=2) + "\n")
        else:
            result = checkpoint_result(model_path, device)
            output.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
    combined = aggregate(results)
    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps({
        "device": device,
        "class_alarm_rates": combined["class_alarm_rates"],
        "physical_v0_agreement": combined.get("physical_v0_agreement"),
        "robust_blind_topple_count": combined["robust_blind_topple_count"],
        "output": str(aggregate_path),
    }, indent=2))


if __name__ == "__main__":
    main()
