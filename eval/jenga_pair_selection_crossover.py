"""DEV crossover: evaluate frozen D2 trajectories on physically selected nested paths."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import boundary_refinement_alarm, result_dict  # noqa: E402
from consequence_monitor import (result_dict as consequence_result_dict,  # noqa: E402
                                 whole_trajectory_consequence_alarm)
from jenga_action_boundary_refine import _projector  # noqa: E402
from jenga_action_branch import sampled_predicted_pose  # noqa: E402
from jenga_bench import BENCH_FILE, EVAL_CONFIG, sha256_file, verify as verify_benchmark  # noqa: E402
from jenga_generic_regime_audit import reconstruct_endpoints  # noqa: E402
from jenga_regime_monitor_v0_d2 import probe_windows  # noqa: E402
from jenga_regime_monitor_v0_freeze import verify as verify_v0  # noqa: E402
from jenga_selective_router_dev import nested_noises  # noqa: E402
from jenga_w5_eval import load_model, predict  # noqa: E402


def d2_on_selected_paths(model, model_scale, start, reference_windows, snippets,
                         pair_details, device):
    """Run D2 on all physically selected levels, using a D2-fitted trajectory projector."""
    original = predict(model, model_scale, start, reference_windows, device)
    original_features = sampled_predicted_pose(original, model)
    early_project = _projector(original_features[:, :2])
    full_project = _projector(original_features)

    noises = np.stack([nested_noises(snippets, pair) for pair in pair_details])
    flat_noises = noises.reshape(-1, *noises.shape[-2:])
    windows = probe_windows(reference_windows, snippets, flat_noises)
    trajectories = predict(model, model_scale, start, windows, device).reshape(
        len(pair_details), 6, 2, -1, 61)
    flat_features = sampled_predicted_pose(
        trajectories.reshape(-1, trajectories.shape[-2], 61), model)
    features = flat_features.reshape(len(pair_details), 6, 2, *flat_features.shape[1:])
    early, full = [], []
    for pair in features:
        pair_early, pair_full = [], []
        for level in pair:
            early_coordinates = early_project(level[:, :2])
            full_coordinates = full_project(level)
            pair_early.append(float(np.linalg.norm(
                early_coordinates[0] - early_coordinates[1])))
            pair_full.append(float(np.linalg.norm(
                full_coordinates[0] - full_coordinates[1])))
        early.append(pair_early); full.append(pair_full)
    boundary = boundary_refinement_alarm(early, full)
    final_traces = trajectories[:, -1]
    endpoints = np.stack([reconstruct_endpoints(snippets, [pair])[0]
                          for pair in pair_details])
    consequence = whole_trajectory_consequence_alarm(
        final_traces[:, :, :, :45], endpoints[:, 0] - endpoints[:, 1],
        boundary.early_delta_bic, boundary.full_delta_bic)
    return {
        "initial_alarm": True, "boundary_refined_alarm": bool(boundary.alarm),
        "boundary_evidence": result_dict(boundary),
        **consequence_result_dict(consequence),
        "early_gaps": np.asarray(early).tolist(), "full_gaps": np.asarray(full).tolist(),
    }


def summarize(rows):
    classes = ("topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    output = {}
    for name in classes:
        selected = [row for row in rows if row["class"] == name]
        output[name] = {
            "states": len(selected),
            "initial_candidates": sum(row["initial_alarm"] for row in selected),
            "boundary_candidates": sum(row.get("boundary_refined_alarm", False)
                                       for row in selected),
            "commitment_majority": sum(row.get("commitment_pairs", 0) >= 2 for row in selected),
            "persistence_majority": sum(row.get("persistence_pairs", 0) >= 2 for row in selected),
            "alarms": sum(row.get("alarm", False) for row in selected),
        }
    return output


def pair_overlap(selection_rows, d2_rows, physical_rows):
    selection = {(row["episode_id"], row["chunk_start"]): row for row in selection_rows}
    physical = {(row["episode_id"], row["chunk_start"]): row for row in physical_rows}
    classes = ("all", "topple_fork", "quiet", "nudge_fork", "small_split", "weak")
    output = {}
    for class_name in classes:
        chosen = [row for row in d2_rows if row["initial_alarm"] and
                  (class_name == "all" or row["class"] == class_name)]
        channels = {}
        for channel in ("boundary", "commitment", "persistence"):
            d2_values, physical_values = [], []
            for row in chosen:
                key = (row["episode_id"], row["chunk_start"])
                if channel == "boundary":
                    d2_evidence = row["boundary_evidence"]
                    physical_evidence = selection[key]
                    d2_values.extend((np.asarray(d2_evidence["early_delta_bic"]) > 0) &
                                     (np.asarray(d2_evidence["full_delta_bic"]) > 0))
                    physical_values.extend((np.asarray(physical_evidence["early_delta_bic"]) > 0) &
                                           (np.asarray(physical_evidence["full_delta_bic"]) > 0))
                elif channel == "commitment":
                    d2_values.extend(np.asarray(row["commitment_delta_bic"]) > 0)
                    physical_values.extend(np.asarray(
                        physical[key]["commitment_delta_bic"]) > 0)
                else:
                    d2_values.extend(np.asarray(row["persistence_delta_bic"]) > 0)
                    physical_values.extend(np.asarray(
                        physical[key]["persistence_delta_bic"]) > 0)
            left, right = np.asarray(d2_values, bool), np.asarray(physical_values, bool)
            channels[channel] = {
                "pairs": int(len(left)), "d2_positive": int(left.sum()),
                "physical_positive": int(right.sum()), "both_positive": int((left & right).sum()),
                "d2_only": int((left & ~right).sum()),
                "physical_only": int((~left & right).sum()),
                "both_negative": int((~left & ~right).sum()),
                "agreement": float(np.mean(left == right)) if len(left) else None,
            }
        output[class_name] = channels
    return output


def compact_existing_summary(result):
    source = result["summary"]["dev"]["1.0"]
    return {name: {"states": value["states"], "alarms": value["alarms"]}
            for name, value in source.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(
        ROOT / "results/jenga/pair_selection_crossover_protocol.json"))
    parser.add_argument("--selection", default=str(
        ROOT / "results/jenga/action_boundary_refine_dev.json"))
    parser.add_argument("--physical", default=str(
        ROOT / "results/jenga/whole_trajectory_consequence_dev.json"))
    parser.add_argument("--d2-selection", default=str(
        ROOT / "results/jenga/trajectory_level_ablation_summary.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results/jenga/w6_cw_d2_s1.pt"))
    parser.add_argument("--output", default=str(
        ROOT / "results/jenga/pair_selection_crossover_summary.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    verify_benchmark(); verify_v0()
    selection_result = json.loads(Path(args.selection).read_text())
    physical_result = json.loads(Path(args.physical).read_text())
    d2_selection = json.loads(Path(args.d2_selection).read_text())
    bench = np.load(BENCH_FILE, allow_pickle=False)
    snippets = bench["snippets"]
    index = {(str(ep), int(chunk)): i for i, (ep, chunk) in enumerate(
        zip(bench["dev_episode"], bench["dev_chunk"]))}
    model, model_scale = load_model(args.checkpoint, args.device)
    rows = []
    for number, prior in enumerate(selection_result["rows"], 1):
        row = {key: prior[key] for key in (
            "split", "episode_id", "chunk_start", "scale", "class", "topple_count")}
        row["initial_alarm"] = bool(prior["initial_alarm"])
        row["boundary_refined_alarm"] = False
        row["alarm"] = False
        if "pair_details" in prior:
            i = index[(row["episode_id"], row["chunk_start"])]
            windows = bench["dev_windows"][i, EVAL_CONFIG["scales"].index(1.0)]
            row.update(d2_on_selected_paths(
                model, model_scale, bench["dev_start"][i], windows,
                snippets, prior["pair_details"], args.device))
        rows.append(row)
        if number % 20 == 0:
            print(f"D2 physical-selected paths {number}/{len(selection_result['rows'])}", flush=True)

    new_summary = summarize(rows)
    physical_summary = compact_existing_summary(physical_result)
    d2_summaries = d2_selection["summaries"]
    matrix = {
        "d2_selection_d2_trajectory": d2_summaries["d2_boundary_d2_consequence"],
        "d2_selection_physical_trajectory": d2_summaries[
            "physical_boundary_physical_consequence"],
        "physical_selection_d2_trajectory": new_summary,
        "physical_selection_physical_trajectory": physical_summary,
    }
    rule = protocol["near_reference_rule"]
    required_topples = math.ceil(
        rule["topple_alarm_fraction_of_physical_min"] *
        physical_summary["topple_fork"]["alarms"])
    near_reference = (
        new_summary["topple_fork"]["alarms"] >= required_topples and
        new_summary["quiet"]["alarms"] <= physical_summary["quiet"]["alarms"])
    output = {
        "protocol": protocol, "protocol_sha256": sha256_file(args.protocol),
        "source_sha256": {
            "selection": sha256_file(args.selection), "physical": sha256_file(args.physical),
            "d2_selection": sha256_file(args.d2_selection),
            "checkpoint": sha256_file(args.checkpoint),
        },
        "matrix": matrix, "physical_selection_d2_rows": rows,
        "pair_overlap_on_physical_selection": pair_overlap(
            selection_result["rows"], rows, physical_result["rows"]),
        "near_physical_reference": near_reference,
        "near_reference_required_topple_alarms": required_topples,
        "decision": ("D2 is near the physical reference when handed physical-selected paths"
                     if near_reference else
                     "D2 remains weak on physical-selected paths; selection and dynamics both fail"),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"matrix": matrix, "near_physical_reference": near_reference,
                      "required_topples": required_topples,
                      "decision": output["decision"]}, indent=2))


if __name__ == "__main__":
    main()
