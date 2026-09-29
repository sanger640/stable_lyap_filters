"""Read-only D2 re-observation/interface analysis on held-out TRAIN trajectories.

No model is trained and no DEV/TEST task label is read.  Short forecasts are repeatedly started
from recorded physical states, but their outputs remain model predictions.  The construction is
therefore teacher-forced/oracle re-anchoring: it diagnoses error accumulation and does not claim
that hypothetical counterfactual branches can be observed during deployment.
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

from action_branch_monitor import (action_branch_partition, boundary_refinement_alarm,
                                   smooth_vs_branch_alarm)
from consequence_monitor import whole_trajectory_consequence_alarm
from jenga_d4_train import partition_agreement
from jenga_w5_eval import load_model
from neighborhood_topology_loss import neighborhood_terms
from state_dynamics import rollout


DATA = ROOT / "results/jenga/d4_neighborhood_data_large.npz"
NESTED = ROOT / "results/jenga/d3_data"
PROTOCOL = ROOT / "results/jenga/reobservation_interface_protocol.json"
OUTPUT = ROOT / "results/jenga/reobservation_interface_result.json"
INTERVALS = (1, 2, 4, 8, 38)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_rows(episode_ids, fraction=.12):
    episodes = np.asarray(episode_ids).astype(str)
    unique = sorted(set(episodes), key=int)
    heldout = unique[:max(1, int(np.ceil(len(unique) * fraction)))]
    return np.flatnonzero(np.isin(episodes, heldout)), heldout


def load_nested(directory, heldout):
    starts, actions, traces, sources = [], [], [], []
    heldout = set(map(str, heldout))
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        episode = path.stem.split("_")[0][2:]
        if episode not in heldout:
            continue
        data = np.load(path, allow_pickle=False)
        if not len(data["start"]):
            continue
        starts.append(data["start"])
        actions.append(data["actions"])
        traces.append(data["traces"])
        sources.extend([path.stem] * len(data["start"]))
    if not starts:
        raise RuntimeError("no held-out nested groups")
    return (np.concatenate(starts), np.concatenate(actions), np.concatenate(traces), sources)


def reanchored_rollout(model, model_scale, starts, actions, truth, interval, device,
                       batch_size=4096, rollout_function=rollout):
    """Predict every state while resetting each segment's input to recorded physical state.

    ``truth[:, t]`` is the state after action ``t``.  A segment beginning at ``t`` starts from the
    common initial state when t=0 and from ``truth[:, t-1]`` otherwise.  Predicted segment endpoints
    are retained in the returned trajectory; only the next segment's hidden input is re-anchored.
    Thus interval=1 measures one-step prediction rather than replacing predictions with truth.
    """
    starts = np.asarray(starts, np.float32)
    actions = np.asarray(actions, np.float32)
    truth = np.asarray(truth, np.float32)
    if actions.ndim != 3 or truth.shape != (*actions.shape[:2], 61):
        raise ValueError("actions/truth must have shapes (N,T,4)/(N,T,61)")
    if starts.shape != (len(actions), 61) or interval < 1:
        raise ValueError("invalid starts or interval")
    predicted = np.empty_like(truth)
    for first in range(0, actions.shape[1], interval):
        stop = min(first + interval, actions.shape[1])
        segment_start = starts if first == 0 else truth[:, first - 1]
        for begin in range(0, len(actions), batch_size):
            end = min(begin + batch_size, len(actions))
            state = torch.as_tensor(segment_start[begin:end], device=device)
            control = torch.as_tensor(actions[begin:end, first:stop], device=device)
            with torch.no_grad():
                values = rollout_function(model, state, control, model_scale)
            predicted[begin:end, first:stop] = values.cpu().numpy()
    return predicted


def binary_metrics(reference, predicted):
    reference = np.asarray(reference, bool)
    predicted = np.asarray(predicted, bool)
    tp = int(np.sum(reference & predicted)); fn = int(np.sum(reference & ~predicted))
    fp = int(np.sum(~reference & predicted)); tn = int(np.sum(~reference & ~predicted))
    return {
        "states": int(len(reference)), "reference_positive": int(reference.sum()),
        "predicted_positive": int(predicted.sum()), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "agreement": float((tp + tn) / max(len(reference), 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "added_positive_rate": float(fp / max(fp + tn, 1)),
    }


def candidate_rows(actions, physical, predicted):
    physical_decisions, predicted_decisions, partitions = [], [], []
    valid = []
    for index, (control, truth, estimate) in enumerate(zip(actions, physical, predicted)):
        try:
            physical_evidence = smooth_vs_branch_alarm(control, truth[:, :18, :45], truth[:, :, :45])
            predicted_evidence = smooth_vs_branch_alarm(
                control, estimate[:, :18, :45], estimate[:, :, :45])
            _, physical_labels = action_branch_partition(control, truth[:, :18, :45],
                                                         truth[:, :, :45])
            _, predicted_labels = action_branch_partition(control, estimate[:, :18, :45],
                                                          estimate[:, :, :45])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        physical_decisions.append(physical_evidence.alarm)
        predicted_decisions.append(predicted_evidence.alarm)
        partitions.append(partition_agreement(physical_labels, predicted_labels))
        valid.append(index)
    result = binary_metrics(physical_decisions, predicted_decisions)
    result["valid_states"] = len(valid)
    result["mean_partition_agreement"] = float(np.mean(partitions)) if partitions else None
    return result


def gap_rows(traces, state_scale):
    """Task-free nested-width boundary/commitment/persistence decisions per group."""
    scale = np.asarray(state_scale, float)[:45]
    rows = []
    for group in traces:
        normalized = group[..., :45] / np.maximum(scale, 1e-6)
        difference = normalized[:, 0] - normalized[:, 1]
        early = np.sqrt(np.mean(difference[:, :18] ** 2, axis=(1, 2)))
        full = np.sqrt(np.mean(difference ** 2, axis=(1, 2)))
        boundary = boundary_refinement_alarm(early[None], full[None])
        rows.append((boundary, early, full))
    return rows


def nested_stage_rows(actions, traces, state_scale):
    boundaries = gap_rows(traces, state_scale)
    rows = []
    for control, group, (boundary, _, _) in zip(actions, traces, boundaries):
        narrow = group[-1:, :, :, :45]
        action_difference = (control[-1, 0, :8] - control[-1, 1, :8])[None]
        final = whole_trajectory_consequence_alarm(
            narrow, action_difference, boundary.early_delta_bic, boundary.full_delta_bic)
        rows.append({
            "boundary": bool(boundary.alarm),
            "commitment": bool(final.commitment_pairs >= 1),
            "persistence": bool(final.persistence_pairs >= 1),
            "alarm": bool(final.alarm),
        })
    return rows


def compare_nested(reference, predicted):
    return {stage: binary_metrics([row[stage] for row in reference],
                                  [row[stage] for row in predicted])
            for stage in ("boundary", "commitment", "persistence", "alarm")}


def continuous_metrics(predicted, truth, actions, state_scale, device, states_per_batch=4):
    totals = []
    with torch.no_grad():
        for first in range(0, len(predicted), states_per_batch):
            stop = min(first + states_per_batch, len(predicted))
            terms = neighborhood_terms(
                torch.as_tensor(predicted[first:stop], device=device),
                torch.as_tensor(truth[first:stop], device=device),
                torch.as_tensor(actions[first:stop], device=device), state_scale)
            totals.append((stop - first, float(terms.response), float(terms.topology),
                           float(terms.commitment)))
    count = sum(row[0] for row in totals)
    return {
        "response": float(sum(row[0] * row[1] for row in totals) / count),
        "topology": float(sum(row[0] * row[2] for row in totals) / count),
        "commitment": float(sum(row[0] * row[3] for row in totals) / count),
    }


def range_summary(values):
    values = np.asarray(values, float)
    return {"median": float(np.median(values)), "min": float(np.min(values)),
            "max": float(np.max(values))}


def aggregate(per_seed):
    summary = {}
    for interval in INTERVALS:
        key = str(interval)
        candidate_fields = ("agreement", "recall", "added_positive_rate",
                            "mean_partition_agreement")
        entry = {"candidate": {field: range_summary([
            row["intervals"][key]["candidate"][field] for row in per_seed])
            for field in candidate_fields}}
        entry["continuous"] = {field: range_summary([
            row["intervals"][key]["continuous"][field] for row in per_seed])
            for field in ("response", "topology", "commitment")}
        entry["nested"] = {stage: {field: range_summary([
            row["intervals"][key]["nested"][stage][field] for row in per_seed])
            for field in ("agreement", "recall", "added_positive_rate")}
            for stage in ("boundary", "commitment", "persistence", "alarm")}
        summary[key] = entry
    return summary


def decide(summary):
    open_loop = summary["38"]
    rows = []
    for interval in (4, 2, 1):
        current = summary[str(interval)]
        candidate_gain = (current["candidate"]["recall"]["median"]
                          - open_loop["candidate"]["recall"]["median"])
        alarm_gain = (current["nested"]["alarm"]["recall"]["median"]
                      - open_loop["nested"]["alarm"]["recall"]["median"])
        candidate_added = (current["candidate"]["added_positive_rate"]["median"]
                           - open_loop["candidate"]["added_positive_rate"]["median"])
        alarm_added = (current["nested"]["alarm"]["added_positive_rate"]["median"]
                       - open_loop["nested"]["alarm"]["added_positive_rate"]["median"])
        passed = (candidate_gain >= .10 and alarm_gain >= .10
                  and candidate_added <= .05 and alarm_added <= .05)
        rows.append({"interval": interval, "candidate_recall_gain": candidate_gain,
                     "final_recall_gain": alarm_gain,
                     "candidate_added_positive_change": candidate_added,
                     "final_added_positive_change": alarm_added, "passes": bool(passed)})
    passing = [row for row in rows if row["passes"]]
    if passing:
        chosen = max(passing, key=lambda row: row["interval"])["interval"]
        conclusion = ("Repeated short-horizon monitoring is supported; design a causal runtime "
                      f"version beginning with interval {chosen}.")
        direction = "short_horizon_reobservation"
    else:
        chosen = None
        conclusion = ("Re-anchoring does not pass the frozen recovery rule; prefer a direct local "
                      "action-response/boundary model over another long open-loop dynamics model.")
        direction = "direct_local_response"
    return {"comparisons": rows, "selected_interval": chosen,
            "direction": direction, "conclusion": conclusion}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--nested", default=str(NESTED))
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--models", nargs="+", default=[
        str(ROOT / f"results/jenga/w6_cw_d2_s{seed}.pt") for seed in range(1, 11)])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; use --force deliberately")
    protocol = json.loads(Path(args.protocol).read_text())
    if protocol.get("status") != "frozen before execution" or protocol["data"]["test_opened"]:
        raise SystemExit("invalid or non-frozen protocol")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    source = np.load(args.data, allow_pickle=False)
    rows, heldout = validation_rows(source["episode_id"])
    starts = source["start"][rows]
    actions = source["actions"][rows]
    truth = source["traces"][rows]
    nested_start, nested_actions, nested_truth, nested_sources = load_nested(args.nested, heldout)
    flat_start = np.concatenate([
        np.repeat(starts[:, None], actions.shape[1], axis=1).reshape(-1, 61),
        np.repeat(nested_start[:, None], nested_actions.shape[1] * 2, axis=1).reshape(-1, 61),
    ])
    flat_actions = np.concatenate([
        actions.reshape(-1, actions.shape[2], 4),
        nested_actions.reshape(-1, nested_actions.shape[3], 4),
    ])
    flat_truth = np.concatenate([
        truth.reshape(-1, truth.shape[2], 61),
        nested_truth.reshape(-1, nested_truth.shape[3], 61),
    ])
    neighborhood_count = len(starts) * actions.shape[1]

    per_seed = []
    for model_number, model_path in enumerate(args.models, 1):
        model, model_scale = load_model(model_path, device)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_scale = checkpoint["state_scale"].to(device)
        physical_nested = nested_stage_rows(nested_actions, nested_truth,
                                             state_scale.cpu().numpy())
        intervals = {}
        for interval in INTERVALS:
            predicted = reanchored_rollout(
                model, model_scale, flat_start, flat_actions, flat_truth, interval,
                device, args.batch_size)
            predicted_neighborhood = predicted[:neighborhood_count].reshape(truth.shape)
            predicted_nested = predicted[neighborhood_count:].reshape(nested_truth.shape)
            intervals[str(interval)] = {
                "candidate": candidate_rows(actions, truth, predicted_neighborhood),
                "continuous": continuous_metrics(predicted_neighborhood, truth, actions,
                                                   state_scale, device),
                "nested": compare_nested(
                    physical_nested, nested_stage_rows(
                        nested_actions, predicted_nested, state_scale.cpu().numpy())),
            }
            print(f"model {model_number}/{len(args.models)} interval {interval} complete", flush=True)
        resolved_model = Path(model_path).resolve()
        per_seed.append({"model": str(resolved_model.relative_to(ROOT)),
                         "model_sha256": sha256(model_path),
                         "seed": checkpoint.get("seed"), "intervals": intervals})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = aggregate(per_seed)
    result = {
        "protocol": protocol,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": sha256(args.protocol),
        "data_sha256": sha256(args.data),
        "heldout_episodes": heldout,
        "neighborhood_states": int(len(starts)),
        "nested_groups": int(len(nested_start)),
        "nested_sources": nested_sources,
        "intervals": list(INTERVALS),
        "per_seed": per_seed,
        "summary": summary,
        "decision": decide(summary),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": summary, "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
