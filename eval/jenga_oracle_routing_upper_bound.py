"""TRAIN-only oracle routing upper bound for reusable physical scale-curve regimes.

The oracle is deliberately non-deployable: it sees the physical future to choose one of three
reusable correction prototypes for D2's predicted nested-width log-gap curve. Prototypes are fit on
training episodes with capacity-balanced hard assignment and scored on held-out training episodes.
This is a necessary-condition test before another causal discrete router is justified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from jenga_w5_eval import load_model  # noqa: E402
from state_dynamics import rollout  # noqa: E402

PHASES = (slice(0, 8), slice(8, 18), slice(18, 38))
PHASE_NAMES = ("action", "early_hold", "late_hold")
EPSILON = 1e-4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_id(path):
    return Path(path).stem.replace("ep", "").replace("_seed", ":")


def validation_ids(trace_directory, fraction=.12):
    identities = sorted({source_id(path) for path in Path(trace_directory).glob("ep*_seed*.npz")})
    return set(identities[:max(1, int(len(identities) * fraction))])


def load_groups(directory):
    starts, actions, traces, sources = [], [], [], []
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        data = np.load(path, allow_pickle=False)
        if not len(data["start"]):
            continue
        starts.append(data["start"]); actions.append(data["actions"])
        traces.append(data["traces"]); sources.extend([source_id(path)] * len(data["start"]))
    return (np.concatenate(starts), np.concatenate(actions), np.concatenate(traces),
            np.asarray(sources))


def log_gap_curve(trajectories, state_scale):
    """Return five relative log-gap levels x three equally weighted temporal phases."""
    effect = (trajectories[:, :, 1, :, :45] - trajectories[:, :, 0, :, :45]) / state_scale[:45]
    gaps = np.stack([
        np.sqrt(np.mean(effect[:, :, phase] ** 2, axis=(2, 3)) + 1e-12)
        for phase in PHASES
    ], axis=-1)
    return np.log((gaps[:, 1:] + EPSILON) / (gaps[:, :1] + EPSILON))


def predict_groups(checkpoint, starts, actions, device, batch_size=8):
    model, delta_scale = load_model(checkpoint, device)
    predictions = []
    with torch.no_grad():
        for first in range(0, len(starts), batch_size):
            start = torch.as_tensor(starts[first:first + batch_size], device=device)
            action = torch.as_tensor(actions[first:first + batch_size], device=device)
            count, levels, endpoints, horizon = action.shape[:4]
            flat_start = start.repeat_interleave(levels * endpoints, 0)
            flat_action = action.reshape(count * levels * endpoints, horizon, 4)
            predicted = rollout(model, flat_start, flat_action, delta_scale)
            predictions.append(predicted.reshape(count, levels, endpoints, horizon, 61).cpu().numpy())
    return np.concatenate(predictions)


def balanced_assign(values, centroids):
    """Minimum-cost hard assignment with mode capacities differing by at most one."""
    count, modes = len(values), len(centroids)
    capacities = np.full(modes, count // modes, int)
    capacities[:count % modes] += 1
    slots = np.repeat(np.arange(modes), capacities)
    cost = cdist(values, centroids, metric="sqeuclidean")[:, slots]
    rows, columns = linear_sum_assignment(cost)
    assignment = np.empty(count, int); assignment[rows] = slots[columns]
    return assignment


def balanced_kmeans(values, modes=3, maximum_iterations=50):
    """Deterministic farthest-first initialization followed by balanced hard EM."""
    mean = values.mean(0, keepdims=True)
    chosen = [int(np.argmax(((values - mean) ** 2).sum(1)))]
    while len(chosen) < modes:
        distance = cdist(values, values[chosen], metric="sqeuclidean").min(1)
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    centroids = values[chosen].copy()
    previous = None
    for iteration in range(1, maximum_iterations + 1):
        assignment = balanced_assign(values, centroids)
        centroids = np.stack([values[assignment == mode].mean(0) for mode in range(modes)])
        if previous is not None and np.array_equal(assignment, previous):
            break
        previous = assignment.copy()
    # Stable ordering makes mode descriptions reproducible despite permutation symmetry.
    order = np.argsort(centroids[:, -1])
    inverse = np.empty_like(order); inverse[order] = np.arange(modes)
    return centroids[order], inverse[assignment], iteration


def huber(values, delta=1.0):
    absolute = np.abs(values)
    return float(np.mean(np.where(absolute <= delta, .5 * values ** 2,
                                  delta * (absolute - .5 * delta))))


def losses(predicted, truth):
    global_error = predicted - truth
    predicted_full = np.concatenate([np.zeros_like(predicted[:, :1]), predicted], axis=1)
    truth_full = np.concatenate([np.zeros_like(truth[:, :1]), truth], axis=1)
    local_error = np.diff(predicted_full, axis=1) - np.diff(truth_full, axis=1)
    return {"global_scale_huber": huber(global_error),
            "local_scale_huber": huber(local_error)}


def classification(predicted, truth):
    predicted_ratio, truth_ratio = np.exp(predicted[:, -1]), np.exp(truth[:, -1])
    true_persistent = truth_ratio.max(1) > .25
    predicted_persistent = predicted_ratio.max(1) > .25
    true_smooth = truth_ratio.max(1) < .1
    predicted_smooth = predicted_ratio.max(1) < .1
    return {
        "persistent_groups": int(true_persistent.sum()),
        "persistent_recall": float(predicted_persistent[true_persistent].mean())
        if true_persistent.any() else None,
        "smooth_groups": int(true_smooth.sum()),
        "smooth_recall": float(predicted_smooth[true_smooth].mean()) if true_smooth.any() else None,
        "median_final_ratio": {phase: float(np.median(predicted_ratio[:, index]))
                               for index, phase in enumerate(PHASE_NAMES)},
    }


def evaluate_split(predicted, truth, centroids):
    residual = (truth - predicted).reshape(len(truth), -1)
    assignment = cdist(residual, centroids, metric="sqeuclidean").argmin(1)
    corrected = predicted + centroids[assignment].reshape(-1, 5, 3)
    baseline_loss, oracle_loss = losses(predicted, truth), losses(corrected, truth)
    reductions = {name: 1.0 - oracle_loss[name] / max(baseline_loss[name], 1e-12)
                  for name in baseline_loss}
    return {"groups": len(truth), "assignment": assignment.tolist(),
            "mode_occupancy": (np.bincount(assignment, minlength=len(centroids))
                               / len(assignment)).tolist(),
            "baseline_loss": baseline_loss, "oracle_loss": oracle_loss,
            "loss_reduction": reductions,
            "baseline_classification": classification(predicted, truth),
            "oracle_classification": classification(corrected, truth)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "results/jenga/d3_data"))
    parser.add_argument("--trace", default=str(ROOT / "results/jenga/trace_data"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results/jenga/w6_cw_d2_s1.pt"))
    parser.add_argument("--protocol", default=str(
        ROOT / "results/jenga/oracle_routing_upper_bound_protocol.json"))
    parser.add_argument("--output", default=str(
        ROOT / "results/jenga/oracle_routing_upper_bound_summary.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    starts, actions, truth_trajectories, sources = load_groups(args.data)
    heldout_ids = validation_ids(args.trace)
    heldout = np.asarray([source in heldout_ids for source in sources])
    fit_groups = int((~heldout).sum())
    heldout_groups = int(heldout.sum())
    if fit_groups != protocol["fit_groups_expected"]:
        raise ValueError(f"expected {protocol['fit_groups_expected']} fit groups, got {fit_groups}")
    if heldout_groups != protocol["heldout_groups_expected"]:
        raise ValueError(
            f"expected {protocol['heldout_groups_expected']} held-out groups, got {heldout_groups}")
    fit_ids = set(sources[~heldout])
    observed_heldout_ids = set(sources[heldout])
    if fit_ids & observed_heldout_ids:
        raise ValueError("episode leakage between fit and held-out TRAIN groups")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_scale = checkpoint["state_scale"].numpy()
    predicted_trajectories = predict_groups(
        args.checkpoint, starts, actions, args.device, args.batch_size)
    predicted_curve = log_gap_curve(predicted_trajectories, state_scale)
    truth_curve = log_gap_curve(truth_trajectories, state_scale)

    training_residual = (truth_curve[~heldout] - predicted_curve[~heldout]).reshape(
        int((~heldout).sum()), -1)
    centroids, training_assignment, iterations = balanced_kmeans(
        training_residual, protocol["modes"], protocol["optimization"]["maximum_iterations"])
    training = evaluate_split(predicted_curve[~heldout], truth_curve[~heldout], centroids)
    # Report the actual capacity-balanced fit assignment rather than nearest-centroid reassignment.
    training["balanced_fit_occupancy"] = (
        np.bincount(training_assignment, minlength=len(centroids)) / len(training_assignment)).tolist()
    heldout_result = evaluate_split(predicted_curve[heldout], truth_curve[heldout], centroids)
    gate = protocol["feasibility_gate"]
    gate_result = {
        "global_scale_loss_reduction": (
            heldout_result["loss_reduction"]["global_scale_huber"]
            >= gate["global_scale_loss_reduction_min"]),
        "local_scale_loss_reduction": (
            heldout_result["loss_reduction"]["local_scale_huber"]
            >= gate["local_scale_loss_reduction_min"]),
        "heldout_mode_occupancy": (
            min(heldout_result["mode_occupancy"]) >= gate["each_heldout_mode_occupancy_min"]),
    }
    output = {
        "protocol": protocol, "protocol_sha256": sha256(args.protocol),
        "checkpoint_sha256": sha256(args.checkpoint),
        "split_audit": {
            "fit_groups": fit_groups,
            "heldout_groups": heldout_groups,
            "fit_episode_ids": sorted(fit_ids),
            "heldout_episode_ids": sorted(observed_heldout_ids),
            "episode_overlap": [],
        },
        "fit_iterations": iterations,
        "centroids": centroids.reshape(len(centroids), 5, 3).tolist(),
        "training": training, "heldout": heldout_result,
        "feasibility_gate": gate_result, "passes": bool(all(gate_result.values())),
        "decision": ("oracle scale regimes are reusable; proceed to a causal router test"
                     if all(gate_result.values()) else
                     "oracle scale regimes fail the TRAIN-only upper-bound gate"),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
