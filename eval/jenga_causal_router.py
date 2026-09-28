"""TRAIN-only causal router for the three oracle scale-correction regimes.

The router never sees a physical or predicted future.  It receives the current privileged state,
the intended action and one sampled execution-error direction.  The task is to identify which of
three task-label-free correction prototypes would best repair frozen D2's nested scale curve.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_oracle_routing_upper_bound import (  # noqa: E402
    balanced_kmeans, classification, load_groups, log_gap_curve, losses, predict_groups,
    sha256, validation_ids,
)


def load_alphas(directory):
    values = []
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        data = np.load(path, allow_pickle=False)
        if len(data["start"]):
            values.append(data["alphas"])
    return np.concatenate(values)


def causal_features(starts, actions, alphas):
    """Recover only pre-future state, nominal action, and uncertainty direction.

    Only level zero is read.  Deeper endpoint locations encode physical bisection decisions and are
    intentionally prohibited.  For A(alpha)=nominal+alpha*residual, the initial endpoint actions
    and their scalar alphas recover nominal and residual exactly.
    """
    left = actions[:, 0, 0, :8]
    right = actions[:, 0, 1, :8]
    left_alpha = alphas[:, 0, 0, None, None]
    right_alpha = alphas[:, 0, 1, None, None]
    width = right_alpha - left_alpha
    if np.any(np.abs(width) < 1e-8):
        raise ValueError("initial alpha endpoints must be distinct")
    residual_xyz = (right[..., :3] - left[..., :3]) / width
    nominal = left.copy()
    nominal[..., :3] = left[..., :3] - left_alpha * residual_xyz
    return np.concatenate([
        np.asarray(starts, np.float32),
        nominal.reshape(len(starts), -1),
        residual_xyz.reshape(len(starts), -1),
    ], axis=1)


def standardize_fit(fit, heldout):
    mean = fit.mean(0)
    scale = fit.std(0)
    scale[scale < 1e-6] = 1.0
    return (fit - mean) / scale, (heldout - mean) / scale, mean, scale


class CausalRouter(nn.Module):
    def __init__(self, inputs, modes=3, residual_dimensions=15):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(inputs, 64), nn.GELU(), nn.Dropout(.1),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(.1),
        )
        self.mode = nn.Linear(32, modes)
        self.residual = nn.Linear(32, residual_dimensions)

    def forward(self, values):
        hidden = self.body(values)
        return self.mode(hidden), self.residual(hidden)


def train_router(features, assignments, residuals, seed, epochs=400, batch_size=32):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(assignments, dtype=torch.long)
    r = torch.as_tensor(residuals, dtype=torch.float32)
    model = CausalRouter(x.shape[1], int(y.max()) + 1, r.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        model.train()
        for indices in torch.randperm(len(x), generator=generator).split(batch_size):
            logits, predicted_residual = model(x[indices])
            loss = F.cross_entropy(logits, y[indices]) + .5 * F.smooth_l1_loss(
                predicted_residual, r[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def corrected_metrics(predicted_curve, truth_curve, centroids, assignment):
    corrected = predicted_curve + centroids[assignment].reshape(-1, 5, 3)
    baseline_loss = losses(predicted_curve, truth_curve)
    corrected_loss = losses(corrected, truth_curve)
    details = classification(corrected, truth_curve)
    details["smooth_false_positive_rate"] = smooth_false_positive_rate(corrected, truth_curve)
    return corrected_loss, details


def smooth_false_positive_rate(predicted, truth):
    """Fraction of truly smooth groups called persistent, leaving the middle band unlabelled."""
    predicted_ratio = np.exp(predicted[:, -1]).max(1)
    truth_ratio = np.exp(truth[:, -1]).max(1)
    true_smooth = truth_ratio < .1
    return float((predicted_ratio[true_smooth] > .25).mean()) if true_smooth.any() else None


def evaluate_assignment(predicted_curve, truth_curve, centroids, assignment, oracle_assignment):
    corrected_loss, details = corrected_metrics(
        predicted_curve, truth_curve, centroids, assignment)
    baseline_loss = losses(predicted_curve, truth_curve)
    oracle_loss, _ = corrected_metrics(
        predicted_curve, truth_curve, centroids, oracle_assignment)
    gain_fraction = {
        name: ((baseline_loss[name] - corrected_loss[name]) /
               max(baseline_loss[name] - oracle_loss[name], 1e-12))
        for name in baseline_loss
    }
    return {
        "oracle_assignment_accuracy": float(np.mean(assignment == oracle_assignment)),
        "mode_occupancy": (np.bincount(assignment, minlength=len(centroids)) /
                           len(assignment)).tolist(),
        "loss": corrected_loss,
        "oracle_gain_fraction": gain_fraction,
        "classification": details,
        "assignment": assignment.tolist(),
    }


def summarize_seed_results(results):
    def values(path):
        output = []
        for result in results:
            value = result
            for key in path:
                value = value[key]
            output.append(float(value))
        return np.asarray(output)

    paths = {
        "oracle_assignment_accuracy": ("oracle_assignment_accuracy",),
        "global_oracle_gain_fraction": ("oracle_gain_fraction", "global_scale_huber"),
        "local_oracle_gain_fraction": ("oracle_gain_fraction", "local_scale_huber"),
        "persistent_recall": ("classification", "persistent_recall"),
        "smooth_false_positive_rate": ("classification", "smooth_false_positive_rate"),
    }
    summary = {}
    for name, path in paths.items():
        metric = values(path)
        summary[name] = {
            "median": float(np.median(metric)), "mean": float(metric.mean()),
            "minimum": float(metric.min()), "maximum": float(metric.max()),
        }
    occupancy = np.asarray([result["mode_occupancy"] for result in results])
    summary["mode_occupancy_median"] = np.median(occupancy, axis=0).tolist()
    summary["minimum_mode_occupancy_across_seeds"] = float(occupancy.min())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "results/jenga/d3_data"))
    parser.add_argument("--trace", default=str(ROOT / "results/jenga/trace_data"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results/jenga/w6_cw_d2_s1.pt"))
    parser.add_argument("--protocol", default=str(
        ROOT / "results/jenga/causal_router_protocol.json"))
    parser.add_argument("--output", default=str(
        ROOT / "results/jenga/causal_router_summary.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="D2 rollout batch size; router training remains fixed by protocol")
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    starts, actions, truth_trajectories, sources = load_groups(args.data)
    alphas = load_alphas(args.data)
    heldout_ids = validation_ids(args.trace)
    heldout = np.asarray([source in heldout_ids for source in sources])
    fit = ~heldout
    if int(fit.sum()) != protocol["fit_groups_expected"] or int(heldout.sum()) != protocol[
            "heldout_groups_expected"]:
        raise ValueError("dataset does not match prospectively fixed group counts")
    if set(sources[fit]) & set(sources[heldout]):
        raise ValueError("episode leakage between fit and held-out TRAIN groups")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_scale = checkpoint["state_scale"].numpy()
    predicted_trajectories = predict_groups(
        args.checkpoint, starts, actions, args.device, args.batch_size)
    predicted_curve = log_gap_curve(predicted_trajectories, state_scale)
    truth_curve = log_gap_curve(truth_trajectories, state_scale)
    all_residual = (truth_curve - predicted_curve).reshape(len(starts), -1)
    centroids, fit_assignment, iterations = balanced_kmeans(
        all_residual[fit], protocol["modes"])
    oracle_assignment = np.argmin(
        ((all_residual[heldout, None] - centroids[None]) ** 2).sum(-1), axis=1)

    features = causal_features(starts, actions, alphas)
    fit_x, heldout_x, feature_mean, feature_scale = standardize_fit(
        features[fit], features[heldout])
    residual_mean = all_residual[fit].mean(0)
    residual_scale = all_residual[fit].std(0)
    residual_scale[residual_scale < 1e-6] = 1.0
    fit_residual = (all_residual[fit] - residual_mean) / residual_scale

    seed_results = []
    router = protocol["router"]
    for seed in router["seeds"]:
        model = train_router(
            fit_x, fit_assignment, fit_residual, seed, router["epochs"])
        model.eval()
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(heldout_x, dtype=torch.float32))
        assignment = logits.argmax(1).numpy()
        result = evaluate_assignment(
            predicted_curve[heldout], truth_curve[heldout], centroids,
            assignment, oracle_assignment)
        result["seed"] = seed
        seed_results.append(result)

    majority_mode = int(np.bincount(fit_assignment, minlength=len(centroids)).argmax())
    majority_assignment = np.full(int(heldout.sum()), majority_mode)
    majority = evaluate_assignment(
        predicted_curve[heldout], truth_curve[heldout], centroids,
        majority_assignment, oracle_assignment)
    oracle = evaluate_assignment(
        predicted_curve[heldout], truth_curve[heldout], centroids,
        oracle_assignment, oracle_assignment)
    baseline_loss = losses(predicted_curve[heldout], truth_curve[heldout])
    baseline_classification = classification(predicted_curve[heldout], truth_curve[heldout])
    baseline_classification["smooth_false_positive_rate"] = smooth_false_positive_rate(
        predicted_curve[heldout], truth_curve[heldout])

    summary = summarize_seed_results(seed_results)
    gate = protocol["feasibility_gate"]
    gate_result = {
        "oracle_assignment_accuracy": summary["oracle_assignment_accuracy"]["median"] >=
        gate["oracle_assignment_accuracy_min"],
        "global_oracle_gain_fraction": summary["global_oracle_gain_fraction"]["median"] >=
        gate["global_oracle_gain_fraction_min"],
        "local_oracle_gain_fraction": summary["local_oracle_gain_fraction"]["median"] >=
        gate["local_oracle_gain_fraction_min"],
        "persistent_recall": summary["persistent_recall"]["median"] >=
        gate["persistent_recall_min"],
        "smooth_false_positive_rate": summary["smooth_false_positive_rate"]["median"] <=
        gate["smooth_false_positive_rate_max"],
        "each_mode_occupancy": summary["minimum_mode_occupancy_across_seeds"] >=
        gate["each_mode_occupancy_min"],
    }
    output = {
        "protocol": protocol,
        "protocol_sha256": sha256(args.protocol),
        "checkpoint_sha256": sha256(args.checkpoint),
        "split_audit": {
            "fit_groups": int(fit.sum()), "heldout_groups": int(heldout.sum()),
            "fit_episode_ids": sorted(set(sources[fit])),
            "heldout_episode_ids": sorted(set(sources[heldout])), "episode_overlap": [],
        },
        "feature_audit": {
            "dimensions": int(features.shape[1]),
            "fit_standardization_only": True,
            "deeper_nested_actions_read": False,
            "feature_mean_sha256": hashlib.sha256(feature_mean.tobytes()).hexdigest(),
            "feature_scale_sha256": hashlib.sha256(feature_scale.tobytes()).hexdigest(),
        },
        "prototype_fit_iterations": iterations,
        "prototype_centroids": centroids.reshape(len(centroids), 5, 3).tolist(),
        "fit_mode_occupancy": (np.bincount(fit_assignment, minlength=len(centroids)) /
                               len(fit_assignment)).tolist(),
        "baseline": {"loss": baseline_loss, "classification": baseline_classification},
        "majority_router": majority,
        "future_informed_oracle": oracle,
        "causal_router_seeds": seed_results,
        "causal_router_summary": summary,
        "feasibility_gate": gate_result,
        "passes": bool(all(gate_result.values())),
        "decision": ("causal router passes TRAIN-only gate; preregister a new DEV experiment"
                     if all(gate_result.values()) else
                     "causal router fails TRAIN-only gate; do not run DEV or TEST"),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "causal_router_summary": summary, "feasibility_gate": gate_result,
        "passes": output["passes"], "decision": output["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
