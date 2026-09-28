"""TRAIN-only causal option-cost router with an explicit no-correction choice."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_causal_router import (  # noqa: E402
    causal_features, evaluate_assignment, load_alphas, smooth_false_positive_rate,
    standardize_fit,
)
from jenga_oracle_routing_upper_bound import (  # noqa: E402
    balanced_kmeans, classification, load_groups, log_gap_curve, losses, predict_groups,
    sha256, validation_ids,
)


def elementwise_huber(values, delta=1.0):
    absolute = np.abs(values)
    return np.where(absolute <= delta, .5 * values ** 2, delta * (absolute - .5 * delta))


def option_cost_components(predicted, truth, corrections):
    """Return per-group/per-option global and adjacent-scale physical curve costs."""
    options = predicted[:, None] + corrections[None]
    global_cost = elementwise_huber(options - truth[:, None]).mean(axis=(2, 3))
    option_full = np.concatenate([np.zeros_like(options[:, :, :1]), options], axis=2)
    truth_full = np.concatenate([np.zeros_like(truth[:, :1]), truth], axis=1)
    local_error = np.diff(option_full, axis=2) - np.diff(truth_full[:, None], axis=2)
    local_cost = elementwise_huber(local_error).mean(axis=(2, 3))
    return global_cost, local_cost


def normalized_option_cost(global_cost, local_cost, normalizers):
    return (global_cost / normalizers["global"] +
            local_cost / normalizers["local"])


class SelectiveCostRouter(nn.Module):
    def __init__(self, inputs, options=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 64), nn.GELU(), nn.Dropout(.1),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(.1),
            nn.Linear(32, options),
        )

    def forward(self, values):
        return self.network(values)


def train_router(features, log_costs, seed, epochs=400, batch_size=32):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    target = torch.as_tensor(log_costs, dtype=torch.float32)
    model = SelectiveCostRouter(x.shape[1], target.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        model.train()
        for indices in torch.randperm(len(x), generator=generator).split(batch_size):
            predicted = model(x[indices])
            loss = F.smooth_l1_loss(predicted, target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def smooth_weak_selection_rate(assignment, truth_curve):
    true_smooth = np.exp(truth_curve[:, -1]).max(1) < .1
    return float(np.isin(assignment[true_smooth], [0, 1]).mean()) if true_smooth.any() else None


def evaluate_options(predicted, truth, corrections, assignment, oracle_assignment):
    result = evaluate_assignment(predicted, truth, corrections, assignment, oracle_assignment)
    result["cost_oracle_option_accuracy"] = result.pop("oracle_assignment_accuracy")
    result["identity_or_weak_mode_on_smooth"] = smooth_weak_selection_rate(assignment, truth)
    return result


def summarize(results):
    paths = {
        "cost_oracle_option_accuracy": ("cost_oracle_option_accuracy",),
        "global_oracle_gain_fraction": ("oracle_gain_fraction", "global_scale_huber"),
        "local_oracle_gain_fraction": ("oracle_gain_fraction", "local_scale_huber"),
        "persistent_recall": ("classification", "persistent_recall"),
        "smooth_false_positive_rate": ("classification", "smooth_false_positive_rate"),
        "identity_or_weak_mode_on_smooth": ("identity_or_weak_mode_on_smooth",),
    }
    output = {}
    for name, path in paths.items():
        values = []
        for result in results:
            value = result
            for key in path:
                value = value[key]
            values.append(float(value))
        values = np.asarray(values)
        output[name] = {
            "median": float(np.median(values)), "mean": float(values.mean()),
            "minimum": float(values.min()), "maximum": float(values.max()),
        }
    occupancy = np.asarray([result["mode_occupancy"] for result in results])
    output["option_occupancy_median"] = np.median(occupancy, axis=0).tolist()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "results/jenga/d3_data"))
    parser.add_argument("--trace", default=str(ROOT / "results/jenga/trace_data"))
    parser.add_argument("--checkpoint", default=str(ROOT / "results/jenga/w6_cw_d2_s1.pt"))
    parser.add_argument("--protocol", default=str(
        ROOT / "results/jenga/selective_cost_router_protocol.json"))
    parser.add_argument("--output", default=str(
        ROOT / "results/jenga/selective_cost_router_summary.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    starts, actions, truth_trajectories, sources = load_groups(args.data)
    alphas = load_alphas(args.data)
    heldout_ids = validation_ids(args.trace)
    heldout = np.asarray([source in heldout_ids for source in sources])
    fit = ~heldout
    if (int(fit.sum()) != protocol["fit_groups_expected"] or
            int(heldout.sum()) != protocol["heldout_groups_expected"]):
        raise ValueError("dataset does not match prospectively fixed group counts")
    if set(sources[fit]) & set(sources[heldout]):
        raise ValueError("episode leakage between fit and held-out TRAIN groups")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_scale = checkpoint["state_scale"].numpy()
    predicted_trajectories = predict_groups(
        args.checkpoint, starts, actions, args.device, args.batch_size)
    predicted_curve = log_gap_curve(predicted_trajectories, state_scale)
    truth_curve = log_gap_curve(truth_trajectories, state_scale)
    residual = (truth_curve - predicted_curve).reshape(len(starts), -1)
    centroids, _, iterations = balanced_kmeans(residual[fit], modes=3)
    corrections = np.concatenate([np.zeros((1, 15)), centroids], axis=0).reshape(4, 5, 3)

    global_cost, local_cost = option_cost_components(predicted_curve, truth_curve, corrections)
    normalizers = {
        "global": float(global_cost[fit, 0].mean()),
        "local": float(local_cost[fit, 0].mean()),
    }
    option_cost = normalized_option_cost(global_cost, local_cost, normalizers)
    log_cost = np.log(np.maximum(option_cost, 1e-6))
    oracle_assignment = option_cost[heldout].argmin(1)

    features = causal_features(starts, actions, alphas)
    fit_x, heldout_x, feature_mean, feature_scale = standardize_fit(
        features[fit], features[heldout])
    seed_results = []
    for seed in protocol["router"]["seeds"]:
        model = train_router(
            fit_x, log_cost[fit], seed, protocol["router"]["epochs"])
        model.eval()
        with torch.no_grad():
            predicted_cost = model(torch.as_tensor(heldout_x, dtype=torch.float32)).numpy()
        assignment = predicted_cost.argmin(1)
        result = evaluate_options(
            predicted_curve[heldout], truth_curve[heldout], corrections,
            assignment, oracle_assignment)
        result["seed"] = seed
        seed_results.append(result)

    constant_option = int(option_cost[fit].mean(0).argmin())
    constant_assignment = np.full(int(heldout.sum()), constant_option)
    constant = evaluate_options(
        predicted_curve[heldout], truth_curve[heldout], corrections,
        constant_assignment, oracle_assignment)
    oracle = evaluate_options(
        predicted_curve[heldout], truth_curve[heldout], corrections,
        oracle_assignment, oracle_assignment)
    baseline_classification = classification(predicted_curve[heldout], truth_curve[heldout])
    baseline_classification["smooth_false_positive_rate"] = smooth_false_positive_rate(
        predicted_curve[heldout], truth_curve[heldout])
    summary = summarize(seed_results)
    gate = protocol["feasibility_gate"]
    gate_result = {
        "cost_oracle_option_accuracy": summary["cost_oracle_option_accuracy"]["median"] >=
        gate["cost_oracle_option_accuracy_min"],
        "global_oracle_gain_fraction": summary["global_oracle_gain_fraction"]["median"] >=
        gate["global_oracle_gain_fraction_min"],
        "local_oracle_gain_fraction": summary["local_oracle_gain_fraction"]["median"] >=
        gate["local_oracle_gain_fraction_min"],
        "persistent_recall": summary["persistent_recall"]["median"] >=
        gate["persistent_recall_min"],
        "smooth_false_positive_rate": summary["smooth_false_positive_rate"]["median"] <=
        gate["smooth_false_positive_rate_max"],
        "identity_or_weak_mode_on_smooth": summary[
            "identity_or_weak_mode_on_smooth"]["median"] >=
        gate["identity_or_weak_mode_on_smooth_min"],
    }
    output = {
        "protocol": protocol, "protocol_sha256": sha256(args.protocol),
        "checkpoint_sha256": sha256(args.checkpoint),
        "split_audit": {
            "fit_groups": int(fit.sum()), "heldout_groups": int(heldout.sum()),
            "fit_episode_ids": sorted(set(sources[fit])),
            "heldout_episode_ids": sorted(set(sources[heldout])), "episode_overlap": [],
        },
        "feature_audit": {
            "dimensions": int(features.shape[1]), "fit_standardization_only": True,
            "deeper_nested_actions_read": False,
        },
        "prototype_fit_iterations": iterations,
        "corrections": corrections.tolist(), "cost_normalizers": normalizers,
        "fit_option_occupancy_oracle": (
            np.bincount(option_cost[fit].argmin(1), minlength=4) / int(fit.sum())).tolist(),
        "heldout_option_occupancy_oracle": (
            np.bincount(oracle_assignment, minlength=4) / int(heldout.sum())).tolist(),
        "baseline": {
            "loss": losses(predicted_curve[heldout], truth_curve[heldout]),
            "classification": baseline_classification,
        },
        "constant_router": {"fit_selected_option": constant_option, **constant},
        "future_informed_cost_oracle": oracle,
        "selective_router_seeds": seed_results,
        "selective_router_summary": summary,
        "feasibility_gate": gate_result,
        "passes": bool(all(gate_result.values())),
        "decision": ("selective cost router passes TRAIN-only gate; preregister DEV"
                     if all(gate_result.values()) else
                     "selective cost router fails TRAIN-only gate; do not run DEV or TEST"),
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "selective_router_summary": summary, "feasibility_gate": gate_result,
        "passes": output["passes"], "decision": output["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
