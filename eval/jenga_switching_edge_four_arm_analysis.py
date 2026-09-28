"""Analyse the prospectively frozen one-seed switching-edge DEV experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from jenga_bench import BENCH_FILE, EVAL_CONFIG  # noqa: E402
from jenga_w5_eval import load_model  # noqa: E402
from state_dynamics import rollout  # noqa: E402


ARMS = {
    "soft_d2": "w6_cw_d2_continue_pilot_s1",
    "hard_d2": "w6_four_arm_hard_d2_s1",
    "switch_d2": "w6_four_arm_switch_d2_s1",
    "switch_d3": "w6_four_arm_switch_d3_s1",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def median_pair_ratios(rows, class_name):
    values = []
    for row in rows:
        if row["class"] != class_name:
            continue
        for pair in row.get("pair_details", []):
            values.append([pair["early_gaps"][-1] / pair["early_gaps"][0],
                           pair["full_gaps"][-1] / pair["full_gaps"][0]])
    values = np.asarray(values, float)
    if not len(values):
        return {"pairs": 0, "early_median": None, "full_median": None}
    return {"pairs": len(values), "early_median": float(np.median(values[:, 0])),
            "full_median": float(np.median(values[:, 1]))}


def monitor_result(path):
    result = json.loads(Path(path).read_text())
    return {
        "checkpoint": result["model"],
        "class_rates": result["summary"]["dev"]["1.0"],
        "stage_counts": result["candidate_summary"],
        "topple_pair_ratios": median_pair_ratios(result["rows"], "topple_fork"),
        "quiet_pair_ratios": median_pair_ratios(result["rows"], "quiet"),
    }


def mode_summary(probabilities):
    probabilities = np.concatenate(probabilities, axis=0)
    usage = probabilities.mean(axis=(0, 1, 2))
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(-1)
    entropy /= np.log(probabilities.shape[-1])
    choices = probabilities.argmax(-1)
    switch_rate = (choices[:, 1:] != choices[:, :-1]).mean()
    return {"soft_occupancy": usage.tolist(),
            "hard_occupancy": (np.bincount(choices.reshape(-1),
                                           minlength=probabilities.shape[-1])
                               / choices.size).tolist(),
            "normalized_entropy_mean": float(entropy.mean()),
            "argmax_switch_rate": float(switch_rate)}


def audit_modes(checkpoint, device):
    model, scale = load_model(checkpoint, device)
    if not hasattr(model, "modes"):
        return None
    benchmark = np.load(BENCH_FILE, allow_pickle=False)
    scale_index = EVAL_CONFIG["scales"].index(1.0)
    by_class = {}
    with torch.no_grad():
        for index, start in enumerate(benchmark["dev_start"]):
            windows = benchmark["dev_windows"][index, scale_index]
            state = torch.as_tensor(np.repeat(start[None], len(windows), axis=0),
                                    dtype=torch.float32, device=device)
            actions = torch.as_tensor(windows[:, 2:], dtype=torch.float32, device=device)
            gates = []
            rollout(model, state, actions, scale, collect=gates)
            probabilities = torch.stack(gates, dim=1).cpu().numpy()
            class_name = str(benchmark["dev_classes"][index, scale_index])
            by_class.setdefault(class_name, []).append(probabilities)
    all_probabilities = [value for values in by_class.values() for value in values]
    return {"all": mode_summary(all_probabilities),
            "by_class": {name: mode_summary(values) for name, values in by_class.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(
        ROOT / "results/jenga/regime_monitor_v0/switching_edge_four_arm_dev"))
    parser.add_argument("--protocol", default=str(
        ROOT / "results/jenga/switching_edge_four_arm_protocol.json"))
    parser.add_argument("--output", default=str(
        ROOT / "results/jenga/switching_edge_four_arm_summary.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    directory = Path(args.results)
    arms = {}
    for name, stem in ARMS.items():
        result_path = directory / f"{stem}.json"
        arm = monitor_result(result_path)
        arm["mode_diagnostics"] = audit_modes(arm["checkpoint"]["path"], args.device)
        arms[name] = arm
    gate = {name: (arm["class_rates"]["topple_fork"]["alarms"] >= 10
                   and arm["class_rates"]["quiet"]["alarms"] <= 3)
            for name, arm in arms.items() if name.startswith("switch")}
    output = {
        "protocol": json.loads(Path(args.protocol).read_text()),
        "protocol_sha256": sha256(args.protocol),
        "arms": arms,
        "feasibility_gate_pass": gate,
        "decision": "switching arms fail one-seed DEV feasibility gate; do not expand or run TEST",
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
