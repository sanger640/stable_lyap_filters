"""Summarise the prospectively matched D3 pilot without touching frozen TEST."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


PHASES = (slice(0, 8), slice(8, 18), slice(18, 38))
PHASE_NAMES = ("action", "early_hold", "late_hold")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    return {str(q): float(v) for q, v in zip(
        (0, .1, .25, .5, .75, .9, 1), np.quantile(values, (0, .1, .25, .5, .75, .9, 1)))}


def dataset_audit(directory, checkpoint):
    scale = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_scale"][:45].numpy()
    rows = []
    files = sorted(Path(directory).glob("ep*_seed*.npz"))
    for path in files:
        data = np.load(path, allow_pickle=False)
        for traces, contact in zip(data["traces"], data["contact"]):
            effect = (traces[:, 1, :, :45] - traces[:, 0, :, :45]) / scale
            gaps = np.stack([
                np.sqrt(np.mean(effect[:, phase] ** 2, axis=(1, 2))) for phase in PHASES
            ], axis=1)
            rows.append([bool(contact), *(gaps[-1] / np.maximum(gaps[0], 1e-12))])
    rows = np.asarray(rows, float)
    result = {"groups": len(rows), "contact_groups": int(rows[:, 0].sum()),
              "quiet_groups": int((rows[:, 0] == 0).sum()), "ratios": {}}
    for name, mask in (("all", np.ones(len(rows), bool)), ("contact", rows[:, 0] > 0),
                       ("quiet", rows[:, 0] == 0)):
        result["ratios"][name] = {
            phase: {"quantiles": quantiles(rows[mask, index + 1]),
                    "smooth_below_0.1": float((rows[mask, index + 1] < .1).mean()),
                    "persistent_above_0.25": float((rows[mask, index + 1] > .25).mean())}
            for index, phase in enumerate(PHASE_NAMES)
        }
    maximum = rows[:, 1:].max(axis=1)
    result["all_phases_smooth_below_0.1"] = int((maximum < .1).sum())
    result["any_phase_persistent_above_0.25"] = int((maximum > .25).sum())
    result["late_hold_persistent_above_0.25"] = int((rows[:, 3] > .25).sum())
    result["meta_sha256"] = sha256(Path(directory) / "meta.json")
    digest = hashlib.sha256()
    for path in files:
        digest.update(bytes.fromhex(sha256(path)))
    result["ordered_npz_digest_sha256"] = digest.hexdigest()
    return result


def pair_ratios(result, class_name):
    values = []
    for row in result["rows"]:
        if row["class"] != class_name:
            continue
        for pair in row.get("pair_details", []):
            values.append([pair["early_gaps"][-1] / pair["early_gaps"][0],
                           pair["full_gaps"][-1] / pair["full_gaps"][0]])
    values = np.asarray(values, float)
    return {"pairs": len(values), "early_median": float(np.median(values[:, 0])),
            "full_median": float(np.median(values[:, 1]))}


def monitor_summary(path):
    result = json.loads(Path(path).read_text())
    rates = result["summary"]["dev"]["1.0"]
    return {"checkpoint": result["model"], "class_rates": rates,
            "stage_counts": result["candidate_summary"],
            "topple_pair_ratios": pair_ratios(result, "topple_fork"),
            "quiet_pair_ratios": pair_ratios(result, "quiet")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/jenga/d3_data")
    parser.add_argument("--scale-checkpoint", default="results/jenga/w6_cw_d2_s1.pt")
    parser.add_argument("--control", default=(
        "results/jenga/regime_monitor_v0/d3_pilot_dev/w6_cw_d2_continue_pilot_s1.json"))
    parser.add_argument("--d3", default=(
        "results/jenga/regime_monitor_v0/d3_pilot_dev/w6_cw_d3_pilot_s1.json"))
    parser.add_argument("--hard-probe", default=(
        "results/jenga/regime_monitor_v0/d2_hard_probe_dev/w6_cw_d2_hard_probe_s1.json"))
    parser.add_argument("--output", default="results/jenga/d3_pilot_summary.json")
    args = parser.parse_args()
    output = {
        "protocol": {
            "split": "DEV only; frozen TEST untouched",
            "comparison": "same D2 seed-1 initialization and three continuation epochs",
            "control": "D2 objective only",
            "treatment": "D2 plus nested-width D3 objective",
            "selection": "fixed three-epoch endpoint; no best-epoch selection",
        },
        "dataset": dataset_audit(args.data, args.scale_checkpoint),
        "control": monitor_summary(args.control),
        "d3": monitor_summary(args.d3),
        "hard_contact_diagnostic": monitor_summary(args.hard_probe),
        "decision": "D3 pilot fails DEV gate; do not expand seeds or run frozen TEST",
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
