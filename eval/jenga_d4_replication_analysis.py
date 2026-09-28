"""Aggregate the prospectively fixed five-seed D4 TRAIN replication."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("response", "topology", "commitment", "candidate_recall",
           "quiet_false_rate", "alarm_agreement", "partition_agreement")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(reports):
    arms = {}
    for arm in ("frozen_d2", "matched_continuation", "full_d4"):
        arms[arm] = {}
        for metric in METRICS:
            values = [report["arms"][arm]["validation"][metric] for report in reports]
            arms[arm][metric] = {"values": values, "median": float(np.median(values)),
                                 "min": float(np.min(values)), "max": float(np.max(values))}
    matched, full, frozen = arms["matched_continuation"], arms["full_d4"], arms["frozen_d2"]
    changes = {metric: 100 * (full[metric]["median"] / matched[metric]["median"] - 1)
               for metric in ("response", "topology", "commitment")}
    gates = {
        "lower_topology": full["topology"]["median"] < matched["topology"]["median"],
        "lower_commitment": full["commitment"]["median"] < matched["commitment"]["median"],
        "retains_frozen_candidate_recall": (full["candidate_recall"]["median"]
                                             >= frozen["candidate_recall"]["median"]),
        "response_within_25_percent": changes["response"] <= 25.0,
    }
    return arms, changes, gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(
        ROOT / "results/jenga/d4_replication_protocol.json"))
    parser.add_argument("--input-pattern", default=str(
        ROOT / "results/jenga/d4_replication_s{seed}.json"))
    parser.add_argument("--output", default=str(
        ROOT / "results/jenga/d4_replication_summary.json"))
    args = parser.parse_args()
    paths = [Path(args.input_pattern.format(seed=seed)) for seed in range(1, 6)]
    reports = [json.loads(path.read_text()) for path in paths]
    arms, changes, gates = aggregate(reports)
    result = {
        "protocol": json.loads(Path(args.protocol).read_text()),
        "protocol_sha256": sha256(args.protocol),
        "report_sha256": {path.name: sha256(path) for path in paths},
        "validation": {"states": 36, "episodes": 6,
                       "physical_candidates": 24, "physical_quiet": 12},
        "arms": arms, "full_d4_vs_matched_median_percent": changes,
        "gates": gates, "passed": all(gates.values()),
        "conclusion": ("Advance to a frozen matched DEV comparison; TEST remains closed."
                       if all(gates.values()) else
                       "Do not open DEV; test one multimodal trajectory architecture on TRAIN."),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"changes": changes, "gates": gates,
                      "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
