"""Build a leakage-controlled D3+D12 manifest from the frozen D12 source split."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d12_verify import fold_for, source_identity
from jenga_d6_set_response import sha256


SPLIT_PROTOCOL = ROOT / "results/jenga/d12_coverage_split_protocol.json"
INTEGRITY = ROOT / "results/jenga/d12_coverage_integrity.json"
OUTPUT = ROOT / "results/jenga/d12_combined_manifest.json"


def records_for(directory, dataset, split):
    records = []
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        episode, seed = source_identity(path.stem)
        with np.load(path, allow_pickle=False) as data:
            for index, (state, probe, step) in enumerate(zip(
                    data["state_index"], data["probe_index"], data["contact_step"])):
                record = {"dataset": dataset, "file": path.name, "index": index,
                          "episode": episode, "reset_seed": seed,
                          "state_index": int(state), "probe_index": int(probe),
                          "contact_step": int(step), "fold": fold_for(episode, seed, split)}
                if "event_stratum" in data.files:
                    record["event_stratum"] = str(data["event_stratum"][index])
                records.append(record)
    return records


def validate(records, split):
    errors = []
    exact = set()
    by_source = defaultdict(set)
    held_episodes = set(split["episode_axis"]["validation_episode_ids"])
    held_seeds = set(split["configuration_axis"]["validation_reset_seeds"])
    for row in records:
        key = (row["file"], row["state_index"], row["probe_index"], row["contact_step"])
        if key in exact:
            errors.append(f"duplicate exact group: {key}")
        exact.add(key)
        by_source[(row["episode"], row["reset_seed"])].add(row["fold"])
        if row["fold"] == "fit" and (row["episode"] in held_episodes or
                                      row["reset_seed"] in held_seeds):
            errors.append(f"reserved source in fit: {key}")
    for source, folds in by_source.items():
        if len(folds) != 1:
            errors.append(f"source split across folds: {source}: {sorted(folds)}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3", default=str(ROOT / "results/jenga/d3_data"))
    parser.add_argument("--d12", default=str(ROOT / "results/jenga/d12_coverage_data"))
    parser.add_argument("--split-protocol", default=str(SPLIT_PROTOCOL))
    parser.add_argument("--integrity", default=str(INTEGRITY))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    split = json.loads(Path(args.split_protocol).read_text())
    integrity = json.loads(Path(args.integrity).read_text())
    if not integrity["passed"]:
        raise SystemExit("D12 integrity report does not pass")
    if integrity["split_protocol_sha256"] != sha256(args.split_protocol):
        raise SystemExit("integrity report and split protocol differ")
    records = records_for(args.d3, "d3", split) + records_for(args.d12, "d12", split)
    records.sort(key=lambda row: (row["dataset"], row["file"], row["index"]))
    errors = validate(records, split)
    fold_counts = Counter(row["fold"] for row in records)
    dataset_fold_counts = {dataset: dict(Counter(row["fold"] for row in records
                                                  if row["dataset"] == dataset))
                           for dataset in ("d3", "d12")}
    output = {"split_protocol_sha256": sha256(args.split_protocol),
              "d12_integrity_sha256": sha256(args.integrity),
              "passed": not errors, "errors": errors, "groups": len(records),
              "fold_counts": dict(fold_counts), "dataset_fold_counts": dataset_fold_counts,
              "source_level_exclusion": True,
              "task_failure_or_alarm_labels_used": False, "records": records}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
