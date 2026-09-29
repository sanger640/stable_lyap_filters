"""Verify D12 response data and materialize its frozen two-axis split manifest."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d12_coverage_data import STRATA, prior_keys
from jenga_d6_set_response import sha256


DATA = ROOT / "results/jenga/d12_coverage_data"
PRIOR = ROOT / "results/jenga/d3_data"
PROTOCOL = ROOT / "results/jenga/d12_coverage_protocol.json"
SPLIT_PROTOCOL = ROOT / "results/jenga/d12_coverage_split_protocol.json"
REPORT = ROOT / "results/jenga/d12_coverage_integrity.json"
MANIFEST = ROOT / "results/jenga/d12_coverage_split_manifest.json"


def source_identity(stem):
    episode, seed = stem.removeprefix("ep").split("_seed")
    return int(episode), int(seed)


def fold_for(episode, seed, split_protocol):
    episode_reserved = episode in set(split_protocol["episode_axis"]["validation_episode_ids"])
    configuration_reserved = seed in set(
        split_protocol["configuration_axis"]["validation_reset_seeds"])
    if episode_reserved and configuration_reserved:
        return "joint_validation"
    if episode_reserved:
        return "episode_validation"
    if configuration_reserved:
        return "configuration_validation"
    return "fit"


def directory_sha256(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob("*.npz")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify(data_directory, prior_directory, expected_groups=576, expected_per_stratum=96):
    prior = prior_keys(prior_directory)
    prior_paths = {key[:3] for key in prior}
    records = []
    exact_keys = set()
    path_keys = set()
    strata = Counter()
    files = sorted(Path(data_directory).glob("ep*_seed*.npz"))
    errors = []
    required = {"start", "actions", "traces", "alphas", "midpoint_sides", "coarse_score",
                "contact", "state_index", "probe_index", "contact_step", "event_stratum"}
    for path in files:
        episode, seed = source_identity(path.stem)
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != required:
                errors.append(f"{path.name}: keys differ: {sorted(set(data.files) ^ required)}")
                continue
            groups = len(data["start"])
            expected_shapes = {
                "start": (groups, 61), "actions": (groups, 6, 2, 38, 4),
                "traces": (groups, 6, 2, 38, 61), "alphas": (groups, 6, 2),
                "midpoint_sides": (groups, 5), "coarse_score": (groups,),
                "contact": (groups,), "state_index": (groups,), "probe_index": (groups,),
                "contact_step": (groups,), "event_stratum": (groups,)
            }
            for name, shape in expected_shapes.items():
                if data[name].shape != shape:
                    errors.append(f"{path.name}:{name} has {data[name].shape}, expected {shape}")
                if np.issubdtype(data[name].dtype, np.number) and not np.all(np.isfinite(data[name])):
                    errors.append(f"{path.name}:{name} contains non-finite values")
            for index in range(groups):
                state = int(data["state_index"][index]); probe = int(data["probe_index"][index])
                step = int(data["contact_step"][index]); stratum = str(data["event_stratum"][index])
                exact = (path.stem, state, probe, step); path_key = exact[:3]
                if exact in prior:
                    errors.append(f"prior overlap: {exact}")
                if exact in exact_keys:
                    errors.append(f"duplicate exact key: {exact}")
                if path_key in path_keys:
                    errors.append(f"duplicate source-state-probe path: {path_key}")
                if stratum not in STRATA:
                    errors.append(f"unknown stratum: {stratum}")
                exact_keys.add(exact); path_keys.add(path_key); strata[stratum] += 1
                records.append({"file": path.name, "index": index, "episode": episode,
                                "reset_seed": seed, "state_index": state,
                                "probe_index": probe, "contact_step": step,
                                "event_stratum": stratum})
    if len(records) != expected_groups:
        errors.append(f"group count {len(records)} != {expected_groups}")
    expected_counts = {stratum: expected_per_stratum for stratum in STRATA}
    if dict(strata) != expected_counts:
        errors.append(f"stratum counts {dict(strata)} != {expected_counts}")
    return records, errors, {"files": len(files), "groups": len(records),
                             "strata": dict(strata), "prior_exact_overlap": 0 if not any(
                                 error.startswith("prior overlap") for error in errors) else None,
                             "prior_source_state_probe_overlap": len(path_keys & prior_paths),
                             "unique_source_state_probe_paths": len(path_keys)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA)); parser.add_argument("--prior", default=str(PRIOR))
    parser.add_argument("--protocol", default=str(PROTOCOL)); parser.add_argument(
        "--split-protocol", default=str(SPLIT_PROTOCOL))
    parser.add_argument("--report", default=str(REPORT)); parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    split = json.loads(Path(args.split_protocol).read_text())
    if split["source_protocol_sha256"] != sha256(args.protocol):
        raise SystemExit("split protocol does not reference the active D12 protocol")
    records, errors, summary = verify(
        args.data, args.prior, protocol["selection"]["total_groups"],
        protocol["selection"]["groups_per_stratum"])
    for record in records:
        record["fold"] = fold_for(record["episode"], record["reset_seed"], split)
    records.sort(key=lambda row: (row["file"], row["index"]))
    fold_counts = Counter(row["fold"] for row in records)
    fold_strata = {fold: dict(Counter(row["event_stratum"] for row in records
                                      if row["fold"] == fold))
                   for fold in split["folds"]}
    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "passed": not errors, "errors": errors,
              "protocol_sha256": sha256(args.protocol),
              "split_protocol_sha256": sha256(args.split_protocol),
              "data_sha256": directory_sha256(args.data), **summary,
              "fold_counts": dict(fold_counts), "fold_strata": fold_strata,
              "task_failure_or_alarm_labels_used": False}
    if errors:
        print(json.dumps(report, indent=2)); raise SystemExit(1)
    Path(args.manifest).write_text(json.dumps({"split_protocol_sha256": sha256(args.split_protocol),
                                               "records": records}, indent=2) + "\n")
    report["manifest_sha256"] = sha256(args.manifest)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
