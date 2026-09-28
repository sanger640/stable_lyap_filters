"""Apply the frozen D4 DEV gates to per-checkpoint Regime Monitor v0 decisions."""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_key(row):
    return str(row["episode_id"]), int(row["chunk_start"])


def summarize(paths, physical):
    rows = []
    for path in paths:
        result = json.loads(Path(path).read_text())
        decisions = result["rows"]
        topples = [row for row in decisions if row["class"] == "topple_fork"]
        quiet = [row for row in decisions if row["class"] == "quiet"]
        agreement = np.mean([bool(row["alarm"]) == physical[state_key(row)]
                             for row in decisions])
        rows.append({"checkpoint_result": str(Path(path).relative_to(ROOT)),
                     "result_sha256": sha256(path),
                     "topple_alarms": int(sum(bool(row["alarm"]) for row in topples)),
                     "topple_states": len(topples),
                     "quiet_alarms": int(sum(bool(row["alarm"]) for row in quiet)),
                     "quiet_states": len(quiet),
                     "physical_v0_agreement": float(agreement)})
    return {"per_seed": rows,
            "median_topple_alarms": float(np.median([row["topple_alarms"] for row in rows])),
            "median_quiet_alarms": float(np.median([row["quiet_alarms"] for row in rows])),
            "median_physical_v0_agreement": float(np.median(
                [row["physical_v0_agreement"] for row in rows]))}


def gates(matched, full):
    return {
        "recall": full["median_topple_alarms"] > matched["median_topple_alarms"],
        "specificity": (full["median_quiet_alarms"]
                        <= matched["median_quiet_alarms"] + 2),
        "physical_agreement": (full["median_physical_v0_agreement"]
                               >= matched["median_physical_v0_agreement"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(ROOT / "results/jenga/d4_dev_protocol.json"))
    parser.add_argument("--physical", default=str(
        ROOT / "results/jenga/whole_trajectory_consequence_dev.json"))
    parser.add_argument("--matched", default=str(
        ROOT / "results/jenga/d4_dev/matched/*_s?.json"))
    parser.add_argument("--full", default=str(
        ROOT / "results/jenga/d4_dev/full_d4/*_s?.json"))
    parser.add_argument("--original", default=str(
        ROOT / "results/jenga/d4_dev/original_d2/w6_cw_d2_s1.json"))
    parser.add_argument("--output", default=str(ROOT / "results/jenga/d4_dev_summary.json"))
    args = parser.parse_args()
    physical_rows = json.loads(Path(args.physical).read_text())["rows"]
    physical = {state_key(row): bool(row["alarm"]) for row in physical_rows}
    matched_paths, full_paths = sorted(glob.glob(args.matched)), sorted(glob.glob(args.full))
    if len(matched_paths) != 5 or len(full_paths) != 5:
        raise SystemExit("expected exactly five matched and five full-D4 results")
    matched, full = summarize(matched_paths, physical), summarize(full_paths, physical)
    original = summarize([args.original], physical)
    decisions = gates(matched, full)
    result = {"protocol": json.loads(Path(args.protocol).read_text()),
              "protocol_sha256": sha256(args.protocol),
              "physical_reference_sha256": sha256(args.physical),
              "original_d2": original, "matched_continuation": matched, "full_d4": full,
              "gates": decisions, "passed": all(decisions.values()),
              "conclusion": ("D4 passes DEV; proceed to the separately governed next stage."
                             if all(decisions.values()) else
                             "D4 fails matched DEV. Keep TEST closed and test one multimodal "
                             "trajectory architecture under the same universal supervision.")}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"matched": matched, "full_d4": full,
                      "gates": decisions, "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
