"""Apply the frozen D5 one-seed TRAIN pilot gate."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gates(deterministic, multimodal):
    occupancy = multimodal["mode_occupancy"]
    return {
        "noncollapse": min(occupancy) >= .05,
        "lower_topology": multimodal["topology"] < deterministic["topology"],
        "lower_commitment": multimodal["commitment"] < deterministic["commitment"],
        "candidate_recall": (multimodal["candidate_recall"]
                             >= deterministic["candidate_recall"]),
        "quiet": multimodal["quiet_false_rate"] <= deterministic["quiet_false_rate"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(
        ROOT / "results/jenga/d5_multimodal_protocol.json"))
    parser.add_argument("--d4", default=str(ROOT / "results/jenga/d4_replication_s1.json"))
    parser.add_argument("--d5", default=str(ROOT / "results/jenga/d5_multimodal_s1_train.json"))
    parser.add_argument("--output", default=str(
        ROOT / "results/jenga/d5_multimodal_pilot_summary.json"))
    args = parser.parse_args()
    d4_result = json.loads(Path(args.d4).read_text())
    d5_result = json.loads(Path(args.d5).read_text())
    deterministic = d4_result["arms"]["full_d4"]["validation"]
    multimodal = d5_result["validation"]
    decisions = gates(deterministic, multimodal)
    result = {"protocol": json.loads(Path(args.protocol).read_text()),
              "protocol_sha256": sha256(args.protocol),
              "deterministic_d4_result_sha256": sha256(args.d4),
              "multimodal_train_result_sha256": sha256(args.d5),
              "deterministic_full_d4_seed1": deterministic,
              "multimodal_d5_seed1": multimodal,
              "relative_percent": {
                  name: 100 * (multimodal[name] / deterministic[name] - 1)
                  for name in ("response", "topology", "commitment")},
              "gates": decisions, "passed": all(decisions.values()),
              "conclusion": ("Replicate seeds 1-5 before a new DEV run."
                             if all(decisions.values()) else
                             "Stop privileged Jenga architecture search; do not replicate or "
                             "open DEV/TEST. Reassess the monitor/world-model interface.")}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"relative_percent": result["relative_percent"],
                      "gates": decisions, "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
