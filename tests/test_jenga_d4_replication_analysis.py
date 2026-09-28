from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d4_replication_analysis import aggregate  # noqa: E402


def report(response, topology, commitment, recall):
    metrics = {"response": response, "topology": topology, "commitment": commitment,
               "candidate_recall": recall, "quiet_false_rate": .1,
               "alarm_agreement": .8, "partition_agreement": .8}
    return {"arms": {"frozen_d2": {"validation": {**metrics, "candidate_recall": .6}},
                     "matched_continuation": {"validation": metrics},
                     "full_d4": {"validation": {**metrics, "response": response * 1.1,
                                                  "topology": topology * .8,
                                                  "commitment": commitment * .9,
                                                  "candidate_recall": .6}}}}


def test_replication_gate_uses_medians_and_all_conditions():
    _, changes, gates = aggregate([report(1., 1., 1., .5) for _ in range(5)])
    assert all(gates.values())
    assert round(changes["response"], 6) == 10.0
    assert round(changes["topology"], 6) == -20.0
