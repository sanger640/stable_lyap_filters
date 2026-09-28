from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d5_multimodal_analysis import gates  # noqa: E402


def test_every_pilot_gate_is_required():
    deterministic = {"topology": .05, "commitment": .05, "candidate_recall": .6,
                     "quiet_false_rate": .2}
    multimodal = {"topology": .04, "commitment": .04, "candidate_recall": .6,
                  "quiet_false_rate": .2, "mode_occupancy": [.2, .3, .5]}
    assert all(gates(deterministic, multimodal).values())
    multimodal["commitment"] = .06
    assert not gates(deterministic, multimodal)["lower_commitment"]
    multimodal["commitment"] = .04; multimodal["mode_occupancy"] = [.01, .49, .5]
    assert not gates(deterministic, multimodal)["noncollapse"]
