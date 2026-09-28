from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT / "src"))

from ground_truth_monitor_alarm_videos import consequential_pair


def test_consequential_pair_requires_all_four_positive_evidence_terms():
    row = {"boundary_evidence": {"early_delta_bic": [1, 1, -1],
                                 "full_delta_bic": [1, 1, 1]},
           "commitment_delta_bic": [-1, 2, 2], "persistence_delta_bic": [2, 3, 3]}
    assert consequential_pair(row) == 1
