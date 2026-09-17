import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_expand_controls import report


def test_report_separates_silent_and_hard_negatives():
    rows = [{"episode_id": "0", "stratum": "silent_episode", "physical_boundary": False,
             "alarm": False, "physical_counts": {"toppled": 0, "intact": 50}},
            {"episode_id": "0", "stratum": "silent_episode", "physical_boundary": False,
             "alarm": True, "physical_counts": {"toppled": 0, "intact": 50}},
            {"episode_id": "7", "stratum": "moving_nominal_non_topple",
             "physical_boundary": True, "alarm": True,
             "physical_counts": {"toppled": 25, "intact": 25}}]
    result = report(rows, ["0"])
    assert result["groups"]["silent_episode"]["false_alarms"] == 1
    assert result["groups"]["moving_nominal_non_topple"]["mixed_outcome_chunks"] == 1
    assert result["silent_episode_alarm_counts"] == {"0": 1}
