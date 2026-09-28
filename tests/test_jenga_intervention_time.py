import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_intervention_time", ROOT / "eval/jenga_intervention_time.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_timed_outcome_uses_first_joint_lift_and_lateral_crossing():
    positions = np.zeros((4, 3, 3))
    positions[1, 0] = [.03, 0, .01]
    positions[2, 0] = [.03, 0, .03]
    positions[3, 0] = [.04, 0, .04]
    trace = {"positions": list(positions), "tilts": list(np.zeros((4, 3))),
             "original_indices": [9, 10, 11, 12]}
    result = MODULE.timed_outcome(trace, np.zeros((3, 3)))
    assert result["pick_success"]
    assert result["time_to_pick_from_episode_start_s"] == .30000000000000004
    assert result["time_to_pick_after_baseline_s"] == .2


def test_time_evaluator_cannot_select_partial_episode_count():
    source = (ROOT / "eval/jenga_intervention_time.py").read_text()
    assert "--episodes" not in source
    assert "--force" not in source
