import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_action_branch_ground_truth", ROOT / "eval/jenga_action_branch_ground_truth.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ground_truth_protocol_uses_same_recorded_times_as_d2():
    assert MODULE.RECORD_AT == (5, 10, 20, 29, 30)
