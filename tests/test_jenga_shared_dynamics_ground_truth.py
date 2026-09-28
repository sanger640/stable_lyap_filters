import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_shared_dynamics_ground_truth",
    ROOT / "eval/jenga_shared_dynamics_ground_truth.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_module_exposes_dev_evaluator():
    assert callable(MODULE.evaluate)
    assert callable(MODULE.prepare)
