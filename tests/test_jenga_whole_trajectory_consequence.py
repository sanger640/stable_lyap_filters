import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_whole_trajectory_consequence",
    ROOT / "eval/jenga_whole_trajectory_consequence.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_module_exposes_dev_evaluator():
    assert callable(MODULE.prepare)
    assert callable(MODULE.evaluate)
