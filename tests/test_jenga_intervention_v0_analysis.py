import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_intervention_v0_analysis", ROOT / "eval/jenga_intervention_v0_analysis.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_comparison_counts_direction():
    rows = [
        {"arms": {"no_monitor": {"neighbor_failure": True},
                  "wrapper": {"neighbor_failure": False}}},
        {"arms": {"no_monitor": {"neighbor_failure": False},
                  "wrapper": {"neighbor_failure": False}}},
    ]
    indices = np.array([[0, 1], [0, 0]])
    result = MODULE.paired_comparison(
        rows, "no_monitor", "wrapper", "neighbor_failure", indices)
    assert result["arm_minus_reference"] == -.5
    assert result["reference_only"] == 1
    assert result["arm_only"] == 0
