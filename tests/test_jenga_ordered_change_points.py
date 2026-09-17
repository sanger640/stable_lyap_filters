import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_ordered_change_points import (confusion, physical_boundary,
                                         physical_split_scalars)


def test_physical_boundary_requires_two_on_each_side():
    assert physical_boundary([False, False, True, True])
    assert not physical_boundary([False, False, False, True])


def test_physical_split_scalars_finds_each_switch():
    splits = physical_split_scalars([-2, -1, 0, 1], [False, False, True, False])
    assert splits == [-0.5, 0.5]


def test_confusion_metrics():
    result = confusion([True, True, False, False], [True, False, True, False])
    assert result["tp"] == result["fp"] == result["fn"] == result["tn"] == 1
    assert result["precision"] == result["recall"] == 0.5
    assert result["false_positive_rate"] == 0.5
