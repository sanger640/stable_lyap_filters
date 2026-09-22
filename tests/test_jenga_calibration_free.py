import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_calibration_free", ROOT / "eval/jenga_calibration_free.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UnitStepModel:
    substeps = 1


def trajectories(groups_at_10, groups_at_30):
    out = np.zeros((len(groups_at_10), 38, 61), np.float32)
    for held, labels in ((10, groups_at_10), (30, groups_at_30)):
        index = MODULE.hold_index(UnitStepModel(), held)
        out[:, index, 0] = np.asarray(labels) * 0.02
    return out


def test_persistent_two_modes_alarm_without_reference_data():
    labels = np.r_[np.zeros(32), np.ones(32)]
    result = MODULE.calibration_free_test(trajectories(labels, labels), UnitStepModel())
    assert result["alarm"]
    assert result["pc1_persistent_alarm"]
    assert result["groups_hold10"] == result["groups_hold30"] == 2


def test_transient_or_single_mode_does_not_alarm():
    split = np.r_[np.zeros(32), np.ones(32)]
    one = np.zeros(64)
    assert not MODULE.calibration_free_test(trajectories(split, one), UnitStepModel())["alarm"]
