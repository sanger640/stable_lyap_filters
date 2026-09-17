import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_physical_jump_diagnostics import adjacent_ratio, quaternion_angle_deg


def test_quaternion_angle_ignores_sign():
    identity = np.array([1.0, 0, 0, 0])
    np.testing.assert_allclose(quaternion_angle_deg(identity, -identity), 0, atol=1e-8)
    half = np.sqrt(0.5)
    np.testing.assert_allclose(
        quaternion_angle_deg(identity, np.array([half, 0, 0, half])), 90, atol=1e-8)


def test_adjacent_ratio_identifies_jump():
    values = np.array([0, 1, 2, 10, 11], np.float64)
    jump, ratio = adjacent_ratio(values, 2)
    assert jump == 8
    assert ratio == 8
