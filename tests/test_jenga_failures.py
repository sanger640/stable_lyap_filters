import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from jenga_failures import boundary_mask, failure_masks


def test_all_three_blocks_can_fail():
    tilt = np.array([[50, 0, 0], [0, 50, 0], [0, 0, 50], [0, 0, 0]])
    masks = failure_masks(tilt)
    np.testing.assert_array_equal(masks["middle_failure"], [True, False, False, False])
    np.testing.assert_array_equal(masks["neighbor_failure"], [False, True, True, False])
    np.testing.assert_array_equal(masks["any_failure"], [True, True, True, False])


def test_middle_fall_below_table_is_failure_but_lift_is_not():
    tilt = np.zeros((3, 3))
    z = np.array([[0.46, 0.46, 0.46], [0.40, 0.46, 0.46], [0.55, 0.46, 0.46]])
    np.testing.assert_array_equal(
        failure_masks(tilt, z)["middle_failure"], [False, True, False])


def test_boundary_requires_two_probes_each_side():
    assert boundary_mask([[False, False, True, True]])[0]
    assert not boundary_mask([[False, False, False, True]])[0]
