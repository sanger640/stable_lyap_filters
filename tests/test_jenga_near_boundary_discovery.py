import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from jenga_near_boundary_discovery import nearest_transition, offset_probe_chunks


def test_nearest_transition_prefers_one_near_zero():
    scalars = np.array([-2, -1, 0, 1, 2])
    outcomes = np.array([False, True, True, False, False])
    assert nearest_transition(scalars, outcomes) == .5


def test_offset_probe_family_keeps_first_pose_and_gripper_semantics():
    chunk = np.array([[1, 2, 3, -1], [2, 4, 6, 1]], np.float32)
    probes = offset_probe_chunks(chunk, [-1, 1], .1, .2)
    np.testing.assert_allclose(probes[:, 0, :3], np.repeat(chunk[None, 0, :3], 2, axis=0))
    np.testing.assert_array_equal(probes[:, :, 3], np.repeat(chunk[None, :, 3], 2, axis=0))
    np.testing.assert_allclose(probes[0, 1, :3], chunk[1, :3] + .1 * (chunk[1, :3] - chunk[0, :3]))
