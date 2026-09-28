import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jenga_action_branch", ROOT / "eval/jenga_action_branch.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_action_errors_keep_only_variable_h8_xyz():
    windows = np.zeros((4, 40, 4), np.float32)
    windows[:, 2:10, :3] = np.arange(4)[:, None, None]
    windows[:, :, 3] = np.arange(4)[:, None]
    errors = MODULE.action_errors(windows)
    assert errors.shape == (4, 8, 3)
    assert np.allclose(errors.mean(0), 0)


def test_pose_features_are_shared_dimensionless_environment_state():
    pose = np.ones((3, 5, 27), np.float32)
    features = MODULE.pose_features(pose)
    assert features.shape == (3, 5, 27)
    assert np.allclose(features[0, 0, :9], 20.0)
    assert np.allclose(features[0, 0, 9:], 1.0)


def test_predicted_pose_uses_the_five_ground_truth_hold_times():
    class Model:
        substeps = 1
    predicted = np.zeros((2, 38, 61), np.float32)
    predicted[:, :, 0] = np.arange(38)
    features = MODULE.sampled_predicted_pose(predicted, Model())
    assert features.shape == (2, 5, 27)
    assert np.allclose(features[0, :, 0] * .05, [12, 17, 27, 36, 37])
