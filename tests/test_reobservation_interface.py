from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_reobservation_interface import binary_metrics, reanchored_rollout


def additive_rollout(_model, state, actions, _scale):
    values = []
    current = state.clone()
    for action in actions.transpose(0, 1):
        current = current + action[:, :1]
        values.append(current.clone())
    return torch.stack(values, dim=1)


def test_reanchored_rollout_retains_predictions_but_resets_segment_input():
    starts = np.zeros((1, 61), np.float32)
    actions = np.zeros((1, 4, 4), np.float32)
    actions[:, :, 0] = 1.
    truth = np.zeros((1, 4, 61), np.float32)
    truth[0, :, :] = np.asarray([10., 20., 30., 40.])[:, None]

    result = reanchored_rollout(
        None, None, starts, actions, truth, interval=2, device="cpu",
        rollout_function=additive_rollout)

    # The first segment predicts 1,2.  The second starts from observed state 20 and predicts 21,22.
    assert np.allclose(result[0, :, 0], [1., 2., 21., 22.])


def test_interval_one_is_teacher_forced_prediction_not_copied_truth():
    starts = np.zeros((1, 3), np.float32)
    actions = np.zeros((1, 3, 4), np.float32); actions[:, :, 0] = 1.
    truth = np.asarray([[[10., 10., 10.], [20., 20., 20.], [30., 30., 30.]]], np.float32)

    # Reuse a three-dimensional dummy state; shape checking is deliberately exercised separately
    # by the production 61-D call, so pad here to the required state layout.
    padded_start = np.zeros((1, 61), np.float32)
    padded_truth = np.zeros((1, 3, 61), np.float32)
    padded_truth[:, :, :3] = truth
    result = reanchored_rollout(
        None, None, padded_start, actions, padded_truth, interval=1, device="cpu",
        rollout_function=additive_rollout)
    assert np.allclose(result[0, :, 0], [1., 11., 21.])
    assert not np.allclose(result[0, :, 0], padded_truth[0, :, 0])


def test_binary_metrics_uses_physical_decision_as_reference():
    result = binary_metrics([True, True, False, False], [True, False, True, False])
    assert result["tp"] == result["fn"] == result["fp"] == result["tn"] == 1
    assert result["agreement"] == result["recall"] == result["added_positive_rate"] == .5
