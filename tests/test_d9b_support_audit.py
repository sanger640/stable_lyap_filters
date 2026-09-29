from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from jenga_d9b_support_audit import (action_features, pairwise_rms, percentile_ranks,
                                     spearman)


def test_action_features_are_endpoint_swap_invariant():
    rng = np.random.default_rng(3)
    actions = rng.normal(size=(2, 6, 2, 38, 4)).astype(np.float32)
    mean = np.zeros((38, 4), np.float32); scale = np.ones((38, 4), np.float32)
    np.testing.assert_allclose(action_features(actions, mean, scale),
                               action_features(actions[:, :, ::-1], mean, scale))


def test_pairwise_rms_and_percentile_ranks():
    left = np.asarray([[0., 0.], [2., 0.]])
    right = np.asarray([[1., 0.], [3., 0.], [2., 2.]])
    distance = pairwise_rms(left, right)
    np.testing.assert_allclose(distance[0], [np.sqrt(.5), np.sqrt(4.5), 2.])
    np.testing.assert_allclose(percentile_ranks([1., 2., 3.]), [1 / 3, 2 / 3, 1.])


def test_spearman_handles_monotonic_and_constant_values():
    assert spearman([1, 2, 3], [10, 20, 30]) == 1.
    assert spearman([1, 1, 1], [2, 3, 4]) == 0.
