import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "jenga_causal_router.py"
SPEC = importlib.util.spec_from_file_location("jenga_causal_router", SCRIPT)
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)


def test_causal_features_reconstruct_nominal_and_residual_without_deeper_levels():
    starts = np.arange(122, dtype=np.float32).reshape(2, 61)
    nominal = np.arange(64, dtype=np.float32).reshape(2, 8, 4) / 10
    residual = np.arange(48, dtype=np.float32).reshape(2, 8, 3) / 100
    alphas = np.zeros((2, 6, 2), dtype=np.float32)
    alphas[:, 0] = [-1.5, .5]
    actions = np.zeros((2, 6, 2, 38, 4), dtype=np.float32)
    for endpoint in range(2):
        actions[:, 0, endpoint, :8] = nominal
        actions[:, 0, endpoint, :8, :3] += alphas[:, 0, endpoint, None, None] * residual
    first = router.causal_features(starts, actions, alphas)
    actions[:, 1:] = 12345  # Future-informed bisection levels must be invisible to the router.
    second = router.causal_features(starts, actions, alphas)

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[:, :61], starts)
    np.testing.assert_allclose(first[:, 61:93], nominal.reshape(2, -1), atol=1e-6)
    np.testing.assert_allclose(first[:, 93:], residual.reshape(2, -1), atol=1e-6)


def test_causal_router_has_hard_mode_and_auxiliary_residual_heads():
    model = router.CausalRouter(117, modes=3, residual_dimensions=15)
    logits, residual = model(torch.zeros(5, 117))

    assert logits.shape == (5, 3)
    assert residual.shape == (5, 15)
    assert logits.argmax(1).shape == (5,)


def test_fit_only_standardization_does_not_use_heldout_statistics():
    fit = np.asarray([[0.0, 2.0], [2.0, 4.0]])
    heldout = np.asarray([[100.0, 200.0]])
    fit_z, heldout_z, mean, scale = router.standardize_fit(fit, heldout)

    np.testing.assert_allclose(mean, [1.0, 3.0])
    np.testing.assert_allclose(scale, [1.0, 1.0])
    np.testing.assert_allclose(fit_z.mean(0), [0.0, 0.0])
    np.testing.assert_allclose(heldout_z, [[99.0, 197.0]])


def test_smooth_false_positive_requires_a_persistent_prediction():
    truth = np.zeros((2, 5, 3), dtype=float)
    truth[:, -1] = np.log([.05, .08])[:, None]
    predicted = truth.copy()
    predicted[0, -1] = np.log(.15)  # Neither strictly smooth nor persistent: not an alarm.
    predicted[1, -1] = np.log(.30)  # Persistent prediction on a smooth target: false alarm.

    assert router.smooth_false_positive_rate(predicted, truth) == .5
