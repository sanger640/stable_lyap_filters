from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d9b_shared_amplitude import factor_targets, factor_losses
from set_response_model import GroupAmplitudeShapeTemporalCurveModel


def test_group_factor_uses_one_amplitude_and_reconstructs():
    target = torch.zeros(2, 6, 41); target[1] = torch.rand(6, 41)
    amplitude, shape, nonzero = factor_targets(target)
    assert amplitude.shape == (2, 1, 1)
    assert torch.equal(amplitude[0], torch.zeros_like(amplitude[0]))
    assert not torch.any(nonzero[0])
    assert torch.allclose(amplitude * shape, target, atol=1e-6)
    losses = factor_losses(amplitude, shape, amplitude, shape, nonzero)
    assert all(float(value) == 0. for value in losses.values())


def test_group_model_shares_amplitude_and_preserves_permutation_equivariance():
    model = GroupAmplitudeShapeTemporalCurveModel(
        steps=5, action_horizon=3, curve_steps=6, hidden=32, heads=4, layers=1).eval()
    state = torch.randn(2, 61); actions = torch.randn(2, 4, 2, 5, 4)
    norms = (torch.zeros(61), torch.ones(61), torch.zeros(5, 4), torch.ones(5, 4))
    permutation = torch.tensor([2, 0, 3, 1]); inverse = torch.argsort(permutation)
    with torch.no_grad():
        prediction, amplitude, shape = model(state, actions, *norms)
        reordered, reordered_amplitude, reordered_shape = model(
            state, actions[:, permutation], *norms)
    assert amplitude.shape == (2, 1, 1)
    assert torch.allclose(amplitude, reordered_amplitude, atol=1e-6)
    assert torch.allclose(prediction, reordered[:, inverse], atol=1e-6)
    assert torch.allclose(shape, reordered_shape[:, inverse], atol=1e-6)


def test_group_model_zero_amplitude_zeros_every_level():
    model = GroupAmplitudeShapeTemporalCurveModel(
        steps=5, action_horizon=3, curve_steps=6, hidden=32, heads=4, layers=1).eval()
    with torch.no_grad():
        model.amplitude_head[-1].weight.zero_(); model.amplitude_head[-1].bias.fill_(-1.)
        state = torch.randn(1, 61); actions = torch.randn(1, 6, 2, 5, 4)
        norms = (torch.zeros(61), torch.ones(61), torch.zeros(5, 4), torch.ones(5, 4))
        prediction, amplitude, _ = model(state, actions, *norms)
    assert torch.equal(amplitude, torch.zeros_like(amplitude))
    assert torch.equal(prediction, torch.zeros_like(prediction))
