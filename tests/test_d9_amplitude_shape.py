from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_d9_amplitude_shape import factor_losses, factor_targets, gate
from set_response_model import AmplitudeShapeTemporalCurveModel


def test_factor_targets_preserve_exact_zero_and_reconstruct():
    target = torch.zeros(2, 3, 41); target[1] = torch.rand(3, 41)
    amplitude, shape, nonzero = factor_targets(target)
    assert torch.equal(amplitude[0], torch.zeros_like(amplitude[0]))
    assert not torch.any(nonzero[0])
    assert torch.allclose(amplitude * shape, target, atol=1e-6)
    losses = factor_losses(amplitude, shape, amplitude, shape, nonzero)
    assert all(float(value) == 0. for value in losses.values())


def test_amplitude_shape_model_emits_exact_zero_for_negative_amplitude_logits():
    model = AmplitudeShapeTemporalCurveModel(steps=5, action_horizon=3, curve_steps=6,
                                              hidden=32, heads=4, layers=1).eval()
    with torch.no_grad():
        model.amplitude_head[-1].weight.zero_(); model.amplitude_head[-1].bias.fill_(-1.)
        state = torch.randn(1, 61); actions = torch.randn(1, 4, 2, 5, 4)
        norms = (torch.zeros(61), torch.ones(61), torch.zeros(5, 4), torch.ones(5, 4))
        prediction, amplitude, _ = model(state, actions, *norms)
    assert torch.equal(amplitude, torch.zeros_like(amplitude))
    assert torch.equal(prediction, torch.zeros_like(prediction))


def test_gate_requires_exact_zero_amplitude():
    errors = {name: 0. for name in ("early_gap", "full_gap", "curve_action",
                                     "curve_early_hold", "curve_late_hold")}
    nested = {"boundary": {"agreement": 1.}, "alarm": {"agreement": 1.}}
    checks = {"endpoint_swap_max_abs": 0., "level_permutation_max_abs": 0.}
    target = torch.tensor([[[0.]], [[1.]]]); predicted = target.clone()
    assert gate(errors, nested, checks, predicted, target)["passes"]
    predicted[0] = 1e-12
    assert not gate(errors, nested, checks, predicted, target)["passes"]
