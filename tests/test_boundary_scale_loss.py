import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from boundary_scale_loss import boundary_scale_terms, phase_gaps  # noqa: E402


def trajectories(profile, steps=38):
    profile = torch.as_tensor(profile, dtype=torch.float32)
    out = torch.zeros(1, len(profile), 2, steps, 45)
    out[0, :, 1, :, 0] = profile[:, None]
    return out


def test_phase_gaps_weight_phases_equally():
    effect = torch.zeros(1, 38, 2)
    effect[:, :8] = 1.0
    effect[:, 8:18] = 2.0
    effect[:, 18:] = 3.0
    gaps = phase_gaps(effect, epsilon=0)
    assert torch.allclose(gaps, torch.tensor([[1.0, 2.0, 3.0]]))


def test_exact_smooth_scaling_has_zero_d3_loss():
    truth = trajectories([1.0, .5, .25, .125, .0625, .03125])
    terms = boundary_scale_terms(truth.clone(), truth, torch.ones(45))
    assert float(terms.total()) == 0.0


def test_exact_persistent_plateau_has_zero_d3_loss():
    truth = trajectories([1.0] * 6)
    terms = boundary_scale_terms(truth.clone(), truth, torch.ones(45))
    assert float(terms.total()) == 0.0


def test_smooth_prediction_is_penalised_for_physical_plateau():
    truth = trajectories([1.0] * 6)
    predicted = trajectories([1.0, .5, .25, .125, .0625, .03125])
    terms = boundary_scale_terms(predicted, truth, torch.ones(45))
    assert float(terms.global_scale) > 0.5
    assert float(terms.local_scale) > 0.05
    assert float(terms.total()) > 0


def test_plateau_prediction_is_penalised_for_smooth_physics():
    truth = trajectories([1.0, .5, .25, .125, .0625, .03125])
    predicted = trajectories([1.0] * 6)
    terms = boundary_scale_terms(predicted, truth, torch.ones(45))
    assert float(terms.global_scale) > 0.5
    assert float(terms.local_scale) > 0.05


def test_d3_scale_loss_backpropagates():
    truth = trajectories([1.0] * 6)
    predicted = trajectories([1.0, .5, .25, .125, .0625, .03125]).requires_grad_()
    terms = boundary_scale_terms(predicted, truth, torch.ones(45))
    terms.total().backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum()) > 0
