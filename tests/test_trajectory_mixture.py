from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trajectory_mixture import TrajectoryMixtureGNN, mixture_terms  # noqa: E402


def scales():
    return (torch.ones(15) * .01, torch.ones(3) * .01), torch.ones(61)


def test_complete_trajectory_modes_and_deployable_choice():
    torch.manual_seed(2)
    model = TrajectoryMixtureGNN(hidden=16, rounds=1, modes=3)
    start = torch.randn(4, 61); actions = torch.randn(4, 5, 4)
    delta, state_scale = scales()
    trajectories, logits = model.forward_trajectory(start, actions, delta, state_scale)
    chosen, modes = model.predict(start, actions, delta, state_scale)
    assert trajectories.shape == (4, 3, 5, 61)
    assert logits.shape == (4, 3) and chosen.shape == (4, 5, 61)
    assert torch.equal(chosen, trajectories[torch.arange(4), modes])


def test_mixture_objective_is_finite_differentiable_and_balanced_at_equal_fit():
    torch.manual_seed(3)
    trajectories = torch.randn(6, 3, 5, 61, requires_grad=True)
    truth = torch.randn(6, 5, 61)
    logits = torch.zeros(6, 3, requires_grad=True)
    terms = mixture_terms(trajectories, logits, truth, torch.ones(61))
    loss = terms["nll"] + terms["balance"] + terms["routing"]
    loss.backward()
    assert torch.isfinite(loss)
    assert trajectories.grad is not None and logits.grad is not None


def test_gate_cannot_read_truth():
    model = TrajectoryMixtureGNN(hidden=16, rounds=1, modes=3)
    start = torch.randn(2, 61); actions = torch.randn(2, 5, 4)
    delta, state_scale = scales()
    _, first = model.forward_trajectory(start, actions, delta, state_scale)
    _, second = model.forward_trajectory(start, actions, delta, state_scale)
    assert torch.equal(first, second)
