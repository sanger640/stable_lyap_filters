from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from neighborhood_topology_loss import (commitment_loss, neighborhood_terms, topology_loss)


def fixture():
    torch.manual_seed(4)
    truth = torch.randn(2, 6, 38, 61)
    actions = torch.randn(2, 6, 38, 4)
    scale = torch.ones(61)
    return truth, actions, scale


def test_identical_neighborhood_has_zero_losses():
    truth, actions, scale = fixture()
    terms = neighborhood_terms(truth, truth, actions, scale)
    assert float(terms.response) == 0.0
    assert float(terms.topology) == 0.0
    assert float(terms.commitment) == 0.0


def test_collapsing_responses_is_penalised():
    truth, actions, scale = fixture()
    collapsed = truth.mean(1, keepdim=True).expand_as(truth).clone()
    assert topology_loss(collapsed, truth, scale) > 0
    assert commitment_loss(collapsed, truth, actions, scale) > 0


def test_losses_are_differentiable():
    truth, actions, scale = fixture()
    predicted = (truth + .1 * torch.randn_like(truth)).requires_grad_()
    neighborhood_terms(predicted, truth, actions, scale).total().backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
