from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.slot_dynamics import SlotDynamicsRepresentation, patch_grid


def test_patch_grid_requires_square_and_spans_normalized_image():
    grid = patch_grid(16)
    assert grid.shape == (16, 2)
    assert torch.allclose(grid.amin(0), torch.tensor([-1., -1.]))
    assert torch.allclose(grid.amax(0), torch.tensor([1., 1.]))


def test_slot_model_reconstructs_and_exposes_generic_geometry():
    torch.manual_seed(2)
    model = SlotDynamicsRepresentation(input_dim=12, slot_dim=16, slots=4, iterations=2)
    out = model(torch.randn(3, 16, 12), torch.randn(3, 4))
    assert out["reconstruction"].shape == (3, 16, 12)
    assert out["masks"].shape == (3, 4, 16)
    assert out["geometry"].shape == (3, 4, 6)
    assert out["predicted_slots"].shape == (3, 4, 16)
    assert torch.allclose(out["masks"].sum(1), torch.ones(3, 16), atol=1e-6)
    assert torch.allclose(out["geometry"][..., -1].sum(1), torch.ones(3), atol=1e-6)


def test_action_conditioned_transition_is_differentiable():
    model = SlotDynamicsRepresentation(input_dim=8, slot_dim=12, slots=3, iterations=1)
    slots = torch.randn(2, 3, 12, requires_grad=True)
    prediction = model.predict(slots, torch.randn(2, 4))
    prediction.square().mean().backward()
    assert slots.grad is not None and torch.isfinite(slots.grad).all()
