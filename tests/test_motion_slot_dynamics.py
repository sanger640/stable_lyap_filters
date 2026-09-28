from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from models.motion_slot_dynamics import MotionSlotDynamicsRepresentation
import panda_push_slot_train_v1 as training


def test_motion_slots_reconstruct_descriptor_change():
    model = MotionSlotDynamicsRepresentation(descriptor_dim=8, slot_dim=12, slots=3, iterations=1)
    start = torch.randn(2, 16, 8); current = start + .1 * torch.randn_like(start)
    out = model(current, start, torch.randn(2, 4))
    assert out["reconstruction"].shape == current.shape
    assert torch.allclose(out["target_delta"], current - start)
    assert out["geometry"].shape == (2, 3, 6)


def test_v1_objective_is_label_free_and_differentiable():
    model = MotionSlotDynamicsRepresentation(descriptor_dim=8, slot_dim=12, slots=3, iterations=1)
    descriptors = torch.randn(2, 2, 4, 16, 8)
    loss, terms = training.objective(model, descriptors, torch.randn(2, 2, 3, 4),
                                     training.LOSS_WEIGHTS)
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == set(training.LOSS_WEIGHTS)
    assert "topple" in " ".join(training.PROTOCOL["excluded"])
