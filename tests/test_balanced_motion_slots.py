from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from models.balanced_motion_slots import BalancedMotionSlotRepresentation, balanced_assignment
import panda_push_slot_train_v2 as training


def test_balanced_assignment_prevents_slot_monopoly():
    masks = balanced_assignment(torch.randn(3, 8, 64), iterations=10)
    assert torch.allclose(masks.sum(1), torch.ones(3, 64), atol=1e-5)
    mass = masks.mean(-1)
    assert torch.allclose(mass, torch.full_like(mass, 1 / 8), atol=3e-3)


def test_balanced_motion_model_shapes_and_protocol_gate():
    model = BalancedMotionSlotRepresentation(descriptor_dim=8, slot_dim=12, slots=4, iterations=1)
    start = torch.randn(2, 16, 8); out = model(start + .1, start)
    assert out["masks"].shape == (2, 4, 16)
    assert training.QUALITY_GATES["minimum_effective_slots"] == 6.0
    assert "topple" in " ".join(training.PROTOCOL["excluded"])
