from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from enhanced_hybrid_dynamics import (EnhancedContactModeGNN, apply_enhanced_step,
                                      contact_modes)


def test_torch_contact_modes_cover_all_transitions():
    current = torch.tensor([[0., 1., 0., 1.]])
    following = torch.tensor([[0., 1., 1., 0.]])
    assert contact_modes(current, following).tolist() == [[0, 1, 2, 3]]


def test_enhanced_model_shapes_and_step_are_finite():
    model = EnhancedContactModeGNN(hidden=32, rounds=1)
    state = torch.zeros(2, 181); state[:, 9:27] = torch.tensor(
        [1., 0., 0., 1., 0., 0.] * 3)
    action = torch.zeros(2, 4); mean = torch.zeros(181); scale = torch.ones(181)
    output = model(state, action, mean, scale, torch.zeros(4), torch.ones(4))
    assert output["contact_mode_logits"].shape == (2, 12, 4)
    assert output["proprio_delta"].shape == (2, 24)
    scales = (torch.ones(15), torch.ones(3), torch.ones(24), torch.zeros(96), torch.ones(96))
    following = apply_enhanced_step(state, action, output, scales)
    assert following.shape == (2, 181)
    assert torch.isfinite(following).all()
