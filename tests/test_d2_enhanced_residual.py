from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from d2_enhanced_residual import D2EnhancedResidual, active_contact_logits
from state_dynamics import StepGraphNet


def test_zero_residual_exactly_preserves_d2_physical_outputs_and_contacts():
    base = StepGraphNet(hidden=32, rounds=1)
    state_dict = {"hidden": 32, "rounds": 1, "model": base.state_dict()}
    d2_block = torch.linspace(.1, 1.5, 15); d2_grip = torch.tensor([.2, .3, .4])
    target_block = torch.linspace(.2, 1.6, 15); target_grip = torch.tensor([.4, .5, .6])
    residual = D2EnhancedResidual(state_dict, d2_block, d2_grip, target_block, target_grip,
                                  hidden=32)
    state = torch.zeros(4, 181); state[:, 9:27] = torch.tensor([1., 0., 0., 1., 0., 0.] * 3)
    state[:, 45:57] = torch.randint(0, 2, (4, 12)).float(); action = torch.zeros(4, 4)
    with torch.no_grad():
        expected = base(state[:, :61], action)
        actual = residual(state, action, torch.zeros(181), torch.ones(181),
                          torch.zeros(4), torch.ones(4))
    assert torch.allclose(actual["block_mean"] * target_block,
                          expected["block_mean"] * d2_block, atol=1e-7)
    assert torch.allclose(actual["gripper_mean"] * target_grip,
                          expected["gripper_mean"] * d2_grip, atol=1e-7)
    assert torch.allclose(actual["block_physical"], expected["block_mean"] * d2_block, atol=1e-7)
    assert torch.allclose(actual["gripper_physical"], expected["gripper_mean"] * d2_grip, atol=1e-7)
    active = torch.isin(actual["contact_mode_logits"].argmax(-1), torch.tensor([1, 2]))
    assert torch.equal(active, active_contact_logits(expected) > 0)
