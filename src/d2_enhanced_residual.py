"""Zero-initialized enhanced residual around the frozen D2 dynamics model."""
import torch
import torch.nn as nn

from enhanced_hybrid_dynamics import ENHANCED_DIM, mlp
from state_dynamics import CONTACT, N_BLOCKS, StepGraphNet


def _zero_last(module):
    layer = next(layer for layer in reversed(module) if isinstance(layer, nn.Linear))
    nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)


def active_contact_logits(output):
    """Map D2 pair and per-block contact logits into the canonical twelve-contact order."""
    values = torch.zeros((len(output["pair_contact_logits"]), 12),
                         device=output["pair_contact_logits"].device,
                         dtype=output["pair_contact_logits"].dtype)
    values[:, :3] = output["pair_contact_logits"]
    per_block = output["block_contact_logits"]
    for block in range(N_BLOCKS):
        values[:, 3 + block] = per_block[:, block, 0]
        values[:, 6 + block] = per_block[:, block, 1]
        values[:, 9 + block] = per_block[:, block, 2]
    return values


def base_mode_logits(current_contacts, active_logits):
    """Four-way logits whose argmax exactly matches D2's next-contact threshold."""
    current = current_contacts > .5
    absent = torch.where(current, torch.full_like(active_logits, -20.), -active_logits)
    persistent = torch.where(current, active_logits, torch.full_like(active_logits, -20.))
    created = torch.where(current, torch.full_like(active_logits, -20.), active_logits)
    lost = torch.where(current, -active_logits, torch.full_like(active_logits, -20.))
    return torch.stack((absent, persistent, created, lost), dim=-1)


class D2EnhancedResidual(nn.Module):
    def __init__(self, d2_state, d2_block_scale, d2_grip_scale, target_block_scale,
                 target_grip_scale, hidden=192):
        super().__init__()
        self.base = StepGraphNet(d2_state["hidden"], d2_state["rounds"])
        self.base.load_state_dict(d2_state["model"])
        for parameter in self.base.parameters(): parameter.requires_grad_(False)
        self.context = mlp(ENHANCED_DIM + 4, hidden, hidden)
        self.block_residual = mlp(hidden, 45, hidden)
        self.grip_residual = mlp(hidden, 3, hidden)
        self.mode_residual = mlp(hidden, 48, hidden)
        self.proprio_delta = mlp(hidden, 24, hidden)
        self.force_next = mlp(hidden, 96, hidden)
        for head in (self.block_residual, self.grip_residual, self.mode_residual):
            _zero_last(head)
        self.register_buffer("d2_block_scale", torch.as_tensor(d2_block_scale))
        self.register_buffer("d2_grip_scale", torch.as_tensor(d2_grip_scale))
        self.register_buffer("target_block_scale", torch.as_tensor(target_block_scale))
        self.register_buffer("target_grip_scale", torch.as_tensor(target_grip_scale))

    def forward(self, state, action, state_mean, state_scale, action_mean, action_scale):
        with torch.no_grad():
            base = self.base(state[:, :61], action)
        context = self.context(torch.cat(((state - state_mean) / state_scale,
                                          (action - action_mean) / action_scale), dim=-1))
        block_residual = self.block_residual(context).view(-1, 3, 15)
        grip_residual = self.grip_residual(context)
        active_base = active_contact_logits(base)
        mode_residual = self.mode_residual(context).view(-1, 12, 4)
        mode_logits = base_mode_logits(state[:, CONTACT], active_base) + mode_residual
        active_correction = ((mode_residual[..., 1] + mode_residual[..., 2])
                             - (mode_residual[..., 0] + mode_residual[..., 3])) * .5
        block_physical = base["block_mean"] * self.d2_block_scale + (
            block_residual * self.target_block_scale)
        grip_physical = base["gripper_mean"] * self.d2_grip_scale + (
            grip_residual * self.target_grip_scale)
        return {
            "block_mean": block_physical / self.target_block_scale,
            "gripper_mean": grip_physical / self.target_grip_scale,
            "block_physical": block_physical,
            "gripper_physical": grip_physical,
            "base_block_physical": base["block_mean"] * self.d2_block_scale,
            "base_gripper_physical": base["gripper_mean"] * self.d2_grip_scale,
            "base_contact_active_logits": active_base,
            "contact_mode_logits": mode_logits,
            "contact_active_logits": active_base + active_correction,
            "proprio_delta": self.proprio_delta(context),
            "force_next": self.force_next(context),
            "continuous_residual": torch.cat((block_residual.reshape(len(state), -1),
                                               grip_residual), dim=-1),
        }
