"""D20 action-conditioned dynamics with enhanced Markov context and supervised contact modes."""
import torch
import torch.nn as nn

from state_dynamics import (CONTACT, GRIPPER, N_BLOCKS, POS, ROT, VEL, StepGraphNet,
                            orthonormalise_rotation)

BASE_DIM = 61
PROPRIO = slice(61, 85)
FORCE = slice(85, 181)
ENHANCED_DIM = 181
CONTINUOUS = tuple(range(45)) + (57, 58, 59)


def mlp(inp, out, hidden):
    return nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                         nn.GELU(), nn.Linear(hidden, out))


def contact_modes(current, following):
    """Torch absent/persistent/created/lost labels for twelve generic contacts."""
    current = current > .5; following = following > .5
    labels = torch.zeros_like(current, dtype=torch.long)
    labels[current & following] = 1
    labels[~current & following] = 2
    labels[current & ~following] = 3
    return labels


class EnhancedContactModeGNN(nn.Module):
    """Graph dynamics conditioned on generic proprioception and force/impulse observations.

    Contact transitions are four supervised physical modes, not unnamed latent experts. The
    model never consumes task, failure, topple, success, or monitor labels.
    """

    def __init__(self, hidden=192, rounds=3):
        super().__init__()
        self.hidden = hidden; self.rounds = rounds
        self.graph = StepGraphNet(hidden, rounds)
        self.context = mlp(ENHANCED_DIM + 4, hidden, hidden)
        self.node_fusion = nn.LayerNorm(hidden)
        self.edge_fusion = nn.LayerNorm(hidden)
        self.contact_mode_head = mlp(2 * hidden, 12 * 4, hidden)
        self.proprio_delta_head = mlp(2 * hidden, 24, hidden)
        self.force_next_head = mlp(2 * hidden, 96, hidden)

    def forward(self, state, action, state_mean, state_scale, action_mean, action_scale):
        base = state[:, :BASE_DIM]
        nodes, edges, senders, receivers, _ = self.graph.trunk(base, action)
        normalized = torch.cat(((state - state_mean) / state_scale,
                                (action - action_mean) / action_scale), dim=-1)
        context = self.context(normalized)
        nodes = self.node_fusion(nodes + context[:, None])
        edges = self.edge_fusion(edges + context[:, None])
        block = self.graph.block_head(nodes[:, :N_BLOCKS])
        shared = self.graph.shared_heads(nodes, edges, senders, receivers)
        pooled = torch.cat((nodes.mean(1), context), dim=-1)
        return {
            "block_mean": block[..., :15],
            "gripper_mean": shared["gripper_mean"],
            "contact_mode_logits": self.contact_mode_head(pooled).view(-1, 12, 4),
            "proprio_delta": self.proprio_delta_head(pooled),
            "force_next": self.force_next_head(pooled),
        }


def continuous_deltas(current, following):
    delta = following - current
    blocks = torch.cat((delta[:, POS].view(-1, 3, 3), delta[:, ROT].view(-1, 3, 6),
                        delta[:, VEL].view(-1, 3, 6)), dim=-1)
    return blocks, delta[:, 57:60], delta[:, PROPRIO]


def apply_enhanced_step(state, action, output, scales, hard_contacts=True):
    block_scale, grip_scale, proprio_scale, force_mean, force_scale = scales
    new = state.clone(); batch = len(state)
    block = output["block_mean"] * block_scale
    new[:, POS] = state[:, POS] + block[..., :3].reshape(batch, 9)
    rotation = state[:, ROT] + block[..., 3:9].reshape(batch, 18)
    new[:, ROT] = orthonormalise_rotation(rotation)
    new[:, VEL] = state[:, VEL] + block[..., 9:15].reshape(batch, 18)
    new[:, 57:60] = state[:, 57:60] + output["gripper_mean"] * grip_scale
    closed = state[:, 60].clone()
    closed = torch.where(action[:, 3] > .9, torch.ones_like(closed), closed)
    closed = torch.where(action[:, 3] < -.9, torch.zeros_like(closed), closed)
    new[:, 60] = closed
    probabilities = output["contact_mode_logits"].softmax(-1)
    active = (torch.sigmoid(output["contact_active_logits"])
              if "contact_active_logits" in output
              else probabilities[..., 1] + probabilities[..., 2])
    new[:, CONTACT] = (active > .5).to(active.dtype) if hard_contacts else active
    new[:, PROPRIO] = state[:, PROPRIO] + output["proprio_delta"] * proprio_scale
    force_log = output["force_next"] * force_scale + force_mean
    new[:, FORCE] = torch.expm1(force_log).clamp_min(0.)
    return new
