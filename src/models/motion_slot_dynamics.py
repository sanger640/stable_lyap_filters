"""Motion-conditioned anonymous slots for label-free physical-dynamics representation.

Unlike the v0 appearance autoencoder, this model observes both the current descriptor and its
change from the common pre-action image, and reconstructs only that change.  Position enters the
attention keys but cannot by itself explain a sample-specific motion residual.
"""
import math

import torch
from torch import nn

from models.slot_dynamics import SlotAttention, SlotDynamicsRepresentation, patch_grid


class MotionSlotDynamicsRepresentation(SlotDynamicsRepresentation):
    def __init__(self, descriptor_dim=64, slot_dim=48, slots=8, iterations=3, action_dim=4):
        self.descriptor_dim = descriptor_dim
        super().__init__(input_dim=2 * descriptor_dim, slot_dim=slot_dim, slots=slots,
                         iterations=iterations, action_dim=action_dim)
        self.position = nn.Linear(6, 2 * descriptor_dim, bias=False)
        self.decoder = nn.Sequential(nn.Linear(slot_dim + 6, 2 * slot_dim), nn.GELU(),
                                     nn.Linear(2 * slot_dim, descriptor_dim + 1))

    def encode(self, current, start):
        if current.shape != start.shape:
            start = torch.broadcast_to(start, current.shape)
        delta = current - start
        coordinates = patch_grid(current.shape[1], current.device, current.dtype)
        positional = torch.cat([coordinates, coordinates.square(),
                                torch.sin(math.pi * coordinates)], dim=-1)
        inputs = torch.cat([current, delta], dim=-1) + self.position(positional)[None]
        slots, _ = self.attention(inputs)
        reconstruction, masks = self.decode(slots, current.shape[1])
        return {"slots": slots, "reconstruction": reconstruction, "masks": masks,
                "geometry": self.geometry(masks), "target_delta": delta}

    def forward(self, current, start, actions=None):
        out = self.encode(current, start)
        if actions is not None:
            out["predicted_slots"] = self.predict(out["slots"], actions)
        return out
