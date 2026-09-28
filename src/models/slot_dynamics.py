"""Anonymous slot representation and action-conditioned latent dynamics.

The module has no task-specific heads.  It groups generic patch descriptors into anonymous slots,
reconstructs those descriptors, and predicts one-step slot evolution conditioned on the action.
Slot masks expose position, second moment and mass as generic image-space geometry.
"""
import math

import torch
from torch import nn


def patch_grid(count, device=None, dtype=torch.float32):
    side = int(round(math.sqrt(count)))
    if side * side != count:
        raise ValueError("patch count must form a square grid")
    axis = torch.linspace(-1., 1., side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


class SlotAttention(nn.Module):
    def __init__(self, input_dim=64, slot_dim=48, slots=8, iterations=3):
        super().__init__()
        self.slots = slots
        self.iterations = iterations
        self.scale = slot_dim ** -0.5
        self.input_norm = nn.LayerNorm(input_dim)
        self.slot_norm = nn.LayerNorm(slot_dim)
        self.mlp_norm = nn.LayerNorm(slot_dim)
        self.keys = nn.Linear(input_dim, slot_dim, bias=False)
        self.values = nn.Linear(input_dim, slot_dim, bias=False)
        self.queries = nn.Linear(slot_dim, slot_dim, bias=False)
        self.initial = nn.Parameter(torch.randn(1, slots, slot_dim) * .02)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, 2 * slot_dim), nn.GELU(),
                                 nn.Linear(2 * slot_dim, slot_dim))

    def forward(self, inputs):
        values = self.values(self.input_norm(inputs))
        keys = self.keys(self.input_norm(inputs))
        slots = self.initial.expand(len(inputs), -1, -1)
        attention = None
        for _ in range(self.iterations):
            previous = slots
            queries = self.queries(self.slot_norm(slots)) * self.scale
            logits = torch.einsum("bsd,bpd->bsp", queries, keys)
            # Pixels compete across slots; each slot then averages its assigned pixels.
            attention = logits.softmax(dim=1) + 1e-8
            weights = attention / attention.sum(dim=2, keepdim=True)
            updates = torch.einsum("bsp,bpd->bsd", weights, values)
            slots = self.gru(updates.reshape(-1, updates.shape[-1]),
                             previous.reshape(-1, previous.shape[-1])).reshape_as(previous)
            slots = slots + self.mlp(self.mlp_norm(slots))
        return slots, attention


class SlotDynamicsRepresentation(nn.Module):
    def __init__(self, input_dim=64, slot_dim=48, slots=8, iterations=3, action_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.slots = slots
        self.attention = SlotAttention(input_dim, slot_dim, slots, iterations)
        self.decoder = nn.Sequential(nn.Linear(slot_dim + 6, 2 * slot_dim), nn.GELU(),
                                     nn.Linear(2 * slot_dim, input_dim + 1))
        self.transition = nn.Sequential(nn.Linear(slot_dim + action_dim, 2 * slot_dim), nn.GELU(),
                                        nn.Linear(2 * slot_dim, slot_dim))

    def decode(self, slots, patches):
        coordinates = patch_grid(patches, slots.device, slots.dtype)
        position = torch.cat([coordinates, coordinates.square(),
                              torch.sin(math.pi * coordinates)], dim=-1)
        position = position[None, None].expand(len(slots), self.slots, -1, -1)
        broadcast = slots[:, :, None].expand(-1, -1, patches, -1)
        decoded = self.decoder(torch.cat([broadcast, position], dim=-1))
        features, mask_logits = decoded[..., :-1], decoded[..., -1]
        masks = mask_logits.softmax(dim=1)
        reconstruction = torch.sum(masks[..., None] * features, dim=1)
        return reconstruction, masks

    def geometry(self, masks):
        coordinates = patch_grid(masks.shape[-1], masks.device, masks.dtype)
        weights = masks / masks.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        centre = torch.einsum("bsp,pd->bsd", weights, coordinates)
        difference = coordinates[None, None] - centre[:, :, None]
        xx = torch.sum(weights * difference[..., 0].square(), dim=-1)
        yy = torch.sum(weights * difference[..., 1].square(), dim=-1)
        xy = torch.sum(weights * difference[..., 0] * difference[..., 1], dim=-1)
        mass = masks.mean(dim=-1)
        return torch.cat([centre, xx[..., None], yy[..., None], xy[..., None], mass[..., None]], -1)

    def encode(self, inputs):
        slots, _ = self.attention(inputs)
        reconstruction, masks = self.decode(slots, inputs.shape[1])
        return {"slots": slots, "reconstruction": reconstruction, "masks": masks,
                "geometry": self.geometry(masks)}

    def predict(self, slots, actions):
        action = actions[:, None].expand(-1, self.slots, -1)
        return slots + self.transition(torch.cat([slots, action], dim=-1))

    def forward(self, inputs, actions=None):
        out = self.encode(inputs)
        if actions is not None:
            out["predicted_slots"] = self.predict(out["slots"], actions)
        return out


def slot_geometry_trajectory(model, descriptors, action_mean=None, action_std=None, actions=None):
    """Encode ``(B,T,P,D)`` descriptors and return slot geometry/latents by frame."""
    batch, time, patches, dims = descriptors.shape
    encoded = model.encode(descriptors.reshape(batch * time, patches, dims))
    return {name: value.reshape(batch, time, *value.shape[1:])
            for name, value in encoded.items() if name in ("slots", "geometry", "masks")}
