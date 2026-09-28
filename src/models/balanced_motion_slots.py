"""Motion-conditioned slots with differentiable balanced patch assignment."""
import torch

from models.motion_slot_dynamics import MotionSlotDynamicsRepresentation
from models.slot_dynamics import patch_grid


def balanced_assignment(logits, iterations=6):
    """Approximately doubly-stochastic slot-by-patch assignments.

    Every patch sums to one across slots while every slot receives approximately the same total
    mass.  This structurally prevents the one-slot monopoly observed in motion-slot v1.
    """
    weights = (logits - logits.amax(dim=1, keepdim=True)).exp().clamp_min(1e-8)
    target_mass = logits.shape[-1] / logits.shape[1]
    for _ in range(iterations):
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8) * target_mass
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)


class BalancedMotionSlotRepresentation(MotionSlotDynamicsRepresentation):
    def decode(self, slots, patches):
        coordinates = patch_grid(patches, slots.device, slots.dtype)
        position = torch.cat([coordinates, coordinates.square(),
                              torch.sin(torch.pi * coordinates)], dim=-1)
        position = position[None, None].expand(len(slots), self.slots, -1, -1)
        broadcast = slots[:, :, None].expand(-1, -1, patches, -1)
        decoded = self.decoder(torch.cat([broadcast, position], dim=-1))
        features, mask_logits = decoded[..., :-1], decoded[..., -1]
        masks = balanced_assignment(mask_logits)
        reconstruction = torch.sum(masks[..., None] * features, dim=1)
        return reconstruction, masks
