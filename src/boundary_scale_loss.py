"""Task-agnostic nested action-width supervision for counterfactual world models.

The loss never sees failure, task, or object-identity labels. It compares predicted and physical
responses for nested pairs of actions whose width halves at every level. Matching the normalized
gap curve forces a model to preserve whichever local topology physics exhibits: proportional
collapse in a smooth region or a nonzero plateau across a hybrid regime boundary.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BoundaryScaleTerms:
    response: torch.Tensor
    effect: torch.Tensor
    global_scale: torch.Tensor
    local_scale: torch.Tensor

    def total(self, local_weight=1.0):
        return self.response + self.effect + self.global_scale + local_weight * self.local_scale


def _phase_slices(steps, action_horizon=8, early_hold=10):
    if not 0 < action_horizon < action_horizon + early_hold < steps:
        raise ValueError("trajectory must contain action, early-hold, and late-hold phases")
    return (slice(0, action_horizon),
            slice(action_horizon, action_horizon + early_hold),
            slice(action_horizon + early_hold, steps))


def phase_gaps(pair_effect, action_horizon=8, early_hold=10, epsilon=1e-6):
    """RMS separation in the action, early-hold, and late-hold phases.

    ``pair_effect`` has shape ``(..., time, features)`` and should already be normalized by generic
    per-channel training scales. Equal phase treatment prevents the longer hold from numerically
    overwhelming the action/contact interval.
    """
    if pair_effect.ndim < 2:
        raise ValueError("pair_effect must contain time and feature dimensions")
    phases = _phase_slices(pair_effect.shape[-2], action_horizon, early_hold)
    return torch.stack([
        torch.sqrt(pair_effect[..., phase, :].square().mean(dim=(-2, -1)) + epsilon ** 2)
        for phase in phases
    ], dim=-1)


def boundary_scale_terms(predicted, truth, state_scale, action_horizon=8, early_hold=10,
                         epsilon=1e-4, huber_delta=1.0):
    """D3 terms for nested counterfactual endpoint trajectories.

    ``predicted`` and ``truth`` have shape ``(groups, levels, 2, time, features)``. Level zero is
    the widest bracket; each following level is one midpoint bisection narrower. The trajectories
    at every level start from the same state and use opposite bracket endpoints.

    ``response`` and ``effect`` are D2-style absolute and counterfactual-vector matching on the
    nested examples. ``global_scale`` matches every level's gap relative to level zero.
    ``local_scale`` matches shrinkage between adjacent levels, identifying the exact bisection at
    which a predicted physical branch is smoothed away.
    """
    if predicted.shape != truth.shape or predicted.ndim != 5 or predicted.shape[2] != 2:
        raise ValueError("predicted and truth must have shape (groups, levels, 2, time, features)")
    if predicted.shape[-1] < 45:
        raise ValueError("trajectories must contain at least 45 continuous state fields")
    if predicted.shape[1] < 2:
        raise ValueError("at least two nested action-width levels are required")
    scale = torch.as_tensor(state_scale, dtype=predicted.dtype, device=predicted.device)[:45]
    if scale.shape != (45,) or torch.any(scale <= 0):
        raise ValueError("state_scale must provide 45 positive continuous scales")
    predicted_z = predicted[..., :45] / scale
    truth_z = truth[..., :45] / scale

    response = (predicted_z - truth_z).square().mean()
    predicted_effect = predicted_z[:, :, 1] - predicted_z[:, :, 0]
    truth_effect = truth_z[:, :, 1] - truth_z[:, :, 0]
    effect = (predicted_effect - truth_effect).square().mean()

    predicted_gap = phase_gaps(predicted_effect, action_horizon, early_hold, epsilon)
    truth_gap = phase_gaps(truth_effect, action_horizon, early_hold, epsilon)
    predicted_global = torch.log((predicted_gap[:, 1:] + epsilon)
                                 / (predicted_gap[:, :1] + epsilon))
    truth_global = torch.log((truth_gap[:, 1:] + epsilon)
                             / (truth_gap[:, :1] + epsilon))
    global_scale = F.huber_loss(predicted_global, truth_global, delta=huber_delta)

    predicted_local = torch.log((predicted_gap[:, 1:] + epsilon)
                                / (predicted_gap[:, :-1] + epsilon))
    truth_local = torch.log((truth_gap[:, 1:] + epsilon)
                            / (truth_gap[:, :-1] + epsilon))
    local_scale = F.huber_loss(predicted_local, truth_local, delta=huber_delta)
    return BoundaryScaleTerms(response, effect, global_scale, local_scale)
