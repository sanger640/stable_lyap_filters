"""Task-free losses for preserving a counterfactual action neighbourhood.

The monitor does not consume one trajectory in isolation.  It compares the response to many
nearby actions, so a useful world model must preserve the *relations* among those responses.  The
functions here operate only on actions and anonymous continuous state coordinates; they contain no
Jenga labels, object roles, failure definitions or monitor thresholds.
"""
from dataclasses import dataclass

import torch


CONTINUOUS = tuple(range(45)) + (57, 58, 59)
PHASES = ((0, 8), (8, 18), (18, 38))


@dataclass(frozen=True)
class NeighborhoodTerms:
    response: torch.Tensor
    topology: torch.Tensor
    commitment: torch.Tensor

    def total(self, topology_weight=1.0, commitment_weight=1.0):
        return (self.response + topology_weight * self.topology
                + commitment_weight * self.commitment)


def _coordinates(trajectories, state_scale):
    index = torch.as_tensor(CONTINUOUS, device=trajectories.device)
    scale = state_scale[index].clamp_min(1e-6)
    return trajectories[..., index] / scale


def _pairwise_phase_geometry(values, phases):
    """RMS trajectory distance for every probe pair in each temporal phase."""
    probes = values.shape[1]
    left, right = torch.triu_indices(probes, probes, offset=1, device=values.device)
    matrices = []
    for start, stop in phases:
        stop = min(stop, values.shape[2])
        if start >= stop:
            continue
        difference = values[:, left, start:stop] - values[:, right, start:stop]
        matrices.append(torch.sqrt(difference.square().mean(dim=(-1, -2)) + 1e-12))
    if not matrices:
        raise ValueError("trajectory is shorter than every requested phase")
    return torch.stack(matrices, dim=-1)


def topology_loss(predicted, truth, state_scale, phases=PHASES):
    """Match the complete pairwise response geometry, not only independent trajectories.

    A per-group physical scale makes the loss invariant to units while retaining relative
    distances and therefore clusters, narrow branches and the ordering of nearby responses.
    """
    predicted_geometry = _pairwise_phase_geometry(_coordinates(predicted, state_scale), phases)
    true_geometry = _pairwise_phase_geometry(_coordinates(truth, state_scale), phases)
    normalizer = true_geometry.mean(dim=1, keepdim=True).detach().clamp_min(1e-4)
    return ((torch.log1p(predicted_geometry / normalizer)
             - torch.log1p(true_geometry / normalizer)) ** 2).mean()


def _local_action_pairs(actions, neighbours):
    """Undirected k-nearest pairs in action-window space, selected without outcomes."""
    flat = actions[..., :3].flatten(2)
    flat = (flat - flat.mean(dim=1, keepdim=True))
    scale = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    flat = flat / scale
    distance = torch.cdist(flat, flat)
    distance.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    nearest = distance.topk(min(neighbours, actions.shape[1] - 1), largest=False).indices
    pairs = []
    for group in range(actions.shape[0]):
        unique = set()
        for left in range(actions.shape[1]):
            for right in nearest[group, left].tolist():
                unique.add(tuple(sorted((left, right))))
        pairs.append(sorted(unique))
    return pairs


def commitment_loss(predicted, truth, actions, state_scale, neighbours=3):
    """Match when locally neighbouring actions separate and whether that separation persists.

    This is a dense trajectory-level signal.  Pairs are chosen solely by action proximity; the
    target is their normalized separation curve over time.  It therefore does not presuppose two
    clusters and remains applicable to any controlled dynamical system.
    """
    p = _coordinates(predicted, state_scale)
    q = _coordinates(truth, state_scale)
    losses = []
    ramp = torch.linspace(0.25, 1.0, predicted.shape[2], device=predicted.device)
    for group, pairs in enumerate(_local_action_pairs(actions, neighbours)):
        left = torch.as_tensor([pair[0] for pair in pairs], device=predicted.device)
        right = torch.as_tensor([pair[1] for pair in pairs], device=predicted.device)
        p_gap = torch.sqrt((p[group, left] - p[group, right]).square().mean(-1) + 1e-12)
        q_gap = torch.sqrt((q[group, left] - q[group, right]).square().mean(-1) + 1e-12)
        normalizer = q_gap.mean().detach().clamp_min(1e-4)
        losses.append((ramp * (torch.log1p(p_gap / normalizer)
                               - torch.log1p(q_gap / normalizer)) ** 2).mean())
    return torch.stack(losses).mean()


def neighborhood_terms(predicted, truth, actions, state_scale, neighbours=3):
    """Return matched dynamics, neighborhood topology and local commitment terms."""
    p, q = _coordinates(predicted, state_scale), _coordinates(truth, state_scale)
    response = ((p - q) ** 2).mean()
    topology = topology_loss(predicted, truth, state_scale)
    commitment = commitment_loss(predicted, truth, actions, state_scale, neighbours)
    return NeighborhoodTerms(response, topology, commitment)
