"""Action-conditioned mixture over complete trajectories.

Unlike a per-step expert switch, each mode represents one coherent future over the complete action
window.  The gate sees only the initial state and intended action sequence.  Physical futures are
used only to train a proper mixture likelihood; task and failure labels never enter.
"""
import torch
import torch.nn as nn

from state_dynamics import (ACTION_SCALE_M, GRIPPER, StepGraphNet, orthonormalise_rotation,
                            rollout)

CONTINUOUS = tuple(range(45)) + (57, 58, 59)


class TrajectoryMixtureGNN(nn.Module):
    """A shared GNN rollout plus K coherent, action-conditioned residual trajectory modes."""

    def __init__(self, hidden=128, rounds=3, modes=3):
        super().__init__()
        if modes < 2:
            raise ValueError("trajectory mixture requires at least two modes")
        self.hidden, self.rounds, self.modes = hidden, rounds, modes
        self.base = StepGraphNet(hidden, rounds)
        self.action_encoder = nn.GRU(4, hidden, batch_first=True)
        self.context = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden))
        self.mode_embeddings = nn.Parameter(torch.randn(modes, hidden) * .05)
        self.residual = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, len(CONTINUOUS)))
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, modes))
        # Corrections start small but nonidentical, avoiding a tied winner-take-all collapse.
        self.residual_scale = .1
        self.orthonormalise = True
        self.hard_contacts = False
        self.substeps = 1

    def action_features(self, start, actions):
        xyz = (actions[..., :3] - start[:, None, GRIPPER.start:GRIPPER.start + 3]) / ACTION_SCALE_M
        command = torch.sign(actions[..., 3:4]) * (actions[..., 3:4].abs() > .9)
        return torch.cat([xyz, command], dim=-1)

    def forward_trajectory(self, start, actions, delta_scale, state_scale):
        """Return mode trajectories ``(B,K,T,61)`` and gate logits ``(B,K)``."""
        baseline = rollout(self.base, start, actions, delta_scale)
        nodes, _, _, _, _ = self.base.trunk(start, actions[:, 0])
        state_context = nodes.mean(dim=1)
        action_sequence, action_final = self.action_encoder(self.action_features(start, actions))
        context = self.context(torch.cat([state_context, action_final[-1]], dim=-1))
        logits = self.gate(torch.cat([state_context, action_final[-1]], dim=-1))

        batch, steps = actions.shape[:2]
        sequence = action_sequence[:, None].expand(batch, self.modes, steps, self.hidden)
        modes = self.mode_embeddings[None, :, None].expand(batch, self.modes, steps, self.hidden)
        residual = self.residual(torch.cat([sequence + context[:, None, None], modes], dim=-1))
        index = torch.as_tensor(CONTINUOUS, device=start.device)
        correction = self.residual_scale * residual * state_scale[index]
        full_correction = torch.zeros(
            batch, self.modes, steps, 61, dtype=baseline.dtype, device=baseline.device
        ).index_copy(-1, index, correction)
        trajectories = baseline[:, None] + full_correction
        if self.orthonormalise:
            rotation = trajectories[..., 9:27].reshape(-1, 18)
            rotation = orthonormalise_rotation(rotation).reshape(batch, self.modes, steps, 18)
            trajectories = torch.cat(
                [trajectories[..., :9], rotation, trajectories[..., 27:]], dim=-1)
        return trajectories, logits

    def predict(self, start, actions, delta_scale, state_scale):
        """Deployable deterministic prediction: choose a mode from state/actions alone."""
        trajectories, logits = self.forward_trajectory(start, actions, delta_scale, state_scale)
        selected = logits.argmax(-1)
        return trajectories[torch.arange(len(start), device=start.device), selected], selected


def initialise_from_stepnet(model, checkpoint):
    """Warm-start the shared dynamics exactly from a deterministic StepGraphNet checkpoint."""
    if checkpoint.get("kind", "single") != "single":
        raise ValueError("trajectory mixture currently requires a single-GNN initialization")
    model.base.load_state_dict(checkpoint["model"])
    model.base.orthonormalise = bool(checkpoint.get("orthonormalise", False))
    model.base.hard_contacts = bool(checkpoint.get("hard_contacts", False))
    model.orthonormalise = model.base.orthonormalise
    model.hard_contacts = model.base.hard_contacts
    return model


def mixture_terms(trajectories, logits, truth, state_scale, temperature=.05):
    """Proper finite-mixture likelihood and task-free anti-collapse diagnostics."""
    index = torch.as_tensor(CONTINUOUS, device=truth.device)
    difference = ((trajectories[..., index] - truth[:, None, :, index])
                  / state_scale[index].clamp_min(1e-6))
    error = difference.square().mean(dim=(-1, -2))
    log_probability = logits.log_softmax(-1) - error / temperature
    nll = -torch.logsumexp(log_probability, dim=-1).mean()
    posterior = log_probability.softmax(-1)
    usage = posterior.mean(0).clamp_min(1e-9)
    balance = (usage * (usage * trajectories.shape[1]).log()).sum()
    target = posterior.detach().argmax(-1)
    routing = nn.functional.cross_entropy(logits, target)
    gate_probability = logits.softmax(-1).clamp_min(1e-9)
    entropy = -(gate_probability * gate_probability.log()).sum(-1).mean()
    selected = (posterior.detach()[..., None, None] * trajectories).sum(1)
    return {"nll": nll, "balance": balance, "routing": routing, "entropy": entropy,
            "selected": selected, "posterior": posterior, "error": error}
