"""
Generalized teacher forcing.

On chaotic data a free-running rollout has gradients that grow like exp(lambda_max * T), so
backprop through a long rollout is numerically meaningless. Pure teacher forcing removes the
explosion but the model never sees its own error and drifts when run freely. GTF interpolates
at every step:

    s_t  <-  alpha * s_t_data  +  (1 - alpha) * s_t_model

alpha = 0 is free-running, alpha = 1 is pure teacher forcing. Heuristic starting point once
lambda_max is known: alpha ~ 1 - exp(-lambda_max * dt). The sweep is a first-class experiment,
not a footnote -- it is the most likely single cause of a failed spectrum.
"""
import torch


def gtf_rollout_loss(model, seq, alpha, action_seq=None):
    """seq: (B, L, d) ground-truth states. Returns mean one-step prediction loss under GTF."""
    B, L, d = seq.shape
    s = seq[:, 0]
    total = 0.0
    for t in range(L - 1):
        a = action_seq[:, t] if action_seq is not None else None
        pred = model(s, a)
        total = total + ((pred - seq[:, t + 1]) ** 2).mean()
        s = alpha * seq[:, t + 1] + (1.0 - alpha) * pred
    return total / (L - 1)


def gtf_rollout_loss_latent(model, seq_obs, alpha, action_seq=None):
    """GTF for a model whose latent is larger than its observation.

    The crucial difference from the full-state version: data is injected into the OBSERVED
    coordinates only. The auxiliary latent coordinates are never forced -- they evolve freely
    and are shaped purely by the loss on the observed readout. Forcing the whole state (which
    is what the obs_dim == d case does) is a different, stronger constraint."""
    B, L, N = seq_obs.shape
    z = model.lift(seq_obs[:, 0])
    total = 0.0
    for t in range(L - 1):
        a = action_seq[:, t] if action_seq is not None else None
        z = model(z, a)
        pred = model.observe(z)
        total = total + ((pred - seq_obs[:, t + 1]) ** 2).mean()
        forced = alpha * seq_obs[:, t + 1] + (1.0 - alpha) * pred
        z = torch.cat([forced, z[..., N:]], dim=-1)
    return total / (L - 1)


def free_rollout(model, s0, steps, action_seq=None):
    """No forcing at all -- what evaluation uses."""
    s, out = s0, []
    for t in range(steps):
        s = model(s, action_seq[:, t] if action_seq is not None else None)
        out.append(s)
    return torch.stack(out, 1)
