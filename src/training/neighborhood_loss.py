"""
Separation-matching loss.

Trajectory accuracy does not imply Jacobian accuracy: two models can predict equally well and
have completely different derivatives. FTLE is built entirely from derivatives, so the usual
prediction loss never supervises the quantity we actually measure. This does, directly:

    L_sep = || Phi^T(s + d0) - Phi^T(s) - dT ||^2 / ||d0||^2

For small d0, Phi^T(s + d0) - Phi^T(s) ~ J^T d0, so this supervises exactly the product of
Jacobians that the FTLE integrates. Normalising by ||d0||^2 makes it scale-free -- it asks
about the RATIO of growth, which is what a Lyapunov exponent is.
"""
import torch


def separation_loss(model, states, pairs, horizon, second_order_eps=0.0, seed=0):
    i, j = pairs["i"], pairs["j"]
    s_i, s_j = states[i], states[j]
    d0 = s_j - s_i
    dT = states[j + horizon] - states[i + horizon]

    def rollout(x):
        for _ in range(horizon):
            x = model(x)
        return x

    pred_sep = rollout(s_i + d0) - rollout(s_i)
    denom = (d0 ** 2).sum(-1).clamp_min(1e-12)
    loss = (((pred_sep - dT) ** 2).sum(-1) / denom).mean()

    if second_order_eps > 0:
        # evaluate the same consistency at perturbed base states: matching Jacobians at
        # nearby points penalises curvature mismatch at first-order cost, which is what
        # stops a model being locally right and globally wrong
        g = torch.Generator(device=states.device).manual_seed(seed)
        eps = torch.randn(s_i.shape, generator=g, dtype=s_i.dtype,
                          device=s_i.device) * second_order_eps
        pred2 = rollout(s_i + eps + d0) - rollout(s_i + eps)
        loss = loss + (((pred2 - dT) ** 2).sum(-1) / denom).mean()
    return loss
