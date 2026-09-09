"""
Smooth-dynamics baselines for Phase 1b — the controls for the piecewise-linear claim.

PLAN.md commits to a piecewise-linear head on the argument that smooth networks cannot
represent a discontinuity, only steepen it. That is a *design commitment*, asserted rather
than measured. These baselines measure it.

Important scoping note. A ReLU network is itself CONTINUOUS, so it cannot represent a jump
discontinuity either. What shPLRNN actually buys is a piecewise-CONSTANT Jacobian with hard
switching, plus explicit inspectable switching surfaces. So the honest comparison is:

  * on Lorenz (smooth, no guard): do the two recover the spectrum equally well? If yes, the
    piecewise-linear head has no intrinsic advantage on smooth dynamics -- which is the
    expected and fair result.
  * on the bouncing ball (Phase 2, hard guard): does the smooth model degrade while shPLRNN
    holds? THAT contrast is the architecture argument, and it cannot be made on Lorenz alone.

Both baselines expose the same interface as ShPLRNN (`forward`, `lift`, `observe`, `.d`) so the
Benettin/QR code runs on them unchanged. Their Jacobians come from autograd rather than an
analytic formula -- correct, but no longer exact-by-construction, which is itself part of what
is being compared.
"""
import torch
import torch.nn as nn


class SmoothRNN(nn.Module):
    """tanh recurrence: s_{t+1} = s + dt_scale * tanh-MLP(s). Smooth everywhere.

    Written in residual form for the same reason shPLRNN is initialised near identity: the
    dt-flow map of a continuous system is near-identity at dt=0.01, and a model that has to
    learn that from scratch spends its first epochs fighting the integrator."""

    def __init__(self, d=3, H=128, action_dim=0, obs_dim=None, hidden_layers=2, scale=0.1):
        super().__init__()
        self.d, self.action_dim = d, action_dim
        self.obs_dim = d if obs_dim is None else obs_dim
        self.scale = scale
        layers, inp = [], d + action_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(inp, H), nn.Tanh()]
            inp = H
        layers += [nn.Linear(inp, d)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():                      # start at identity
            self.net[-1].weight.mul_(0.01); self.net[-1].bias.zero_()

    def forward(self, s, a=None):
        x = s if a is None else torch.cat([s, a], -1)
        return s + self.scale * self.net(x)

    step = forward

    def observe(self, z):
        return z[..., :self.obs_dim]

    def lift(self, x):
        if self.obs_dim == self.d:
            return x
        pad = torch.zeros(*x.shape[:-1], self.d - self.obs_dim, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], -1)


class GRUCellModel(nn.Module):
    """GRU-style gated recurrence, also smooth (sigmoid/tanh gates).

    Included because gating is the usual answer to "the RNN cannot switch behaviour" -- a GRU
    can approximate a switch with a saturated sigmoid. Whether a *saturated* gate is good
    enough at a guard surface is exactly the open question, so it belongs in the comparison."""

    def __init__(self, d=3, H=128, action_dim=0, obs_dim=None, scale=0.1):
        super().__init__()
        self.d, self.action_dim = d, action_dim
        self.obs_dim = d if obs_dim is None else obs_dim
        self.scale = scale
        self.cell = nn.GRUCell(d + action_dim, H)
        self.out = nn.Linear(H, d)
        with torch.no_grad():
            self.out.weight.mul_(0.01); self.out.bias.zero_()
        self.H = H

    def forward(self, s, a=None):
        x = s if a is None else torch.cat([s, a], -1)
        h = self.cell(x, torch.zeros(*x.shape[:-1], self.H, dtype=x.dtype, device=x.device))
        return s + self.scale * self.out(h)

    step = forward

    def observe(self, z):
        return z[..., :self.obs_dim]

    def lift(self, x):
        if self.obs_dim == self.d:
            return x
        pad = torch.zeros(*x.shape[:-1], self.d - self.obs_dim, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], -1)


def autograd_jacobian(s, model):
    """Jacobian for models without an analytic form. Correct, but computed rather than exact --
    and that difference is part of what Phase 1b measures."""
    return torch.autograd.functional.jacobian(lambda x: model.step(x), s, vectorize=True)
