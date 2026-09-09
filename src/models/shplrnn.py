"""
Shallow piecewise-linear RNN (shPLRNN) — the dynamics head whose Jacobians are exact.

    s_{t+1} = A @ s_t + W1 @ relu(W2 @ s_t + h2) + h1 + C @ a_t

Each hidden unit i defines a switching hyperplane  W2[i]·s + h2[i] = 0. Those hyperplanes
carve state space into polyhedral cells; inside a cell the map is exactly affine, so the
Jacobian is analytic and piecewise constant. That is the whole point: a smooth network can
only *steepen* a discontinuity, never represent it, which is why the existing causal-ViT
head's Jacobian is correct but useless at the operating perturbation scale (PLAN.md §0).

Keep d in 20-60. The FTLE step does d x d QR at every evaluation point and high-d Jacobian
estimates are noise-dominated at realistic data scale.
"""
import torch
import torch.nn as nn


class ShPLRNN(nn.Module):
    def __init__(self, d=32, H=32, action_dim=0):
        super().__init__()
        self.d, self.H, self.action_dim = d, H, action_dim
        self.A = nn.Parameter(torch.rand(d) * 0.5 + 0.5)      # diagonal, init near-contractive
        self.W1 = nn.Parameter(torch.randn(d, H) / H ** 0.5)
        self.W2 = nn.Parameter(torch.randn(H, d) / d ** 0.5)
        self.h1 = nn.Parameter(torch.zeros(d))
        self.h2 = nn.Parameter(torch.zeros(H))
        self.C = nn.Parameter(torch.randn(d, action_dim) / max(action_dim, 1) ** 0.5) \
            if action_dim else None

    def gate(self, s):
        """Active-set indicator: which side of each hyperplane s falls on. (…, H) in {0,1}."""
        return (s @ self.W2.T + self.h2 > 0).to(s.dtype)

    def forward(self, s, a=None):
        out = s * self.A + torch.relu(s @ self.W2.T + self.h2) @ self.W1.T + self.h1
        if self.C is not None and a is not None:
            out = out + a @ self.C.T
        return out

    step = forward

    def cell_id(self, s):
        """Integer id of the polyhedral cell (hashable active set), for cell-crossing analysis."""
        g = self.gate(s).to(torch.int64)
        return tuple(g.tolist()) if g.dim() == 1 else [tuple(r.tolist()) for r in g]
