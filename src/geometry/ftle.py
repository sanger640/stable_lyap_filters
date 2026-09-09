"""
Flow-map FTLE and the full Lyapunov spectrum, plus a saturating separation measure.

Two rules from PLAN.md that this module exists to enforce:

* **Return the full spectrum, not just lambda_max.** The ratio of positive to negative
  exponents separates a saddle (recoverable) from a repeller (not), and that distinction is
  what the monitor's downstream decision actually needs.
* **Never report an FTLE without its horizon T.** At a genuine guard surface the exponent
  DIVERGES with T instead of converging -- it is a singularity, not a finite ridge. Every
  function here returns T alongside the value, and `finite_separation` is the honest
  companion quantity to report there.
"""
import torch

from jacobian import batched_jacobian, jacobian


def lyapunov_spectrum(s0, model, T, dt=1.0, reorth_every=10, action=None):
    """Full spectrum (descending) via QR reorthonormalisation of the deformation matrix.

    Reorthonormalising is not optional: the raw Jacobian product overflows/collapses to the
    leading direction within tens of steps, which silently turns every exponent into
    lambda_max."""
    d = model.d
    Q = torch.eye(d, dtype=s0.dtype, device=s0.device)
    logs = torch.zeros(d, dtype=s0.dtype, device=s0.device)
    s = s0.clone()
    for t in range(T):
        Q = jacobian(s, model) @ Q
        s = model.step(s, action[t] if action is not None else None)
        if (t + 1) % reorth_every == 0:
            Q, R = torch.linalg.qr(Q)
            logs = logs + torch.log(torch.abs(torch.diagonal(R)) + 1e-300)
    Q, R = torch.linalg.qr(Q)
    logs = logs + torch.log(torch.abs(torch.diagonal(R)) + 1e-300)
    # Benettin/QR exponents are only asymptotically ordered; sort so callers can rely on
    # spec[0] == lambda_max. NOTE this breaks the correspondence between exponent i and
    # column i of Q -- if you need the covariant directions, use the unsorted logs.
    return torch.sort(logs / (T * dt), descending=True).values


def ftle(s0, model, T, dt=1.0, **kw):
    """Leading exponent. Returns (value, T) -- the horizon is part of the number."""
    return float(lyapunov_spectrum(s0, model, T, dt, **kw)[0].detach()), T


def finite_separation(s0, model, T, eps=1e-4, cap=1.0, n_dirs=32, action=None, seed=0):
    """max over sampled unit directions of min(||Phi_T(s0+eps*v) - Phi_T(s0)||, cap).

    Stays finite where the exponent diverges, so this is the quantity to report at a guard
    surface. `cap` should be set to the scale beyond which 'already failed' is the honest
    description -- not left at its default without thought."""
    g = torch.Generator(device=s0.device).manual_seed(seed)
    V = torch.randn(n_dirs, model.d, generator=g, dtype=s0.dtype, device=s0.device)
    V = V / V.norm(dim=-1, keepdim=True)
    S = torch.cat([s0.unsqueeze(0), s0.unsqueeze(0) + eps * V], 0)
    for t in range(T):
        a = action[t].expand(S.shape[0], -1) if action is not None else None
        S = model.step(S, a)
    sep = (S[1:] - S[0:1]).norm(dim=-1)
    return float(torch.clamp(sep, max=cap).max().detach()), T


def ftle_field(states, model, T, dt=1.0, **kw):
    """FTLE at each of (N,d) states. Returns (N,) values and the horizon."""
    return torch.tensor([lyapunov_spectrum(s, model, T, dt, **kw)[0] for s in states]), T


def divergence_with_horizon(s0, model, horizons=(5, 10, 20, 50), dt=1.0, **kw):
    """Characterise whether FTLE diverges with T at this point (guard) or converges (regular).

    Report this, not a single number, anywhere near a suspected boundary."""
    return {int(T): {"ftle": float(lyapunov_spectrum(s0, model, T, dt, **kw)[0].detach()),
                     "finite_sep": finite_separation(s0, model, T, **kw)[0]}
            for T in horizons}
