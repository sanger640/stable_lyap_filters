"""
Lorenz-63 — the calibration system for Phase 1.

    dx/dt = sigma (y - x)
    dy/dt = x (rho - z) - y
    dz/dt = x y - beta z

Used because its Lyapunov spectrum is known and, more importantly, because two properties
let us validate ANY computed spectrum with no external reference at all:

  1. sum of exponents == trace of the Jacobian == -(sigma + 1 + beta) == -13.6667 (constant)
  2. one exponent must be exactly 0 (the direction along the flow neither grows nor shrinks)

Published reference: (0.906, 0.000, -14.572). Note 0.906 + 0 - 14.572 = -13.666, consistent
with (1). We recompute it ourselves anyway rather than trusting the number.
"""
import torch

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
PUBLISHED_SPECTRUM = (0.906, 0.000, -14.572)


def trace_of_jacobian(sigma=SIGMA, beta=BETA):
    """Constant divergence of the vector field; the spectrum MUST sum to this."""
    return -(sigma + 1.0 + beta)


def f(s, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return torch.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dim=-1)


def jac_f(s, sigma=SIGMA, rho=RHO, beta=BETA):
    """Analytic Jacobian of the vector field (continuous time)."""
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    return torch.stack([
        torch.stack([-sigma * one, sigma * one, zero], -1),
        torch.stack([rho - z, -one, -x], -1),
        torch.stack([y, x, -beta * one], -1),
    ], dim=-2)


def rk4_step(s, dt, **kw):
    k1 = f(s, **kw); k2 = f(s + dt / 2 * k1, **kw)
    k3 = f(s + dt / 2 * k2, **kw); k4 = f(s + dt * k3, **kw)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def flow_jacobian(s, dt, **kw):
    """Jacobian of the dt-flow map, by integrating the variational equation dPhi/dt = J Phi
    with the same RK4 scheme used for the state. Using I + J*dt instead would introduce an
    O(dt^2) bias that shows up directly in the exponents."""
    d = s.shape[-1]
    I = torch.eye(d, dtype=s.dtype, device=s.device)

    def deriv(state, Phi):
        return f(state, **kw), jac_f(state, **kw) @ Phi

    k1s, k1P = deriv(s, I)
    k2s, k2P = deriv(s + dt / 2 * k1s, I + dt / 2 * k1P)
    k3s, k3P = deriv(s + dt / 2 * k2s, I + dt / 2 * k2P)
    k4s, k4P = deriv(s + dt * k3s, I + dt * k3P)
    return I + dt / 6 * (k1P + 2 * k2P + 2 * k3P + k4P)


def simulate(n_steps, dt=0.01, s0=None, burn_in=5000, seed=0, **kw):
    """Trajectory on the attractor. burn_in is discarded so we are not measuring the transient."""
    g = torch.Generator().manual_seed(seed)
    s = (torch.randn(3, generator=g, dtype=torch.float64) if s0 is None
         else torch.as_tensor(s0, dtype=torch.float64))
    for _ in range(burn_in):
        s = rk4_step(s, dt, **kw)
    out = torch.empty(n_steps, 3, dtype=torch.float64)
    for i in range(n_steps):
        out[i] = s
        s = rk4_step(s, dt, **kw)
    return out


def true_spectrum(T=100_000, dt=0.01, seed=0, reorth_every=10, **kw):
    """Ground truth from the TRUE equations, via the same Benettin/QR code the learned model
    will use. Computing this ourselves (rather than quoting the literature) is what makes a
    Phase-1 failure diagnosable: wrong here => our FTLE code is broken; right here but wrong
    on the learned model => the model's Jacobians are."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "geometry"))
    from ftle import lyapunov_spectrum_general

    s0 = simulate(1, dt=dt, seed=seed, **kw)[0]
    return lyapunov_spectrum_general(
        s0, lambda s: rk4_step(s, dt, **kw), lambda s: flow_jacobian(s, dt, **kw),
        T=T, dt=dt, reorth_every=reorth_every)


def validate_spectrum(spec, tol_trace=1e-2, tol_zero=1e-2):
    """The two reference-free checks. Returns (ok, report)."""
    spec = torch.as_tensor(spec, dtype=torch.float64)
    tr, got = trace_of_jacobian(), float(spec.sum())
    zero_err = float(spec.abs().min())
    ok = abs(got - tr) < tol_trace and zero_err < tol_zero
    return ok, {
        "spectrum": [round(float(v), 4) for v in spec],
        "sum": round(got, 4), "expected_sum_trace": round(tr, 4),
        "sum_error": round(abs(got - tr), 5),
        "closest_to_zero": round(zero_err, 5),
        "sum_check_passed": bool(abs(got - tr) < tol_trace),
        "zero_exponent_check_passed": bool(zero_err < tol_zero),
    }
