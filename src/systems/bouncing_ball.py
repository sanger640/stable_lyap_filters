"""
Bouncing ball on a sinusoidally driven table — Phase 2's system.

The cheapest system with BOTH properties we need: chaos *and* a hard guard surface. Lorenz
gave us chaos but is perfectly smooth; Jenga toppling has a real guard but is expensive and
opaque. This has both, in low dimension, with an exactly known impact condition.

    table:      s(t)   = A sin(phi),  phi = omega * t
    free fall:  xdot   = v,  vdot = -g,  phidot = omega
    GUARD:      x = A sin(phi)                       <- the discontinuity
    impact:     v_out  = (1+e) * v_table - e * v_in  <- e = restitution

The impact law is the moving-surface form: the relative velocity reverses and is scaled by e,
so v_out - v_table = -e (v_in - v_table). Using the stationary-surface law (v_out = -e v_in)
would silently remove the energy injection that makes this system chaotic at all.

State is (x, v, phi) with phi kept on [0, 2pi). Chaos comes from the table phase at impact:
hit slightly earlier or later and the outgoing velocity differs a lot.

Dimensionless units: g = 1, A = 1. The controlling parameter is Gamma = A omega^2 / g =
omega^2 -- the table's peak acceleration relative to gravity. Chaos requires Gamma well above 1.
"""
import numpy as np
import torch

G, A, E_DEFAULT = 1.0, 1.0, 0.8
# |v - v_table| below this at impact counts as contact rather than a bounce. It cannot be
# arbitrarily small: relative velocity decays by a factor e per impact, so reaching 1e-6 from
# O(1) takes ~60 impacts. Paired with max_impacts=64 below, 1e-4 is reachable (~41 impacts at
# e=0.8) while still being far below any physically meaningful bounce.
STICK_TOL = 1e-4

# PHASE-2 OPERATING POINT, chosen by eval/tune_bouncing_ball.py. Gamma = omega^2 = 2.0.
# Gamma=4 (omega=2) is also chaotic but throws the ball to x~47 against table amplitude 1, so
# impacts land on 0.27% of samples -- the guard surface, the entire object of Phase 2, would be
# almost never seen. At omega=1.4, dt=0.2 impacts occupy ~2.8% of samples (~35 steps apart)
# with x_med~6, and lambda_max stays comfortably positive. dt, not e, is the lever for impact
# DENSITY: lowering e instead drives the system into inelastic collapse (see `step`).
OMEGA_DEFAULT = 1.4
DT_DEFAULT = 0.20
TWO_PI = 2.0 * np.pi


def table_pos(phi, A_=A):
    return A_ * np.sin(phi)


def table_vel(phi, omega, A_=A):
    return A_ * omega * np.cos(phi)


def gap(state, omega, A_=A):
    """x - table(phi). The GUARD SURFACE is exactly gap == 0."""
    x, _, phi = state
    return x - table_pos(phi, A_)


def contact_force(phi, omega, A_=A):
    """Normal force (per unit mass) needed to hold the ball ON the table: g + a_table, with
    a_table = -A omega^2 sin(phi). Contact is only possible while this is >= 0."""
    return G - A_ * omega * omega * np.sin(phi)


def _detach_time(phi0, omega, rem, A_=A, tol=1e-13):
    """First t in (0, rem] where contact_force goes negative, else None (bisection)."""
    if contact_force(phi0 + omega * rem, omega, A_) >= 0:
        n = 64                                    # force can dip and recover within `rem`
        prev = 0.0
        for i in range(1, n + 1):
            t = rem * i / n
            if contact_force(phi0 + omega * t, omega, A_) < 0:
                lo, hi = prev, t
                break
            prev = t
        else:
            return None
    else:
        lo, hi = 0.0, rem
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if contact_force(phi0 + omega * mid, omega, A_) >= 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi


def _free_flight(state, dt, omega):
    x, v, phi = state
    return np.array([x + v * dt - 0.5 * G * dt * dt, v - G * dt, phi + omega * dt])


def step(state, dt, omega=OMEGA_DEFAULT, e=E_DEFAULT, A_=A, max_impacts=64, tol=1e-13):
    """Advance dt, resolving any impacts EXACTLY inside the interval by bisection.

    Event detection is not optional here. With a fixed step and no bisection the impact gets
    smeared over up to dt, which blurs the very surface Phase 2 is trying to localise -- and
    the blur would look like model error rather than a data-generation artefact."""
    s = np.asarray(state, dtype=np.float64)
    remaining = dt
    for _ in range(max_impacts):
        cand = _free_flight(s, remaining, omega)
        if gap(cand, omega, A_) >= 0 or remaining <= 0:
            s = cand
            break
        lo, hi = 0.0, remaining                       # gap>=0 at lo, <0 at hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if gap(_free_flight(s, mid, omega), omega, A_) >= 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        s = _free_flight(s, lo, omega)                # sit exactly on the guard
        vt = table_vel(s[2], omega, A_)
        s[1] = (1.0 + e) * vt - e * s[1]              # <- the discontinuous reset
        s[0] = table_pos(s[2], A_)                    # kill numerical penetration
        remaining -= lo

        # CONTACT / STICKING PHASE. Without this the model is ill-posed: the impact map alone
        # produces inelastic collapse (infinitely many impacts in finite time), measured here
        # as a single step needing >4000 impacts with the ball at x=-0.22. Physically the ball
        # does not stay stuck -- with Gamma>1 it rides the table and DETACHES the moment the
        # table falls away faster than gravity (contact force < 0). Resampling collapsed runs
        # away would have biased out a real part of the dynamics; this represents it instead.
        if abs(s[1] - vt) < STICK_TOL and contact_force(s[2], omega, A_) >= 0 and remaining > tol:
            td = _detach_time(s[2], omega, remaining, A_)
            ride = remaining if td is None else min(td, remaining)
            s[2] = s[2] + omega * ride                # ride the table exactly
            s[0] = table_pos(s[2], A_)
            s[1] = table_vel(s[2], omega, A_)
            remaining -= ride
        if remaining <= tol:
            break
    else:
        # Loop ran to max_impacts without finishing the interval: INELASTIC COLLAPSE (Zeno).
        # Infinitely many impacts in finite time, which happens when dissipation outruns the
        # table's energy injection. Measured at e<=0.5: the ball settles onto the table, the
        # impact detector reads ZERO impacts, x_max goes negative (riding the table), and
        # lambda_max comes back as ~190. Silently truncating produced all of that nonsense, so
        # record it on the state and let callers check.
        step.collapsed = True
    s[2] = s[2] % TWO_PI
    return s


step.collapsed = False                  # module-level so getattr is never needed


def simulate(n_steps, dt=DT_DEFAULT, s0=None, burn_in=2000, omega=OMEGA_DEFAULT, e=E_DEFAULT,
             seed=0, return_impacts=False):
    step.collapsed = False              # per-call: the flag is sticky and shared
    rng = np.random.default_rng(seed)
    s = np.array([2.0 + rng.uniform(0, 1), rng.uniform(-1, 1), rng.uniform(0, TWO_PI)]) \
        if s0 is None else np.asarray(s0, dtype=np.float64)
    for _ in range(burn_in):
        s = step(s, dt, omega, e)
    out = np.empty((n_steps, 3))
    impacts = np.zeros(n_steps, dtype=bool)
    for i in range(n_steps):
        out[i] = s
        prev_v = s[1]
        s = step(s, dt, omega, e)
        impacts[i] = (s[1] - prev_v) > G * dt * 1.5   # velocity jumped UP => an impact occurred
    simulate.collapsed = step.collapsed
    return (torch.from_numpy(out), torch.from_numpy(impacts)) if return_impacts \
        else torch.from_numpy(out)


def lambda_max_two_particle(n_steps=200_000, dt=DT_DEFAULT, omega=OMEGA_DEFAULT, e=E_DEFAULT,
                            eps=1e-9, seed=0, burn_in=2000):
    """Ground-truth lambda_max by the two-particle method with renormalisation.

    Deliberately NOT the Jacobian/Benettin method used for Lorenz. Across an impact the
    correct tangent map is not the smooth flow Jacobian -- it needs the saltation matrix,
    which accounts for the guard being crossed at a state-dependent time. Getting that wrong
    silently biases the exponent. The two-particle estimator sidesteps the issue entirely and
    is the standard choice for hybrid systems, at the cost of only giving lambda_max.

    Returns NaN if the run hit inelastic collapse. That is not defensive padding: this system
    has COEXISTING attractors -- a chaotic bouncing one and a sticking one. The renormalisation
    s2 <- s + diff*(eps/d) puts the perturbed copy at an artificial state, and at some seeds it
    falls into the sticking basin. The estimator then measures the gap between a bouncing and a
    stuck trajectory, which is not an exponent at all: measured seed 3 returns 3.95 where its
    neighbours return 0.14. Raising max_impacts 8 -> 64 does NOT fix it (3.95 -> 5.06), proving
    it is a genuine basin event and not truncation. Use `true_lambda_max` to pool seeds."""
    step.collapsed = False
    rng = np.random.default_rng(seed)
    s = np.array([2.0 + rng.uniform(0, 1), rng.uniform(-1, 1), rng.uniform(0, TWO_PI)])
    for _ in range(burn_in):
        s = step(s, dt, omega, e)
    d0 = np.array([eps, 0.0, 0.0])
    s2 = s + d0
    total, n = 0.0, 0
    for _ in range(n_steps):
        s = step(s, dt, omega, e)
        s2 = step(s2, dt, omega, e)
        if step.collapsed:
            return float("nan")      # estimate is already invalid; grinding on just burns
                                     # ~480 gap evaluations per step in the sticking regime
        diff = s2 - s
        diff[2] = (diff[2] + np.pi) % TWO_PI - np.pi     # phase is circular
        d = np.linalg.norm(diff)
        if d > 0:
            total += np.log(d / eps)
            n += 1
            s2 = s + diff * (eps / d)                     # renormalise
    if step.collapsed or not n:
        return float("nan")
    return total / (n * dt)


def true_lambda_max(n_seeds=8, n_steps=40_000, dt=DT_DEFAULT, omega=OMEGA_DEFAULT,
                    e=E_DEFAULT, min_ok=5):
    """Ground-truth lambda_max: median over seeds whose perturbed copy stayed on the chaotic
    attractor. Pooling is required because single-seed estimates are contaminated by the
    basin-escape mode documented above."""
    vals = [lambda_max_two_particle(n_steps=n_steps, dt=dt, omega=omega, e=e, seed=sd)
            for sd in range(n_seeds)]
    ok = [v for v in vals if np.isfinite(v)]
    assert len(ok) >= min_ok, f"only {len(ok)}/{n_seeds} seeds survived: {vals}"
    return float(np.median(ok)), float(np.std(ok)), vals


def impact_surface_normal():
    """The guard x - A sin(phi) = 0 is not a plane, so a single ReLU hyperplane cannot match
    it globally. Phase 2's criterion must therefore be LOCAL: does some learned hyperplane
    align with the true guard's tangent plane near the states the system actually visits?
    Returns grad(gap) = (1, 0, -A cos(phi)) as a function of phi."""
    return lambda phi: np.array([1.0, 0.0, -A * np.cos(phi)])


# ---------------------------------------------------------------------------------------
# Observation coordinates for learning
# ---------------------------------------------------------------------------------------

def to_obs(traj):
    """(x, v, phi) -> (x, v, cos phi, sin phi).

    phi is stored mod 2pi, so the raw coordinate leaps by ~2pi every 22 steps at the Phase-2
    regime -- measured at 4.5% of transitions, MORE OFTEN than real impacts (2.7%). A model
    given raw phi spends capacity on a coordinate artefact, and worse, the guard-alignment test
    is contaminated: a ReLU hyperplane can align with the wrap at phi=0 instead of the true
    guard, which would read as a Phase-2 result while being an artefact.

    The embedding also makes the guard EXACTLY LINEAR in the observed coordinates:
    x - A sin(phi) = 0 becomes x - A * o_4 = 0, a genuine hyperplane one ReLU can match
    exactly. That is what turns 'does a learned hyperplane find the guard' into a crisp
    measurement with a known target normal, instead of asking a model to tile a curved surface.
    Note this does NOT smooth the dynamics: the velocity reset is still discontinuous (|dv| up
    to 14.7 in one step), which is the discontinuity the architecture argument is about."""
    t = np.asarray(traj, dtype=np.float64)
    return np.stack([t[..., 0], t[..., 1], np.cos(t[..., 2]), np.sin(t[..., 2])], axis=-1)


def guard_normal_obs(A_=A):
    """Unit normal of the guard in observation coordinates: x - A*sin(phi) = 0, so the normal
    is (1, 0, 0, -A) normalised. This is the known target for the hyperplane-alignment test."""
    n = np.array([1.0, 0.0, 0.0, -A_])
    return n / np.linalg.norm(n)


# ---------------------------------------------------------------------------------------
# Ground-truth spectrum
# ---------------------------------------------------------------------------------------

def _wrap_diff(a, b):
    d = (a - b).clone()
    d[2] = (d[2] + np.pi) % TWO_PI - np.pi
    return d


def true_spectrum(n_seeds=8, T=20_000, dt=DT_DEFAULT, omega=OMEGA_DEFAULT, e=E_DEFAULT,
                  eps=1e-9, burn_in=2000, zero_tol=0.01):
    """Full spectrum by FINITE-DIFFERENCE Benettin, pooled over seeds.

    No Jacobian is used anywhere: at the guard the correct tangent map needs the saltation
    matrix, and finite differences of the flow map sidestep it. The estimator is validated
    against the analytic-Jacobian spectrum on Lorenz, where the two agree to machine precision.

    `eps` is critical and NOT a free knob. It must be small enough that the perturbed particle
    stays on the SAME SIDE of the guard as the reference; at eps=1e-7 seed 3 returns
    lambda_1 = -0.030 (a non-chaotic answer for a demonstrably chaotic trajectory), and only
    at eps <= 1e-8 does it recover 0.148. Lorenz shows no such sensitivity because it is
    smooth -- the guard is what makes eps matter.

    T MATTERS: lambda_2 converges to 0 only as ~1/T. At T=4000 every seed reads -0.02..-0.03
    and the check below correctly rejects all of them; by T=20000 it settles near -0.004. Do
    not widen zero_tol to make a short run pass -- that discards the convergence check.

    Seeds are accepted only if |lambda_2| < zero_tol. lambda_2 is the flow direction and must
    be 0, so this is an internal validity check in the same spirit as Lorenz's zero-exponent
    test, and it is what rejects the seeds whose exponents are corrupted."""
    import torch
    from ftle import lyapunov_spectrum_finite_diff

    step_fn = lambda s: torch.from_numpy(step(s.numpy(), dt, omega, e))   # noqa: E731
    rows = []
    for sd in range(n_seeds):
        rng = np.random.default_rng(sd)
        s = np.array([2.0 + rng.uniform(0, 1), rng.uniform(-1, 1), rng.uniform(0, TWO_PI)])
        for _ in range(burn_in):
            s = step(s, dt, omega, e)
        spec = lyapunov_spectrum_finite_diff(torch.tensor(s, dtype=torch.float64), step_fn,
                                             T=T, dt=dt, eps=eps, diff_fn=_wrap_diff)
        rows.append(spec.tolist())
    arr = np.array(rows)
    ok = np.abs(arr[:, 1]) < zero_tol
    assert ok.sum() >= 3, f"only {ok.sum()} seeds passed the lambda_2~0 check:\n{arr}"
    med = np.median(arr[ok], axis=0)
    return med.tolist(), dict(per_seed=rows, accepted=ok.tolist(),
                              n_accepted=int(ok.sum()), std=arr[ok].std(0).tolist())


def validate_spectrum(spec, lam_max_ref=None, tol_zero=0.01, tol_max=0.02):
    """Two independent checks on a candidate true spectrum, mirroring Lorenz's Stage 1."""
    rep = {"zero_exponent": float(spec[1]), "zero_ok": bool(abs(spec[1]) < tol_zero),
           "lambda_max": float(spec[0]), "sum": float(sum(spec))}
    if lam_max_ref is not None:
        rep["lambda_max_ref_two_particle"] = float(lam_max_ref)
        rep["lambda_max_agrees"] = bool(abs(spec[0] - lam_max_ref) < tol_max)
    rep["ok"] = rep["zero_ok"] and rep.get("lambda_max_agrees", True)
    return rep["ok"], rep
