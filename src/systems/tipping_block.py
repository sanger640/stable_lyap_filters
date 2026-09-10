"""
Rocking / tipping block — Phase 3's system, and the first one that models a FAILURE.

Why this replaces the bouncing ball. The ball had a genuine guard, but no unsafe region: every
trajectory lives on one attractor, "divergence" is recurrent, nothing is destroyed, and
separation saturates at the attractor scale and then OSCILLATES (measured: swings down to 17%
of its own max). That last property wrecked every detection experiment, because two trajectories
that had diverged kept coming back together.

    Jenga                              bouncing ball          tipping block
    block topples and STAYS toppled    recurrent              absorbing        <-
    irreversible                       fully reversible       irreversible     <-
    rare, catastrophic                 constant (2.7%)        rare, tunable    <-
    safe/unsafe are different regions  one attractor          two outcomes     <-

This is the classical rocking block (Housner 1963), which is the standard model for exactly the
Jenga question: how hard can you push a free-standing block before it goes over?

    theta      tilt; 0 = flat on the ground, sign = which corner it pivots on
    alpha      slenderness, atan(half-width / half-height). |theta| = alpha is the point where
               the centre of mass passes over the pivot -- past it, gravity TOPPLES rather than
               restores, and the block is gone.

    rocking about a corner (theta > 0), force F applied horizontally at the centre of mass:

        I_O thetadd = F R cos(alpha - theta) - m g R sin(alpha - theta),   I_O = (4/3) m R^2

    GUARD 1  theta = 0     the block slams onto its base and starts rocking the other way,
                           with angular velocity scaled by Housner's e_r = 1 - 1.5 sin^2(alpha)
    GUARD 2  |theta| = alpha   TOPPLE. Absorbing: it falls to lying flat and stays there.

Units: m = g = R = 1, so the only shape parameter is alpha.
"""
import numpy as np

G, R_, M = 1.0, 1.0, 1.0
ALPHA_DEFAULT = 0.35            # ~20 deg: slender enough to topple, squat enough to rock
FALLEN_ANGLE = np.pi / 2        # lying flat -- the absorbing state
DT_DEFAULT = 0.02


def restitution(alpha):
    """Housner's angular-velocity ratio at a base impact. Energy ratio is its square.
    Non-negative only for alpha <= 54.7 deg; beyond that the model does not apply."""
    e = 1.0 - 1.5 * np.sin(alpha) ** 2
    assert e >= 0.0, f"Housner restitution negative at alpha={alpha}: block too squat"
    return e


def critical_force(alpha=ALPHA_DEFAULT):
    """Static push (at the CoM) needed to start the block rocking: F > g tan(alpha).
    Below this it simply does not move, which is the 'safe by construction' regime."""
    return G * np.tan(alpha)


def toppled(state, alpha=ALPHA_DEFAULT):
    """Past the critical angle the CoM is beyond the pivot and the fall is irreversible."""
    return abs(state[0]) >= alpha


def is_fallen(state):
    return abs(state[0]) >= FALLEN_ANGLE - 1e-9


def _accel(theta, F, alpha):
    """Angular acceleration about whichever corner is currently the pivot."""
    c = 3.0 / (4.0 * R_)                       # = m R / I_O with I_O = (4/3) m R^2
    if theta >= 0.0:                           # pivot = right corner
        return c * (F * np.cos(alpha - theta) - G * np.sin(alpha - theta))
    return c * (F * np.cos(alpha + theta) + G * np.sin(alpha + theta))


def step(state, F, dt=DT_DEFAULT, alpha=ALPHA_DEFAULT, tol=1e-12):
    """One step of the hybrid dynamics, resolving both guards exactly.

    Returns the new (theta, theta_dot). Once the block is lying flat it is FROZEN -- that is the
    whole point of this system: unlike the bouncing ball, a failed trajectory never returns, so
    the separation between a failed and a surviving rollout cannot oscillate back to zero."""
    s = np.asarray(state, dtype=np.float64).copy()

    if is_fallen(s):                                   # absorbing
        return np.array([np.sign(s[0]) * FALLEN_ANGLE, 0.0])

    # at rest on the base and the push is too weak to lift a corner -> nothing happens
    if abs(s[0]) < tol and abs(s[1]) < tol and abs(F) <= critical_force(alpha):
        return np.array([0.0, 0.0])

    th, om = s
    a1 = _accel(th, F, alpha)                          # semi-implicit Euler (symplectic-ish)
    om_new = om + a1 * dt
    th_new = th + om_new * dt

    if toppled(np.array([th_new, om_new]), alpha):
        # Past the critical angle gravity drives the fall; run it out to lying flat and freeze.
        sgn = np.sign(th_new if th_new != 0 else om_new)
        return np.array([sgn * FALLEN_ANGLE, 0.0])

    if th * th_new < 0.0:                              # crossed theta = 0: base impact
        lo, hi = 0.0, dt                               # bisect for the crossing time
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            om_m = om + a1 * mid
            th_m = th + om_m * mid
            if th * th_m >= 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-14:
                break
        om_at = om + a1 * lo
        om_after = restitution(alpha) * om_at          # energy lost to the slam
        rem = dt - lo
        a2 = _accel(0.0 if om_after >= 0 else -0.0, F, alpha)
        om_new = om_after + a2 * rem
        th_new = om_new * rem
        if toppled(np.array([th_new, om_new]), alpha):
            return np.array([np.sign(th_new) * FALLEN_ANGLE, 0.0])

    # settled: at rest on the base with too little push to lift off again
    if abs(th_new) < 1e-4 and abs(om_new) < 1e-3 and abs(F) <= critical_force(alpha):
        return np.array([0.0, 0.0])
    return np.array([th_new, om_new])


def simulate(actions, s0=None, dt=DT_DEFAULT, alpha=ALPHA_DEFAULT):
    """Roll an action (force) sequence. Returns states (n+1, 2) and a per-step fallen flag."""
    s = np.array([0.0, 0.0]) if s0 is None else np.asarray(s0, dtype=np.float64).copy()
    n = len(actions)
    out = np.empty((n + 1, 2)); fell = np.zeros(n + 1, dtype=bool)
    out[0], fell[0] = s, is_fallen(s)
    for i, F in enumerate(actions):
        s = step(s, float(F), dt, alpha)
        out[i + 1], fell[i + 1] = s, is_fallen(s)
    return out, fell


def push_profile(n, amp, t_on=10, t_off=None, dt=DT_DEFAULT):
    """A simple 'gripper nudge': a rectangular push of height `amp`. The single scalar `amp`
    is what makes safe/unsafe a clean one-parameter question."""
    a = np.zeros(n)
    a[t_on:(n if t_off is None else t_off)] = amp
    return a


def topple_threshold(n, t_on=10, t_off=60, dt=DT_DEFAULT, alpha=ALPHA_DEFAULT,
                     lo=0.0, hi=3.0, iters=60):
    """Bisect for the push amplitude that just topples the block.

    This is the ground truth the whole toy exists to provide: an exact, sharp boundary in
    ACTION space between a safe push and an unsafe one -- which the bouncing ball never had."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, fell = simulate(push_profile(n, mid, t_on, t_off, dt), dt=dt, alpha=alpha)
        if fell[-1]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
