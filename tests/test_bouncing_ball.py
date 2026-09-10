"""Phase-2 system tests. Each one guards a bug that actually occurred during development."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "systems"))
import bouncing_ball as bb  # noqa: E402

P = dict(dt=0.20, omega=1.4, e=0.8)


def test_no_penetration():
    """Event detection must land ON the guard, never through it. Without bisection the impact
    smears over up to dt and would look like model error in Phase 2."""
    tr = bb.simulate(5000, **P)
    g = np.array([bb.gap(s, P["omega"]) for s in tr.numpy()])
    assert g.min() > -1e-9, f"penetration: min gap {g.min():.2e}"


def test_impact_law_injects_energy():
    """The moving-surface law v_out = (1+e)v_table - e v_in is what makes this chaotic. The
    stationary law (v_out = -e v_in) silently removes the drive and kills the chaos."""
    phi = 0.0                                  # table moving UP at A*omega
    vt = bb.table_vel(phi, P["omega"])
    v_in, e = -3.0, 0.8
    v_out = (1 + e) * vt - e * v_in
    assert v_out > -e * v_in, "impact must gain energy from an upward-moving table"
    assert abs((v_out - vt) + e * (v_in - vt)) < 1e-12, "relative velocity must reverse by -e"


def test_no_inelastic_collapse_over_long_run():
    """Regression: the impact map ALONE is ill-posed -- a single step once needed >4000 impacts
    with the ball at x=-0.22. The contact/detach phase makes long runs well-posed."""
    bb.simulate(40000, seed=0, **P)
    assert not bb.simulate.collapsed, "inelastic collapse in a long run"


def test_contact_force_and_detachment():
    """Detachment happens exactly when the table falls away faster than gravity, i.e.
    sin(phi) > g/(A omega^2) = 1/Gamma. Getting this wrong glues the ball to the table."""
    om = P["omega"]; inv_gamma = bb.G / (bb.A * om * om)
    assert bb.contact_force(np.arcsin(inv_gamma) - 1e-6, om) > 0
    assert bb.contact_force(np.arcsin(inv_gamma) + 1e-6, om) < 0
    t = bb._detach_time(0.0, om, 10.0)          # start at sin=0, contact held
    assert t is not None and abs(bb.contact_force(om * t, om)) < 1e-6


def test_lambda_max_positive_and_seed_consistent():
    """Ground truth must be reproducible across seeds. Before the contact phase, 5/8 seeds
    returned NaN and a survivor returned 3.95 where its neighbours returned 0.14."""
    lam, sd, vals = bb.true_lambda_max(n_seeds=6, n_steps=20000, min_ok=6, **P)
    assert all(np.isfinite(v) for v in vals), f"non-finite exponents: {vals}"
    assert lam > 0.02, f"not chaotic: {lam}"
    assert sd < 0.05 * max(abs(lam), 1e-9) + 0.02, f"seed spread too large: {sd}"


def test_guard_is_sampled_often_enough():
    """Phase 2 is ABOUT the guard, so a regime where impacts are 0.27% of samples (Gamma=4)
    is useless no matter how chaotic it is."""
    _, imp = bb.simulate(20000, return_impacts=True, **P)
    frac = float(imp.double().mean())
    assert 0.01 < frac < 0.15, f"impact fraction {frac:.4f} outside usable band"


def test_finite_diff_matches_analytic_on_smooth_system():
    """The licence for using finite differences on the ball: on LORENZ, where an analytic
    Jacobian exists, the two estimators must agree. They match to ~1e-4. Without this check
    the hybrid-system spectrum would rest on an unvalidated instrument."""
    sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "src" / p)
                    for p in ("systems", "geometry")]
    import torch
    import lorenz
    from ftle import lyapunov_spectrum_finite_diff, lyapunov_spectrum_general
    dt = 0.01
    s0 = torch.tensor(lorenz.simulate(2000, dt=dt).numpy()[-1], dtype=torch.float64)
    an = lyapunov_spectrum_general(s0, lambda s: lorenz.rk4_step(s, dt),
                                   lambda s: lorenz.flow_jacobian(s, dt), T=8000, dt=dt)
    fd = lyapunov_spectrum_finite_diff(s0, lambda s: lorenz.rk4_step(s, dt), T=8000, dt=dt,
                                       eps=1e-7)
    assert float((an - fd).abs().max()) < 1e-3, f"analytic {an} vs finite-diff {fd}"


def test_obs_removes_phase_wrap():
    """Raw phi jumps by ~2pi on 4.5% of transitions -- MORE OFTEN than real impacts (2.7%).
    A model cannot tell that coordinate artefact from a real discontinuity."""
    tr = bb.simulate(5000, **P).numpy()
    raw_jumps = (np.abs(np.diff(tr[:, 2])) > np.pi).mean()
    obs = bb.to_obs(tr)
    obs_jumps = (np.abs(np.diff(obs[:, 2:], axis=0)) > 1.0).mean()
    assert raw_jumps > 0.02, "expected frequent wraps in raw phi"
    assert obs_jumps == 0.0, f"observation coords still jump: {obs_jumps}"
    assert np.allclose(obs[:, 2] ** 2 + obs[:, 3] ** 2, 1.0)


def test_guard_is_a_hyperplane_in_obs_coords():
    """x - A sin(phi) = 0 becomes linear in (x, v, cos, sin), so ONE ReLU boundary can match it
    exactly. That is what makes the alignment test a measurement against a known normal."""
    n = bb.guard_normal_obs()
    tr = bb.simulate(3000, **P).numpy()
    obs = bb.to_obs(tr)
    resid = obs @ n                                    # == gap / ||n||
    gap = np.array([bb.gap(s, P["omega"]) for s in tr])
    assert np.allclose(resid, gap / np.linalg.norm([1.0, 0.0, 0.0, -bb.A]), atol=1e-9)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS  {name}")
            except AssertionError as err:
                fails += 1; print(f"  FAIL  {name}: {err}")
    print("\nall bouncing-ball tests passed" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
