"""Correctness tests for the analytic Jacobian and the FTLE machinery.

The Jacobian test is the load-bearing one: everything downstream is a product of Jacobians,
so an error here is invisible later and contaminates every result.
"""
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path[:0] = [str(SRC), str(SRC / "models"), str(SRC / "geometry")]

from shplrnn import ShPLRNN          # noqa: E402
from jacobian import jacobian, batched_jacobian, on_hyperplane  # noqa: E402
from ftle import lyapunov_spectrum, finite_separation, divergence_with_horizon  # noqa: E402

torch.manual_seed(0)


def _model(d=8, H=8):
    torch.manual_seed(0)
    return ShPLRNN(d=d, H=H).double()


def test_jacobian_matches_autograd():
    m = _model()
    for trial in range(20):
        s = torch.randn(m.d, dtype=torch.float64) * 1.5
        if on_hyperplane(s, m, tol=1e-3):
            continue                                   # undefined exactly on a boundary
        J_analytic = jacobian(s, m)
        J_auto = torch.autograd.functional.jacobian(lambda x: m.step(x), s)
        err = (J_analytic - J_auto).abs().max().item()
        assert err < 1e-6, f"trial {trial}: analytic vs autograd mismatch {err:.2e}"


def test_batched_matches_single():
    m = _model()
    S = torch.randn(7, m.d, dtype=torch.float64)
    Jb = batched_jacobian(S, m)
    for i in range(S.shape[0]):
        assert (Jb[i] - jacobian(S[i], m)).abs().max() < 1e-10


def test_piecewise_constant_within_cell():
    """The Jacobian must be identical for two points in the same polyhedral cell."""
    m = _model()
    s = torch.randn(m.d, dtype=torch.float64)
    same, tries = 0, 0
    while same < 3 and tries < 500:
        tries += 1
        s2 = s + torch.randn(m.d, dtype=torch.float64) * 1e-4
        if m.cell_id(s) == m.cell_id(s2):
            assert (jacobian(s, m) - jacobian(s2, m)).abs().max() < 1e-12
            same += 1
    assert same >= 3, "could not find same-cell pairs to compare"


def test_spectrum_of_known_linear_system():
    """With all ReLU gates forced off, the map is exactly diag(A): spectrum = log|A|."""
    m = _model()
    with torch.no_grad():
        m.h2.fill_(-1e6)                                # gates always closed
        m.h1.zero_()
        m.A.copy_(torch.tensor([0.9, 0.8, 1.1, 0.5, 1.05, 0.7, 0.95, 1.2], dtype=torch.float64))
    # s0 tiny and T modest so the expanding modes (|A|>1) cannot grow the state far enough
    # to reopen the gates -- otherwise this stops being a linear system mid-test and the
    # "known answer" is no longer known.
    s0 = torch.full((m.d,), 1e-6, dtype=torch.float64)
    spec = lyapunov_spectrum(s0, m, T=50, dt=1.0)
    assert m.gate(s0).sum() == 0, "gates must stay closed for this to be the linear case"
    expected = torch.log(m.A.detach().abs()).sort(descending=True).values
    err = (spec.sort(descending=True).values - expected).abs().max().item()
    assert err < 1e-6, f"spectrum error {err:.2e}"


def test_spectrum_is_descending_and_full():
    m = _model()
    spec = lyapunov_spectrum(torch.randn(m.d, dtype=torch.float64), m, T=100)
    assert spec.shape == (m.d,), "must return the FULL spectrum, not just lambda_max"
    assert torch.all(spec[:-1] >= spec[1:] - 1e-12), "spectrum must be returned descending"


def test_finite_separation_saturates():
    m = _model()
    s0 = torch.randn(m.d, dtype=torch.float64)
    v, T = finite_separation(s0, m, T=50, eps=1e-3, cap=0.25)
    assert v <= 0.25 + 1e-12 and T == 50


def test_divergence_with_horizon_reports_every_T():
    m = _model()
    out = divergence_with_horizon(torch.randn(m.d, dtype=torch.float64), m, horizons=(5, 10))
    assert set(out) == {5, 10}
    assert all({"ftle", "finite_sep"} <= set(v) for v in out.values())


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS  {name}")
            except AssertionError as e:
                fails += 1; print(f"  FAIL  {name}: {e}")
    print("\nall geometry tests passed" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
