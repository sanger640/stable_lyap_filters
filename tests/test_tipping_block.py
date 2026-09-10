"""Tipping block tests. The first four guard the properties the bouncing ball LACKED."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "systems"))
import tipping_block as tb  # noqa: E402

N, T_ON, T_OFF = 300, 10, 60


def _thr():
    return tb.topple_threshold(N, T_ON, T_OFF)


def test_failure_is_absorbing():
    """THE property this system exists for. In the bouncing ball a diverged pair came back
    together (separation fell to 17% of its max), which made every detection test ambiguous.
    Once this block is over it is over.

    Note the block keeps MOVING after the flag fires -- it is falling, not frozen. What must
    hold is that it never comes back: the flag is monotone, |theta| only grows until the block
    is flat, and it is frozen from then on."""
    S, fell = tb.simulate(tb.push_profile(N, _thr() * 1.1, T_ON, T_OFF))
    i = int(np.argmax(fell))
    assert fell[i:].all(), "toppled flag turned back off"
    assert (np.diff(np.abs(S[i:, 0])) >= -1e-12).all(), "block rotated back toward upright"
    flat = np.abs(S[:, 0]) >= tb.FALLEN_ANGLE - 1e-9
    j = int(np.argmax(flat))
    assert flat.any() and np.allclose(S[j:, 0], S[j, 0]), "not frozen after landing"


def test_safe_and_failed_never_reconverge():
    thr = _thr()
    Ss, _ = tb.simulate(tb.push_profile(N, thr * 0.95, T_ON, T_OFF))
    Su, fu = tb.simulate(tb.push_profile(N, thr * 1.05, T_ON, T_OFF))
    d = np.abs(Ss[:, 0] - Su[:, 0])
    # Right at the crossing the two are still CLOSE (one is at alpha, the other just below);
    # the gap opens as the block falls. The claim is that it never closes again.
    j = int(np.argmax(np.abs(Su[:, 0]) >= tb.FALLEN_ANGLE - 1e-9))
    assert d[j:].min() > 0.5, "separation collapsed after the block landed"
    assert d[-1] > 1.0, f"final separation only {d[-1]:.3f}"


def test_boundary_is_sharp_in_action_space():
    """An exact safe/unsafe boundary in ACTION space -- the ground truth the bouncing ball
    could never provide, because it had no unsafe region at all."""
    thr = _thr()
    assert not tb.simulate(tb.push_profile(N, thr * 0.999, T_ON, T_OFF))[1][-1]
    assert tb.simulate(tb.push_profile(N, thr * 1.001, T_ON, T_OFF))[1][-1]


def test_weak_push_does_not_move_the_block():
    """Below g*tan(alpha) no corner lifts, so the block is safe by statics, not by luck."""
    S, fell = tb.simulate(tb.push_profile(N, tb.critical_force() * 0.9, T_ON, T_OFF))
    assert not fell[-1] and np.abs(S[:, 0]).max() < 1e-9


def test_rocking_loses_energy_and_settles():
    """Base impacts must dissipate (Housner e_r < 1), otherwise the block rocks forever and
    'safe' would never be a settled state."""
    S, fell = tb.simulate(tb.push_profile(N, _thr() * 0.8, T_ON, T_OFF))
    assert not fell[-1]
    peaks = np.abs(S[:, 0])
    assert peaks[200:].max() < peaks[:200].max(), "rocking amplitude did not decay"
    assert 0.0 < tb.restitution(tb.ALPHA_DEFAULT) < 1.0


def test_fall_is_integrated_not_teleported():
    """Regression: step() used to jump straight from |theta|>=alpha to lying flat in ONE step,
    which both looked wrong and erased the fall dynamics a world model would have to learn.
    The fall must take real time and accelerate."""
    S, fell = tb.simulate(tb.push_profile(N, _thr() * 1.05, T_ON, T_OFF))
    i = int(np.argmax(fell))
    flat = np.abs(S[:, 0]) >= tb.FALLEN_ANGLE - 1e-9
    assert flat.any(), "block never reached the ground"
    j = int(np.argmax(flat))
    assert j - i > 20, f"fall took only {j-i} steps -- teleporting again?"
    d = np.diff(np.abs(S[i:j + 1, 0]))
    assert d[-1] > d[0], "fall did not accelerate under gravity"


def test_critical_angle_matches_geometry():
    """|theta| = alpha is exactly where the CoM crosses the pivot -- check against geometry."""
    a = tb.ALPHA_DEFAULT
    piv = np.array([np.sin(a), 0.0])
    c, s = np.cos(-a), np.sin(-a)
    com = (np.array([0.0, np.cos(a)]) - piv) @ np.array([[c, -s], [s, c]]).T + piv
    assert abs(com[0] - piv[0]) < 1e-12, "CoM is not above the pivot at theta = alpha"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS  {name}")
            except AssertionError as err:
                fails += 1; print(f"  FAIL  {name}: {err}")
    print("\nall tipping-block tests passed" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
