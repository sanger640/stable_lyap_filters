from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import monitor_v1_eval as mv  # noqa: E402


def test_trace_endings_use_hold_10_and_30_and_millimetres():
    traces = np.zeros((64, 38, 45))
    traces[:, 17, 0] = 0.001
    traces[:, 37, 0] = 0.002
    early, late = mv.trace_endings(traces, slice(0, 9))
    assert np.allclose(early[:, 0], 1.0) and np.allclose(late[:, 0], 2.0)


def test_two_separated_persistent_groups_alarm_and_a_smooth_spread_does_not():
    rng = np.random.default_rng(0)
    groups = np.r_[np.zeros(54), np.ones(10)]
    split = np.c_[groups * 40.0 + rng.normal(0, 0.5, 64), rng.normal(0, 0.5, (64, 8))]
    assert mv.decide(split, split + rng.normal(0, 0.1, split.shape))[0]
    smooth = np.c_[np.linspace(-5, 5, 64), rng.normal(0, 0.5, (64, 8))]
    assert not mv.decide(smooth, smooth)[0]


def test_pusher_is_excluded_on_u1_pushing():
    assert mv.NON_ROBOT["u1_pushing"] == slice(3, 9)
