import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from action_branch_monitor import (boundary_refinement_alarm,
                                   cross_validated_smooth_vs_branch_alarm,
                                   shared_vs_branch_dynamics_alarm,
                                   smooth_vs_branch_alarm)


def probes(seed=2):
    return np.random.default_rng(seed).normal(size=(64, 4))


def trajectory(x, branch=False):
    time = np.linspace(0.0, 1.0, 12)
    smooth = np.stack([
        x[:, 0, None] * time,
        (0.4 * x[:, 1, None] ** 2 + 0.2 * x[:, 2, None]) * time ** 2,
    ], axis=2)
    if branch:
        side = (x[:, 0] > 0.15).astype(float)
        smooth[:, :, 1] += side[:, None] * (time[None] > 0.35) * 4.0
    return smooth


def test_smooth_nonlinear_response_is_not_a_branch():
    x = probes()
    y = trajectory(x)
    result = smooth_vs_branch_alarm(x, y[:, :7], y)
    assert not result.alarm


def test_connected_persistent_action_branch_alarms():
    x = probes()
    y = trajectory(x, branch=True)
    result = smooth_vs_branch_alarm(x, y[:, :7], y)
    assert result.alarm
    assert result.action_connected
    assert min(result.region_sizes) >= 2


def test_transient_branch_does_not_alarm():
    x = probes()
    full = trajectory(x, branch=True)
    early = trajectory(x)[:, :7]
    result = smooth_vs_branch_alarm(x, early, full)
    assert not result.alarm


def test_curved_action_boundary_is_allowed_but_charged():
    x = probes()
    time = np.linspace(0.0, 1.0, 12)
    side = np.sum(x[:, :2] ** 2, axis=1) < 1.2
    y = np.zeros((len(x), len(time), 2))
    y[:, :, 0] = x[:, 0, None] * time
    y[:, :, 1] = side[:, None] * (time[None] > 0.35) * 4.0
    result = smooth_vs_branch_alarm(x, y[:, :7], y)
    assert result.alarm
    assert result.partition_description_length > 0


def test_predictive_monitor_rejects_smooth_response():
    x = probes()
    y = trajectory(x)
    result = cross_validated_smooth_vs_branch_alarm(x, y[:, :7], y)
    assert not result.alarm
    assert result.folds == 8


def test_predictive_monitor_accepts_generalising_branch():
    x = probes()
    y = trajectory(x, branch=True)
    result = cross_validated_smooth_vs_branch_alarm(x, y[:, :7], y)
    assert result.alarm
    assert result.early_relative_improvement > 0
    assert result.full_relative_improvement > 0


def test_predictive_monitor_rejects_transient_branch():
    x = probes()
    full = trajectory(x, branch=True)
    early = trajectory(x)[:, :7]
    result = cross_validated_smooth_vs_branch_alarm(x, early, full)
    assert not result.alarm


def test_boundary_refinement_rejects_gap_that_decays_to_zero():
    decay = np.array([[1.0, .5, .25, .125, .0625, .03125],
                      [.8, .4, .2, .1, .05, .025],
                      [1.2, .6, .3, .15, .075, .0375]])
    result = boundary_refinement_alarm(decay, decay)
    assert not result.alarm
    assert result.early_branch_pairs == 0


def test_boundary_refinement_accepts_persistent_plateau():
    levels = np.arange(6)
    plateau = np.stack([.7 + scale * .5 ** levels for scale in (1.0, .8, 1.2)])
    result = boundary_refinement_alarm(plateau, plateau)
    assert result.alarm
    assert result.early_branch_pairs == 3


def test_boundary_refinement_requires_early_and_late_persistence():
    levels = np.arange(6)
    decay = np.stack([scale * .5 ** levels for scale in (1.0, .8, 1.2)])
    plateau = np.stack([.7 + scale * .5 ** levels for scale in (1.0, .8, 1.2)])
    assert not boundary_refinement_alarm(decay, plateau).alarm


def dynamics_pair(kind, static_offset=0., steps=38):
    """Synthetic pose/velocity traces with an action endpoint at index seven."""
    pair = np.zeros((2, steps, 45))
    pair[1, :, 0] = static_offset
    hold = np.arange(steps - 7, dtype=float)
    shared = .2 * (1. - np.exp(-hold / 7.))
    pair[:, 7:, 0] += shared
    pair[:, 7:, 27] = np.gradient(shared)
    if kind == "different":
        branch = .015 * hold ** 1.5
        pair[1, 7:, 1] += branch
        pair[1, 7:, 28] = np.gradient(branch)
    elif kind == "transient":
        branch = .3 * np.sin(np.pi * np.minimum(hold, 10.) / 10.)
        branch[hold > 10] = 0.
        pair[1, 7:, 1] += branch
        pair[1, 7:, 28] = np.gradient(branch)
    return pair


def test_dynamics_monitor_rejects_static_offset_with_shared_evolution():
    pairs = np.stack([dynamics_pair("shared", static_offset=value)
                      for value in (.05, .1, .2)])
    assert not shared_vs_branch_dynamics_alarm(pairs).alarm


def test_dynamics_monitor_accepts_persistent_different_laws():
    pairs = np.stack([dynamics_pair("different", static_offset=value)
                      for value in (.02, .04, .06)])
    result = shared_vs_branch_dynamics_alarm(pairs)
    assert result.alarm
    assert result.persistent_branch_pairs >= 2


def test_dynamics_monitor_rejects_early_difference_that_converges():
    pairs = np.stack([dynamics_pair("transient", static_offset=value)
                      for value in (.02, .04, .06)])
    assert not shared_vs_branch_dynamics_alarm(pairs).alarm
