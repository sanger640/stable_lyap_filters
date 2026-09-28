import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from consequence_monitor import (amplification_delta_bic,
                                 boundary_amplification_persistence_alarm,
                                 multiresolution_consequence_alarm,
                                 persistence_delta_bic, separation_curves,
                                 whole_trajectory_consequence_alarm)


def physical_pair(kind, amplitude=1., steps=38):
    """Anonymous pose/velocity mechanism with H=8 ending at trace index seven."""
    pair = np.zeros((2, steps, 45), float)
    hold = np.arange(steps - 7, dtype=float)
    if kind == "static":
        pair[1, :, 0] = amplitude
    elif kind == "smooth_shrinking":
        pair[1, 7:, 0] = amplitude * (1. - np.exp(-hold / 4.))
    elif kind == "wobble":
        response = amplitude * np.sin(np.pi * np.minimum(hold, 20.) / 20.)
        response[hold > 20] = 0.
        pair[1, 7:, 0] = response
        pair[1, 7:, 27] = np.gradient(response)
    elif kind == "diverge":
        response = amplitude * (hold / 30.) ** 2
        pair[1, 7:, 0] = response
        pair[1, 7:, 27] = np.gradient(response)
    elif kind == "stick_slip":
        response = amplitude * np.maximum(hold - 4., 0.) / 26.
        pair[1, 7:, 1] = response
        pair[1, 7:, 28] = np.gradient(response)
    elif kind == "free_flight":
        response = -.5 * amplitude * (hold / 30.) ** 2
        pair[1, 7:, 2] = response
        pair[1, 7:, 29] = np.gradient(response)
    else:
        raise ValueError(kind)
    return pair


def decision(kind, boundary=True):
    pairs = np.stack([physical_pair(kind, amplitude) for amplitude in (.8, 1., 1.2)])
    evidence = np.ones(3) if boundary else -np.ones(3)
    return boundary_amplification_persistence_alarm(pairs, evidence, evidence)


def test_static_offset_produces_zero_effect_curve_and_is_rejected():
    pairs = np.stack([physical_pair("static", value) for value in (.1, 1., 10.)])
    assert np.allclose(separation_curves(pairs), 0.)
    assert not decision("static").alarm


def test_smooth_response_without_boundary_is_rejected():
    result = decision("smooth_shrinking", boundary=False)
    assert not result.alarm
    assert result.boundary_pairs == 0


def test_large_recoverable_wobble_is_not_persistent():
    curve = separation_curves(np.stack([physical_pair("wobble")]))[0]
    assert persistence_delta_bic(curve) < 0
    assert not decision("wobble").alarm


def test_sustained_divergence_passes():
    result = decision("diverge")
    assert result.alarm
    assert result.consequential_pairs == 3


def test_stick_slip_passes():
    assert decision("stick_slip").alarm


def test_supported_free_flight_passes():
    assert decision("free_flight").alarm


def whole_pair(kind, amplitude=1., steps=38):
    pair = np.zeros((2, steps, 45), float)
    t = np.arange(1, steps + 1, dtype=float)
    exposure = np.minimum(t, 8.) / 8.
    if kind == "passive":
        response = amplitude * exposure
    elif kind == "recover":
        response = amplitude * exposure * np.exp(-np.maximum(t - 8., 0.) / 4.)
    elif kind == "commit":
        # A physically realizable committed transition rises over several control steps rather
        # than teleporting configuration in one sample.
        response = amplitude * exposure + 6. * amplitude * (
            1. - np.exp(-np.maximum(t - 4., 0.) / 3.))
    elif kind == "stick_slip":
        response = amplitude * exposure + 4. * amplitude * (
            1. - np.exp(-np.maximum(t - 4., 0.) / 2.))
    elif kind == "free_flight":
        response = amplitude * exposure + amplitude * np.maximum(t - 6., 0.) ** 2 / 20.
    else:
        raise ValueError(kind)
    pair[1, :, 0] = response
    pair[1, :, 27] = np.gradient(response)
    return pair


def whole_decision(kind, boundary=True):
    pairs = np.stack([whole_pair(kind, value) for value in (.8, 1., 1.2)])
    action = np.ones((3, 8, 3)); action[:, :, 1:] = 0.
    evidence = np.ones(3) if boundary else -np.ones(3)
    return whole_trajectory_consequence_alarm(pairs, action, evidence, evidence)


def test_whole_trajectory_rejects_smooth_passive_static_displacement():
    assert not whole_decision("passive").alarm


def test_whole_trajectory_rejects_recoverable_forced_response():
    assert not whole_decision("recover").alarm


def test_whole_trajectory_accepts_during_action_commitment():
    assert whole_decision("commit").alarm


def test_whole_trajectory_accepts_stick_slip_commitment():
    assert whole_decision("stick_slip").alarm


def test_whole_trajectory_accepts_supported_free_flight_commitment():
    assert whole_decision("free_flight").alarm


def multiresolution_inputs(stable=True):
    pairs, actions = [], []
    for amplitude in (.8, 1., 1.2):
        levels, level_actions = [], []
        for level in range(6):
            width = .5 ** level
            if stable:
                # A smooth width-dependent component vanishes around one fixed commitment shape.
                trace = whole_pair("stick_slip", amplitude)
                trace[1, :, 0] += width * np.minimum(np.arange(1, 39), 8.) / 8.
            else:
                # Alternating commitment mechanisms do not approach one local limiting response.
                kind = "stick_slip" if level % 2 == 0 else "free_flight"
                trace = whole_pair(kind, amplitude)
            levels.append(trace)
            action = np.ones((8, 3)) * width
            action[:, 1:] = 0.
            level_actions.append(action)
        pairs.append(levels); actions.append(level_actions)
    return np.asarray(pairs), np.asarray(actions)


def test_multiresolution_accepts_stable_commitment_limit():
    traces, actions = multiresolution_inputs(True)
    result = multiresolution_consequence_alarm(traces, actions, np.ones(3), np.ones(3))
    assert result.alarm
    assert result.consequential_pairs == 3


def test_multiresolution_rejects_nonconvergent_commitment_shape():
    traces, actions = multiresolution_inputs(False)
    result = multiresolution_consequence_alarm(traces, actions, np.ones(3), np.ones(3))
    assert not result.alarm
