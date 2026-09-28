"""Calibration-free structural test for an amplified, persistent action boundary.

The monitor operates on refined pairs of nearby counterfactual trajectories.  It asks three
separate questions without using task labels or a reference population: does the response retain a
finite local action boundary, does the physical difference grow after the differing actions stop,
and does that growth persist rather than reconverge?
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares, nnls


DECAY_TIMES = (.1, .25, .5, 1., 2.)


@dataclass(frozen=True)
class ConsequenceEvidence:
    alarm: bool
    consequential_pairs: int
    pairs: int
    boundary_pairs: int
    amplification_pairs: int
    persistence_pairs: int
    amplification_delta_bic: tuple
    persistence_delta_bic: tuple
    curves: tuple


@dataclass(frozen=True)
class WholeTrajectoryConsequenceEvidence:
    """Evidence for a boundary that commits during the action or immediate response."""
    alarm: bool
    consequential_pairs: int
    pairs: int
    boundary_pairs: int
    commitment_pairs: int
    persistence_pairs: int
    commitment_delta_bic: tuple
    persistence_delta_bic: tuple
    commitment_onsets: tuple
    curves: tuple


@dataclass(frozen=True)
class MultiresolutionConsequenceEvidence:
    """Evidence that commitment converges to a stable local boundary as width shrinks."""
    alarm: bool
    consequential_pairs: int
    pairs: int
    boundary_pairs: int
    stable_commitment_pairs: int
    persistence_pairs: int
    shape_convergence_pairs: int
    commitment_delta_bic_by_level: tuple
    commitment_onsets_by_level: tuple
    persistence_delta_bic: tuple
    shape_convergence_delta_bic: tuple


def _bic(residual, parameters, scale):
    residual = np.asarray(residual, float)
    n = len(residual)
    # Relative numerical floor only; multiplying every observation by any positive constant leaves
    # all BIC differences unchanged. This avoids turning simulator roundoff into infinite evidence.
    floor = 1e-12 * max(float(scale * scale), 1e-24)
    rss = max(float(np.sum(residual ** 2)), n * floor)
    return float(n * np.log(rss / n) + parameters * np.log(n))


def _best_amplification_bic(curve, growing):
    """Constant/decay null, optionally augmented by sustained nonnegative growth."""
    values = np.asarray(curve, float)
    t = np.linspace(0., 1., len(values))
    scale = max(float(np.sqrt(np.mean(values ** 2))), 1e-12)
    best = np.inf
    for decay_time in DECAY_TIMES:
        columns = [np.ones_like(t), np.exp(-t / decay_time)]
        if growing:
            columns.append(t ** 2)
        design = np.stack(columns, axis=1)
        coefficients, _ = nnls(design, values)
        best = min(best, _bic(values - design @ coefficients, len(columns), scale))
    return float(best)


def amplification_delta_bic(curve):
    """Positive means sustained post-action growth earns its extra parameter."""
    return _best_amplification_bic(curve, False) - _best_amplification_bic(curve, True)


def _best_persistence_bic(curve, persistent, late_start=10):
    """Late decay-to-zero null versus a decay/growth curve with a nonzero asymptote."""
    values = np.asarray(curve, float)[late_start:]
    if len(values) < 5:
        raise ValueError("persistence interval must contain at least five samples")
    t = np.linspace(0., 1., len(values))
    scale = max(float(np.sqrt(np.mean(values ** 2))), 1e-12)
    best = np.inf
    for decay_time in DECAY_TIMES:
        decay = np.exp(-t / decay_time)
        if not persistent:
            coefficient, _ = nnls(decay[:, None], values)
            residual = values - decay * coefficient[0]
            score = _bic(residual, 1, scale)
        else:
            # c + a exp(-t/tau): c>=0, while a may be positive (settling to a plateau) or
            # negative (continuing to approach one). Both are persistent late responses.
            design = np.stack([np.ones_like(t), decay], axis=1)
            fit = least_squares(lambda p: values - design @ p, [values[-1], values[0]-values[-1]],
                                bounds=([0., -np.inf], [np.inf, np.inf]))
            score = _bic(fit.fun, 2, scale)
        best = min(best, score)
    return float(best)


def persistence_delta_bic(curve, late_start=10):
    """Positive means a persistent late component earns its extra parameter."""
    return (_best_persistence_bic(curve, False, late_start)
            - _best_persistence_bic(curve, True, late_start))


def separation_curves(pair_traces, action_end=7):
    """Return internally normalised post-action difference-of-displacement curves.

    Each endpoint's state at H=8 is its own origin, so an already-existing static offset contributes
    exactly zero. Four generic channel families (position, rotation, linear and angular velocity)
    receive equal total weight using only variation inside the current probe set. This is invariant
    to measurement units and uses neither quiet examples nor task outcomes.
    """
    traces = np.asarray(pair_traces, float)
    if traces.ndim != 4 or traces.shape[1] != 2 or traces.shape[3] < 45:
        raise ValueError("pair_traces must have shape (pairs, 2, time, at least 45 fields)")
    state = traces[:, :, :, :45].copy()
    groups = (slice(0, 9), slice(9, 27), slice(27, 36), slice(36, 45))
    post = state[:, :, action_end:]
    relative = post.copy()
    # Static offsets are meaningful only for configuration. Velocity is already an instantaneous
    # rate: subtracting its H=8 value would turn ordinary damping to rest into a permanent gap.
    relative[..., :27] -= post[:, :, :1, :27]
    normalised = np.zeros_like(relative)
    for group in groups:
        values = relative[..., group]
        # One scalar per generic family preserves its internal geometry. A relative floor prevents
        # a perfectly still family from being promoted through division by roundoff.
        rms = float(np.sqrt(np.mean(values ** 2)))
        global_rms = float(np.sqrt(np.mean(relative ** 2)))
        if rms > max(global_rms, 1.) * 1e-10:
            normalised[..., group] = values / rms
    difference = normalised[:, 0] - normalised[:, 1]
    return np.linalg.norm(difference, axis=2)


def whole_trajectory_separation_curves(pair_traces):
    """Separation from the common pre-action state over all H=8 plus hold steps.

    The pre-action difference is exactly zero and is prepended explicitly because dense replay
    starts after the first perturbed command. Generic channel families are internally normalised,
    making the curve invariant to measurement units and a common rescaling of a representation.
    """
    traces = np.asarray(pair_traces, float)
    if traces.ndim != 4 or traces.shape[1] != 2 or traces.shape[3] < 45:
        raise ValueError("pair_traces must have shape (pairs, 2, time, at least 45 fields)")
    difference = traces[:, 0, :, :45] - traces[:, 1, :, :45]
    groups = (slice(0, 9), slice(9, 27), slice(27, 36), slice(36, 45))
    normalised = np.zeros_like(difference)
    global_rms = float(np.sqrt(np.mean(difference ** 2)))
    for group in groups:
        values = difference[..., group]
        rms = float(np.sqrt(np.mean(values ** 2)))
        if rms > max(global_rms, 1.) * 1e-10:
            normalised[..., group] = values / rms
    curves = np.linalg.norm(normalised, axis=2)
    return np.concatenate([np.zeros((len(curves), 1)), curves], axis=1)


def _action_exposure(action_difference, total_steps):
    delta = np.asarray(action_difference, float).reshape(len(action_difference), -1)
    energy = np.sum(delta ** 2, axis=1)
    cumulative = np.sqrt(np.r_[0., np.cumsum(energy)])
    if cumulative[-1] <= 1e-15:
        raise ValueError("refined action endpoints are identical")
    cumulative /= cumulative[-1]
    if total_steps < len(cumulative):
        raise ValueError("trajectory is shorter than the action sequence")
    return np.r_[cumulative, np.ones(total_steps - len(cumulative))]


def _best_passive_bic(curve, exposure, committed=False, horizon=8, immediate_hold=5):
    """Smooth forced response, optionally plus a searched autonomous commitment event."""
    values = np.asarray(curve, float)
    q = np.asarray(exposure, float)
    if values.shape != q.shape:
        raise ValueError("curve and action exposure must have matching lengths")
    time = np.arange(len(values), dtype=float)
    post = np.maximum(time - horizon, 0.)
    scale = max(float(np.sqrt(np.mean(values ** 2))), 1e-12)
    null_scores = []
    designs = []
    # q and q^2 allow ordinary nonlinear accumulation during the action. The decaying copy allows
    # a forced transient to relax after H while q itself supplies a passive static displacement.
    for decay_time in (2., 5., 10., 20.):
        design = np.stack([q, q ** 2, q * np.exp(-post / decay_time)], axis=1)
        coefficients, _ = nnls(design, values)
        null_scores.append(_bic(values - design @ coefficients, 3, scale))
        designs.append(design)
    if not committed:
        return float(min(null_scores)), None

    best_score, best_onset = np.inf, None
    candidates = []
    # An outcome may become committed during the perturbation or in the immediate five-step
    # response. Multiple fixed rise shapes cover abrupt contact switches and gradual falls.
    for onset in range(1, horizon + immediate_hold + 1):
        elapsed = np.maximum(time - onset, 0.)
        candidates.extend((onset, shape) for shape in (
            (elapsed > 0).astype(float),
            1. - np.exp(-elapsed / 2.),
            1. - np.exp(-elapsed / 5.)))
    search_charge = 2. * np.log(len(candidates))
    for design in designs:
        for onset, event in candidates:
            augmented = np.column_stack([design, event])
            coefficients, _ = nnls(augmented, values)
            score = _bic(values - augmented @ coefficients, 4, scale) + search_charge
            if score < best_score:
                best_score, best_onset = score, onset
    return float(best_score), int(best_onset)


def commitment_delta_bic(curve, action_difference, horizon=8, immediate_hold=5):
    """Positive evidence means cumulative action exposure cannot explain a committed response."""
    exposure = _action_exposure(action_difference, len(curve))
    passive, _ = _best_passive_bic(curve, exposure, False, horizon, immediate_hold)
    committed, onset = _best_passive_bic(curve, exposure, True, horizon, immediate_hold)
    return float(passive - committed), onset


def whole_trajectory_consequence_alarm(pair_traces, action_differences,
                                       early_boundary_bic, full_boundary_bic,
                                       horizon=8, immediate_hold=5, late_hold=10):
    """Require local scaling, action-to-consequence commitment, and late persistence."""
    curves = whole_trajectory_separation_curves(pair_traces)
    actions = np.asarray(action_differences, float)
    early = np.asarray(early_boundary_bic, float)
    full = np.asarray(full_boundary_bic, float)
    if actions.shape[0] != len(curves) or early.shape != (len(curves),) or full.shape != early.shape:
        raise ValueError("actions and boundary evidence must provide one entry per pair")
    boundary = (early > 0) & (full > 0)
    commitment_rows = [commitment_delta_bic(curve, action, horizon, immediate_hold)
                       for curve, action in zip(curves, actions)]
    commitment = np.asarray([row[0] for row in commitment_rows])
    onsets = tuple(row[1] for row in commitment_rows)
    # Index `horizon` is the state after all H differing actions; test whether its consequence
    # remains through the late hold independently of when commitment occurred.
    persistence = np.asarray([
        persistence_delta_bic(curve[horizon:], late_hold) for curve in curves])
    consequential = boundary & (commitment > 0) & (persistence > 0)
    needed = len(curves) // 2 + 1
    return WholeTrajectoryConsequenceEvidence(
        bool(np.sum(consequential) >= needed), int(np.sum(consequential)), len(curves),
        int(np.sum(boundary)), int(np.sum(commitment > 0)), int(np.sum(persistence > 0)),
        tuple(map(float, commitment)), tuple(map(float, persistence)), onsets,
        tuple(tuple(map(float, curve)) for curve in curves))


def _shape_convergence_delta_bic(curves):
    """Positive means successive normalized-curve changes vanish with bracket width."""
    values = np.asarray(curves, float)
    norms = np.sqrt(np.mean(values ** 2, axis=1))
    if np.any(norms <= 1e-12):
        return -np.inf
    unit = values / norms[:, None]
    gaps = np.sqrt(np.mean(np.diff(unit, axis=0) ** 2, axis=1))
    if len(gaps) < 4:
        raise ValueError("at least five bracket levels are required")
    width = .5 ** np.arange(len(gaps), dtype=float)

    def score(plateau):
        if plateau:
            def residual(params):
                offset, linear, quadratic = params
                return gaps - (offset + linear * width + quadratic * width ** 2)
            fit = least_squares(residual, [max(gaps[-1], 0.), gaps[0]-gaps[-1], 0.],
                                bounds=([0., -np.inf, -np.inf],
                                        [np.inf, np.inf, np.inf]))
            parameters = 3
        else:
            def residual(params):
                linear, quadratic = params
                return gaps - (linear * width + quadratic * width ** 2)
            fit = least_squares(residual, [gaps[0], 0.])
            parameters = 2
        return _bic(fit.fun, parameters,
                    max(float(np.sqrt(np.mean(gaps ** 2))), 1e-12))
    return float(score(True) - score(False))


def multiresolution_consequence_alarm(level_pair_traces, level_action_differences,
                                      early_boundary_bic, full_boundary_bic,
                                      horizon=8, immediate_hold=5, late_hold=10,
                                      final_levels=3):
    """Require a stable commitment limit over successively narrower action brackets.

    Inputs have shape ``(pairs, levels, 2, time, features)``. At the final three widths the
    commitment model must win at every level, its discrete onset may move by at most one recorded
    control step, and the entire normalized separation curve must converge toward a limiting shape.
    The one-step tolerance is the temporal resolution of the measurement, not a fitted score.
    """
    traces = np.asarray(level_pair_traces, float)
    actions = np.asarray(level_action_differences, float)
    if traces.ndim != 5 or traces.shape[2] != 2 or actions.shape[:2] != traces.shape[:2]:
        raise ValueError("expected traces (pairs, levels, 2, time, features) and matching actions")
    if traces.shape[1] < final_levels or final_levels < 2:
        raise ValueError("not enough refinement levels")
    early = np.asarray(early_boundary_bic, float)
    full = np.asarray(full_boundary_bic, float)
    if early.shape != (len(traces),) or full.shape != early.shape:
        raise ValueError("one early/full boundary value is required per pair")
    boundary = (early > 0) & (full > 0)
    all_commitment, all_onsets, final_persistence, shape_delta = [], [], [], []
    stable = []
    for pair_traces, pair_actions in zip(traces, actions):
        curves = whole_trajectory_separation_curves(pair_traces)
        rows = [commitment_delta_bic(curve, action, horizon, immediate_hold)
                for curve, action in zip(curves, pair_actions)]
        delta = np.asarray([row[0] for row in rows])
        onsets = np.asarray([row[1] for row in rows], int)
        tail_delta = delta[-final_levels:]
        tail_onsets = onsets[-final_levels:]
        all_commitment.append(tuple(map(float, delta)))
        all_onsets.append(tuple(map(int, onsets)))
        stable.append(bool(np.all(tail_delta > 0)
                           and np.ptp(tail_onsets) <= 1))
        final_persistence.append(float(persistence_delta_bic(
            curves[-1, horizon:], late_hold)))
        shape_delta.append(_shape_convergence_delta_bic(curves))
    persistence = np.asarray(final_persistence) > 0
    convergence = np.asarray(shape_delta) > 0
    consequential = boundary & np.asarray(stable) & persistence & convergence
    needed = len(traces) // 2 + 1
    return MultiresolutionConsequenceEvidence(
        bool(np.sum(consequential) >= needed), int(np.sum(consequential)), len(traces),
        int(np.sum(boundary)), int(np.sum(stable)), int(np.sum(persistence)),
        int(np.sum(convergence)), tuple(all_commitment), tuple(all_onsets),
        tuple(final_persistence), tuple(shape_delta))


def boundary_amplification_persistence_alarm(pair_traces, early_boundary_bic,
                                              full_boundary_bic, action_end=7,
                                              late_start=10):
    """Require the same majority of refined brackets to pass all three structural tests."""
    curves = separation_curves(pair_traces, action_end)
    early = np.asarray(early_boundary_bic, float)
    full = np.asarray(full_boundary_bic, float)
    if early.shape != (len(curves),) or full.shape != early.shape:
        raise ValueError("one early/full boundary evidence value is required per pair")
    boundary = (early > 0) & (full > 0)
    amplification = np.asarray([amplification_delta_bic(curve) for curve in curves])
    persistence = np.asarray([persistence_delta_bic(curve, late_start) for curve in curves])
    consequential = boundary & (amplification > 0) & (persistence > 0)
    needed = len(curves) // 2 + 1
    return ConsequenceEvidence(
        bool(np.sum(consequential) >= needed), int(np.sum(consequential)), len(curves),
        int(np.sum(boundary)), int(np.sum(amplification > 0)), int(np.sum(persistence > 0)),
        tuple(map(float, amplification)), tuple(map(float, persistence)),
        tuple(tuple(map(float, curve)) for curve in curves))


def result_dict(result):
    return asdict(result)
