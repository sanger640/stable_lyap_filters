"""Ordered, multivariate change-point detection for local counterfactual probes.

The detector asks whether an ordered response is better described by a continuous
piecewise polynomial or by the same model with a step discontinuity.  BIC pays for
the extra per-output jump parameters; no outcome labels or cluster labels enter the
fit.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChangePointResult:
    alarm: bool
    split_index: int | None
    split_scalar: float | None
    jump_evidence_bic: float
    jump_norm: float
    adjacent_jump_ratio: float
    null_bic: float
    continuous_bic: float
    discontinuous_bic: float
    coverage: float
    valid_probes: int


def _design(s, degree, split=None, discontinuous=False):
    """Polynomial or continuous truncated-power spline, optionally with a step."""
    s = np.asarray(s, np.float64)
    columns = [s ** power for power in range(degree + 1)]
    if split is not None:
        right = np.maximum(s - float(split), 0.0)
        if discontinuous:
            columns.append((s > split).astype(np.float64))
        # Starting at power one guarantees continuity when no step is present.
        columns.extend(right ** power for power in range(1, degree + 1))
    return np.stack(columns, axis=1)


def _fit_bic(design, y):
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    rss = max(float(np.square(residual).sum()), np.finfo(np.float64).tiny)
    observations = int(y.size)
    parameters = int(design.shape[1] * y.shape[1])
    bic = observations * np.log(rss / observations) + parameters * np.log(observations)
    return float(bic), coefficients


def robust_component_scale(responses, floor_fraction=0.1):
    """Label-free component scaling with a floor for nearly constant dimensions."""
    x = np.asarray(responses, np.float64)
    flat = x.reshape(-1, x.shape[-1])
    scale = np.std(flat, axis=0)
    positive = scale[scale > np.finfo(np.float64).eps]
    reference = float(np.median(positive)) if len(positive) else 1.0
    return np.maximum(scale, float(floor_fraction) * reference)


def detect_ordered_jump(scalars, response, *, degree=3, min_side=8, scale=None,
                        min_jump_ratio=3.0):
    """Compare smooth/continuous and discontinuous fits along ordered probes.

    Non-finite probes are removed and reported through ``coverage``.  The split is
    searched only where both sides have ``min_side`` observations.  An alarm means
    that BIC prefers the discontinuous model to both the global smooth null and the
    best continuous piecewise model.
    """
    s = np.asarray(scalars, np.float64)
    y = np.asarray(response, np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if len(s) != len(y):
        raise ValueError("scalars and response must have the same length")
    valid = np.isfinite(s) & np.isfinite(y).all(axis=1)
    coverage = float(valid.mean()) if len(valid) else 0.0
    s, y = s[valid], y[valid]
    if scale is not None:
        component_scale = np.asarray(scale, np.float64)
        if component_scale.shape != (y.shape[1],):
            raise ValueError("scale must have one value per response dimension")
        y = y / component_scale
    order = np.argsort(s)
    s, y = s[order], y[order]
    minimum = max(int(min_side), int(degree) + 2)
    if len(s) < 2 * minimum:
        return ChangePointResult(False, None, None, 0.0, 0.0, 0.0,
                                 float("inf"), float("inf"), float("inf"),
                                 coverage, len(s))

    null_bic, _ = _fit_bic(_design(s, degree), y)
    best_continuous = (float("inf"), None, None)
    best_discontinuous = (float("inf"), None, None)
    for split_index in range(minimum, len(s) - minimum + 1):
        split = float((s[split_index - 1] + s[split_index]) / 2.0)
        continuous = _design(s, degree, split, False)
        discontinuous = _design(s, degree, split, True)
        continuous_bic, continuous_coef = _fit_bic(continuous, y)
        discontinuous_bic, discontinuous_coef = _fit_bic(discontinuous, y)
        if continuous_bic < best_continuous[0]:
            best_continuous = (continuous_bic, split_index, continuous_coef)
        if discontinuous_bic < best_discontinuous[0]:
            best_discontinuous = (discontinuous_bic, split_index, discontinuous_coef)

    # A split model was allowed to search K locations.  The same 2 log(K)
    # change-point search penalty applies to both split families, so it cancels
    # in their direct comparison but prevents either from beating the no-split
    # null merely because it had many chances.
    candidates = len(s) - 2 * minimum + 1
    search_penalty = 2.0 * np.log(candidates)
    best_continuous = (best_continuous[0] + search_penalty,
                       best_continuous[1], best_continuous[2])
    best_discontinuous = (best_discontinuous[0] + search_penalty,
                          best_discontinuous[1], best_discontinuous[2])
    discontinuous_bic, split_index, coefficients = best_discontinuous
    continuous_bic = min(null_bic, best_continuous[0])
    evidence = float(continuous_bic - discontinuous_bic)
    split = float((s[split_index - 1] + s[split_index]) / 2.0)
    # The step column follows the global polynomial columns.
    jump = coefficients[degree + 1]
    jump_norm = float(np.linalg.norm(jump))
    adjacent = np.linalg.norm(np.diff(y, axis=0), axis=1)
    local = np.delete(adjacent, split_index - 1)
    baseline = float(np.median(local)) if len(local) else 0.0
    adjacent_ratio = float(adjacent[split_index - 1] / max(baseline, 1e-12))
    # BIC can reward a step for approximating a globally smooth function between
    # finitely sampled points.  Requiring the observed adjacent change to exceed
    # ordinary local increments makes the alarm about a resolved jump, not merely
    # a somewhat better two-piece approximation to curvature.
    alarm = bool(evidence > 0.0 and discontinuous_bic < null_bic
                 and adjacent_ratio >= float(min_jump_ratio))
    return ChangePointResult(alarm, int(split_index), split, evidence, jump_norm,
                             adjacent_ratio, null_bic, float(best_continuous[0]),
                             float(discontinuous_bic), coverage, len(s))
