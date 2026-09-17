"""Local, unlabeled persistent response-branch detection on ordered action probes.

At each candidate gap, compare a continuous piecewise-linear response with the
same model plus a discontinuous step. The BIC penalty prices the extra step
parameter; only gaps favored at both short tails count as persistent branches.
No outcome labels or global basin assignments enter the detector.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalBranchResult:
    status: str
    split_index: int | None
    lower_abs_coefficient: float | None
    upper_abs_coefficient: float | None
    direction: int | None
    evidence_by_tail: tuple[float, float] | None
    detectable_gaps: int


def gap_bic_evidence(coefficients, response, split_index, *, side_points=4):
    """Positive evidence favors a local jump over a continuous local bend."""
    s = np.asarray(coefficients, np.float64)
    y = np.asarray(response, np.float64)
    if y.ndim != 2 or len(s) != len(y):
        raise ValueError("response must be (strength, feature)")
    if split_index < 3 or len(s) - split_index < 3:
        return None
    lo = max(0, split_index - side_points)
    hi = min(len(s), split_index + side_points)
    x = s[lo:hi]
    values = y[lo:hi]
    if split_index - lo < 3 or hi - split_index < 3:
        return None
    if not np.isfinite(values).all() or not np.all(np.diff(x) > 0):
        return None
    boundary = float((s[split_index - 1] + s[split_index]) / 2)
    # Centering makes the step column the physical discontinuity at the gap.
    centered = x - boundary
    hinge = np.maximum(centered, 0.0)
    step = (centered > 0).astype(np.float64)
    continuous = np.column_stack([np.ones(len(x)), centered, hinge])
    discontinuous = np.column_stack([continuous, step])

    def bic(design):
        residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
        # At machine precision, a perfectly smooth curve must not appear to
        # favor a step because one fit happened to round a little better.
        numerical_floor = np.finfo(np.float64).eps * max(
            float(np.square(values - values.mean(axis=0)).sum()), 1.0)
        rss = max(float(np.square(residual).sum()), numerical_floor)
        observations = int(values.size)
        parameters = int(design.shape[1] * values.shape[1])
        return observations * np.log(rss / observations) + parameters * np.log(observations)

    return float(bic(continuous) - bic(discontinuous))


def detect_local_persistent_branch(coefficients, responses, *, side_points=4):
    """Return the closest BIC-supported split that persists at both held tails.

    responses must have shape (strength, 2, feature). All strengths are compared
    to the original nominal command at coefficient zero. If no split is favored,
    the result means only that no branch was *resolved on this finite grid*.
    """
    s = np.asarray(coefficients, np.float64)
    y = np.asarray(responses, np.float64)
    if y.ndim != 3 or y.shape[:2] != (len(s), 2):
        raise ValueError("responses must be (strength, 2 tails, feature)")
    if len(s) < 7 or not np.all(np.diff(s) > 0) or np.sum(np.isclose(s, 0)) != 1:
        raise ValueError("strengths must be sorted, include one zero, and have >=7 points")
    zero = int(np.flatnonzero(np.isclose(s, 0))[0])
    candidates = []
    count = 0
    for split in range(3, len(s) - 2):
        left = gap_bic_evidence(s, y[:, 0], split, side_points=side_points)
        right = gap_bic_evidence(s, y[:, 1], split, side_points=side_points)
        if left is None or right is None:
            continue
        count += 1
        if left > 0 and right > 0:
            direction = -1 if split <= zero else 1
            lower = float(min(abs(s[split - 1]), abs(s[split])))
            upper = float(max(abs(s[split - 1]), abs(s[split])))
            # A gap crossing nominal itself has zero lower distance.
            candidates.append((upper, min(left, right), split, lower, direction,
                               (float(left), float(right))))
    if not candidates:
        return LocalBranchResult("not_resolved_on_grid", None, None, None, None, None, count)
    _, _, split, lower, direction, evidence = min(candidates,
                                                 key=lambda item: (item[0], -item[1]))
    upper = float(max(abs(s[split - 1]), abs(s[split])))
    return LocalBranchResult("bracketed", split, lower, upper, direction, evidence, count)
