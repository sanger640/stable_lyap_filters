"""Label-free multiresolution test for a candidate response branch.

Two interval bisections follow the half with the larger physical response
separation. A finite jump should retain separation; a locally smooth response
should lose separation as the action interval halves. The two one-parameter
models have equal complexity, so no task-calibrated alarm threshold is used.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleResult:
    status: str
    distances_by_tail: tuple[tuple[float, float, float], tuple[float, float, float]]
    constant_rss_by_tail: tuple[float, float]
    shrinking_rss_by_tail: tuple[float, float]
    retained_fraction_by_tail: tuple[float, float]


def choose_larger_half(left, midpoint, right):
    """Choose one interval for both tails using their combined separation."""
    a, m, b = (np.asarray(x, np.float64) for x in (left, midpoint, right))
    if a.shape != m.shape or a.shape != b.shape or a.ndim != 2 or a.shape[0] != 2:
        raise ValueError("each endpoint must have shape (2 tails, features)")
    left_energy = float(np.square(m - a).sum())
    right_energy = float(np.square(b - m).sum())
    return 0 if left_energy >= right_energy else 1


def compare_scales(distances):
    """Compare constant separation with D proportional to interval width."""
    values = np.asarray(distances, np.float64)
    if values.shape != (2, 3) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("distances must be finite, nonnegative (2 tails, 3 scales)")
    constant = np.ones(3)
    shrinking = np.array([1.0, .5, .25])
    constant_rss, shrinking_rss = [], []
    for y in values:
        c = float(np.dot(y, constant) / np.dot(constant, constant))
        q = float(np.dot(y, shrinking) / np.dot(shrinking, shrinking))
        constant_rss.append(float(np.square(y - c * constant).sum()))
        shrinking_rss.append(float(np.square(y - q * shrinking).sum()))
    retained = tuple(float(y[2] / y[0]) if y[0] > 0 else 0.0 for y in values)
    status = "persistent_separation" if all(a < b for a, b in zip(
        constant_rss, shrinking_rss)) else "shrinks_or_ambiguous"
    return ScaleResult(status, tuple(tuple(float(v) for v in y) for y in values),
                       tuple(constant_rss), tuple(shrinking_rss), retained)
