"""Label-free multi-peak local expansion along ordered counterfactual probes."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalExpansionResult:
    alarm: bool
    score: float
    peak_indices: tuple[int, ...]
    peak_scalars: tuple[float, ...]
    peak_evidence: tuple[float, ...]
    adjacent_gain: tuple[float, ...]
    coverage: float
    valid_probes: int


def _design(s, split, discontinuous):
    """Continuous piecewise linear local response, optionally with a step."""
    x = np.asarray(s, np.float64) - float(split)
    columns = [np.ones(len(x)), x, np.maximum(x, 0.0)]
    if discontinuous:
        columns.append((x > 0.0).astype(np.float64))
    return np.stack(columns, axis=1)


def _bic(design, response):
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ coefficients
    rss = max(float(np.square(residual).sum()), np.finfo(np.float64).tiny)
    observations = int(response.size)
    parameters = int(design.shape[1] * response.shape[1])
    return float(observations * np.log(rss / observations)
                 + parameters * np.log(observations))


def detect_local_expansion(scalars, response, *, half_window=8, scale=None):
    """Find any number of locally discontinuous expansion peaks.

    At every supported adjacent pair, compare a continuous piecewise-linear fit
    with the same fit plus a step.  Each fit sees only ``half_window`` points on
    either side, so a second transition elsewhere cannot invalidate the first.
    BIC pays for the per-component step and ``2 log(K)`` pays for searching K
    adjacent locations.  Positive adjusted evidence is the parameter-free alarm
    rule; labels never enter the calculation.
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
    order = np.argsort(s); s, y = s[order], y[order]
    width = int(half_window)
    if width < 3:
        raise ValueError("half_window must be at least three")
    candidates = len(s) - 2 * width + 1
    if candidates <= 0:
        return LocalExpansionResult(False, 0.0, (), (), (), (),
                                    coverage, len(s))
    # A constant counterfactual response has exactly zero expansion.  Without
    # this invariant, BIC compares two round-off-sized residuals and can assign
    # enormous evidence to a meaningless step in deterministic simulator data.
    if np.allclose(y, y[:1], rtol=1e-7, atol=1e-9):
        return LocalExpansionResult(False, 0.0, (), (), (), (),
                                    coverage, len(s))
    search_penalty = 2.0 * np.log(candidates)
    evidence, gain = [], []
    for index in range(width, len(s) - width + 1):
        first, last = index - width, index + width
        local_s, local_y = s[first:last], y[first:last]
        split = float((s[index - 1] + s[index]) / 2.0)
        continuous = _bic(_design(local_s, split, False), local_y)
        stepped = _bic(_design(local_s, split, True), local_y)
        evidence.append(float(continuous - stepped - search_penalty))
        adjacent = float(np.linalg.norm(y[index] - y[index - 1]))
        ordinary = np.linalg.norm(np.diff(local_y, axis=0), axis=1)
        ordinary = np.delete(ordinary, width - 1)
        gain.append(adjacent / max(float(np.median(ordinary)), 1e-12))
    evidence = np.asarray(evidence); gain = np.asarray(gain)
    # Adjacent positive locations are one broad peak, represented by its maximum.
    positive = np.flatnonzero(evidence > 0.0); peaks = []
    for index in positive:
        if not peaks or index > peaks[-1][-1] + 1:
            peaks.append([int(index)])
        else:
            peaks[-1].append(int(index))
    representatives = [max(group, key=lambda i: evidence[i]) for group in peaks]
    split_indices = tuple(int(i + width) for i in representatives)
    return LocalExpansionResult(
        bool(representatives), float(np.max(evidence)), split_indices,
        tuple(float((s[i - 1] + s[i]) / 2.0) for i in split_indices),
        tuple(float(evidence[i - width]) for i in split_indices),
        tuple(float(value) for value in gain), coverage, len(s))
