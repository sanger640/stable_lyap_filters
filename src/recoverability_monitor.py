"""Calibration-free local recoverability test for counterfactual physical branches.

Each side of a refined action boundary receives the same bounded library of corrective actions.
The test compares the two resulting reachable sets using their own sampling resolution. It has no
task outcome, safe reference population, named object role, or fitted physical-distance threshold.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RecoverabilityEvidence:
    alarm: bool
    unrecoverable_pairs: int
    pairs: int
    boundary_pairs: int
    cross_distance: tuple
    reachable_resolution: tuple
    overlap_ratio: tuple


def _generic_coordinates(outcomes):
    """Equal-family internal units for anonymous pose and velocity endpoint states."""
    values = np.asarray(outcomes, float)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[2] < 45:
        raise ValueError("outcomes must have shape (2, corrections, at least 45 fields)")
    values = values[:, :, :45]
    groups = (slice(0, 9), slice(9, 27), slice(27, 36), slice(36, 45))
    result = np.zeros_like(values)
    centred_all = values - values.mean(axis=(0, 1), keepdims=True)
    global_rms = float(np.sqrt(np.mean(centred_all ** 2)))
    for group in groups:
        centred = centred_all[..., group]
        rms = float(np.sqrt(np.mean(centred ** 2)))
        if rms > max(global_rms, 1.) * 1e-10:
            result[..., group] = centred / rms
    return result


def reachable_overlap(outcomes):
    """Cross-set distance divided by within-set nearest-neighbour sampling resolution.

    A ratio <=1 means the sets approach one another at least as closely as each set samples its own
    corrective response. If corrections have no effect, identical sets are recoverable and any
    nonzero separation is conservatively unresolved/unrecoverable.
    """
    z = _generic_coordinates(outcomes)
    distance = np.linalg.norm(z[:, :, None] - z[:, None, :], axis=3)
    corrections = z.shape[1]
    if corrections < 2:
        raise ValueError("at least two corrective actions are required")
    within = []
    for branch in range(2):
        matrix = distance[branch].copy()
        np.fill_diagonal(matrix, np.inf)
        within.append(float(np.median(np.min(matrix, axis=1))))
    cross = np.linalg.norm(z[0, :, None] - z[1, None, :], axis=2)
    cross_distance = float(np.min(cross))
    resolution = float(np.sqrt(within[0] * within[1]))
    if resolution <= 1e-12:
        ratio = 0. if cross_distance <= 1e-12 else np.inf
    else:
        ratio = cross_distance / resolution
    return cross_distance, resolution, float(ratio)


def recoverability_alarm(pair_outcomes, early_boundary_bic, full_boundary_bic):
    """Alarm when most refined boundary pairs have non-overlapping correction-reachable sets."""
    outcomes = np.asarray(pair_outcomes, float)
    if outcomes.ndim != 4 or outcomes.shape[1] != 2:
        raise ValueError("pair_outcomes must have shape (pairs, 2, corrections, features)")
    early = np.asarray(early_boundary_bic, float)
    full = np.asarray(full_boundary_bic, float)
    if early.shape != (len(outcomes),) or full.shape != early.shape:
        raise ValueError("one early/full boundary value is required per pair")
    boundary = (early > 0) & (full > 0)
    rows = [reachable_overlap(pair) for pair in outcomes]
    cross = np.asarray([row[0] for row in rows])
    resolution = np.asarray([row[1] for row in rows])
    ratio = np.asarray([row[2] for row in rows])
    unrecoverable = boundary & (ratio > 1.)
    needed = len(outcomes) // 2 + 1
    return RecoverabilityEvidence(
        bool(np.sum(unrecoverable) >= needed), int(np.sum(unrecoverable)), len(outcomes),
        int(np.sum(boundary)), tuple(map(float, cross)), tuple(map(float, resolution)),
        tuple(map(float, ratio)))


def result_dict(result):
    return asdict(result)
