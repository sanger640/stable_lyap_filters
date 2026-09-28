"""Visual-native consequence stage for calibration-free action-boundary monitoring.

The state implementation gives equal weight to four known physical channel families.  Visual
latents have no position/rotation/velocity field semantics, so this module treats the complete
representation as one metric space.  Only the separation-curve construction changes; boundary
scaling, action exposure, commitment/persistence BIC comparisons, majority rule and zero decision
boundary are imported unchanged from ``consequence_monitor``.
"""
import numpy as np

from consequence_monitor import (WholeTrajectoryConsequenceEvidence,
                                 commitment_delta_bic, persistence_delta_bic)


def visual_whole_trajectory_separation_curves(pair_trajectories):
    """Return scale- and coordinate-order-invariant visual pair separation over time.

    ``pair_trajectories`` has shape ``(pairs, 2, time, features...)``.  All visual dimensions form
    one representation family and receive one RMS scale estimated solely inside the current set of
    refined counterfactual pairs.  A common pre-action zero is prepended because dense replay starts
    after the first differing command.  No reference population, outcome or task label is used.
    """
    traces = np.asarray(pair_trajectories, float)
    if traces.ndim < 4 or traces.shape[1] != 2:
        raise ValueError("pair_trajectories must have shape (pairs, 2, time, features...)")
    difference = (traces[:, 0] - traces[:, 1]).reshape(traces.shape[0], traces.shape[2], -1)
    rms = float(np.sqrt(np.mean(difference ** 2)))
    if not np.isfinite(rms):
        raise ValueError("visual trajectories contain non-finite values")
    if rms == 0.:
        curves = np.zeros(difference.shape[:2], float)
    else:
        curves = np.linalg.norm(difference / rms, axis=2)
    return np.concatenate([np.zeros((len(curves), 1)), curves], axis=1)


def visual_whole_trajectory_consequence_alarm(pair_trajectories, action_differences,
                                              early_boundary_bic, full_boundary_bic,
                                              horizon=8, immediate_hold=5, late_hold=10):
    """Apply unchanged v0 consequence evidence to a visual-native separation curve."""
    curves = visual_whole_trajectory_separation_curves(pair_trajectories)
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
    persistence = np.asarray([
        persistence_delta_bic(curve[horizon:], late_hold) for curve in curves])
    consequential = boundary & (commitment > 0) & (persistence > 0)
    needed = len(curves)//2 + 1
    return WholeTrajectoryConsequenceEvidence(
        bool(np.sum(consequential) >= needed), int(np.sum(consequential)), len(curves),
        int(np.sum(boundary)), int(np.sum(commitment > 0)), int(np.sum(persistence > 0)),
        tuple(map(float, commitment)), tuple(map(float, persistence)), onsets,
        tuple(tuple(map(float, curve)) for curve in curves))
