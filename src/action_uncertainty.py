"""Unlabeled action-tracking residual model for Jenga replay episodes.

Action[t] is a commanded end-effector target; proprio[t+1] is the observed
position after that action. A linear lag model explains predictable tracking
error from target movement. Its residual covariance is a *proxy*, not a
certified controller-noise model or a failure-calibrated safety threshold.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingErrorModel:
    coefficients: np.ndarray  # (4,3): intercept and command delta
    covariance: np.ndarray  # (3,3) residual positional error
    radial_quantile_90: float
    samples: int

    @property
    def axes(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        order = np.argsort(eigenvalues)[::-1]
        return eigenvectors[:, order] * np.sqrt(eigenvalues[order])[None, :]


def tracking_arrays(episodes):
    design, errors = [], []
    for episode in episodes:
        actions = np.asarray(episode.actions, np.float64)
        proprio = np.asarray(episode.proprio, np.float64)
        if actions.shape != proprio.shape or actions.ndim != 2 or actions.shape[1] < 3:
            raise ValueError("aligned actions/proprio must be (time, >=3)")
        if len(actions) < 3:
            continue
        delta = actions[1:-1, :3] - actions[:-2, :3]
        design.append(np.column_stack([np.ones(len(delta)), delta]))
        errors.append(actions[1:-1, :3] - proprio[2:, :3])
    if not design:
        raise ValueError("no episodes with at least three observations")
    return np.concatenate(design), np.concatenate(errors)


def fit_tracking_error(episodes):
    design, error = tracking_arrays(episodes)
    coefficients = np.linalg.lstsq(design, error, rcond=None)[0]
    residual = error - design @ coefficients
    covariance = np.cov(residual, rowvar=False)
    # Only a numerical regularizer; the physical covariance remains data-driven.
    covariance += np.eye(3) * max(np.trace(covariance) / 3, 1e-12) * 1e-6
    radii = np.sqrt(np.einsum("ni,ij,nj->n", residual,
                              np.linalg.inv(covariance), residual))
    return TrackingErrorModel(coefficients, covariance,
                              float(np.quantile(radii, .90)), len(residual))
