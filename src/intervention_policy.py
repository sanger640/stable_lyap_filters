"""Task-agnostic action fallback used after a persistent Regime Monitor v0 alarm.

The policy sees only the current end-effector command, the intended action chunk, and anonymous
Regime Monitor evidence. It contains no Jenga geometry, object identity, outcome label, or fitted
threshold.
"""
from dataclasses import dataclass

import numpy as np


ACTION_SCALES = (1.0, 0.75, 0.50, 0.25, 0.0)


@dataclass(frozen=True)
class CandidateDecision:
    index: int
    scale: float
    consequential_pairs: int


def scaled_action_candidates(chunk, current_action, scales=ACTION_SCALES):
    """Return fixed progress-scaled alternatives around the current absolute EE command.

    Position targets are contracted toward the current EE position. Gripper commands remain the
    intended commands except for the zero-motion candidate, which keeps the current gripper state.
    """
    actions = np.asarray(chunk, np.float32)
    current = np.asarray(current_action, np.float32)
    if actions.ndim != 2 or actions.shape[1] != 4 or current.shape != (4,):
        raise ValueError("expected chunk (H,4) and current_action (4,)")
    if not scales or scales[0] != 1.0 or scales[-1] != 0.0:
        raise ValueError("candidate scales must start at 1 and end at 0")
    candidates = []
    for scale in scales:
        candidate = actions.copy()
        candidate[:, :3] = current[:3] + float(scale) * (actions[:, :3] - current[:3])
        if scale == 0:
            candidate[:, 3] = current[3]
        candidates.append(candidate)
    return np.stack(candidates)


def choose_robust_candidate(evidence, scales=ACTION_SCALES):
    """Minimize v0-positive pairs, breaking ties toward the intended action.

    This is a rank, not a calibrated alarm threshold. The frozen v0 decision remains untouched.
    """
    if len(evidence) != len(scales):
        raise ValueError("one evidence record is required per candidate scale")
    pairs = [int(row.get("consequential_pairs", 0)) for row in evidence]
    index = min(range(len(scales)), key=lambda i: (pairs[i], abs(1.0 - scales[i]), i))
    return CandidateDecision(index, float(scales[index]), pairs[index])


def choose_oracle_candidate(grades, scales=ACTION_SCALES):
    """Privileged local upper bound: avoid a new neighbor topple, then preserve task action.

    `grades` contains `(new_neighbor_topple, peak_neighbor_tilt_deg)`. Tilt is only a final
    deterministic tie-break after safety and deviation; it never enters the deployable policy.
    """
    if len(grades) != len(scales):
        raise ValueError("one physical grade is required per candidate scale")
    index = min(range(len(scales)), key=lambda i: (
        bool(grades[i][0]), abs(1.0 - scales[i]), float(grades[i][1]), i))
    return index


def hold_action(current_action):
    """One generic reobservation step that preserves pose and gripper state."""
    current = np.asarray(current_action, np.float32)
    if current.shape != (4,):
        raise ValueError("current_action must have shape (4,)")
    return current.copy()
