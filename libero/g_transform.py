"""Formulation G on LIBERO delta actions (same four universal transformations as eval/opt3.py on Jenga).

A LIBERO chunk is T delta actions a_t = (dxyz, drot, gripper). G acts on the cumulative plan P_t = sum_{j<t} a_j
(t = 0..T), exactly as Jenga's G acts on absolute targets: speed v in [0,1] and pause k in [0,8] steps re-time the plan
(tau_t = v * max(0, t - k)), amplitude s in [0,1] shrinks it toward the start, back-off b in [0,30] mm adds a ramped
move along the reverse of the chunk's own net direction. Rotation is re-timed with the arm, not shrunk or backed off.
Gripper stays on the policy's clock (formulation G; G2 was rejected in Stage 4b).
"""
import numpy as np

LO, HI = np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 8.0, 1.0, 30.0])
IDENTITY = np.array([1.0, 0.0, 1.0, 0.0])
MM_PER_UNIT = 50.0           # robosuite OSC_POSE: action 1.0 = 5 cm per step (output_max 0.05 m)
B_MIN_MOVE_MM = 2.0


def transform(chunk, theta):
    """chunk (T, 7) deltas, theta (4,) -> (T, 7) deltas."""
    chunk = np.asarray(chunk, np.float64); T = len(chunk)
    v, k, s, b = theta; b = b / MM_PER_UNIT
    P = np.vstack([np.zeros((1, 6)), np.cumsum(chunk[:, :6], 0)])               # (T+1, 6)
    t = np.arange(T + 1)
    tau = np.clip(v * np.maximum(t - k, 0), 0, T)
    i = np.clip(np.floor(tau).astype(int), 0, T - 1); w = (tau - i)[:, None]
    Q = P[i] * (1 - w) + P[i + 1] * w
    Q[:, :3] *= s
    net = P[T, :3]; nrm = np.linalg.norm(net)
    if nrm * MM_PER_UNIT >= B_MIN_MOVE_MM:
        Q[:, :3] -= b * (np.minimum(t, 8) / 8)[:, None] * (net / nrm)
    out = chunk.copy(); out[:, :6] = np.diff(Q, axis=0)
    return out


def exec_noise(rng, T, sigma_xyz=0.15, sigma_rot=0.05, smooth=3):
    """Smooth random perturbation of the delta actions (execution noise), (T, 7)."""
    n = rng.normal(size=(T + smooth - 1, 6)) * np.r_[[sigma_xyz] * 3, [sigma_rot] * 3]
    k = np.ones(smooth) / smooth
    n = np.stack([np.convolve(n[:, j], k, mode="valid") for j in range(6)], 1)
    return np.hstack([n, np.zeros((T, 1))])
