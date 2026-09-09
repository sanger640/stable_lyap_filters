"""
Near-neighbour pair mining: the data that supervises Jacobians rather than trajectories.

Keep (i, j) when the states are close AND the actions over the horizon nearly match. The
action filter is not optional -- two states that separate because different actions were
commanded say nothing about the flow map. (Lorenz is autonomous, so there the filter is a
no-op; it matters from Phase 3 onward.)
"""
import numpy as np
import torch


def mine_pairs(states, horizon, radius, actions=None, action_tol=None,
               max_pairs=200_000, exclude_temporal=50, seed=0):
    """states: (N, d) tensor from ONE contiguous trajectory.

    exclude_temporal drops pairs that are close merely because they are adjacent in time --
    those carry no information about the flow map's sensitivity, only about its continuity.
    """
    from scipy.spatial import cKDTree
    S = states.detach().cpu().numpy()
    N = len(S)
    tree = cKDTree(S)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return None
    i, j = pairs[:, 0], pairs[:, 1]
    keep = (np.abs(i - j) > exclude_temporal) & (i + horizon < N) & (j + horizon < N)
    i, j = i[keep], j[keep]
    if actions is not None and action_tol is not None:
        A = actions.detach().cpu().numpy()
        seg = lambda k: np.stack([A[k + h] for h in range(horizon)], 1)
        keep2 = np.linalg.norm(seg(i) - seg(j), axis=(1, 2)) < action_tol
        i, j = i[keep2], j[keep2]
    if len(i) > max_pairs:
        sel = np.random.default_rng(seed).choice(len(i), max_pairs, replace=False)
        i, j = i[sel], j[sel]
    return {"i": torch.as_tensor(i, dtype=torch.long),
            "j": torch.as_tensor(j, dtype=torch.long),
            "horizon": horizon, "n_pairs": len(i)}
