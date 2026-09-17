"""Label-free test for distinct outcome modes among noisy executions of one action chunk.

Given the settled endings of N executions that differ only by execution noise, decide whether
they fall into two separated groups rather than one continuous spread. All constants are fixed
here, before any Jenga result was seen:

* project onto the endings' first principal component, best two-group split, each side >= 2;
* two hard-assigned Gaussians must beat one on BIC;
* Ashman's D > 2 (standard bimodality criterion);
* the groups' mean endings must differ by more than the injected execution perturbation
  (a consequence larger than its cause; in the same millimetre units);
* persistence: the same split (up to one probe) must pass at an earlier settle time.
"""
from dataclasses import dataclass, asdict

import numpy as np

BOX_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
ASHMAN_D = 2.0
MIN_SIDE = 2


def corner_displacements_mm(start_pose, delta_pose, half_size, blocks):
    """Displacement of each box corner, (..., len(blocks)*8*3) in mm.

    Pose layout (per jenga_predictive_regime_probe.all_block_pose): 3x3 positions then the
    first two rotation-matrix columns of each block as 3x2, flattened.
    """
    start = np.asarray(start_pose, float)
    end = start + np.asarray(delta_pose, float)

    def corners(pose):
        lead = pose.shape[:-1]
        position = pose[..., :9].reshape(*lead, 3, 3)
        cols = pose[..., 9:27].reshape(*lead, 3, 3, 2)
        c0, c1 = cols[..., 0], cols[..., 1]
        rotation = np.stack([c0, c1, np.cross(c0, c1)], axis=-1)
        local = BOX_SIGNS * np.asarray(half_size, float)
        world = position[..., :, None, :] + np.einsum("...bij,kj->...bki", rotation, local)
        return world[..., blocks, :, :]

    moved = corners(end) - corners(start)
    return 1000 * moved.reshape(*moved.shape[:-3], -1)


@dataclass
class ModeTest:
    alarm: bool
    bic_prefers_two: bool
    ashman_d: float
    minority: int
    separation_rms_mm: float
    spread_rms_mm: float
    labels: np.ndarray

    def summary(self):
        out = asdict(self)
        out.pop("labels")
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in out.items()}


def _neg2ll(values):
    var = max(float(np.var(values)), 1e-12)
    return len(values) * (np.log(2 * np.pi * var) + 1)


def two_mode_test(endings, perturbation_mm):
    """endings (n, D) in mm. perturbation_mm: scale of the injected execution noise."""
    x = np.asarray(endings, float)
    n = len(x)
    centered = x - x.mean(0)
    spread = float(np.sqrt(np.mean(np.sum(centered ** 2, 1)) / max(x.shape[1] // 3, 1)))
    if n < 2 * MIN_SIDE or not np.any(centered):
        return ModeTest(False, False, 0.0, 0, 0.0, spread, np.zeros(n, int))
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    y = centered @ axis
    order = np.argsort(y)
    ys = y[order]
    best, split = -1.0, MIN_SIDE
    for i in range(MIN_SIDE, n - MIN_SIDE + 1):
        a, b = ys[:i], ys[i:]
        between = i * (n - i) / n * (a.mean() - b.mean()) ** 2
        if between > best:
            best, split = between, i
    labels = np.zeros(n, int)
    labels[order[split:]] = 1
    a, b = y[labels == 0], y[labels == 1]
    pooled = np.sqrt(a.var() + b.var())
    d = float(np.inf if pooled == 0 else np.sqrt(2) * abs(a.mean() - b.mean()) / pooled)
    sizes = np.array([len(a), len(b)])
    bic1 = _neg2ll(y) + 2 * np.log(n)
    bic2 = (_neg2ll(a) + _neg2ll(b) - 2 * np.sum(sizes * np.log(sizes / n))
            + 5 * np.log(n))
    separation = float(np.linalg.norm(x[labels == 0].mean(0) - x[labels == 1].mean(0))
                       / np.sqrt(max(x.shape[1] // 3, 1)))
    prefers = bool(bic2 < bic1)
    alarm = bool(prefers and d > ASHMAN_D and sizes.min() >= MIN_SIDE
                 and separation > perturbation_mm)
    return ModeTest(alarm, prefers, d, int(sizes.min()), separation, spread, labels)


def same_partition(a, b, tolerance=1):
    a, b = np.asarray(a), np.asarray(b)
    return int(min(np.sum(a != b), np.sum(a != 1 - b))) <= tolerance


# ---------------------------------------------------------------------------
# Multi-direction, multi-group version. Constants fixed before any Jenga result.
MULTI_DIMS = 5
MAX_GROUPS = 4
N_INIT = 10
EM_ITERS = 200


def _diag_gmm(x, k, rng):
    """One EM run of a diagonal-covariance Gaussian mixture. Returns (loglik, resp, mu, var)."""
    n, d = x.shape
    floor = max(float(x.var(0).mean()), 1e-12) * 1e-6  # numerical guard only
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):  # k-means++ seeding
        dist = np.min([np.sum((x - c) ** 2, 1) for c in centres], axis=0)
        total = dist.sum()
        centres.append(x[rng.choice(n, p=dist / total)] if total > 0 else x[rng.integers(n)])
    mu = np.array(centres)
    var = np.tile(x.var(0) + floor, (k, 1))
    weight = np.full(k, 1.0 / k)
    previous = -np.inf
    for _ in range(EM_ITERS):
        log_p = (np.log(weight)[None]
                 - 0.5 * np.sum(np.log(2 * np.pi * var), 1)[None]
                 - 0.5 * np.sum((x[:, None] - mu[None]) ** 2 / var[None], 2))
        top = log_p.max(1, keepdims=True)
        log_norm = top + np.log(np.exp(log_p - top).sum(1, keepdims=True))
        resp = np.exp(log_p - log_norm)
        loglik = float(log_norm.sum())
        nk = resp.sum(0) + 1e-12
        weight = nk / n
        mu = (resp.T @ x) / nk[:, None]
        var = (resp.T @ x ** 2) / nk[:, None] - mu ** 2 + floor
        var = np.maximum(var, floor)
        if abs(loglik - previous) < 1e-8 * max(1.0, abs(loglik)):
            break
        previous = loglik
    return loglik, resp, mu, var


def _pairwise_gaps_ok(x, labels, k):
    """Every group has >= MIN_SIDE runs, and for every pair, projected onto the line joining
    their means: two hard groups beat one on 1-D BIC (a halved continuous cloud does not)
    and Ashman D > 2."""
    if np.bincount(labels, minlength=k).min() < MIN_SIDE:
        return False
    for a in range(k):
        for b in range(a + 1, k):
            xa, xb = x[labels == a], x[labels == b]
            axis = xb.mean(0) - xa.mean(0)
            norm = np.linalg.norm(axis)
            if norm == 0:
                return False
            pa, pb = xa @ axis / norm, xb @ axis / norm
            both = np.concatenate([pa, pb])
            n = len(both)
            sizes = np.array([len(pa), len(pb)])
            bic1 = _neg2ll(both) + 2 * np.log(n)
            bic2 = (_neg2ll(pa) + _neg2ll(pb) - 2 * np.sum(sizes * np.log(sizes / n))
                    + 5 * np.log(n))
            pooled = np.sqrt(pa.var() + pb.var())
            d = np.inf if pooled == 0 else np.sqrt(2) * abs(pa.mean() - pb.mean()) / pooled
            if not (bic2 < bic1 and d > ASHMAN_D):
                return False
    return True


def multi_mode_test(endings, dims=MULTI_DIMS, max_groups=MAX_GROUPS, seed=0):
    """Number of well-separated outcome groups among noisy executions (1 = no fork)."""
    x = np.asarray(endings, float)
    n = len(x)
    centred = x - x.mean(0)
    if not np.any(centred):
        return {"groups": 1, "labels": np.zeros(n, int), "sizes": [n], "bic": {}}
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    z = (u * s)[:, :min(dims, len(s))]
    d = z.shape[1]
    rng = np.random.default_rng(seed)
    bic, best = {}, {}
    for k in range(1, max_groups + 1):
        params = k * 2 * d + (k - 1)
        candidates = []
        for _ in range(N_INIT if k > 1 else 1):
            loglik, resp, mu, var = _diag_gmm(z, k, rng)
            labels = resp.argmax(1)
            if k == 1 or _pairwise_gaps_ok(z, labels, k):
                candidates.append((loglik, labels))
        if candidates:
            loglik, labels = max(candidates, key=lambda c: c[0])
            bic[k] = -2 * loglik + params * np.log(n)
            best[k] = labels
    k = min(bic, key=bic.get)
    labels = best[k]
    return {"groups": int(k), "labels": labels,
            "sizes": sorted(np.bincount(labels).tolist(), reverse=True),
            "bic": {int(a): float(b) for a, b in bic.items()}}


def groupings_persist(a, b, tolerance=1):
    """True if one grouping maps onto the other with <= tolerance probes misplaced
    (checked both directions, so a merge of groups over time is allowed)."""
    a, b = np.asarray(a), np.asarray(b)

    def misplaced(src, dst):
        return sum(int(np.sum(src == g)) - int(np.bincount(dst[src == g]).max())
                   for g in np.unique(src))
    return min(misplaced(a, b), misplaced(b, a)) <= tolerance


def coarse_persistent_fork(labels_a, labels_b, min_shared=MIN_SIDE, tolerance=1):
    """Revised persistence (declared after Stage 2b, to be judged on fresh data only).

    Link a step-10 group to a step-30 group when they share >= min_shared runs; runs whose
    (step-10, step-30) pair is shared by fewer runs are strays. Linked groups merge into
    components. A fork persists if strays <= tolerance and >= 2 components each hold
    >= min_shared runs: some split holds over time even if finer groups reshuffle.
    """
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    pairs, counts = np.unique(np.stack([a, b], 1), axis=0, return_counts=True)
    strong = pairs[counts >= min_shared]
    strays = int(counts[counts < min_shared].sum())
    parent = {}

    def find(node):
        while parent.setdefault(node, node) != node:
            node = parent[node]
        return node
    for ga, gb in strong:
        parent[find(("a", ga))] = find(("b", gb))
    sizes = {}
    for (ga, gb), n in zip(pairs, counts):
        if n >= min_shared:
            root = find(("a", ga))
            sizes[root] = sizes.get(root, 0) + int(n)
    big = sum(v >= min_shared for v in sizes.values())
    return bool(strays <= tolerance and big >= 2)
