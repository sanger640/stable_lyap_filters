"""Unsupervised terminal-basin discovery shared by the toy and Jenga monitors.

The old pipeline used a single-linkage distance plateau.  That happened to work on the
imbalanced tipping-block basins but one bridging Jenga scene welded its two basins together.
HDBSCAN is the current method: PCA first, then density-persistent clusters with an explicit
noise label.

Scikit-learn's HDBSCAN does not expose out-of-sample prediction.  Runtime assignment therefore
uses nearest centroids *inside density support*: a point must also be close to a training member
of that cluster.  The support radius is the unsupervised 99th percentile of within-cluster
nearest-neighbour distances.  Points outside support retain label -1 and are reported separately
as noise coverage; they do not participate in the basin-dissent vote.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ProprioResidualizer:
    """Remove visual variation linearly explained by proprio and its quadratic terms."""
    coefficients: np.ndarray

    @staticmethod
    def design(proprio):
        p = np.asarray(proprio, dtype=np.float32)
        if p.ndim != 2:
            raise ValueError(f"proprio must be 2-D, got {p.shape}")
        return np.hstack([p, p ** 2, np.ones((len(p), 1), np.float32)])

    @classmethod
    def fit(cls, endings, proprio):
        e = np.asarray(endings, dtype=np.float32)
        a = cls.design(proprio)
        if len(e) != len(a):
            raise ValueError(f"endings/proprio length mismatch: {len(e)} != {len(a)}")
        return cls(np.linalg.lstsq(a, e, rcond=None)[0].astype(np.float32))

    def transform(self, endings, proprio):
        e = np.asarray(endings, dtype=np.float32)
        a = self.design(proprio)
        if len(e) != len(a) or e.shape[1] != self.coefficients.shape[1]:
            raise ValueError("endings do not match the fitted residualizer")
        return e - a @ self.coefficients


@dataclass
class BasinModel:
    mean: np.ndarray
    components: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    members: tuple
    support_radii: np.ndarray
    min_cluster_size: int
    support_quantile: float

    @property
    def n_clusters(self):
        return len(self.centers)

    @property
    def coverage(self):
        return float((self.labels >= 0).mean())

    @property
    def cluster_sizes(self):
        return [int((self.labels == i).sum()) for i in range(self.n_clusters)]

    def transform(self, endings):
        x = np.asarray(endings, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected endings shaped (n,{self.mean.shape[0]}), got {x.shape}")
        return (x - self.mean) @ self.components.T

    def predict_pca(self, x):
        """Assign PCA-space points to a basin, preserving -1 outside density support."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.centers.shape[1]:
            raise ValueError(f"expected PCA points shaped (n,{self.centers.shape[1]}), got {x.shape}")
        dc = ((x[:, None] - self.centers[None]) ** 2).sum(-1)
        nearest = dc.argmin(1)
        out = np.full(len(x), -1, dtype=int)
        for c in range(self.n_clusters):
            q = np.where(nearest == c)[0]
            if not len(q):
                continue
            d_support = np.sqrt(((x[q, None] - self.members[c][None]) ** 2).sum(-1)).min(1)
            out[q[d_support <= self.support_radii[c] * (1.0 + 1e-6)]] = c
        return out

    def predict(self, endings):
        return self.predict_pca(self.transform(endings))


def _support_radius(points, quantile):
    if len(points) < 2:
        return 0.0
    d2 = ((points[:, None] - points[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nearest = np.sqrt(d2.min(1))
    return float(np.quantile(nearest, quantile))


def fit_basin_model(endings, pca_dim=8, min_cluster_fraction=0.05,
                    support_quantile=0.99):
    """Fit PCA + HDBSCAN without supplying the basin count or using outcome labels."""
    from sklearn.cluster import HDBSCAN

    e = np.asarray(endings, dtype=np.float32)
    if e.ndim != 2 or len(e) < 3:
        raise ValueError(f"endings must be a 2-D array with at least 3 rows, got {e.shape}")
    if not 0 < min_cluster_fraction <= 1:
        raise ValueError("min_cluster_fraction must be in (0,1]")
    if not 0 < support_quantile <= 1:
        raise ValueError("support_quantile must be in (0,1]")

    mean = e.mean(0)
    centered = e - mean
    k = min(int(pca_dim), len(e) - 1, e.shape[1])
    if k < 1:
        raise ValueError("pca_dim leaves no components")
    components = np.linalg.svd(centered, full_matrices=False)[2][:k].astype(np.float32)
    x = centered @ components.T
    mcs = max(3, int(np.ceil(min_cluster_fraction * len(e))))
    labels = HDBSCAN(min_cluster_size=mcs).fit_predict(x).astype(int)
    ids = sorted(set(labels) - {-1})
    if not ids:
        raise ValueError(
            f"HDBSCAN found no basin with min_cluster_size={mcs} ({min_cluster_fraction:.1%} of n)")

    # HDBSCAN labels are already contiguous in sklearn, but normalise explicitly so downstream
    # counting never relies on that implementation detail.
    remap = {old: new for new, old in enumerate(ids)}
    labels = np.array([remap.get(v, -1) for v in labels], dtype=int)
    members = tuple(x[labels == c].copy() for c in range(len(ids)))
    centers = np.stack([p.mean(0) for p in members]).astype(np.float32)
    radii = np.array([_support_radius(p, support_quantile) for p in members], np.float32)
    return BasinModel(mean.astype(np.float32), components, labels, centers, members, radii,
                      mcs, support_quantile)


def dissent_count(labels, n_clusters):
    """Number of known-basin probes outside the plurality known basin.

    HDBSCAN noise is excluded from this vote.  An all-noise probe set therefore has k=0 and must
    be interpreted together with its separately reported known-basin coverage.
    """
    lab = np.asarray(labels, dtype=int)
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    known = lab[(lab >= 0) & (lab < n_clusters)]
    if not len(known):
        return 0
    plurality = max(int((known == c).sum()) for c in range(n_clusters))
    return int(len(known) - plurality)


def known_coverage(labels, n_clusters):
    """Fraction of probes assigned to a discovered basin; noise is not a dissent vote."""
    lab = np.asarray(labels, dtype=int)
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    return float(((lab >= 0) & (lab < n_clusters)).mean()) if len(lab) else 0.0
