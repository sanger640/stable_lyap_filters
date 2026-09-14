"""Which clustering method finds the basins on BOTH systems, with no k supplied?

Neither method tried so far is universal:
  * single-linkage + merge plateau  -- works on the TOY (3 imbalanced basins), one bridging pair
    destroys it on JENGA
  * Ward + merge-height gap         -- works on JENGA (2 balanced basins), its bias toward
    EQUAL-SIZED clusters destroys it on the TOY (212/50/38)

Picking per task by which gives the right answer is using the labels, so the calibration-free claim
needs a method that handles both.

HDBSCAN is the candidate. Its selection rule keeps the clusters that PERSIST over the widest range
of density levels -- conceptually the same idea as our merge-distance plateau, but built on density
rather than raw distance, and with an explicit NOISE label so a single bridging point is discarded
instead of welding two groups together.

`min_cluster_size` is the one knob. It is set from the data as a fraction of n, not tuned against
labels: "a basin must contain at least this share of the episodes to count as a basin."
"""
import sys
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import HDBSCAN

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval")]
from phase_e_attractors import pdist, plateau, agreement           # noqa: E402
from jenga_linkage import count_by_gap                             # noqa: E402


def score(lab, truth):
    """Agreement, ignoring HDBSCAN noise points but counting them against coverage."""
    m = lab >= 0
    if m.sum() == 0:
        return 0.0, 0.0, 0
    k = int(lab[m].max()) + 1
    ag = agreement(lab[m], truth[m]) if k <= 6 else float("nan")
    return ag, m.mean(), k


def run(name, X, truth, frac=0.10):
    n = len(X)
    D = pdist(X); sc = float(np.linalg.norm(X - X.mean(0), axis=1).mean())
    true_k = int(truth.max()) + 1
    sizes_true = [int((truth == j).sum()) for j in range(true_k)]
    print(f"\n=== {name} ===  n={n}, true k={true_k}, sizes {sizes_true}")
    print(f"{'method':>26}{'k found':>9}{'agreement':>12}{'coverage':>10}   sizes")

    _, k_sl, _, _ = plateau(D, sc)
    print(f"{'single + merge plateau':>26}{k_sl:>9}{'--':>12}{'100%':>10}")

    Zl = linkage(X, method="ward")
    k_w, _, _ = count_by_gap(Zl)
    lab = fcluster(Zl, k_w, criterion="maxclust") - 1
    ag = agreement(lab, truth) if k_w <= 6 else float("nan")
    print(f"{'ward + merge-height gap':>26}{k_w:>9}{ag:>11.1%}{'100%':>10}   "
          f"{sorted(np.bincount(lab).tolist(), reverse=True)[:4]}")

    for f in (0.05, 0.10, 0.15):
        mcs = max(3, int(f * n))
        h = HDBSCAN(min_cluster_size=mcs).fit(X)
        lab = h.labels_
        ag, cov, k = score(lab, truth)
        sizes = sorted(np.bincount(lab[lab >= 0]).tolist(), reverse=True)[:4] if (lab >= 0).any() else []
        print(f"{f'HDBSCAN (min {f:.0%} of n)':>26}{k:>9}{ag:>11.1%}{cov:>9.0%}   {sizes}")


# ---------------------------------------------------------------- TOY
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_d_settle import basin, STRIDE                           # noqa: E402

E = np.load(ROOT / "results/phase_c/phase_e_endings_H20_n300.npz")[
    "predictor_gtf_warm.pt_40"].astype(np.float32)
Zl_ = np.load(ROOT / "results/phase_c/latents.npy", mmap_mode="r")
idx = np.linspace(0, Zl_.shape[0] - 1, 300).astype(int)
rng = np.random.default_rng(0); thr = tb.topple_threshold(450, 10, 60)
acts = []
for _ in range(Zl_.shape[0]):
    a = tb.random_push(rng, 450, thr); tb.simulate(a, dt=tb.DT_DEFAULT)
    br.sample_lighting(rng); acts.append(a)
truth_toy = basin(np.array([tb.simulate(
    np.concatenate([acts[i][:20 * STRIDE], np.zeros(40 * STRIDE)]), dt=tb.DT_DEFAULT)[0][-1, 0]
    for i in idx]))
Xc = (E - E.mean(0)).astype(np.float32)
V = np.linalg.svd(Xc[::2], full_matrices=False)[2]
run("TOY, PCA 8", Xc @ V[:8].T, truth_toy)

# ---------------------------------------------------------------- JENGA
import json, torch                                                 # noqa: E402
from jenga_basins import load, remove_arm                          # noqa: E402
from jenga_geometry import encode                                  # noqa: E402
W = Path("/home/sanger/wksp")
eps, F, P, L = load(W / "panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb",
                    W / "panda_express/labels_noise100.json")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).cuda().eval()
Ej = encode(F, model, "cuda").reshape(len(F), -1).astype(np.float32)
R = remove_arm(Ej, P)
truth_j = np.array([L[e]["outcome"] != "success" for e in eps]).astype(int)
Xc = (R - R.mean(0)).astype(np.float32)
V = np.linalg.svd(Xc[::2], full_matrices=False)[2]
run("JENGA, arm removed, PCA 2", Xc @ V[:2].T, truth_j)

print("\ncoverage = share of points NOT called noise. HDBSCAN discarding a few outliers is")
print("the intended behaviour -- it is what stops one bridging point from welding two basins.")
