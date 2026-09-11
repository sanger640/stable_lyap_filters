"""Can the attractors be recovered at all, and by what procedure?

Phase E's single-linkage merge-distance sweep returned k=300 everywhere -- for the CONTROL
(encoded truth, zero model error) too. That rules out the model and indicts the procedure:
single-linkage relies on LOCAL structure (nearest-neighbour chains), which concentrates badly in
98,304 dimensions. The Phase B geometry test reached 100% nearest-centroid in this same space,
and that relies on GLOBAL structure. So the question is not whether the information is there.

Two levers, separated:
  1. PCA before clustering -- does dropping low-variance nuisance directions open a scale gap?
  2. k-means with the count swept by STABILITY across seeds -- still no k supplied, the same
     "let the data pick the count" principle as the plateau, but centroid-based.

k-means assignment uses ||x-c||^2 = ||x||^2 - 2x.c + ||c||^2 rather than an (n, k, d) broadcast,
which is what timed out at full dimension (and is the same mistake that OOM-killed Phase E).
"""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval"), "/home/sanger/wksp/dino_wm"]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_d_settle import basin, STRIDE                           # noqa: E402
from phase_e_attractors import pdist, agreement                    # noqa: E402

OUT = ROOT / "results" / "phase_c"


def kmeans(X, k, seed=0, iters=50):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    xs = (X ** 2).sum(1)[:, None]
    for _ in range(iters):
        lab = (xs - 2.0 * (X @ C.T) + (C ** 2).sum(1)[None]).argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab


def stable_k(X, ks=(2, 3, 4, 5, 6), seeds=4):
    """Pick k by co-assignment reproducibility across seeds. No k supplied by hand."""
    out = {}
    for k in ks:
        ls = [kmeans(X.copy(), k, seed=s) for s in range(seeds)]
        co = [(l[:, None] == l[None]) for l in ls]
        out[k] = float(np.mean([(co[a] == co[b]).mean()
                                for a in range(seeds) for b in range(a + 1, seeds)]))
    return max(out, key=out.get), out


def main():
    t0 = time.time()
    d = np.load(OUT / "phase_e_endings_H20_n300.npz")
    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    idx = np.linspace(0, Z.shape[0] - 1, 300).astype(int)

    rng = np.random.default_rng(0); thr = tb.topple_threshold(450, 10, 60)
    acts = []
    for _ in range(Z.shape[0]):
        a = tb.random_push(rng, 450, thr); tb.simulate(a, dt=tb.DT_DEFAULT)
        br.sample_lighting(rng); acts.append(a)
    th = np.array([tb.simulate(np.concatenate([acts[i][:20 * STRIDE], np.zeros(40 * STRIDE)]),
                               dt=tb.DT_DEFAULT)[0][-1, 0] for i in idx])
    truth = basin(th)
    print(f"true basins {[int((truth==j).sum()) for j in range(3)]}   "
          f"majority baseline {max(np.bincount(truth))/len(truth):.1%}\n", flush=True)

    sets = {"CONTROL encoded truth": np.stack([np.asarray(Z[i, -1], np.float32).ravel()
                                               for i in idx]),
            "predicted (single-step)": d["predictor.pt_40"].astype(np.float32),
            "predicted (GTF warm)": d["predictor_gtf_warm.pt_40"].astype(np.float32)}

    for name, E in sets.items():
        print(f"{'='*62}\n{name}\n{'='*62}")
        print(f"{'PCA dim':>8}{'sep ratio':>11}{'k=3 agree':>11}{'stable k':>10}"
              f"{'  stability by k':>18}")
        Xc = (E - E.mean(0)).astype(np.float32)
        V = np.linalg.svd(Xc[::2], full_matrices=False)[2]
        for kdim in (0, 2, 4, 8, 16, 32, 64):
            X = Xc if kdim == 0 else Xc @ V[:kdim].T
            D = pdist(X); scale = float(np.linalg.norm(X - X.mean(0), axis=1).mean())
            iu = np.triu_indices(len(X), 1); pw = D[iu] / scale
            same = truth[iu[0]] == truth[iu[1]]
            sep = np.median(pw[~same]) / np.median(pw[same])
            ag = agreement(kmeans(X.copy(), 3), truth)
            bk, st = stable_k(X)
            print(f"{kdim if kdim else 'full':>8}{sep:>11.3f}{ag:>11.1%}{bk:>10}"
                  f"   " + " ".join(f"{k}:{v:.2f}" for k, v in st.items()), flush=True)
        print()
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
