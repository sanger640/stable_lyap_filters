"""JENGA GO/NO-GO: are the basins there, and can they be found without labels?

Run order: this, then `jenga_geometry.py` for the raw-vs-arm-removed comparison.

WHAT THIS ESTABLISHED (100 labelled episodes, final real frames, encoder only -- no world model):

 1. The basins EXIST, more cleanly than on the toy. Peak neighbour tilt is sharply bimodal: 75
    episodes under 19 deg, 25 above 90 deg, and NOTHING in between -- a 72.2 deg gap. The 45 deg
    "topple" threshold sits in empty space, so the label is not a judgement call.

 2. The ARM dominates the raw latents and must be removed. PC1 (25% of variance) tracks
    end-effector pose; the topple signal sits in PC2/PC4 (11.5%, 5.7%) and is entangled with EE x.
    Regressing proprio (plus quadratic terms) out of the latents uses NO topple labels -- proprio is
    the robot's own state, known at runtime -- so it keeps the method calibration-free.

        separation   1.50 -> 2.15     (toy reference: 2.2)
        nearest-centroid  94% -> 98-99%   (base rate 75%)
        corr with peak tilt  0.54 -> 0.77

 3. The representation carries the distinction. 98-99% leave-one-out nearest centroid, and latent
    distance tracks HOW FAR the block tipped at r = 0.77 -- a graded proximity signal, which is
    closer to what this method predicts than the binary outcome is.

 4. DISCOVERY is the blocker, not representation. Two independent count-selection methods fail:
      * merge-distance plateau: ONE bridging pair (ep45 at 94.5 deg vs ep16 at 7.0 deg) sits
        0.120*scale apart while the median within-group nearest-neighbour distance is 0.115*scale
        -- a ratio of 1.04. Single-linkage chains straight through it and returns k=1 or k=100.
      * cross-seed stability: picks k=3 (0.880) over the true k=2 (0.801).
    But k-means AT k=2 agrees with the topple label 98.0%. The structure is findable; the automatic
    count is not.

NEXT: single-linkage is famously bridge-sensitive, so try a linkage that is not -- average or Ward
-- before concluding the count cannot be discovered. That is the one gap between here and a working
Jenga monitor on the representation side.
"""
import argparse, io, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
from phase_e_attractors import pdist, plateau, agreement           # noqa: E402
from jenga_geometry import encode                                  # noqa: E402

W = Path("/home/sanger/wksp")


def load(lmdb_path, labels_path, size=196):
    import lmdb
    L = json.load(open(labels_path))
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
    with env.begin() as t:
        meta = pickle.loads(t.get(b"__metadata__"))["episodes"]
        eps = sorted(meta, key=int)
        F, P = [], []
        for e in eps:
            k = meta[e]["keys"]["cam2"][-1]
            F.append(np.asarray(Image.open(io.BytesIO(t.get(k.encode()))).convert("RGB")
                                .resize((size, size), Image.BICUBIC)))
            P.append(np.asarray(pickle.loads(t.get(f"{e}_proprio".encode())))[-1])
    return eps, np.stack(F), np.stack(P, dtype=np.float64), L


def remove_arm(E, P):
    """Project out everything proprio can explain. Uses no outcome labels."""
    A = np.hstack([P, P ** 2, np.ones((len(P), 1))])
    return E - A @ np.linalg.lstsq(A, E, rcond=None)[0]


def kmeans(Z, k, seed=0, iters=60):
    r = np.random.default_rng(seed)
    C = Z[r.choice(len(Z), k, replace=False)].copy()
    xs = (Z ** 2).sum(1)[:, None]
    for _ in range(iters):
        lab = (xs - 2 * Z @ C.T + (C ** 2).sum(1)[None]).argmin(1)
        for q in range(k):
            if (lab == q).any():
                C[q] = Z[lab == q].mean(0)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(W / "panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb"))
    ap.add_argument("--labels", default=str(W / "panda_express/labels_noise100.json"))
    ap.add_argument("--pca", type=int, default=2)
    args = ap.parse_args()

    eps, F, P, L = load(args.lmdb, args.labels)
    fail = np.array([L[e]["outcome"] != "success" for e in eps]).astype(int)
    tilt = np.array([L[e]["peak_tilt_deg"] for e in eps])
    print(f"{len(eps)} episodes: {fail.sum()} toppled / {(1-fail).sum()} intact")

    srt = np.sort(tilt); g = np.diff(srt); i = int(g.argmax())
    print(f"peak tilt is BIMODAL: largest gap {g[i]:.1f} deg between {srt[i]:.1f} and {srt[i+1]:.1f}; "
          f"{int(((tilt>20)&(tilt<60)).sum())} episodes in 20-60 deg\n")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           verbose=False).cuda().eval()
    E = encode(F, model, "cuda").reshape(len(F), -1).astype(np.float32)

    for name, M in (("raw", E), ("arm removed", remove_arm(E, P))):
        Xc = (M - M.mean(0)).astype(np.float32)
        X = Xc @ np.linalg.svd(Xc[::2], full_matrices=False)[2][:args.pca].T
        D = pdist(X); sc = float(np.linalg.norm(X - X.mean(0), axis=1).mean())
        iu = np.triu_indices(len(X), 1); pw = D[iu] / sc
        same = fail[iu[0]] == fail[iu[1]]
        inter = D[np.ix_(fail == 1, fail == 0)]
        nn = np.array([np.sort(D[i][fail == fail[i]])[1] for i in range(len(X))])
        lab, k, _, _ = plateau(D, sc)
        print(f"=== {name} (PCA {args.pca}) ===")
        print(f"  separation (median between / within) : {np.median(pw[~same])/np.median(pw[same]):.3f}")
        print(f"  single-linkage bridge ratio          : {inter.min()/np.median(nn):.2f}  "
              f"(needs > ~1.5; ONE close pair welds the clumps)")
        print(f"  merge-distance plateau               : k={k}")
        print(f"  k-means at k=2, agreement            : {agreement(kmeans(X,2), fail):.1%}")
        print(f"  corr(distance from intact centroid, peak tilt): "
              f"{np.corrcoef(np.linalg.norm(X - X[fail==0].mean(0), axis=1), tilt)[0,1]:.3f}\n")


if __name__ == "__main__":
    main()
