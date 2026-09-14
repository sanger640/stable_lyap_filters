"""Can the basin COUNT be discovered without labels, using a linkage that is not bridge-sensitive?

Single-linkage merges on the NEAREST pair, so one coincidental close pair welds two clumps into
one. Measured on Jenga: ep45 (block over at 94.5 deg) and ep16 (barely moved, 7.0 deg) sit
0.120*scale apart while the median within-group nearest-neighbour distance is 0.115*scale -- a
ratio of 1.04. Single-linkage chains straight through, and the merge-distance plateau returns
nonsense even though k-means AT k=2 scores 98%.

Average-linkage merges on the MEAN distance between two groups and Ward merges on the increase in
within-group variance. Neither can be bridged by a single pair. This checks whether either recovers
the count with nothing supplied.

COUNT CRITERION, still unsupervised: build the dendrogram, walk the merge heights, and cut at the
LARGEST RELATIVE JUMP. Two well-separated clumps produce many cheap merges (joining points inside a
clump) then one expensive one (joining the clumps), so the jump marks the boundary.

Also writes a contact sheet per discovered cluster so the grouping can be eyeballed rather than
taken on trust.
"""
import argparse, io, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
from phase_e_attractors import pdist, agreement                    # noqa: E402
from jenga_geometry import encode                                  # noqa: E402
from jenga_basins import load, remove_arm, kmeans                   # noqa: E402

W = Path("/home/sanger/wksp")
OUT = ROOT / "results" / "jenga"


def count_by_gap(Zl, kmax=8):
    """Cut the dendrogram at the largest RELATIVE jump in merge height. Nothing supplied."""
    h = Zl[:, 2]
    tail = h[-kmax:]                                # the last kmax merges
    ratios = tail[1:] / np.maximum(tail[:-1], 1e-12)
    j = int(np.argmax(ratios))                      # biggest jump
    k = len(tail) - j - 1                           # clusters remaining just before it
    return max(k, 1), h, ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(W / "panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb"))
    ap.add_argument("--labels", default=str(W / "panda_express/labels_noise100.json"))
    ap.add_argument("--pca", type=int, default=2)
    ap.add_argument("--per-clump", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    eps, F, P, L = load(args.lmdb, args.labels)
    fail = np.array([L[e]["outcome"] != "success" for e in eps]).astype(int)
    tilt = np.array([L[e]["peak_tilt_deg"] for e in eps])
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           verbose=False).cuda().eval()
    E = encode(F, model, "cuda").reshape(len(F), -1).astype(np.float32)
    R = remove_arm(E, P)
    Xc = (R - R.mean(0)).astype(np.float32)
    X = Xc @ np.linalg.svd(Xc[::2], full_matrices=False)[2][:args.pca].T
    D = pdist(X)
    print(f"{len(eps)} episodes, {fail.sum()} toppled / {(1-fail).sum()} intact "
          f"(arm removed, PCA {args.pca})\n")

    cond = squareform(D, checks=False)
    print(f"{'linkage':>10}{'k found':>9}{'agreement':>12}{'sizes':>14}   merge-height jumps (last 6)")
    print("-" * 82)
    results = {}
    for meth in ("single", "average", "complete", "ward"):
        Zl = linkage(X if meth == "ward" else cond, method=meth)
        k, h, ratios = count_by_gap(Zl)
        lab = fcluster(Zl, k, criterion="maxclust") - 1
        ag = agreement(lab, fail) if k <= 6 else float("nan")
        sizes = sorted(np.bincount(lab).tolist(), reverse=True)[:3]
        results[meth] = (k, ag, lab)
        print(f"{meth:>10}{k:>9}{ag:>11.1%}{str(sizes):>14}   "
              + " ".join(f"{r:.2f}" for r in ratios[-6:]))
    print(f"\n{'k-means at k=2 (reference)':>36}{agreement(kmeans(X,2), fail):>11.1%}")
    print("\ncount criterion: cut at the largest RELATIVE jump in merge height -- unsupervised.")

    # ---- contact sheets from the best method that found k=2
    pick = next((m for m in ("ward", "average", "complete") if results[m][0] == 2), None)
    if pick is None:
        print("\nno linkage found k=2; not writing contact sheets"); return
    k, ag, lab = results[pick]
    print(f"\nwriting contact sheets from '{pick}' (k={k}, agreement {ag:.1%})")
    rng = np.random.default_rng(0)
    for c in range(k):
        idx = np.where(lab == c)[0]
        sel = idx[np.argsort(tilt[idx])]                       # spread across the tilt range
        sel = sel[np.linspace(0, len(sel) - 1, min(args.per_clump, len(sel))).astype(int)]
        tiles, labels = [], []
        for i in sel:
            tiles.append(F[i]); labels.append(f"ep{eps[i]} {tilt[i]:.0f}d")
        cols = min(5, len(tiles)); rows = int(np.ceil(len(tiles) / cols))
        h_, w_ = F.shape[1], F.shape[2]; pad = 6
        sheet = np.full((rows * (h_ + 22) + pad, cols * (w_ + pad) + pad, 3), 245, np.uint8)
        for n, t_ in enumerate(tiles):
            r_, c_ = divmod(n, cols)
            y = pad + r_ * (h_ + 22); x = pad + c_ * (w_ + pad)
            sheet[y:y + h_, x:x + w_] = t_
        im = Image.fromarray(sheet)
        from PIL import ImageDraw
        d = ImageDraw.Draw(im)
        for n, lb in enumerate(labels):
            r_, c_ = divmod(n, cols)
            d.text((pad + c_ * (w_ + pad) + 3, pad + r_ * (h_ + 22) + h_ + 4), lb, fill=(20, 20, 20))
        mean_t = tilt[idx].mean(); n_fail = int(fail[idx].sum())
        name = OUT / f"clump{c}_n{len(idx)}_meantilt{mean_t:.0f}deg.png"
        im.save(name)
        print(f"  clump {c}: n={len(idx):>3}  mean tilt {mean_t:5.1f} deg  "
              f"toppled {n_fail}/{len(idx)}  -> {name.name}")


if __name__ == "__main__":
    main()
