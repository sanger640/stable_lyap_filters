"""Phase E of PLAN_DINOWM: do the PREDICTED ending latents form discoverable attractors?

THE FIRST PHASE WHERE THE METHOD, NOT THE MODEL, IS ON TRIAL. Everything so far measured how well
the predictor tracks reality. This asks whether the monitor's own machinery works: cluster the
ending latents with NO k supplied, via the merge-distance plateau, and see whether the count that
emerges is 3 and whether the groups correspond to the three outcomes.

Phase D bounded the difficulty without deciding it. One-step error is 26.9% of ending scale at a
moment when theta decodes to 0.040 rad, so most latent error lives in theta-IRRELEVANT directions
-- shadow pixels, render noise, texture. The open question is whether the error lies in
BASIN-DISCRIMINATIVE directions, and only clustering answers that.

THE CONTROL IS THE POINT. Running the same discovery on ENCODED TRUE endings separates two very
different failures:
    true endings cluster, predicted do not  ->  the MODEL is at fault
    neither clusters                        ->  the REPRESENTATION cannot support the method,
                                                which would also condemn Jenga

Settle length is swept because Phase D showed the latents never stop moving (0.60-0.88% of scale
per step, flatlining rather than converging). Where you truncate the tail changes the ending you
get, so the plateau has to survive that drift.

`merge` and `find_attractors` are imported from eval/phase3_pipeline.py -- the same code that found
3 attractors for the shPLRNN. merge() is O(n^2) in 98k dimensions, so distances are precomputed
once and `merge_D` reproduces the flood fill on the matrix; equivalence is ASSERTED against the
original, not assumed.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "src" / "models"),
                str(ROOT / "eval")]
import tipping_block as tb                                         # noqa: E402
from phase_c_train import Predictor, NUM_HIST                      # noqa: E402
from phase3_pipeline import merge, find_attractors                 # noqa: E402
from phase_d_settle import basin, STRIDE                           # noqa: E402

OUT = ROOT / "results" / "phase_c"


def pdist(X):
    """Pairwise distances via the Gram matrix.

    The obvious `np.linalg.norm(X[:, None] - X[None], axis=-1)` materialises an
    (n, n, 98304) intermediate -- about 35 TB at n=300 -- and gets the process OOM-killed.
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b needs only the (n, n) Gram matrix."""
    G = X @ X.T
    sq = np.diag(G)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))


def merge_D(D, d):
    """Flood fill on a PRECOMPUTED distance matrix. Equivalent to phase3_pipeline.merge(X, d)."""
    n = len(D); lab = -np.ones(n, int); g = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]; lab[i] = g
        while stack:
            j = stack.pop()
            nb = np.where((lab < 0) & (D[j] < d))[0]
            lab[nb] = g; stack.extend(nb.tolist())
        g += 1
    return lab


def plateau(D, scale, fracs=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8)):
    """find_attractors' logic, on a precomputed distance matrix."""
    counts = [(f, merge_D(D, f * scale).max() + 1) for f in fracs]
    runs, cur = [], [counts[0]]
    for prev, c in zip(counts, counts[1:]):
        if c[1] == prev[1]:
            cur.append(c)
        else:
            runs.append(cur); cur = [c]
    runs.append(cur)
    best = max(runs, key=len)
    d = float(np.median([f for f, _ in best]) * scale)
    return merge_D(D, d), best[0][1], len(best), counts


def agreement(lab, truth):
    """Best label permutation, since cluster ids are arbitrary."""
    from itertools import permutations
    ks, kt = int(lab.max()) + 1, int(truth.max()) + 1   # int(): permutations rejects np.int64
    if ks > 6:
        return float("nan")
    best = 0.0
    for p in permutations(range(max(ks, kt)), ks):
        best = max(best, float((np.array(p)[lab] == truth).mean()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--settles", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda"; t0 = time.time()
    L = max(args.settles)

    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    meta = np.load(OUT / "meta.npz"); A = meta["actions"]
    p = np.load(OUT / "phase_c_eval.npz")
    mu, sd, amu, asd = (float(p[k]) for k in ("mu", "sd", "amu", "asd"))
    idx = np.linspace(0, Z.shape[0] - 1, args.n).astype(int)

    print("ground truth basins after H + settle ...", flush=True)
    rng = np.random.default_rng(0); thr = tb.topple_threshold(450, 10, 60)
    import block_render as br
    acts = []
    for _ in range(Z.shape[0]):
        a = tb.random_push(rng, 450, thr); tb.simulate(a, dt=tb.DT_DEFAULT)
        br.sample_lighting(rng); acts.append(a)
    truth = {}
    for s in args.settles:
        th = []
        for i in idx:
            a = np.concatenate([acts[i][:args.H * STRIDE], np.zeros(s * STRIDE)])
            th.append(tb.simulate(a, dt=tb.DT_DEFAULT)[0][-1, 0])
        truth[s] = basin(np.array(th))
        print(f"  settle {s:>2}: true basins {[int((truth[s]==j).sum()) for j in range(3)]}")

    zero_a = (0.0 - amu) / asd
    runs = {}
    cache = OUT / f"phase_e_endings_H{args.H}_n{args.n}.npz"
    if cache.exists():
        d = np.load(cache)
        for tag, ck in (("single-step (dino_wm recipe)", "predictor.pt"),
                        ("GTF warm-started", "predictor_gtf_warm.pt")):
            runs[tag] = {s_: d[f"{ck}_{s_}"] for s_ in args.settles}
        print(f"  loaded cached endings from {cache.name}", flush=True)
    for tag, ck in ([] if runs else (("single-step (dino_wm recipe)", "predictor.pt"),
                                     ("GTF warm-started", "predictor_gtf_warm.pt"))):
        model = Predictor().to(dev); model.load_state_dict(torch.load(OUT / ck)); model.eval()
        E = {s: [] for s in args.settles}
        with torch.no_grad():
            for i in idx:
                z = ((torch.from_numpy(np.asarray(Z[i, :NUM_HIST], np.float32)).to(dev)
                      - mu) / sd)[None]
                aa = (torch.from_numpy(A[i]).to(dev) - amu) / asd
                for t in range(NUM_HIST, args.H):
                    z = torch.cat([z, model(z[:, -NUM_HIST:], aa[None, t-NUM_HIST:t])[:, -1:]], 1)
                az = torch.full((1, NUM_HIST, 1), zero_a, device=dev, dtype=z.dtype)
                for j in range(1, L + 1):
                    z = torch.cat([z, model(z[:, -NUM_HIST:], az)[:, -1:]], 1)
                    if j in E:
                        E[j].append(z[0, -1].float().cpu().numpy().ravel() * sd + mu)
        runs[tag] = {s: np.stack(v) for s, v in E.items()}
        print(f"  rolled {tag} ({time.time()-t0:.0f}s)", flush=True)

    if not cache.exists():
        np.savez(cache, **{f"{ck}_{s_}": runs[tag][s_]
                           for tag, ck in (("single-step (dino_wm recipe)", "predictor.pt"),
                                           ("GTF warm-started", "predictor_gtf_warm.pt"))
                           for s_ in args.settles})
        print(f"  cached endings -> {cache.name}", flush=True)

    # control: ENCODED TRUE endings (upper bound -- no model error at all)
    ctrl = np.stack([np.asarray(Z[i, -1], np.float32).ravel() for i in idx])
    runs["CONTROL: encoded true endings"] = {args.settles[-1]: ctrl}

    # verify merge_D reproduces phase3_pipeline.merge exactly
    sm = ctrl[:40]
    Dm = pdist(sm)
    sc = float(np.linalg.norm(sm - sm.mean(0), axis=1).mean())
    assert (merge_D(Dm, 0.3 * sc) == merge(sm, 0.3 * sc)).all(), "merge_D != phase3_pipeline.merge"
    print("  merge_D verified against phase3_pipeline.merge\n", flush=True)

    print(f"{'model':<32}{'settle':>7}{'k found':>9}{'plateau':>9}{'agree':>8}   counts by d/scale")
    print("-" * 96)
    for tag, byS in runs.items():
        for s, Es in byS.items():
            D = pdist(Es)
            scale = float(np.linalg.norm(Es - Es.mean(0), axis=1).mean())
            lab, k, width, counts = plateau(D, scale)
            tr = truth[s] if s in truth else truth[max(truth)]
            ag = agreement(lab, tr)
            print(f"{tag:<32}{s:>7}{k:>9}{width:>9}{ag:>8.1%}   "
                  + " ".join(f"{f}:{c}" for f, c in counts))
    print(f"\nACCEPTANCE: plateau at k=3 with >= 80% agreement")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
