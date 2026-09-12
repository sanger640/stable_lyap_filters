"""Phase F of PLAN_DINOWM: the monitor, end to end, on DINO-WM.

First time the whole pipeline runs on the image-based model. Comparison target is the shPLRNN's
AUC 0.808 on the same system.

WHAT PHASE E CHANGED. Clustering happens in PCA space, not the raw 98,304-dim latent, and on
PREDICTED endings, never encoded ones. Both are unsupervised, so the calibration-free claim holds:
the attractor count still comes from the merge-distance plateau with nothing supplied, minus
singletons.

WARM START. The monitor's context is the last `num_hist` REAL encoded observations. That is what a
deployed monitor has, and it is required here because a single frame shows theta but not omega.

THE LABEL IS PER-CHUNK, NOT PER-EPISODE. The monitor at time t asks "from the state I am in now,
does this H-step chunk straddle a boundary?". Scoring it against a margin measured on the whole
450-step action from t=0 answers a different question -- an error already made once in this project
(NOTES: "label from the full action while the monitor saw one chunk"). So the oracle bisects the
scaling of THIS chunk from THIS state, at simulator resolution.

ORDER. The false-positive floor runs FIRST. It replaces the vacuous eps=0 null: at the OPERATING
eps, on episodes far from any boundary, no probe should be able to cross one, so any k > 0 is model
error manufacturing dissent. If those already reach k >= 2 the alarm is measuring the model.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval"), "/home/sanger/wksp/dino_wm"]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_c_train import Predictor, NUM_HIST, D                   # noqa: E402
from phase_e_attractors import pdist, plateau                      # noqa: E402

OUT = ROOT / "results" / "phase_c"
STRIDE = 10


def build_centroids(E, pca_dim=8, min_size=3):
    """Phase E's recipe: PCA, merge-distance plateau, drop singletons. Nothing supplied."""
    mean = E.mean(0)
    V = np.linalg.svd((E - mean)[::2], full_matrices=False)[2][:pca_dim]
    X = (E - mean) @ V.T
    Dm = pdist(X); scale = float(np.linalg.norm(X - X.mean(0), axis=1).mean())
    lab, k, width, counts = plateau(Dm, scale)
    keep = [j for j in range(k) if (lab == j).sum() >= min_size]
    C = np.stack([X[lab == j].mean(0) for j in keep])
    return C, mean, V, k, len(keep), width, counts


class Monitor:
    """Basin entropy / dissent count over an H-step chunk, from real observed context."""

    def __init__(self, model, C, mean, V, mu, sd, amu, asd, H, settle, eps, n_probe, dev):
        self.__dict__.update(locals()); del self.self

    @torch.no_grad()
    def score_many(self, ctx_list, chunks, rng):
        """All scoring times for one episode in ONE batched rollout.

        Two speedups over the per-time version, which ran at 55 s/episode:
          * batch is (n_times * n_probe), so the 60 sequential steps are paid once per EPISODE
            rather than once per scoring time
          * the context is a fixed NUM_HIST window, not a tensor grown by torch.cat every step --
            the old version reallocated up to ~800 MB per score for frames it never read
        """
        # Chunk the scoring times so the batch stays within VRAM: 8 times x 32 probes = 256
        # needs ~9.6 GB for attention alone on a 7.5 GB card. Groups of 4 (batch 128) fit.
        if len(chunks) > 4:
            out = []
            for a in range(0, len(chunks), 4):
                out += self.score_many(ctx_list[a:a + 4], chunks[a:a + 4], rng)
            return out
        T, n = len(chunks), self.n_probe
        z = rng.standard_normal((T, n, 1))
        P = np.stack([c[None] * (1.0 + self.eps * z[i]) for i, c in enumerate(chunks)])
        P = P.reshape(T * n, -1)
        P = np.concatenate([P, np.zeros((T * n, self.settle))], axis=1)
        acts = torch.from_numpy(((P[:, :, None] - self.amu) / self.asd).astype(np.float32)
                                ).to(self.dev)
        ctx = np.stack(list(ctx_list))
        zc = (torch.from_numpy(ctx.astype(np.float32)).to(self.dev) - self.mu) / self.sd
        win = zc[:, None].expand(-1, n, -1, -1, -1).reshape(T * n, NUM_HIST, 256, D).contiguous()
        # bf16 for the rollout: the model was TRAINED under bf16 autocast, so these are
        # the same numerics it saw, and it roughly halves the wall clock.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(P.shape[1]):
                a_win = acts[:, max(0, t - NUM_HIST + 1):t + 1]
                if a_win.shape[1] < NUM_HIST:
                    a_win = torch.cat([a_win[:, :1].expand(-1, NUM_HIST - a_win.shape[1], -1),
                                       a_win], 1)
                nxt = self.model(win, a_win)[:, -1:]
                win = torch.cat([win[:, 1:], nxt], 1)
        E = win[:, -1].reshape(T * n, -1).float().cpu().numpy() * self.sd + self.mu
        X = (E - self.mean) @ self.V.T
        lab = ((X[:, None] - self.C[None]) ** 2).sum(-1).argmin(1).reshape(T, n)
        return [int(n - max((lab[i] == j).sum() for j in range(len(self.C))))
                for i in range(T)]


def chunk_margin(s0, chunk_sim, settle_sim, s_max=0.5, coarse=0.04):
    """Smallest fractional scaling of THIS chunk, from THIS state, that flips the outcome."""
    tail = np.zeros(settle_sim)
    run = lambda s: bool(tb.simulate(np.concatenate([chunk_sim * s, tail]),   # noqa: E731
                                     s0=s0, dt=tb.DT_DEFAULT)[1][-1])
    base = run(1.0)
    for i in range(1, int(s_max / coarse) + 1):
        for sg in (+1, -1):
            s = 1.0 + sg * i * coarse
            if s <= 0:
                continue
            if run(s) != base:
                lo, hi = 1.0 + sg * (i - 1) * coarse, s
                for _ in range(12):
                    m = 0.5 * (lo + hi)
                    if run(m) != base:
                        hi = m
                    else:
                        lo = m
                return abs(0.5 * (lo + hi) - 1.0)
    return s_max


def auc(score, lab):
    lab = np.asarray(lab, bool); s = np.asarray(score, float)
    p, q = s[lab], s[~lab]
    if not len(p) or not len(q):
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[order] = np.arange(1, len(s) + 1)
    xs = s[order]; i = 0                                            # average ranks for ties
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[lab].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--settle", type=int, default=40)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--pca", type=int, default=8)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--times", type=int, nargs="+", default=[3, 6, 9, 12, 15, 18, 21, 24])
    ap.add_argument("--near", type=float, default=0.23)
    ap.add_argument("--ckpt", default="predictor_gtf_warm.pt")
    args = ap.parse_args()
    dev = "cuda"; t0 = time.time()

    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    meta = np.load(OUT / "meta.npz"); A = meta["actions"]
    p = np.load(OUT / "phase_c_eval.npz")
    mu, sd, amu, asd = (float(p[k]) for k in ("mu", "sd", "amu", "asd"))
    N = Z.shape[0]

    cache = np.load(OUT / "phase_e_endings_H20_n300.npz")
    E = cache[f"{args.ckpt}_{args.settle}"].astype(np.float32)
    C, pmean, V, k_raw, k_keep, width, counts = build_centroids(E, args.pca)
    print(f"attractors: plateau k={k_raw} (width {width}), {k_keep} after dropping singletons")
    print(f"  counts by d/scale: " + " ".join(f"{f}:{c}" for f, c in counts) + "\n", flush=True)

    model = Predictor().to(dev); model.load_state_dict(torch.load(OUT / args.ckpt)); model.eval()
    mon = Monitor(model, C, pmean, V, mu, sd, amu, asd, args.H, args.settle,
                  args.eps, args.n_probe, dev)

    rng = np.random.default_rng(0); thr = tb.topple_threshold(450, 10, 60)
    acts = []
    for _ in range(N):
        a = tb.random_push(rng, 450, thr); tb.simulate(a, dt=tb.DT_DEFAULT)
        br.sample_lighting(rng); acts.append(a)

    eps_rng = np.random.default_rng(777)
    idx = np.arange(N - args.n_episodes, N)
    print(f"scoring {len(idx)} episodes at t={args.times} "
          f"(H={args.H}, settle={args.settle}, eps={args.eps}, n={args.n_probe})", flush=True)
    rows = []
    for c, i in enumerate(idx):
        st, _ = tb.simulate(acts[i], dt=tb.DT_DEFAULT)
        ts = [t for t in args.times if t + args.H <= Z.shape[1]]
        ctxs = [np.asarray(Z[i, t - NUM_HIST + 1:t + 1], np.float32) for t in ts]
        chunks = [A[i, t:t + args.H, 0].astype(np.float64) for t in ts]
        ks = mon.score_many(ctxs, chunks, eps_rng)
        ms = [chunk_margin(st[t * STRIDE], acts[i][t * STRIDE:(t + args.H) * STRIDE],
                           args.settle * STRIDE) for t in ts]
        rows.append(dict(i=int(i), k=ks, margin=ms,
                         k_max=int(max(ks)), m_min=float(min(ms))))
        if (c + 1) % 10 == 0:
            print(f"  {c+1}/{len(idx)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    K = np.array([r["k_max"] for r in rows]); M = np.array([r["m_min"] for r in rows])
    kk = np.concatenate([r["k"] for r in rows]); mm = np.concatenate([r["margin"] for r in rows])

    print(f"\n{'='*70}\nFALSE-POSITIVE FLOOR (replaces the vacuous eps=0 null)\n{'='*70}")
    for lo in (0.35, 0.40, 0.45):
        far = mm >= lo
        if far.sum():
            print(f"  chunks with margin >= {lo:.2f}  (n={far.sum():4d}):  "
                  f"mean k {kk[far].mean():5.2f}   k=0 on {100*(kk[far]==0).mean():4.0f}%   "
                  f"k>={args.k_min} on {100*(kk[far]>=args.k_min).mean():4.0f}%")
    print(f"  bar: k < {args.k_min} on >= 95% of large-margin chunks")

    print(f"\n{'='*70}\nPER-CHUNK detection (the label the monitor is actually answerable for)"
          f"\n{'='*70}")
    print(f"  {len(mm)} chunks, {(mm < args.near).sum()} near (margin < {args.near})")
    print(f"  AUC of dissent count vs near: {auc(kk, mm < args.near):.3f}")
    print(f"\n  {'near <':>8}{'n':>6}{'prec':>8}{'recall':>8}{'F1':>8}")
    for thr_ in (0.10, 0.15, 0.20, 0.23, 0.30):
        near = mm < thr_; al = kk >= args.k_min
        tp = int((al & near).sum()); fp = int((al & ~near).sum()); fn = int((~al & near).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        print(f"  {thr_:>8.2f}{int(near.sum()):>6}{pr:>8.3f}{rc:>8.3f}"
              f"{2*pr*rc/max(pr+rc,1e-9):>8.3f}")

    print(f"\n{'='*70}\nPER-EPISODE (k_max vs min chunk margin) -- comparable to shPLRNN 0.808"
          f"\n{'='*70}")
    near = M < args.near; al = K >= args.k_min
    tp = int((al & near).sum()); fp = int((al & ~near).sum())
    fn = int((~al & near).sum()); tn = int((~al & ~near).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    print(f"                near     far\n       ALARM {tp:>7}{fp:>8}\n    no alarm {fn:>7}{tn:>8}")
    print(f"  precision {pr:.3f}  recall {rc:.3f}  F1 {2*pr*rc/max(pr+rc,1e-9):.3f}  "
          f"accuracy {(al==near).mean():.3f}")
    print(f"  AUC {auc(K, near):.3f}   (shPLRNN reference 0.808)")
    import json
    json.dump({"rows": rows, "args": vars(args)},
              open(OUT / "phase_f_results_dense.json", "w"), indent=1)
    print(f"\n({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
