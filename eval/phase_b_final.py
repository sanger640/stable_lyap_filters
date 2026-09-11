"""Phase B, corrected. Two questions the first attempt could not answer.

FIX 1 -- numerics. The first probe standardised the PCA components, which divides the
low-variance ones by tiny numbers and turns them into amplified noise, leaving the normal-equation
solve ill-conditioned. The tell was R^2 moving NON-MONOTONICALLY in lambda
(0.972 -> -0.476 -> -0.872 -> 0.717), which ridge never does. Fixed: SVD-based solve, centre only,
no per-feature scaling.

FIX 2 -- lambda is chosen on a VALIDATION split of the training trajectories, never on test.
The earlier sweep reported test R^2 at every lambda, which is test-set selection.

RUN A -- the sampling rate, under FIXED lighting so the lighting confound cannot contaminate it.
RUN B -- how many lighting samples does invariance need? The failing probe saw 32 training
episodes = 32 lighting draws. Phase C will have 600. Sweep the count and find out whether the
varied-lighting failure is a sample-size artefact or a real barrier.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval")]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_b_probe import encode, build, make_trajectories         # noqa: E402

CACHE = ROOT / "results" / "phase_a"
LAMS = np.logspace(-8, 4, 13)


def ridge_svd(Xtr, ytr, Xva, yva, Xte, yte):
    """Centre-only ridge via SVD. lambda picked on VALIDATION, reported on TEST."""
    mu, ym = Xtr.mean(0), ytr.mean(0)
    U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    UtY = U.T @ (ytr - ym)

    def r2(W, X, y):
        p = (X - mu) @ W + ym
        return 1.0 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)

    best = (-np.inf, None, None)
    for lam in LAMS:
        W = Vt.T @ ((S / (S ** 2 + lam))[:, None] * UtY)
        v = r2(W, Xva, yva)
        if v.mean() > best[0]:
            best = (v.mean(), lam, W)
    return r2(best[2], Xte, yte), best[1]


def pca_fit(flat_sub, k=256):
    mu = flat_sub.mean(0)
    _, _, Vt = np.linalg.svd(flat_sub - mu, full_matrices=False)
    return mu, Vt[:k]


def evaluate(F, states, n_tr, n_va, stride, num_hist=3):
    X, Y, G = build(F, states, list(range(len(F))), stride, num_hist)
    tr, va = G < n_tr, (G >= n_tr) & (G < n_tr + n_va)
    te = G >= n_tr + n_va
    return ridge_svd(X[tr], Y[tr], X[va], Y[va], X[te], Y[te])


# ---------------------------------------------------------------- RUN A: sampling rate
print("=== RUN A: sampling rate (cached features, corrected probe) ===", flush=True)
for tag in ("fixed", "varied"):
    d = np.load(CACHE / f"feat_{tag}.npz")
    n = sum(1 for k in d.files if k.startswith("f"))
    flat = [d[f"f{i}"] for i in range(n)]
    states = [d[f"s{i}"] for i in range(n)]
    allf = np.concatenate(flat)
    mu, V = pca_fit(allf[::max(1, len(allf) // 3000)])
    F = [(f - mu) @ V.T for f in flat]
    del allf, flat
    print(f"\n  lighting {tag.upper()}   ({n} traj: 24 train / 8 val / 8 test)")
    print(f"  {'stride':>7}{'dt_eff':>9}{'steps/ep':>10}{'theta R2':>11}{'omega R2':>11}"
          f"{'lambda':>10}")
    for s in (1, 2, 3, 5, 10):
        r2, lam = evaluate(F, states, 24, 8, s)
        print(f"  {s:>7}{s*tb.DT_DEFAULT:>9.3f}{450//s:>10}{r2[0]:>11.3f}{r2[1]:>11.3f}"
              f"{lam:>10.1e}", flush=True)
    del F, d

# ---------------------------------------------------------------- RUN B: lighting sample size
print("\n\n=== RUN B: how many lighting draws does invariance need? ===", flush=True)
N_TRAJ, SUB = 160, 3            # render every 3rd sim step -> 150 frames/episode
t0 = time.time()
trajs = make_trajectories(N_TRAJ, 450, seed=7)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(dev).eval()

raw, states = [], []
for st, light, i in trajs[:15]:                       # first 15 kept raw, to fit PCA
    bg = br.prepare(light, seed=i)
    fr = np.stack([br.render(th, light, bg=bg) for th in st[::SUB, 0]])
    raw.append(encode(fr, model, dev).reshape(len(fr), -1)); states.append(st[::SUB])
sub = np.concatenate(raw)[::5]
mu, V = pca_fit(sub)
F = [(r - mu) @ V.T for r in raw]
del raw, sub
for st, light, i in trajs[15:]:                       # rest projected on the fly
    bg = br.prepare(light, seed=i)
    fr = np.stack([br.render(th, light, bg=bg) for th in st[::SUB, 0]])
    F.append((encode(fr, model, dev).reshape(len(fr), -1) - mu) @ V.T)
    states.append(st[::SUB])
print(f"  {N_TRAJ} episodes x {len(F[0])} frames rendered+encoded ({time.time()-t0:.0f}s)\n",
      flush=True)

print(f"  {'train eps':>10}{'theta R2':>11}{'omega R2':>11}{'lambda':>10}   (32 test eps held out)")
n_te = 32
for n_tr in (8, 16, 32, 64, 96, 128 - 16):
    n_va = 16
    idx = list(range(n_tr + n_va)) + list(range(N_TRAJ - n_te, N_TRAJ))
    Fs = [F[i] for i in idx]; Ss = [states[i] for i in idx]
    r2, lam = evaluate(Fs, Ss, n_tr, n_va, stride=1)   # stride 1 here = 3 sim steps
    print(f"  {n_tr:>10}{r2[0]:>11.3f}{r2[1]:>11.3f}{lam:>10.1e}", flush=True)

print("\nacceptance: theta R^2 > 0.95, omega R^2 > 0.80")
