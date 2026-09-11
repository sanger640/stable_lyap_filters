"""Why did the Phase B probe fail? Diagnose before accepting the kill.

theta is DIRECTLY VISIBLE in a single frame, so a linear probe scoring R^2=0.856 on it is evidence
that the PROBE is wrong, not that the representation is. And omega scoring NEGATIVE R^2 means the
fit does worse than predicting the mean -- a generalisation failure, not an information failure.

Four candidate causes, each separable:
  A. ridge lambda badly chosen           -> sweep it
  B. generalisation across TRAJECTORIES  -> compare trajectory-split vs frame-split
  C. per-episode LIGHTING is the confound-> re-run with lighting held fixed
  D. omega's scale/distribution          -> just look at it

Caches render+encode so the variants are cheap.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval")]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_b_probe import encode, ridge_r2, build, make_trajectories   # noqa: E402

CACHE = ROOT / "results" / "phase_a"
N_TRAJ, STEPS, PCA = 40, 450, 256


def corpus(fixed_light, tag):
    """Render + encode once per lighting condition, then cache."""
    f = CACHE / f"feat_{tag}.npz"
    if f.exists():
        d = np.load(f)
        return [d[f"f{i}"] for i in range(N_TRAJ)], [d[f"s{i}"] for i in range(N_TRAJ)]
    trajs = make_trajectories(N_TRAJ, STEPS, seed=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           verbose=False).to(dev).eval()
    one = br.sample_lighting(np.random.default_rng(12345))
    feats, states = [], []
    t0 = time.time()
    for st, light, i in trajs:
        L = one if fixed_light else light
        bg = br.prepare(L, seed=0 if fixed_light else i)
        fr = np.stack([br.render(th, L, bg=bg) for th in st[:, 0]])
        feats.append(encode(fr, model, dev).reshape(len(fr), -1))
        states.append(st)
    print(f"  [{tag}] rendered+encoded in {time.time()-t0:.0f}s", flush=True)
    np.savez(f, **{f"f{i}": x for i, x in enumerate(feats)},
             **{f"s{i}": x for i, x in enumerate(states)})
    return feats, states


def pca_project(flat, k=PCA):
    allf = np.concatenate(flat)
    sub = allf[::max(1, len(allf) // 3000)]
    mu = sub.mean(0)
    _, _, Vt = np.linalg.svd(sub - mu, full_matrices=False)
    V = Vt[:k]
    return [(f - mu) @ V.T for f in flat]


print("=== D. what does omega actually look like? ===")
_, states0 = corpus(False, "varied")
om = np.concatenate([s[:, 1] for s in states0])
th = np.concatenate([s[:, 0] for s in states0])
print(f"  theta  std {th.std():.3f}  |  pct |.|<0.01: {100*(np.abs(th)<0.01).mean():.0f}%")
print(f"  omega  std {om.std():.3f}  |  pct |.|<0.01: {100*(np.abs(om)<0.01).mean():.0f}%"
      f"  |  max |omega| {np.abs(om).max():.2f}")
print(f"  omega kurtosis {((om-om.mean())**4).mean()/om.var()**2:.1f}  (3 = gaussian)")

for tag, fixed in (("varied", False), ("fixed", True)):
    print(f"\n=== lighting {tag.upper()} ===", flush=True)
    flat, states = corpus(fixed, tag)
    F = pca_project(flat)
    gid = list(range(N_TRAJ))
    X, Y, G = build(F, states, gid, stride=3, num_hist=3)
    ntr = int(0.8 * N_TRAJ)
    traj_m = G < ntr
    rng = np.random.default_rng(0)
    frame_m = rng.random(len(X)) < 0.8                       # B: frame-level split (leaky)

    print(f"{'lambda':>10}{'traj-split th':>15}{'traj-split om':>15}"
          f"{'frame-split th':>16}{'frame-split om':>16}")
    for lam in (1e-2, 1.0, 1e2, 1e4, 1e6):
        a = ridge_r2(X[traj_m], Y[traj_m], X[~traj_m], Y[~traj_m], lam)
        b = ridge_r2(X[frame_m], Y[frame_m], X[~frame_m], Y[~frame_m], lam)
        print(f"{lam:>10.0e}{a[0]:>15.3f}{a[1]:>15.3f}{b[0]:>16.3f}{b[1]:>16.3f}", flush=True)
