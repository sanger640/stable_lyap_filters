"""Phase B of PLAN_DINOWM.md: can (theta, omega) be linearly decoded from DINOv2 patch features?

THE KILL TEST. A single frame shows theta but NOT omega -- angular velocity is invisible in a
static image -- so the model must infer it from `num_hist` frames. If a LINEAR probe cannot
recover omega from three frames of patch features, then the representation does not carry the
information and no predictor trained on it will either. Better to learn that here than after
Phase C.

The sampling rate is the second question, and it is genuinely open. NOTES records that coarsening
450 -> 45 steps was harmless, but that was measured with omega HANDED to the model. From images,
omega comes from differences between frames: coarse sampling means large inter-frame motion (risk
of aliasing during fast rocking) and fine sampling means sub-pixel motion (risk of falling below
the render's resolution). There should be a sweet spot; this finds it.

Probing on a PCA subspace rather than the raw 256x384 grid keeps the regression tractable. PCA is
linear, so this is exactly a linear probe restricted to the leading subspace -- slightly weaker
than a full linear probe, never stronger, so a PASS here is conservative.

A raw-pixel probe runs alongside as a control: if downsampled pixels do as well, DINOv2 is not
contributing anything on this input and the whole representation swap tests less than intended.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval")]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from run_phase3_monitor import T_ON, T_OFF                         # noqa: E402

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def make_trajectories(n_traj, n_steps, seed=0):
    """Trajectories plus their per-episode lighting. Lighting is fixed WITHIN an episode."""
    rng = np.random.default_rng(seed)
    thr = tb.topple_threshold(n_steps, T_ON, T_OFF)
    out = []
    for i in range(n_traj):
        act = tb.random_push(rng, n_steps, thr)
        st, _ = tb.simulate(act, dt=tb.DT_DEFAULT)
        out.append((st[:n_steps], br.sample_lighting(rng), i))
    return out


@torch.no_grad()
def encode(frames, model, dev, batch=96):
    """(N,H,W,3) uint8 -> (N, 256, 384) patch tokens, float32 on CPU."""
    out = []
    for i in range(0, len(frames), batch):
        x = frames[i:i + batch].astype(np.float32) / 255.0
        x = (x - IMNET_MEAN) / IMNET_STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
        out.append(model.forward_features(x)["x_norm_patchtokens"].float().cpu())
    return torch.cat(out).numpy()


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    """Closed-form ridge with an intercept; returns R^2 per target on the test split."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    ym = ytr.mean(0)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ym))
    pred = B @ W + ym
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot


def build(feats, states, traj_id, stride, num_hist=3):
    """Stack `num_hist` frames spaced `stride` apart -> predict (theta, omega) at the last one."""
    X, Y, G = [], [], []
    for f, s, g in zip(feats, states, traj_id):
        need = (num_hist - 1) * stride
        for t in range(need, len(f)):
            X.append(np.concatenate([f[t - k * stride] for k in range(num_hist - 1, -1, -1)]))
            Y.append(s[t]); G.append(g)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=40)
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--num-hist", type=int, default=3)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"generating {args.n_traj} trajectories x {args.steps} steps ...", flush=True)
    trajs = make_trajectories(args.n_traj, args.steps, args.seed)

    print("rendering ...", flush=True)
    frames, states, gid = [], [], []
    for st, light, i in trajs:
        bg = br.prepare(light, seed=i)
        frames.append(np.stack([br.render(th, light, bg=bg) for th in st[:, 0]]))
        states.append(st); gid.append(i)
    print(f"  {sum(len(f) for f in frames):,} frames  ({time.time()-t0:.0f}s)", flush=True)

    print("encoding with dinov2_vits14 ...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(dev).eval()
    te = time.time()
    patch = [encode(f, model, dev) for f in frames]
    n_all = sum(len(p) for p in patch)
    print(f"  {n_all/(time.time()-te):.0f} frames/s", flush=True)

    # PCA on a subsample of flattened patch grids; spatial layout is preserved (it is linear)
    flat = [p.reshape(len(p), -1) for p in patch]
    sub = np.concatenate(flat)[::max(1, n_all // 3000)]
    mu = sub.mean(0)
    _, S, Vt = np.linalg.svd(sub - mu, full_matrices=False)
    V = Vt[:args.pca]
    print(f"  PCA {args.pca} comps, {100*(S[:args.pca]**2).sum()/(S**2).sum():.1f}% variance")
    dino = [(f - mu) @ V.T for f in flat]

    # control: raw 28x28 greyscale pixels
    pix = [f[:, ::8, ::8].mean(-1).reshape(len(f), -1).astype(np.float32) / 255.0 for f in frames]

    ntr = int(0.8 * args.n_traj)
    print(f"\nnum_hist={args.num_hist}, train {ntr} traj / test {args.n_traj-ntr} traj "
          f"(split by TRAJECTORY, so adjacent frames cannot leak)\n")
    print(f"{'stride':>7}{'dt_eff':>9}{'DINO theta':>12}{'DINO omega':>12}"
          f"{'pix theta':>11}{'pix omega':>11}")
    print("-" * 62)
    rows = {}
    for s in args.strides:
        line = [f"{s:>7}{s*tb.DT_DEFAULT:>9.3f}"]
        r = {}
        for name, F in (("dino", dino), ("pix", pix)):
            X, Y, G = build(F, states, gid, s, args.num_hist)
            m = G < ntr
            r2 = ridge_r2(X[m], Y[m], X[~m], Y[~m])
            r[name] = r2
            line.append(f"{r2[0]:>12.3f}{r2[1]:>12.3f}" if name == "dino"
                        else f"{r2[0]:>11.3f}{r2[1]:>11.3f}")
        rows[s] = {k: v.tolist() for k, v in r.items()}
        print("".join(line), flush=True)

    print("-" * 62)
    print("acceptance: theta R^2 > 0.95, omega R^2 > 0.80")
    ok = [s for s in args.strides if rows[s]["dino"][0] > 0.95 and rows[s]["dino"][1] > 0.80]
    if ok:
        print(f"  PASS at strides {ok}; coarsest clearing the bar = {max(ok)} "
              f"(dt_eff {max(ok)*tb.DT_DEFAULT:.3f}s, {args.steps//max(ok)} steps/episode)")
    else:
        print("  FAIL at every stride -- omega is not linearly recoverable. STOP (PLAN Phase B).")
    import json
    json.dump({"rows": rows, "args": vars(args)},
              open(ROOT / "results" / "phase_a" / "phase_b_probe.json", "w"), indent=1)
    print(f"\n({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
