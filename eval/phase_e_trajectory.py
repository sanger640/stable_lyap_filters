"""Basin assignment by TRAJECTORY CONVERGENCE, tested where the monitor actually works.

Two rollouts are in the same basin IF THEY CONVERGE. That is the definition of a basin rather than
a proxy for it, it averages over the whole tail instead of trusting one endpoint, and it is immune
to the drift Phase D found (the ending latents never stop moving, so where you truncate changes the
endpoint you read).

CRUCIALLY, this is tested WITHIN an episode, across probes -- not across episodes. Across episodes
the lighting differs, so two rollouts ending in the same basin can never converge; that is what
crippled Phase E's discovery step. The monitor never faces that: its 32 probes come from one
episode, share lighting exactly, and differ only in action. Convergence is the right instrument for
that setting and the wrong one for the other.

Compared against three alternatives on the same probes, with per-probe ground truth from the
simulator:
    euclid-end   Euclidean on the ending latent          (what Phase E used)
    cosine-end   cosine on the ending latent             (what CLAUDE.md specifies for Jenga)
    pca-end      PCA-8 then Euclidean on the ending      (what rescued Phase E)
    converge     d_ij(end) / d_ij(tail start)            (this idea)
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval"), "/home/sanger/wksp/dino_wm"]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_c_train import Predictor, NUM_HIST                      # noqa: E402
from phase_d_settle import basin, STRIDE                           # noqa: E402

OUT = ROOT / "results" / "phase_c"


def pair_stats(aff, truth, larger_means_different=True):
    """How well does one pairwise score separate same-basin from different-basin probe pairs?"""
    n = len(truth); iu = np.triu_indices(n, 1)
    s = aff[iu]; same = truth[iu[0]] == truth[iu[1]]
    if same.all() or (~same).all():
        return None
    if not larger_means_different:
        s = -s
    order = np.argsort(s); y = (~same)[order]           # AUC: different-basin should score higher
    r = np.arange(1, len(y) + 1)
    p, q = int(y.sum()), int((~y).sum())
    auc = (r[y].sum() - p * (p + 1) / 2) / (p * q)
    best = max(((s < t) == same).mean() for t in np.linspace(s.min(), s.max(), 200))
    return float(auc), float(best), float(same.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--L", type=int, default=40)
    args = ap.parse_args()
    dev = "cuda"; t0 = time.time()

    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    p = np.load(OUT / "phase_c_eval.npz")
    mu, sd, amu, asd = (float(p[k]) for k in ("mu", "sd", "amu", "asd"))
    model = Predictor().to(dev)
    model.load_state_dict(torch.load(OUT / "predictor_gtf_warm.pt")); model.eval()

    rng = np.random.default_rng(0); thr = tb.topple_threshold(450, 10, 60)
    acts = []
    for _ in range(Z.shape[0]):
        a = tb.random_push(rng, 450, thr); tb.simulate(a, dt=tb.DT_DEFAULT)
        br.sample_lighting(rng); acts.append(a)
    idx = np.linspace(0, Z.shape[0] - 1, args.episodes).astype(int)
    prng = np.random.default_rng(7)
    zero_a = (0.0 - amu) / asd
    res = {k: [] for k in ("euclid-end", "cosine-end", "pca-end", "converge")}
    n_split = 0

    for e, i in enumerate(idx):
        # probes: a * (1 + eps*z), one scalar per probe -- the shared-action family
        zs = prng.standard_normal((args.n_probe, 1))
        P = acts[i][None] * (1.0 + args.eps * zs)
        # GROUND TRUTH per probe: drive H frames, then zero force, in the real simulator
        tb_th = []
        for pr in P:
            a = np.concatenate([pr[:args.H * STRIDE], np.zeros(args.L * STRIDE)])
            tb_th.append(tb.simulate(a, dt=tb.DT_DEFAULT)[0][-1, 0])
        truth = basin(np.array(tb_th))
        if len(np.unique(truth)) < 2:
            continue                                     # no split -> nothing to separate
        n_split += 1

        Ap = P[:, :args.H * STRIDE].reshape(args.n_probe, args.H, STRIDE).mean(2)[:, :, None]
        An = (torch.from_numpy(Ap).float().to(dev) - amu) / asd
        z0 = ((torch.from_numpy(np.asarray(Z[i, :NUM_HIST], np.float32)).to(dev) - mu) / sd)
        z = z0[None].expand(args.n_probe, -1, -1, -1).contiguous()
        with torch.no_grad():
            for t in range(NUM_HIST, args.H):
                z = torch.cat([z, model(z[:, -NUM_HIST:], An[:, t-NUM_HIST:t])[:, -1:]], 1)
            az = torch.full((args.n_probe, NUM_HIST, 1), zero_a, device=dev, dtype=z.dtype)
            d_start = d_end = None
            for j in range(args.L):
                z = torch.cat([z, model(z[:, -NUM_HIST:], az)[:, -1:]], 1)
                F = z[:, -1].reshape(args.n_probe, -1)
                D = torch.cdist(F[None], F[None])[0].cpu().numpy()
                if j == 0:
                    d_start = D
                d_end = D
            E = z[:, -1].reshape(args.n_probe, -1).float().cpu().numpy()

        # --- the four candidate pairwise scores
        r = pair_stats(d_end, truth)
        if r: res["euclid-end"].append(r)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        r = pair_stats(1.0 - En @ En.T, truth)
        if r: res["cosine-end"].append(r)
        Xc = E - E.mean(0)
        V = np.linalg.svd(Xc, full_matrices=False)[2][:8]
        X = Xc @ V.T
        r = pair_stats(np.linalg.norm(X[:, None] - X[None], axis=-1), truth)
        if r: res["pca-end"].append(r)
        r = pair_stats(d_end / np.maximum(d_start, 1e-9), truth)     # CONVERGENCE
        if r: res["converge"].append(r)
        if (e + 1) % 10 == 0:
            print(f"  {e+1}/{len(idx)} episodes ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{n_split}/{len(idx)} episodes had probes landing in >1 basin\n")
    print(f"{'method':<14}{'pair AUC':>10}{'best pair acc':>15}")
    print("-" * 40)
    for k, v in res.items():
        if not v:
            continue
        a = np.mean([x[0] for x in v]); b = np.mean([x[1] for x in v])
        print(f"{k:<14}{a:>10.3f}{b:>15.1%}")
    print(f"\nchance pair acc = {np.mean([x[2] for x in res['euclid-end']]):.1%} "
          f"(fraction of pairs that are same-basin)")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
