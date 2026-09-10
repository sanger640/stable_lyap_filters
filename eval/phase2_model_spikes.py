"""
Does the LEARNED model reproduce the local-lambda spikes, and can they be used as a detector?

This is the question the monitor actually depends on. On the true system, pairs whose bounce
timing drifts apart show large bidirectional spikes in local lambda (measured: range
[-2.41, +3.31] vs [-0.30, +0.83] for a clean pair) -- the window in which one ball has bounced
and the other has not, so their relative velocity is ~2v. Test A failed to establish that the
learned model represents the discontinuity sharply, so whether it reproduces those spikes is
unverified.

Design, kept honest by construction:

  * The LABEL comes from the TRUE system (does this pair's separation actually blow up?).
  * The SCORE comes from the MODEL alone, using only what a monitor could compute at runtime.
  * Both members of a pair start from the SAME raw state, mapped through the same observation
    transform, so the comparison is apples to apples.

A detector that works needs the model's spikes to line up with the true system's -- both in
size (does the score separate the classes?) and in time (does it fire at the right step?).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / p) for p in ("systems", "geometry", "models", "training")]
sys.path.insert(0, str(ROOT / "eval"))

import bouncing_ball as bb                                        # noqa: E402
from run_phase2_ball import DT, E, OBS_DIM, OMEGA, make_data, train   # noqa: E402

OUT = ROOT / "results" / "phase2"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "model_spikes.pt"


def local_lambda(D, W, dt):
    o = np.full(len(D), np.nan)
    o[W:] = np.log(D[W:] / np.maximum(D[:-W], 1e-300)) / (W * dt)
    return o


def true_pair(s0, eps, v, n):
    """Roll the pair through the real simulator; separation measured in OBSERVATION space so
    it is directly comparable with the model's, and bounce times for labelling."""
    s, s2 = s0.copy(), s0 + eps * v
    A = np.empty((n, 3)); B = np.empty((n, 3))
    tA, tB = [], []
    for t in range(n):
        A[t], B[t] = s, s2
        p1, p2 = s[1], s2[1]
        s = bb.step(s, DT, OMEGA, E); s2 = bb.step(s2, DT, OMEGA, E)
        if (s[1] - p1) > bb.G * DT * 1.5:
            tA.append(t)
        if (s2[1] - p2) > bb.G * DT * 1.5:
            tB.append(t)
    return bb.to_obs(A), bb.to_obs(B), np.array(tA), np.array(tB)


def model_pair(model, oA0, oB0, mu, sd, n):
    """Roll the same two initial observations through the learned model."""
    with torch.no_grad():
        z = model.lift(((torch.from_numpy(np.stack([oA0, oB0])) - mu) / sd).double())
        out = np.empty((n, 2, OBS_DIM))
        for t in range(n):
            out[t] = model.observe(z).numpy()
            z = model(z)
    return out[:, 0], out[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--impact-weight", type=float, default=5.0)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--n-data", type=int, default=60000)
    ap.add_argument("--n-seq", type=int, default=4000)
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.1)
    args = ap.parse_args()

    data, mu, sd, raw, imp = make_data(args.n_data, seed=0)
    if CKPT.exists():
        import shplrnn
        model = shplrnn.ShPLRNN(d=OBS_DIM, H=args.H, obs_dim=OBS_DIM).double()
        model.load_state_dict(torch.load(CKPT)); print(f"loaded {CKPT}", flush=True)
    else:
        print(f"training (epochs={args.epochs}, alpha={args.alpha}, "
              f"iw={args.impact_weight}) ...", flush=True)
        model, _ = train(0, args.epochs, args.H, OBS_DIM, 0.5, args.n_data, args.n_seq,
                         alpha=args.alpha, impact_weight=args.impact_weight)
        torch.save(model.state_dict(), CKPT)

    n, W = args.horizon, args.window
    rng = np.random.default_rng(0)
    base = bb.simulate(args.n_pairs * 3, dt=DT, omega=OMEGA, e=E, seed=7).numpy()
    rows = []
    for i in range(args.n_pairs):
        s0 = base[rng.integers(len(base))].copy()
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        oA, oB, tA, tB = true_pair(s0, args.eps, v, n)
        mA, mB = model_pair(model, oA[0], oB[0], mu, sd, n)

        d_true = np.linalg.norm(oA - oB, axis=1)
        d_mod = np.linalg.norm((mA - mB) * sd.numpy(), axis=1)     # back to obs units
        if not np.isfinite(d_mod).all() or d_mod[0] <= 0:
            continue
        lt, lm = local_lambda(d_true, W, DT), local_lambda(d_mod, W, DT)
        if not np.isfinite(lm[W:]).all():
            continue

        k = min(len(tA), len(tB))
        mm = np.abs(tA[:k] - tB[:k]) * DT if k else np.array([0.0])
        rows.append(dict(
            true_final_sep=float(d_true[-1]), model_final_sep=float(d_mod[-1]),
            true_maxabs=float(np.nanmax(np.abs(lt))), model_maxabs=float(np.nanmax(np.abs(lm))),
            true_std=float(np.nanstd(lt)), model_std=float(np.nanstd(lm)),
            max_mismatch=float(mm.max()), counts_differ=bool(len(tA) != len(tB)),
            t_true_spike=int(np.nanargmax(np.abs(lt))), t_model_spike=int(np.nanargmax(np.abs(lm))),
        ))

    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    # LABEL from the true system: separation genuinely blew up
    lab = R["true_final_sep"] > np.median(R["true_final_sep"]) * 3.0
    print(f"\n{len(rows)} pairs, {int(lab.sum())} labelled divergent by the TRUE system\n")

    def auc(score, y):
        o = np.argsort(score); y = y[o]
        r = np.arange(1, len(y) + 1)
        p, q = y.sum(), (~y).sum()
        return float((r[y].sum() - p * (p + 1) / 2) / (p * q)) if p and q else float("nan")

    res = {"n_pairs": len(rows), "n_divergent": int(lab.sum())}
    print(f"{'score':<28}{'AUC':>8}   (0.5 = useless)")
    for name, sc in (("TRUE max|local λ|", R["true_maxabs"]),
                     ("MODEL max|local λ|", R["model_maxabs"]),
                     ("MODEL std(local λ)", R["model_std"]),
                     ("MODEL final separation", R["model_final_sep"])):
        a = auc(sc, lab); res[name] = a
        print(f"{name:<28}{a:>8.3f}")

    corr = float(np.corrcoef(R["true_maxabs"], R["model_maxabs"])[0, 1])
    res["corr_true_vs_model_maxabs"] = corr
    dt_spike = np.abs(R["t_true_spike"] - R["t_model_spike"])
    res["median_spike_time_error_steps"] = float(np.median(dt_spike))
    res["spike_within_5_steps_frac"] = float((dt_spike <= 5).mean())
    print(f"\ncorr(true, model) of max|local λ| : {corr:+.3f}")
    print(f"spike TIMING: median error {np.median(dt_spike):.0f} steps, "
          f"{100*(dt_spike<=5).mean():.0f}% within 5 steps  (chance ~{100*10/args.horizon:.0f}%)")
    print(f"\nmax|local λ| magnitude   true {R['true_maxabs'].mean():.2f} "
          f"vs model {R['model_maxabs'].mean():.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].scatter(R["true_maxabs"], R["model_maxabs"], s=12, alpha=0.6, c="#c0392b")
    ax[0].set_xlabel("TRUE max|local λ|"); ax[0].set_ylabel("MODEL max|local λ|")
    ax[0].set_title(f"do they agree per pair?  r={corr:+.2f}")
    for a_, (sc, name) in zip(ax[1:], [(R["true_maxabs"], "TRUE max|local λ|"),
                                       (R["model_maxabs"], "MODEL max|local λ|")]):
        a_.hist(sc[~lab], bins=25, alpha=0.6, label="clean", color="#2980b9")
        a_.hist(sc[lab], bins=25, alpha=0.6, label="divergent", color="#c0392b")
        a_.set_xlabel(name); a_.set_ylabel("pairs"); a_.legend(fontsize=8)
        a_.set_title(f"AUC {auc(sc, lab):.3f}")
    fig.suptitle("Does the learned model reproduce the local-λ spikes?")
    fig.tight_layout(); fig.savefig(FIG / "model_spikes.png", dpi=130); plt.close(fig)

    json.dump(res, open(OUT / "model_spikes.json", "w"), indent=1)
    print(f"\n-> {OUT / 'model_spikes.json'}\n-> {FIG / 'model_spikes.png'}")


if __name__ == "__main__":
    main()
