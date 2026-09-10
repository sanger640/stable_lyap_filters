"""
The learned monitor running live on two action sequences that BOTH SURVIVE.

That is the point of the demo. One is a near miss (a few percent from toppling), the other is
comfortably safe. Their OUTCOMES are identical, so an outcome detector -- "will the block fall?"
-- cannot distinguish them at all. Only a proximity detector can, and that is the case the
project actually cares about: not being anywhere near the edge.

Per row:
    left    the true block (ground truth, for the viewer to check against)
    middle  the LEARNED model's predicted tilt for all probes -- the ensemble fan
    right   the running divergence score against its calibrated threshold

Probes are `shared-action` (a + eps*z*a, one scalar z per probe) at eps=0.20, the best family
from eval/run_phase3_monitor.py. Everything on the middle and right panels comes from the
learned model alone; the left panel is never used by the monitor.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation      # noqa: E402
from matplotlib.patches import Polygon                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / p) for p in ("systems", "models", "training")]
sys.path.insert(0, str(ROOT / "eval"))

import tipping_block as tb                                        # noqa: E402
from shplrnn import ShPLRNN                                       # noqa: E402
from run_phase3_monitor import (ACT_DIM, OBS_DIM, T_OFF, T_ON, make_data,  # noqa: E402
                                margin_of, probes, rollout)

ALPHA = tb.ALPHA_DEFAULT
OUT = ROOT / "results" / "phase3"


def corners(theta):
    a, b = np.sin(ALPHA), np.cos(ALPHA)
    base = np.array([[-a, 0.0], [a, 0.0], [a, 2 * b], [-a, 2 * b]])
    piv = np.array([a if theta >= 0 else -a, 0.0])
    c, s = np.cos(-theta), np.sin(-theta)
    return (base - piv) @ np.array([[c, -s], [s, c]]).T + piv


def find_examples(rng, n, thr, tries=600):
    """Both must SURVIVE; one marginal, one comfortable."""
    near = far = None
    for _ in range(tries):
        a = tb.random_push(rng, n, thr)
        if bool(tb.simulate(a)[1][-1]):
            continue                                   # keep only survivors
        m = margin_of(a)
        if m < 0.06 and near is None:
            near = (a, m)
        if m > 0.35 and far is None:
            far = (a, m)
        if near and far:
            break
    assert near and far, "could not find both examples -- widen the search"
    return near, far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--eps", type=float, default=0.20)
    ap.add_argument("--n-perturb", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(OUT / "monitor_live.mp4"))
    args = ap.parse_args()
    n = args.steps

    thr = tb.topple_threshold(n, T_ON, T_OFF)
    S, A, mu, sd, amu, asd = make_data(40, thr, n, seed=0)          # only for mu/sd
    model = ShPLRNN(d=4, H=128, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
    model.load_state_dict(torch.load(OUT / f"monitor_model_T{n}.pt"))

    cal = json.load(open(OUT / "monitor_final.json"))["rows"]
    thresh = cal[f"{args.eps}|shared-action"]["proximity"]["threshold"]
    print(f"calibrated alarm threshold (best-F1 on the test set): {thresh:.4f}", flush=True)

    rng = np.random.default_rng(7)
    (a_near, m_near), (a_far, m_far) = find_examples(rng, n, thr)
    print(f"POSITIVE: survives, margin {m_near:.1%} (a near miss)")
    print(f"NEGATIVE: survives, margin {m_far:.1%} (comfortable)", flush=True)

    s0 = ((torch.zeros(1, OBS_DIM).double() - mu) / sd)
    rows = []
    for tag, act, marg, col in (("POSITIVE — near miss", a_near, m_near, "#c0392b"),
                                ("NEGATIVE — comfortable", a_far, m_far, "#2980b9")):
        P = np.concatenate([act[None], probes("shared-action", act, thr, args.eps,
                                              args.n_perturb, rng)])
        pn = ((torch.from_numpy(P[:, :, None]).double() - amu) / asd)
        Z = rollout(model, pn, s0.expand(len(P), OBS_DIM), n)
        th_pred = (model.observe(Z)[:, :, 0] * sd[0] + mu[0]).numpy()   # (P, n)
        score = Z.norm(dim=-1).std(dim=0).numpy()                        # running divergence
        true_th = tb.simulate(act)[0][:, 0]
        rows.append((tag, marg, col, th_pred, score, true_th))
        print(f"   {tag}: final score {score[-1]:.4f} -> "
              f"{'ALARM' if score[-1] > thresh else 'quiet'}", flush=True)

    idx = np.arange(0, n, args.stride)
    t = idx * tb.DT_DEFAULT
    smax = max(r[4].max() for r in rows) * 1.15

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.25], hspace=0.34, wspace=0.26)
    arts = []
    for r, (tag, marg, col, th_pred, score, true_th) in enumerate(rows):
        ax0, ax1, ax2 = (fig.add_subplot(gs[r, i]) for i in range(3))
        ax0.set_xlim(-1.6, 1.6); ax0.set_ylim(-0.25, 2.3); ax0.set_aspect("equal")
        ax0.set_xticks([]); ax0.set_yticks([]); ax0.axhline(0, color="#2c3e50", lw=3)
        ax0.set_title(f"{tag}\ntrue margin {marg:.1%}", fontsize=10, color=col)
        blk = Polygon(corners(0.0), closed=True, fc=col, ec="#2c3e50", lw=1.5, alpha=0.8)
        ax0.add_patch(blk)
        txt = ax0.text(0.03, 0.97, "", transform=ax0.transAxes, va="top", fontsize=9,
                       family="monospace")

        ax1.set_xlim(0, t[-1]); ax1.set_ylim(-0.7, 1.9)
        ax1.set_xlabel("time"); ax1.set_ylabel("model-predicted tilt")
        ax1.axhline(ALPHA, color="k", ls="--", lw=0.9)
        ax1.text(t[-1], ALPHA, " topple point ", ha="right", va="bottom", fontsize=7)
        ax1.axhline(-ALPHA, color="k", ls="--", lw=0.9)
        for j in range(1, len(th_pred)):
            ax1.plot(np.arange(n) * tb.DT_DEFAULT, th_pred[j], "-", lw=0.6, color=col,
                     alpha=0.18)
        fan = [ax1.plot([], [], "-", lw=0.9, color=col, alpha=0.5)[0]
               for _ in range(len(th_pred) - 1)]
        nom, = ax1.plot([], [], "-", lw=2.2, color="#2c3e50")
        ax1.set_title("the probe ensemble (learned model)", fontsize=9)

        ax2.set_xlim(0, t[-1]); ax2.set_ylim(0, smax)
        ax2.set_xlabel("time"); ax2.set_ylabel("divergence score")
        ax2.axhline(thresh, color="#e67e22", ls="--", lw=1.5)
        ax2.text(0.02, thresh, " alarm threshold", va="bottom", fontsize=8, color="#e67e22")
        ax2.plot(np.arange(n) * tb.DT_DEFAULT, score, "-", lw=0.8, color=col, alpha=0.25)
        sline, = ax2.plot([], [], "-", lw=2.2, color=col)
        verdict = ax2.text(0.5, 0.93, "", transform=ax2.transAxes, ha="center", fontsize=13,
                           fontweight="bold")
        ax2.set_title("monitor output", fontsize=9)
        arts.append((blk, txt, fan, nom, sline, verdict, th_pred, score, true_th))

    fig.suptitle("Both sequences SURVIVE — identical outcome, different margin. "
                 "Only a proximity monitor can tell them apart.", fontsize=12)

    def update(k):
        i = idx[k]; out = []
        for (blk, txt, fan, nom, sline, verdict, th_pred, score, true_th) in arts:
            blk.set_xy(corners(true_th[i]))
            for j, ln in enumerate(fan):
                ln.set_data(np.arange(i + 1) * tb.DT_DEFAULT, th_pred[j + 1, :i + 1])
            nom.set_data(np.arange(i + 1) * tb.DT_DEFAULT, th_pred[0, :i + 1])
            sline.set_data(np.arange(i + 1) * tb.DT_DEFAULT, score[:i + 1])
            hot = score[i] > thresh
            verdict.set_text("⚠ NEAR BOUNDARY" if hot else "safe margin")
            verdict.set_color("#c0392b" if hot else "#27ae60")
            txt.set_text(f"t {t[k]:6.2f}\nθ_true {true_th[i]:+6.3f}\n"
                         f"score {score[i]:7.4f}")
            out += [blk, txt, *fan, nom, sline, verdict]
        return out

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / args.fps, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=3000))
    plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
