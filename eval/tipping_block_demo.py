"""
Video demo of the tipping block: two pushes either side of the topple boundary.

The whole point is the contrast with the bouncing ball. There, a "diverged" pair kept coming
back together -- separation saturated and then oscillated down to 17% of its own maximum, which
made every detection experiment ambiguous. Here the failure is ABSORBING: once the block is
over, it is over, and the gap between a safe and a failed rollout never closes again.

The two pushes differ by ~10% in amplitude and are visually identical for the first ~1.5
seconds. That is the regime a safety monitor has to work in.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation      # noqa: E402
from matplotlib.patches import Polygon                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "systems"))
import tipping_block as tb                                        # noqa: E402


def block_corners(theta, alpha):
    """Rectangle corners for tilt theta. Pivot is the corner it is rocking on: the right corner
    for theta > 0, the left for theta < 0 -- which is what makes |theta| = alpha the moment the
    centre of mass passes over that pivot."""
    a, b = np.sin(alpha), np.cos(alpha)               # half-width, half-height (R = 1)
    base = np.array([[-a, 0.0], [a, 0.0], [a, 2 * b], [-a, 2 * b]])
    piv = np.array([a if theta >= 0 else -a, 0.0])
    c, s = np.cos(-theta), np.sin(-theta)
    rot = np.array([[c, -s], [s, c]])
    return (base - piv) @ rot.T + piv


def com(theta, alpha):
    a, b = np.sin(alpha), np.cos(alpha)
    piv = np.array([a if theta >= 0 else -a, 0.0])
    c, s = np.cos(-theta), np.sin(-theta)
    return (np.array([0.0, b]) - piv) @ np.array([[c, -s], [s, c]]).T + piv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=320)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--t-on", type=int, default=10)
    ap.add_argument("--t-off", type=int, default=60)
    ap.add_argument("--safe-rel", type=float, default=0.95)
    ap.add_argument("--unsafe-rel", type=float, default=1.05)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase3" / "tipping_block.mp4"))
    args = ap.parse_args()

    n, alpha, dt = args.steps, tb.ALPHA_DEFAULT, tb.DT_DEFAULT
    thr = tb.topple_threshold(n, args.t_on, args.t_off, dt, alpha)
    runs = []
    for rel, name, col in ((args.safe_rel, "SAFE", "#2980b9"),
                           (args.unsafe_rel, "UNSAFE", "#c0392b")):
        F = tb.push_profile(n, thr * rel, args.t_on, args.t_off, dt)
        S, fell = tb.simulate(F, dt=dt, alpha=alpha)
        runs.append((name, rel, col, F, S, fell))
        print(f"  {name:>6}: amp {thr*rel:.4f} ({rel:.0%} of threshold {thr:.4f}), "
              f"max|theta| {np.abs(S[:,0]).max():.4f}, fell={fell[-1]}", flush=True)

    t = np.arange(n + 1) * dt
    sep = np.abs(runs[0][4][:, 0] - runs[1][4][:, 0])
    print(f"  separation safe-vs-unsafe: end {sep[-1]:.3f}, "
          f"min after topple {sep[int(np.argmax(runs[1][5])):].min():.3f}", flush=True)

    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.3, 1.0], hspace=0.35, wspace=0.28)
    arts = []
    for r, (name, rel, col, F, S, fell) in enumerate(runs):
        ax0, ax1, ax2 = (fig.add_subplot(gs[r, i]) for i in range(3))
        ax0.set_xlim(-1.5, 2.2); ax0.set_ylim(-0.25, 2.3); ax0.set_aspect("equal")
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.axhline(0, color="#2c3e50", lw=3)
        ax0.set_title(f"{name} — push at {rel:.0%} of the topple threshold",
                      fontsize=11, color=col)
        poly = Polygon(block_corners(0.0, alpha), closed=True, fc=col, ec="#2c3e50",
                       lw=1.5, alpha=0.75)
        ax0.add_patch(poly)
        cm, = ax0.plot([], [], "o", ms=6, color="#f1c40f", mec="#2c3e50", zorder=6)
        arrow = ax0.annotate("", xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=2.5))
        lbl = ax0.text(0.02, 0.96, "", transform=ax0.transAxes, va="top", fontsize=9,
                       family="monospace")

        ax1.set_xlim(0, t[-1]); ax1.set_ylim(-0.15, 1.75)
        ax1.set_xlabel("time"); ax1.set_ylabel("tilt θ")
        ax1.axhline(alpha, color="k", ls="--", lw=1.0)
        ax1.text(t[-1], alpha, " α = topple point ", ha="right", va="bottom", fontsize=8)
        ax1.axhline(tb.FALLEN_ANGLE, color="grey", ls=":", lw=1.0)
        ax1.text(t[-1], tb.FALLEN_ANGLE, " fallen (absorbing) ", ha="right", va="top", fontsize=8)
        ax1.plot(t, S[:, 0], "-", lw=0.9, color=col, alpha=0.25)
        lth, = ax1.plot([], [], "-", lw=2, color=col)
        ax1.set_title("tilt — crossing α is irreversible", fontsize=9)

        ax2.set_xlim(0, t[-1]); ax2.set_ylim(-0.05, thr * 1.35)
        ax2.set_xlabel("time"); ax2.set_ylabel("push force")
        ax2.axhline(tb.critical_force(alpha), color="grey", ls=":", lw=1.0)
        ax2.text(t[-1], tb.critical_force(alpha), " static lift-off ", ha="right",
                 va="bottom", fontsize=8)
        ax2.plot(t[:len(F)], F, "-", lw=0.9, color=col, alpha=0.25)   # F has n, t has n+1
        lf, = ax2.plot([], [], "-", lw=2, color=col)
        ax2.set_title("the action", fontsize=9)
        arts.append((poly, cm, arrow, lbl, lth, lf, F, S, fell))

    fig.suptitle("Tipping block — two pushes 10% apart. Identical for ~1.5 s, then one is gone.",
                 fontsize=12)

    def update(i):
        out = []
        for (poly, cm, arrow, lbl, lth, lf, F, S, fell) in arts:
            th = S[i, 0]
            poly.set_xy(block_corners(th, alpha))
            c = com(th, alpha); cm.set_data([c[0]], [c[1]])
            if F[min(i, len(F) - 1)] > 0 and not fell[i]:
                arrow.set_position((c[0] - 0.75, c[1]))
                arrow.xy = (c[0] - 0.12, c[1])
            else:
                arrow.set_position((c[0], c[1])); arrow.xy = (c[0], c[1])
            lbl.set_text(f"t {t[i]:5.2f}\nθ {th:+6.3f}\nF {F[min(i,len(F)-1)]:5.3f}"
                         + ("\nTOPPLED" if fell[i] else ""))
            lth.set_data(t[:i + 1], S[:i + 1, 0])
            k = min(i + 1, len(F))
            lf.set_data(t[:k], F[:k])
            out += [poly, cm, arrow, lbl, lth, lf]
        return out

    anim = FuncAnimation(fig, update, frames=n + 1, interval=1000 / args.fps, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2600))
    plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
