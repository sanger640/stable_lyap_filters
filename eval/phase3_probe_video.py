"""
Side-by-side video of the two probe families acting on the same near-boundary push.

    column 1   UNPERTURBED   the action as issued
    column 2   SCALED        one Gaussian per probe, MULTIPLIES the whole sequence
    column 3   ISOTROPIC     one Gaussian per timestep, ADDED

The action is chosen to sit close to the topple boundary, which is where a monitor has to work
and where the two families visibly differ:

  * SCALED probes change HOW HARD the push is, so the ensemble splits -- some blocks go over,
    some do not. That split is the boundary-proximity signal.
  * ISOTROPIC probes change the push's SHAPE while barely changing its size: 450 independent
    nudges largely cancel, so the net force moves by only about eps/sqrt(450) ~ 0.2% even though
    every timestep moved by eps. The ensemble stays bunched and the boundary is never probed.

Force arrows are drawn live: solid for the nominal action, translucent for the extreme probes,
so the WIDTH of the arrow fan is the range of forces actually being tried.
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

ALPHA = tb.ALPHA_DEFAULT


def block_corners(theta):
    a, b = np.sin(ALPHA), np.cos(ALPHA)
    base = np.array([[-a, 0.0], [a, 0.0], [a, 2 * b], [-a, 2 * b]])
    piv = np.array([a if theta >= 0 else -a, 0.0])
    c, s = np.cos(-theta), np.sin(-theta)
    return (base - piv) @ np.array([[c, -s], [s, c]]).T + piv


def com(theta):
    a, b = np.sin(ALPHA), np.cos(ALPHA)
    piv = np.array([a if theta >= 0 else -a, 0.0])
    c, s = np.cos(-theta), np.sin(-theta)
    return (np.array([0.0, b]) - piv) @ np.array([[c, -s], [s, c]]).T + piv


def margin_of(act, s_max=0.5, coarse=0.02):
    base = bool(tb.simulate(act)[1][-1])
    for i in range(1, int(s_max / coarse) + 1):
        for sign in (+1, -1):
            s = 1.0 + sign * i * coarse
            if s <= 0:
                continue
            if bool(tb.simulate(act * s)[1][-1]) != base:
                return abs(s - 1.0)
    return s_max


def pick_action(rng, n, thr, lo=0.02, hi=0.10, tries=400):
    """A push close enough to the boundary that scaled probes actually split the outcome."""
    best = None
    for _ in range(tries):
        a = tb.random_push(rng, n, thr)
        m = margin_of(a)
        if lo <= m <= hi:
            return a, m
        if best is None or abs(m - (lo + hi) / 2) < abs(best[1] - (lo + hi) / 2):
            best = (a, m)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--probes", type=int, default=12)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="render every Nth step")
    ap.add_argument("--out", default=str(ROOT / "results" / "phase3" / "probe_families.mp4"))
    args = ap.parse_args()

    n, NP = args.steps, args.probes
    thr = tb.topple_threshold(n, 10, 60)
    rng = np.random.default_rng(4)
    act, marg = pick_action(rng, n, thr)
    print(f"chosen action: margin {marg:.3f} (near the boundary), threshold {thr:.4f}",
          flush=True)

    scaled = act[None] * (1.0 + args.eps * rng.standard_normal((NP, 1)))
    isotro = act[None] + thr * args.eps * rng.standard_normal((NP, n))

    def roll(A):
        S = np.empty((len(A), n + 1, 2)); F = np.empty((len(A), n + 1))
        for j, a in enumerate(A):
            st, _ = tb.simulate(a, dt=tb.DT_DEFAULT)
            S[j] = st; F[j] = np.append(a, a[-1])
        return S, F

    S0, F0 = roll(act[None])
    Ss, Fs = roll(scaled)
    Si, Fi = roll(isotro)
    for nm, S in (("nominal", S0), ("scaled", Ss), ("isotropic", Si)):
        fell = (np.abs(S[:, -1, 0]) >= ALPHA).sum()
        print(f"  {nm:<10}: {fell}/{len(S)} toppled", flush=True)

    idx = np.arange(0, n + 1, args.stride)
    t = idx * tb.DT_DEFAULT
    cols = [("UNPERTURBED\nthe action as issued", S0, F0, "#2c3e50"),
            (f"SCALED probes  (×(1+{args.eps}z), one z per probe)", Ss, Fs, "#c0392b"),
            (f"ISOTROPIC probes  (+{args.eps}·z per timestep)", Si, Fi, "#2980b9")]

    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 0.85, 0.85], hspace=0.42, wspace=0.22)
    ymax = 2.4
    fmax = max(np.abs(Fs).max(), np.abs(Fi).max()) * 1.15
    arts = []

    for c, (title, S, F, col) in enumerate(cols):
        ax0 = fig.add_subplot(gs[0, c]); ax1 = fig.add_subplot(gs[1, c])
        ax2 = fig.add_subplot(gs[2, c])
        ax0.set_xlim(-2.3, 2.3); ax0.set_ylim(-0.3, ymax); ax0.set_aspect("equal")
        ax0.set_xticks([]); ax0.set_yticks([]); ax0.axhline(0, color="#2c3e50", lw=3)
        ax0.set_title(title, fontsize=10, color=col)

        ghosts = [Polygon(block_corners(0.0), closed=True, fc=col, ec="none", alpha=0.16)
                  for _ in range(len(S) - 1)]
        for g in ghosts:
            ax0.add_patch(g)
        nom = Polygon(block_corners(0.0), closed=True, fc=col, ec="#2c3e50", lw=1.6, alpha=0.85)
        ax0.add_patch(nom)
        # force arrows: solid = nominal, translucent = the extreme probes
        arrow_n, = ax0.plot([], [], "-", lw=3, color="#e67e22", marker=">", ms=9,
                            markevery=[-1], zorder=8)
        arrow_lo, = ax0.plot([], [], "-", lw=1.4, color="#e67e22", alpha=0.45,
                             marker=">", ms=5, markevery=[-1], zorder=7)
        arrow_hi, = ax0.plot([], [], "-", lw=1.4, color="#e67e22", alpha=0.45,
                             marker=">", ms=5, markevery=[-1], zorder=7)
        txt = ax0.text(0.02, 0.97, "", transform=ax0.transAxes, va="top", fontsize=9,
                       family="monospace")

        ax1.set_xlim(0, t[-1]); ax1.set_ylim(-fmax, fmax)
        ax1.set_ylabel("force"); ax1.set_xlabel("time")
        ax1.axhline(0, color="grey", lw=0.6)
        if len(F) > 1:
            ax1.fill_between(np.arange(n + 1) * tb.DT_DEFAULT, F.min(0), F.max(0),
                             color=col, alpha=0.18, lw=0)
        ax1.plot(np.arange(n + 1) * tb.DT_DEFAULT, F0[0], "-", lw=0.8, color="#2c3e50",
                 alpha=0.5)
        fline, = ax1.plot([], [], "-", lw=1.8, color=col)
        ax1.set_title("applied force (band = probe range)", fontsize=8)

        ax2.set_xlim(0, t[-1]); ax2.set_ylim(-0.15, 1.7)
        ax2.set_ylabel("tilt θ"); ax2.set_xlabel("time")
        ax2.axhline(ALPHA, color="k", ls="--", lw=0.9)
        ax2.text(t[-1], ALPHA, " topple point α ", ha="right", va="bottom", fontsize=7)
        for j in range(len(S)):
            ax2.plot(np.arange(n + 1) * tb.DT_DEFAULT, S[j, :, 0], "-", lw=0.7,
                     color=col, alpha=0.25)
        tlines = [ax2.plot([], [], "-", lw=1.0, color=col, alpha=0.55)[0]
                  for _ in range(len(S) - 1)]
        tnom, = ax2.plot([], [], "-", lw=2.0, color=col)
        ax2.set_title("every probe's tilt", fontsize=8)
        arts.append((ghosts, nom, arrow_n, arrow_lo, arrow_hi, txt, fline, tlines, tnom, S, F))

    fig.suptitle(f"Same near-boundary push (margin {marg:.1%}), two ways to probe it — "
                 f"ε={args.eps}, {NP} probes each", fontsize=12)

    def draw_arrow(line, theta, force, scale=1.6):
        """Horizontal arrow at the centre of mass; length ∝ |force|, direction = sign."""
        if abs(force) < 1e-9:
            line.set_data([], []); return
        c = com(theta)
        L = scale * force
        line.set_data([c[0], c[0] + L], [c[1], c[1]])

    def update(k):
        i = idx[k]; out = []
        for (ghosts, nom, arrow_n, arrow_lo, arrow_hi, txt, fline, tlines, tnom,
             S, F) in arts:
            for j, g in enumerate(ghosts):
                g.set_xy(block_corners(S[j + 1, i, 0]))
            nom.set_xy(block_corners(S[0, i, 0]))
            draw_arrow(arrow_n, S[0, i, 0], F[0, i])
            if len(F) > 1:
                lo, hi = F[:, i].min(), F[:, i].max()
                draw_arrow(arrow_lo, S[0, i, 0], lo)
                draw_arrow(arrow_hi, S[0, i, 0], hi)
                nfell = int((np.abs(S[:, i, 0]) >= ALPHA).sum())
                spread = float(np.linalg.norm(S[:, i] - S[:, i].mean(0), axis=1).mean())
                txt.set_text(f"t {t[k]:6.2f}\nF {F[0,i]:+6.3f}\nrange [{lo:+.3f},{hi:+.3f}]\n"
                             f"toppled {nfell}/{len(S)}\nspread {spread:6.3f}")
            else:
                txt.set_text(f"t {t[k]:6.2f}\nF {F[0,i]:+6.3f}\n"
                             f"θ {S[0,i,0]:+6.3f}")
            fline.set_data(np.arange(i + 1) * tb.DT_DEFAULT, F[0, :i + 1])
            for j, ln in enumerate(tlines):
                ln.set_data(np.arange(i + 1) * tb.DT_DEFAULT, S[j + 1, :i + 1, 0])
            tnom.set_data(np.arange(i + 1) * tb.DT_DEFAULT, S[0, :i + 1, 0])
            out += [*ghosts, nom, arrow_n, arrow_lo, arrow_hi, txt, fline, *tlines, tnom]
        return out

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / args.fps, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=3000))
    plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
