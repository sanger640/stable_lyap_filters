"""
Side-by-side video of the Test B result: a CLEAN pair and a STRADDLED pair, with lambda(t)
computed live.

Both rows start from two balls separated by the same eps. The only difference is what happens
at the guard:

  CLEAN      both balls bounce at the same moments. Same event sequence, drifting apart
             smoothly -> lambda(t) settles to a plateau near the true lambda_1 = 0.158.

  STRADDLED  at some approach one ball has bounced and the other has not YET (the table is
             moving, so a hair of difference decides whether contact happens on this pass or
             the next). Their bounce COUNTS diverge, separation jumps, and lambda(t) climbs
             instead of settling.

Note the balls are drawn at two horizontal offsets purely so both are visible; the system is
one-dimensional and both are physically at the same place. "Bounces" counts impacts so far,
and the pair is flagged the moment those counts disagree.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "systems"))
import bouncing_ball as bb                                        # noqa: E402

OMEGA, E = bb.OMEGA_DEFAULT, bb.E_DEFAULT
LAM_TRUE = 0.1585


def roll_pair(s0, eps, direction, n, dt):
    """Evolve a reference and a perturbed ball, recording separation and bounce counts."""
    s, s2 = s0.copy(), s0 + eps * direction
    X = np.empty((n, 2)); TAB = np.empty(n); D = np.empty(n)
    C = np.zeros((n, 2), dtype=int)
    c1 = c2 = 0
    for t in range(n):
        X[t] = (s[0], s2[0]); TAB[t] = bb.table_pos(s[2])
        d = s2 - s
        d[2] = (d[2] + np.pi) % bb.TWO_PI - np.pi
        D[t] = np.linalg.norm(d)
        C[t] = (c1, c2)
        p1, p2 = s[1], s2[1]
        s = bb.step(s, dt, OMEGA, E); s2 = bb.step(s2, dt, OMEGA, E)
        c1 += (s[1] - p1) > bb.G * dt * 1.5
        c2 += (s2[1] - p2) > bb.G * dt * 1.5
    return X, TAB, D, C


def find_pairs(eps, n, dt, n_try=700, seed=3):
    """Search initial conditions for the clearest example of each case."""
    rng = np.random.default_rng(seed)
    base = bb.simulate(3000, dt=dt, omega=OMEGA, e=E, seed=5).numpy()
    best_clean = best_strad = None
    for _ in range(n_try):
        s0 = base[rng.integers(len(base))].copy()
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        X, TAB, D, C = roll_pair(s0, eps, v, n, dt)
        diverged = C[-1, 0] != C[-1, 1]
        lam_end = np.log(D[-1] / eps) / ((n - 1) * dt)
        n_bounce = C[-1].max()
        if diverged and n_bounce >= 1:
            score = lam_end
            if best_strad is None or score > best_strad[0]:
                best_strad = (score, s0, v, X, TAB, D, C)
        elif not diverged and n_bounce >= 2 and X[:, 0].max() < 25:
            score = -abs(lam_end - LAM_TRUE)              # closest to a clean plateau
            if best_clean is None or score > best_clean[0]:
                best_clean = (score, s0, v, X, TAB, D, C)
    assert best_clean and best_strad, "no suitable pair found -- widen the search"
    return best_clean, best_strad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase2" / "clean_vs_straddled.mp4"))
    args = ap.parse_args()

    print("searching for a clean pair and a straddled pair ...", flush=True)
    (sc, _, _, Xc, TABc, Dc, Cc), (ss, _, _, Xs, TABs, Ds, Cs) = find_pairs(
        args.eps, args.frames, args.dt)
    t = np.arange(args.frames) * args.dt
    lam_c = np.concatenate([[np.nan], np.log(Dc[1:] / args.eps) / (t[1:])])
    lam_s = np.concatenate([[np.nan], np.log(Ds[1:] / args.eps) / (t[1:])])
    print(f"  clean     : {Cc[-1]} bounces, final separation {Dc[-1]:.3f}, lambda {lam_c[-1]:+.3f}")
    print(f"  straddled : {Cs[-1]} bounces, final separation {Ds[-1]:.3f}, lambda {lam_s[-1]:+.3f}",
          flush=True)

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.3, 1.3], hspace=0.33, wspace=0.30)
    rows = []
    ymax = max(Xc.max(), Xs.max()) * 1.1 + 1
    dmax = max(Dc.max(), Ds.max()) * 1.3
    lmax = max(np.nanmax(lam_c), np.nanmax(lam_s)) * 1.25

    for r, (name, X, TAB, D, C, lam, col) in enumerate([
            ("CLEAN — both balls bounce together", Xc, TABc, Dc, Cc, lam_c, "#2980b9"),
            ("STRADDLED — bounce counts diverge", Xs, TABs, Ds, Cs, lam_s, "#c0392b")]):
        ax0, ax1, ax2 = (fig.add_subplot(gs[r, i]) for i in range(3))
        ax0.set_xlim(-1.4, 1.4); ax0.set_ylim(-1.6, ymax); ax0.set_xticks([])
        ax0.set_ylabel("height"); ax0.set_title(name, fontsize=10, color=col)
        bA, = ax0.plot([], [], "o", ms=11, color="#e67e22", zorder=5)
        bB, = ax0.plot([], [], "o", ms=11, color=col, zorder=5)
        tab, = ax0.plot([], [], "-", lw=5, color="#2c3e50", solid_capstyle="butt")
        txt = ax0.text(0.03, 0.97, "", transform=ax0.transAxes, va="top", fontsize=9,
                       family="monospace")

        ax1.set_xlim(0, t[-1]); ax1.set_ylim(0, dmax)
        ax1.set_ylabel("separation  d(t)"); ax1.set_xlabel("time")
        ax1.axhline(args.eps, color="grey", ls=":", lw=0.8)
        ax1.plot(t, D, "-", lw=0.9, color=col, alpha=0.25)
        ld, = ax1.plot([], [], "-", lw=2, color=col)

        ax2.set_xlim(0, t[-1]); ax2.set_ylim(-0.15, lmax)
        ax2.set_ylabel("λ(t)"); ax2.set_xlabel("time")
        ax2.axhline(LAM_TRUE, color="k", ls="--", lw=0.9)
        ax2.text(t[-1] * 0.98, LAM_TRUE, " true λ₁", ha="right", va="bottom", fontsize=8)
        ax2.plot(t, lam, "-", lw=0.9, color=col, alpha=0.25)
        ll, = ax2.plot([], [], "-", lw=2, color=col)
        rows.append((bA, bB, tab, txt, ld, ll, X, TAB, D, C, lam))

    fig.suptitle("Same perturbation (ε=%.2f), two outcomes — λ plateaus, or λ climbs"
                 % args.eps, fontsize=12)
    bar = np.array([-1.0, 1.0])

    def update(i):
        arts = []
        for (bA, bB, tab, txt, ld, ll, X, TAB, D, C, lam) in rows:
            bA.set_data([-0.45], [X[i, 0]]); bB.set_data([0.45], [X[i, 1]])
            tab.set_data(bar, [TAB[i], TAB[i]])
            flag = "  <-- DIVERGED" if C[i, 0] != C[i, 1] else ""
            txt.set_text(f"t {t[i]:6.2f}\nbounces {C[i,0]}/{C[i,1]}{flag}\n"
                         f"d {D[i]:8.4f}\nλ {lam[i]:+8.3f}" if i else "")
            ld.set_data(t[:i + 1], D[:i + 1]); ll.set_data(t[:i + 1], lam[:i + 1])
            arts += [bA, bB, tab, txt, ld, ll]
        return arts

    anim = FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2600))
    plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
