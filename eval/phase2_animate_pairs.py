"""
Side-by-side video of the Test B result: a CLEAN pair and a CASCADE pair, with lambda(t) live.

WHAT THIS SHOWS, corrected. An earlier version flagged a pair the first instant its bounce
COUNTS disagreed and highlighted that moment as "the event". That was wrong. In the pair it
picked, the counts differed for 0.1 time units and immediately re-synced -- both balls bounced,
0.1 apart -- and separation did not grow through it at all (d went 0.579 -> 0.549, local lambda
went NEGATIVE). The real divergence came ~9 time units later.

The mechanism is a CASCADE, not one event. Each bounce amplifies the mismatch in WHEN the two
balls hit, because outgoing velocity depends on the table's phase at contact:

    bounce 1:  6.3 vs 6.4   ->  0.1 apart
    bounce 2: 15.1 vs 16.0  ->  0.9 apart
    bounce 3: 19.8 vs never ->  they have stopped bouncing together

Pairs are therefore selected by the GROWTH of that timing mismatch, and every bounce of each
ball is marked on the plots. Note this needs no grazing (tangential contact) -- it is ordinary
phase amplification at impact, and no claim about a grazing singularity is made here.

The balls are drawn at two horizontal offsets purely so both are visible; the system is
one-dimensional and both are physically at the same place.
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
    """Evolve a reference and a perturbed ball, recording separation and every bounce TIME."""
    s, s2 = s0.copy(), s0 + eps * direction
    X = np.empty((n, 2)); TAB = np.empty(n); D = np.empty(n)
    tA, tB = [], []
    for t in range(n):
        X[t] = (s[0], s2[0]); TAB[t] = bb.table_pos(s[2])
        d = s2 - s
        d[2] = (d[2] + np.pi) % bb.TWO_PI - np.pi
        D[t] = np.linalg.norm(d)
        p1, p2 = s[1], s2[1]
        s = bb.step(s, dt, OMEGA, E); s2 = bb.step(s2, dt, OMEGA, E)
        if (s[1] - p1) > bb.G * dt * 1.5:
            tA.append(t * dt)
        if (s2[1] - p2) > bb.G * dt * 1.5:
            tB.append(t * dt)
    return X, TAB, D, np.array(tA), np.array(tB)


def mismatches(tA, tB):
    """Pair bounces in order; return |t_A,k - t_B,k| for each matched pair."""
    k = min(len(tA), len(tB))
    return np.abs(tA[:k] - tB[:k]) if k else np.array([])


def find_pairs(eps, n, dt, n_try=900, seed=3):
    """Select by GROWTH of the timing mismatch, not by a transient count disagreement."""
    rng = np.random.default_rng(seed)
    base = bb.simulate(3000, dt=dt, omega=OMEGA, e=E, seed=5).numpy()
    best_clean = best_casc = None
    for _ in range(n_try):
        s0 = base[rng.integers(len(base))].copy()
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        X, TAB, D, tA, tB = roll_pair(s0, eps, v, n, dt)
        mm = mismatches(tA, tB)
        if len(mm) < 2 or X[:, 0].max() > 26:
            continue
        rec = (X, TAB, D, tA, tB, mm)
        if mm[-1] > 5 * mm[0] and mm[-1] > 0.3:            # mismatch clearly amplified
            score = mm[-1] / (mm[0] + 1e-9)
            if best_casc is None or score > best_casc[0]:
                best_casc = (score, *rec)
        elif mm.max() < 0.15 and len(tA) == len(tB):       # stayed locked together
            score = -mm.max()
            if best_clean is None or score > best_clean[0]:
                best_clean = (score, *rec)
    assert best_clean and best_casc, "no suitable pair found -- widen the search"
    return best_clean, best_casc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--window", type=int, default=10, help="steps for the local-lambda window")
    ap.add_argument("--out", default=str(ROOT / "results" / "phase2" / "clean_vs_cascade.mp4"))
    args = ap.parse_args()

    print("searching (selection = growth of the bounce-timing mismatch) ...", flush=True)
    (_, Xc, TABc, Dc, tAc, tBc, mmc), (_, Xs, TABs, Ds, tAs, tBs, mms) = find_pairs(
        args.eps, args.frames, args.dt)
    t = np.arange(args.frames) * args.dt
    W = args.window

    def cum(D):
        return np.concatenate([[np.nan], np.log(D[1:] / args.eps) / t[1:]])

    def local(D):
        """Rate over a sliding window. Cumulative lambda divides by TOTAL elapsed time, so it
        smears any localised change across the whole history."""
        o = np.full(len(D), np.nan)
        o[W:] = np.log(D[W:] / D[:-W]) / (W * args.dt)
        return o

    lam_c, lam_s, loc_c, loc_s = cum(Dc), cum(Ds), local(Dc), local(Ds)
    print(f"  clean   : bounces A {np.round(tAc,2)}  B {np.round(tBc,2)}")
    print(f"            mismatch {np.round(mmc,3)}  final d {Dc[-1]:.3f}  lambda {lam_c[-1]:+.3f}")
    print(f"  cascade : bounces A {np.round(tAs,2)}  B {np.round(tBs,2)}")
    print(f"            mismatch {np.round(mms,3)}  final d {Ds[-1]:.3f}  lambda {lam_s[-1]:+.3f}",
          flush=True)

    fig = plt.figure(figsize=(16.5, 7.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.9, 1.15, 1.15, 1.15], hspace=0.36, wspace=0.32)
    rows = []
    ymax = max(Xc.max(), Xs.max()) * 1.1 + 1
    dmax = max(Dc.max(), Ds.max()) * 1.25
    lomax = max(np.nanmax(loc_c), np.nanmax(loc_s)) * 1.15
    lomin = min(np.nanmin(loc_c), np.nanmin(loc_s)) * 1.15
    lmax = max(np.nanmax(lam_c[5:]), np.nanmax(lam_s[5:])) * 1.2

    for r, (name, X, TAB, D, tA, tB, lam, loc, col) in enumerate([
            ("CLEAN — bounces stay locked together", Xc, TABc, Dc, tAc, tBc,
             lam_c, loc_c, "#2980b9"),
            ("CASCADE — bounce timing drifts apart", Xs, TABs, Ds, tAs, tBs,
             lam_s, loc_s, "#c0392b")]):
        ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[r, i]) for i in range(4))

        ax0.set_xlim(-1.4, 1.4); ax0.set_ylim(-1.6, ymax); ax0.set_xticks([])
        ax0.set_ylabel("height"); ax0.set_title(name, fontsize=10, color=col)
        bA, = ax0.plot([], [], "o", ms=11, color="#e67e22", zorder=5)
        bB, = ax0.plot([], [], "o", ms=11, color=col, zorder=5)
        tab, = ax0.plot([], [], "-", lw=5, color="#2c3e50", solid_capstyle="butt")
        txt = ax0.text(0.03, 0.97, "", transform=ax0.transAxes, va="top", fontsize=9,
                       family="monospace")

        for a, (series, ylab, lo, hi, title) in zip(
                (ax1, ax2, ax3),
                [(D, "separation  d(t)", 0, dmax, "separation"),
                 (lam, "cumulative λ(t)", -0.15, lmax, "cumulative λ — smears the event"),
                 (loc, f"local λ (window {W})", lomin, lomax,
                  "local λ — resolves each bounce")]):
            a.set_xlim(0, t[-1]); a.set_ylim(lo, hi)
            a.set_ylabel(ylab); a.set_xlabel("time"); a.set_title(title, fontsize=9)
            for bt in tA:                                  # every bounce of each ball
                a.axvline(bt, color="#e67e22", lw=1.0, alpha=0.6)
            for bt in tB:
                a.axvline(bt, color=col, lw=1.0, alpha=0.6, ls="--")
            a.plot(t, series, "-", lw=0.9, color=col, alpha=0.22)
        ax2.axhline(LAM_TRUE, color="k", ls="--", lw=0.9)
        ax3.axhline(LAM_TRUE, color="k", ls="--", lw=0.9)
        ld, = ax1.plot([], [], "-", lw=2, color=col)
        lc, = ax2.plot([], [], "-", lw=2, color=col)
        ll, = ax3.plot([], [], "-", lw=2, color=col)
        rows.append((bA, bB, tab, txt, ld, lc, ll, X, TAB, D, tA, tB, lam, loc))

    fig.suptitle("Same perturbation (ε=%.2f).  Solid orange = ball A bounces, dashed = ball B. "
                 "The gap between them is the whole story." % args.eps, fontsize=11)
    bar = np.array([-1.0, 1.0])

    def update(i):
        arts = []
        for (bA, bB, tab, txt, ld, lc, ll, X, TAB, D, tA, tB, lam, loc) in rows:
            bA.set_data([-0.45], [X[i, 0]]); bB.set_data([0.45], [X[i, 1]])
            tab.set_data(bar, [TAB[i], TAB[i]])
            nA = int((tA <= t[i]).sum()); nB = int((tB <= t[i]).sum())
            k = min(nA, nB)
            mm_now = abs(tA[k - 1] - tB[k - 1]) if k else 0.0
            txt.set_text(f"t {t[i]:6.2f}\nbounces {nA}/{nB}\n"
                         f"Δt bounce {mm_now:6.3f}\nd {D[i]:9.4f}\nλ {lam[i]:+9.3f}"
                         if i else "")
            ld.set_data(t[:i + 1], D[:i + 1])
            lc.set_data(t[:i + 1], lam[:i + 1])
            ll.set_data(t[:i + 1], loc[:i + 1])
            arts += [bA, bB, tab, txt, ld, lc, ll]
        return arts

    anim = FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2600))
    plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
