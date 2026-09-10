"""
Visual inspection of the Phase-2 bouncing ball. Renders the trajectory so the physics can be
checked by eye before any model is trained on it.

Rendered at a FINE dt for smooth motion; the training regime samples the same trajectory at
dt=0.20. Those are not different simulations -- impacts are resolved exactly by bisection
regardless of dt, so dt only sets how often the trajectory is *observed*. The middle panel
marks where the training samples actually land, which is the honest picture of what the model
gets to see: roughly one sample every 37 steps contains an impact.
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


def rollout(n, dt, omega, e, seed, burn_in):
    """Fine-grained trajectory plus per-sample impact flags."""
    rng = np.random.default_rng(seed)
    s = np.array([2.0 + rng.uniform(0, 1), rng.uniform(-1, 1), rng.uniform(0, bb.TWO_PI)])
    for _ in range(burn_in):
        s = bb.step(s, dt, omega, e)
    out = np.empty((n, 3))
    hit = np.zeros(n, dtype=bool)
    for i in range(n):
        out[i] = s
        prev_v = s[1]
        s = bb.step(s, dt, omega, e)
        hit[i] = (s[1] - prev_v) > bb.G * dt * 1.5
    return out, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega", type=float, default=1.4)
    ap.add_argument("--e", type=float, default=0.8)
    ap.add_argument("--dt-fine", type=float, default=0.05, help="render step (smooth motion)")
    ap.add_argument("--dt-train", type=float, default=0.20, help="Phase-2 sampling step")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase2" / "bouncing_ball.mp4"))
    args = ap.parse_args()

    traj, hit = rollout(args.frames, args.dt_fine, args.omega, args.e, args.seed, 2000)
    t = np.arange(args.frames) * args.dt_fine
    x, phi = traj[:, 0], traj[:, 2]
    tab = bb.table_pos(phi)
    every = max(1, int(round(args.dt_train / args.dt_fine)))       # training-sample stride
    gamma = args.omega ** 2
    detach = np.arcsin(min(1.0, bb.G / (bb.A * gamma)))            # sin(phi) = 1/Gamma

    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 2.0], wspace=0.28)
    ax_full, ax_zoom, ax_ts = (fig.add_subplot(gs[i]) for i in range(3))
    fig.suptitle(f"Bouncing ball on a driven table   —   ω={args.omega} (Γ={gamma:.2f}), "
                 f"e={args.e},  λ_max≈0.158", fontsize=12)

    # ---- left: full flight ----------------------------------------------------------------
    ax_full.set_xlim(-1.6, 1.6); ax_full.set_ylim(-1.6, x.max() * 1.12 + 1)
    ax_full.set_title("full flight", fontsize=10)
    ax_full.set_ylabel("height"); ax_full.set_xticks([])
    ball_f, = ax_full.plot([], [], "o", ms=13, color="#c0392b", zorder=5)
    tab_f, = ax_full.plot([], [], "-", lw=5, color="#2c3e50", solid_capstyle="butt")
    trail_f, = ax_full.plot([], [], "-", lw=1, color="#c0392b", alpha=0.35)
    ax_full.axhline(0, color="grey", lw=0.6, ls=":")

    # ---- middle: zoom on the guard ---------------------------------------------------------
    ax_zoom.set_xlim(-1.6, 1.6); ax_zoom.set_ylim(-1.8, 3.2)
    ax_zoom.set_title(f"zoom near the guard  (table amplitude A={bb.A:.0f})", fontsize=10)
    ax_zoom.set_xticks([])
    ball_z, = ax_zoom.plot([], [], "o", ms=15, color="#c0392b", zorder=5)
    tab_z, = ax_zoom.plot([], [], "-", lw=7, color="#2c3e50", solid_capstyle="butt")
    ax_zoom.axhline(bb.A, color="#2c3e50", lw=0.6, ls="--", alpha=0.5)
    ax_zoom.axhline(-bb.A, color="#2c3e50", lw=0.6, ls="--", alpha=0.5)
    flash = ax_zoom.scatter([], [], s=420, facecolors="none", edgecolors="#e67e22", lw=2.5)
    txt = ax_zoom.text(0.03, 0.97, "", transform=ax_zoom.transAxes, va="top", fontsize=9,
                       family="monospace")

    # ---- right: time series ----------------------------------------------------------------
    ax_ts.set_xlim(t[0], t[-1]); ax_ts.set_ylim(-2, x.max() * 1.12 + 1)
    ax_ts.set_title("height vs time — dots mark what the model actually sees (dt=%.2f)"
                    % args.dt_train, fontsize=10)
    ax_ts.set_xlabel("time"); ax_ts.set_ylabel("height")
    ax_ts.plot(t, x, "-", lw=1.0, color="#c0392b", alpha=0.30)
    ax_ts.plot(t, tab, "-", lw=0.8, color="#2c3e50", alpha=0.30)
    ax_ts.plot(t[::every], x[::every], ".", ms=4, color="#7f8c8d", alpha=0.55,
               label=f"training samples (dt={args.dt_train})")
    ax_ts.plot(t[hit], x[hit], "v", ms=7, color="#e67e22", label="impact")
    ax_ts.legend(loc="upper right", fontsize=8, framealpha=0.9)
    live_x, = ax_ts.plot([], [], "-", lw=1.8, color="#c0392b")
    live_t, = ax_ts.plot([], [], "-", lw=1.4, color="#2c3e50")
    cursor = ax_ts.axvline(t[0], color="k", lw=0.8, alpha=0.5)

    bar = np.array([-1.0, 1.0])
    n_hit = np.cumsum(hit)

    def update(i):
        h, tb = x[i], tab[i]
        ball_f.set_data([0], [h]); tab_f.set_data(bar, [tb, tb])
        j = max(0, i - 120)
        trail_f.set_data(np.zeros(i - j + 1), x[j:i + 1])
        ball_z.set_data([0], [h]); tab_z.set_data(bar, [tb, tb])
        flash.set_offsets(np.array([[0.0, tb]]) if hit[i] else np.empty((0, 2)))
        contact = bb.contact_force(phi[i], args.omega)
        txt.set_text(f"t   {t[i]:7.2f}\nv   {traj[i,1]:7.3f}\nφ   {phi[i]:7.3f}\n"
                     f"gap {h - tb:7.3f}\nimpacts {n_hit[i]:3d}\n"
                     f"contact {'holds' if contact >= 0 else 'RELEASE':>7}")
        live_x.set_data(t[:i + 1], x[:i + 1]); live_t.set_data(t[:i + 1], tab[:i + 1])
        cursor.set_xdata([t[i], t[i]])
        return ball_f, tab_f, trail_f, ball_z, tab_z, flash, txt, live_x, live_t, cursor

    anim = FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    plt.close(fig)

    print(f"impacts in clip : {int(hit.sum())} over {t[-1]:.1f} time units")
    print(f"height          : median {np.median(x):.2f}, max {x.max():.2f} "
          f"(table amplitude {bb.A:.0f})")
    print(f"detach phase    : sin(φ) > 1/Γ = {bb.G/(bb.A*gamma):.3f}  ->  φ = {detach:.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
