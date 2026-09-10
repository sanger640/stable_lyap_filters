"""Five live-monitor videos: basin entropy running online, alarming before the block goes over.

This is a RECEDING-HORIZON version of eval/phase3_pipeline.py. Rather than scoring the action
once up front, the monitor re-evaluates every few steps: it takes the CURRENT state, perturbs
the REMAINING action, rolls each probe through the model, lets it settle, assigns each to a
discovered attractor, and reports S = -sum p_j ln p_j. Alarm on S > 0 -- the probes disagreed
about where this ends.

The number that matters is the WARNING TIME: how long before the block actually crosses the
topple angle does the alarm first fire.

The five cases are chosen to include the method's blind spot as well as its successes. A push
far PAST the threshold topples robustly -- every probe agrees it falls -- so the entropy is
LOW and the monitor stays quiet. That is correct behaviour for a proximity detector and wrong
behaviour for a failure detector, and the demo should show it rather than hide it.
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
import torch                                                      # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation      # noqa: E402
from matplotlib.patches import Polygon                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / p) for p in ("systems", "models", "training")]
sys.path.insert(0, str(ROOT / "eval"))

import tipping_block as tb                                        # noqa: E402
from shplrnn import ShPLRNN                                       # noqa: E402
from run_phase3_monitor import ACT_DIM, OBS_DIM, T_OFF, T_ON, make_data, margin_of, rollout
from phase3_pipeline import find_attractors, settle_latents       # noqa: E402

ALPHA = tb.ALPHA_DEFAULT
OUT = ROOT / "results" / "phase3"


def corners(theta):
    a, b = np.sin(ALPHA), np.cos(ALPHA)
    base = np.array([[-a, 0.0], [a, 0.0], [a, 2 * b], [-a, 2 * b]])
    piv = np.array([a if theta >= 0 else -a, 0.0])
    c, s = np.cos(-theta), np.sin(-theta)
    return (base - piv) @ np.array([[c, -s], [s, c]]).T + piv


class Monitor:
    """Basin entropy from an arbitrary current state over the remaining action."""

    def __init__(self, model, C, mu, sd, amu, asd, thr, settle, eps, n_probe, rng):
        self.__dict__.update(model=model, C=C, mu=mu, sd=sd, amu=amu, asd=asd,
                             thr=thr, settle=settle, eps=eps, n_probe=n_probe, rng=rng)

    @torch.no_grad()
    def score(self, state, remaining):
        if len(remaining) < 5:
            return 0.0
        P = remaining[None] * (1.0 + self.eps * self.rng.standard_normal((self.n_probe, 1)))
        long = np.concatenate([P, np.zeros((len(P), self.settle))], axis=1)
        an = ((torch.from_numpy(long[:, :, None]).double() - self.amu) / self.asd)
        s0 = ((torch.from_numpy(np.tile(state, (len(P), 1))).double() - self.mu) / self.sd)
        Z = rollout(self.model, an, s0, long.shape[1])
        E = Z[:, -1].numpy()
        lab = np.argmin(((E[:, None] - self.C[None]) ** 2).sum(-1), axis=1)
        p = np.array([(lab == j).mean() for j in range(len(self.C))])
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())


def pick_cases(rng, n, thr, n_try=1500):
    """Five cases spanning the method's behaviour, INCLUDING its blind spot."""
    want = [("A  topple, right on the edge", lambda f, m: f and m < 0.05),
            ("B  topple, close to the edge", lambda f, m: f and 0.05 <= m < 0.12),
            ("C  survives, but only just", lambda f, m: (not f) and m < 0.05),
            ("D  survives comfortably", lambda f, m: (not f) and m > 0.35),
            ("E  topples decisively (BLIND SPOT)", lambda f, m: f and m > 0.30)]
    found = {}
    for _ in range(n_try):
        a = tb.random_push(rng, n, thr)
        f = bool(tb.simulate(a)[1][-1])
        m = margin_of(a)
        for name, test in want:
            if name not in found and test(f, m):
                found[name] = (a, m, f)
        if len(found) == len(want):
            break
    return [(nm, *found[nm]) for nm, _ in want if nm in found]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--settle", type=int, default=400)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--every", type=int, default=15, help="monitor re-evaluates every N steps")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    n, st = args.steps, args.settle

    thr = tb.topple_threshold(n, T_ON, T_OFF)
    _, _, mu, sd, amu, asd = make_data(40, thr, n, seed=0)
    model = ShPLRNN(d=4, H=128, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
    model.load_state_dict(torch.load(OUT / f"monitor_model_T{n}.pt"))
    s0n = ((torch.zeros(1, OBS_DIM).double() - mu) / sd)
    rng = np.random.default_rng(21)

    print("discovering attractors ...", flush=True)
    off = np.stack([tb.random_push(rng, n, thr) for _ in range(600)])
    E, step = settle_latents(model, off, mu, sd, amu, asd, s0n, n, st)
    scale = np.linalg.norm(E - E.mean(0), axis=1).mean()
    C, d, _, _ = find_attractors(E[step < 0.01 * scale], scale)
    print(f"  {len(C)} attractors (d={d:.3f})\n", flush=True)

    mon = Monitor(model, C, mu, sd, amu, asd, thr, st, args.eps, args.n_probe, rng)
    cases = pick_cases(np.random.default_rng(4), n, thr)
    print(f"found {len(cases)} cases\n", flush=True)

    times = np.arange(0, n, args.every)
    for i, (name, act, marg, fell) in enumerate(cases, 1):
        traj, _ = tb.simulate(act, dt=tb.DT_DEFAULT)
        th = traj[:, 0]
        cross = int(np.argmax(np.abs(th) >= ALPHA)) if (np.abs(th) >= ALPHA).any() else None
        S = np.array([mon.score(traj[t], act[t:]) for t in times])
        alarm = np.where(S > 1e-9)[0]
        first = int(times[alarm[0]]) if len(alarm) else None
        warn = (cross - first) if (cross is not None and first is not None) else None
        print(f"{name}: margin {marg:.1%}, topples={fell}, "
              f"first alarm t={first}, crosses alpha t={cross}, "
              f"warning {warn if warn is not None else '-'} steps", flush=True)

        idx = np.arange(0, n, args.stride)
        tt = idx * tb.DT_DEFAULT
        fig = plt.figure(figsize=(14, 4.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 1.3], wspace=0.28)
        ax0, ax1, ax2 = (fig.add_subplot(gs[j]) for j in range(3))
        col = "#c0392b" if fell else "#2980b9"
        ax0.set_xlim(-1.6, 1.6); ax0.set_ylim(-0.25, 2.3); ax0.set_aspect("equal")
        ax0.set_xticks([]); ax0.set_yticks([]); ax0.axhline(0, color="#2c3e50", lw=3)
        blk = Polygon(corners(0.0), closed=True, fc=col, ec="#2c3e50", lw=1.5, alpha=0.82)
        ax0.add_patch(blk)
        banner = ax0.text(0.5, 0.95, "", transform=ax0.transAxes, ha="center", fontsize=14,
                          fontweight="bold")
        info = ax0.text(0.02, 0.03, "", transform=ax0.transAxes, fontsize=9, family="monospace")

        ax1.set_xlim(0, tt[-1]); ax1.set_ylim(-1.75, 1.75)
        ax1.set_xlabel("time"); ax1.set_ylabel("tilt θ")
        for s_ in (+1, -1):
            ax1.axhline(s_ * ALPHA, color="k", ls="--", lw=0.9)
        ax1.text(tt[-1], ALPHA, " topple point ", ha="right", va="bottom", fontsize=7)
        ax1.plot(np.arange(n) * tb.DT_DEFAULT, th[:n], "-", lw=0.8, color=col, alpha=0.25)
        lth, = ax1.plot([], [], "-", lw=2, color=col)
        if cross is not None:
            ax1.axvline(cross * tb.DT_DEFAULT, color="#c0392b", lw=1.2, ls=":")
        if first is not None:
            ax1.axvline(first * tb.DT_DEFAULT, color="#e67e22", lw=1.2)
        ax1.set_title("what actually happens", fontsize=9)

        ax2.set_xlim(0, tt[-1]); ax2.set_ylim(-0.05, max(1.15, S.max() * 1.2))
        ax2.set_xlabel("time"); ax2.set_ylabel("basin entropy S")
        ax2.axhline(0, color="#e67e22", ls="--", lw=1.5)
        ax2.text(0.02, 0.02, " alarm when S > 0", color="#e67e22", fontsize=8)
        ax2.plot(times * tb.DT_DEFAULT, S, "-", lw=0.8, color="#e67e22", alpha=0.25)
        lS, = ax2.plot([], [], "o-", lw=2, ms=3, color="#e67e22")
        ax2.set_title("the monitor (model only)", fontsize=9)
        fig.suptitle(f"{name}   |   true margin {marg:.1%}"
                     + (f"   |   WARNING {warn} steps before the topple" if warn else ""),
                     fontsize=12)

        def update(k, blk=blk, banner=banner, info=info, lth=lth, lS=lS,
                   th=th, S=S, cross=cross, first=first):
            j = idx[k]
            blk.set_xy(corners(th[j]))
            hot = first is not None and j >= first
            banner.set_text("⚠ NEAR BOUNDARY" if hot else "monitoring")
            banner.set_color("#c0392b" if hot else "#27ae60")
            m = times <= j
            info.set_text(f"t {tt[k]:6.2f}\nθ {th[j]:+6.3f}\n"
                          f"S {S[m][-1] if m.any() else 0:6.3f}")
            lth.set_data(np.arange(j + 1) * tb.DT_DEFAULT, th[:j + 1])
            lS.set_data(times[m] * tb.DT_DEFAULT, S[m])
            return blk, banner, info, lth, lS

        anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / args.fps, blit=False)
        out = OUT / f"live_{i}_{name.split()[0]}.mp4"
        anim.save(out, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
        plt.close(fig)
        print(f"   -> {out}", flush=True)


if __name__ == "__main__":
    main()
