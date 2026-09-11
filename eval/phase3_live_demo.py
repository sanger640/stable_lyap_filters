"""Five live-monitor videos: basin entropy running online, alarming before the block goes over.

Receding-horizon version of eval/phase3_pipeline.py. Every few steps the monitor takes the
CURRENT state, perturbs the REMAINING action, rolls each probe through the model, lets it
settle, assigns each to a discovered attractor, and reports S = -sum p_j ln p_j. Alarm on S > 0.

FOUR PANELS, so the entropy is something you can watch rather than a plotted number:

  1  the true block NOW, plus a ghost for every probe drawn at ITS PREDICTED FINAL TILT,
     coloured by which attractor it was assigned to. Piles of ghosts at two different places
     IS the split that S measures.
  2  the same probe endpoints in LATENT space (PCA to 2-D) with the attractor centres marked.
     This is what the monitor actually sees -- no physical interpretation involved.
  3  ground truth: what the block really does, with the alarm and topple times marked.
  4  the basin entropy trace, with ln2 / ln3 reference lines.

Panels 1 (ghosts), 2 and 4 come from the model alone; only the solid block and panel 3 are
ground truth, and they are never used by the monitor.

The five cases include the method's blind spot as well as its successes: a push far PAST the
threshold topples robustly, every probe agrees it falls, entropy is LOW and the monitor stays
quiet. Correct for a proximity detector, disqualifying for a failure detector.
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
ACOL = ["#8e44ad", "#27ae60", "#d35400"]                          # per-attractor colours


def corners(theta):
    a, b = np.sin(ALPHA), np.cos(ALPHA)
    base = np.array([[-a, 0.0], [a, 0.0], [a, 2 * b], [-a, 2 * b]])
    piv = np.array([a if theta >= 0 else -a, 0.0])
    c, s = np.cos(-theta), np.sin(-theta)
    return (base - piv) @ np.array([[c, -s], [s, c]]).T + piv


class Monitor:
    """Basin entropy from an arbitrary current state over the remaining action.

    Returns the internals as well as the score, so the video can show WHY S is what it is."""

    def __init__(self, model, C, mu, sd, amu, asd, thr, settle, eps, n_probe, rng,
                 lookahead=None):
        """`lookahead` fixes how many steps of the plan are considered. None = receding, i.e.
        all remaining action -- which makes the question SHRINK as the episode proceeds, so a
        safe episode's entropy must decay to zero by construction and carries no information
        late on. A fixed lookahead asks the same question at every t: constant cost AND constant
        meaning. Measured on 100 episodes: fixed H=200 gives AUC 0.836 vs 0.805 receding."""
        self.__dict__.update(model=model, C=C, mu=mu, sd=sd, amu=amu, asd=asd, thr=thr,
                             settle=settle, eps=eps, n_probe=n_probe, rng=rng,
                             lookahead=lookahead)

    @torch.no_grad()
    def score(self, state, remaining):
        if self.lookahead is not None:
            remaining = remaining[:self.lookahead]
        if len(remaining) < 5:
            return 0.0, None, None, None
        P = remaining[None] * (1.0 + self.eps * self.rng.standard_normal((self.n_probe, 1)))
        long = np.concatenate([P, np.zeros((len(P), self.settle))], axis=1)
        an = ((torch.from_numpy(long[:, :, None]).double() - self.amu) / self.asd)
        s0 = ((torch.from_numpy(np.tile(state, (len(P), 1))).double() - self.mu) / self.sd)
        Z = rollout(self.model, an, s0, long.shape[1])
        E = Z[:, -1].numpy()
        th = (self.model.observe(Z)[:, -1, 0] * self.sd[0] + self.mu[0]).numpy()
        lab = np.argmin(((E[:, None] - self.C[None]) ** 2).sum(-1), axis=1)
        p = np.array([(lab == j).mean() for j in range(len(self.C))])
        pp = p[p > 0]
        return float(-(pp * np.log(pp)).sum()), E, lab, th


def pick_cases(rng, n, thr, n_try=1500):
    want = [("A  topple, right on the edge", lambda f, m: f and m < 0.05),
            ("B  topple, close to the edge", lambda f, m: f and 0.05 <= m < 0.12),
            ("C  survives, but only just", lambda f, m: (not f) and m < 0.05),
            ("D  survives comfortably", lambda f, m: (not f) and m > 0.35),
            ("E  topples decisively (BLIND SPOT)", lambda f, m: f and m > 0.30)]
    found = {}
    for _ in range(n_try):
        a = tb.random_push(rng, n, thr)
        f = bool(tb.simulate(a)[1][-1]); m = margin_of(a)
        for name, test in want:
            if name not in found and test(f, m):
                found[name] = (a, m, f)
        if len(found) == len(want):
            break
    return [(nm, *found[nm]) for nm, _ in want if nm in found]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--settle", type=int, default=150)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--every", type=int, default=15)
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
    E0, step = settle_latents(model, off, mu, sd, amu, asd, s0n, n, 400)
    scale = np.linalg.norm(E0 - E0.mean(0), axis=1).mean()
    C, d, _, _ = find_attractors(E0[step < 0.01 * scale], scale)
    # PCA basis from the offline endpoints, so the latent panel is a fixed, honest projection
    Ec = E0 - E0.mean(0)
    basis = np.linalg.svd(Ec, full_matrices=False)[2][:2]
    proj = lambda X: (X - E0.mean(0)) @ basis.T                         # noqa: E731
    with torch.no_grad():
        cth = (model.observe(torch.from_numpy(C).double())[:, 0] * sd[0] + mu[0]).numpy()
    order = np.argsort(cth)                                             # left, upright, right
    cname = {order[0]: "falls left", order[1]: "stays up", order[2]: "falls right"}
    print(f"  {len(C)} attractors, readout tilt {np.round(cth,2)} -> {[cname[j] for j in range(len(C))]}\n",
          flush=True)

    mon = Monitor(model, C, mu, sd, amu, asd, thr, st, args.eps, args.n_probe, rng)
    cases = pick_cases(np.random.default_rng(4), n, thr)
    times = np.arange(0, n, args.every)
    Cp = proj(C)

    for i, (name, act, marg, fell) in enumerate(cases, 1):
        traj, _ = tb.simulate(act, dt=tb.DT_DEFAULT)
        th = traj[:, 0]
        cross = int(np.argmax(np.abs(th) >= ALPHA)) if (np.abs(th) >= ALPHA).any() else None
        S, EE, LL, TH = [], [], [], []
        for t in times:
            s_, e_, l_, t_ = mon.score(traj[t], act[t:])
            S.append(s_); EE.append(e_); LL.append(l_); TH.append(t_)
        S = np.array(S)
        alarm = np.where(S > 1e-9)[0]
        first = int(times[alarm[0]]) if len(alarm) else None
        warn = (cross - first) if (cross is not None and first is not None) else None
        print(f"{name}: margin {marg:.1%}, topples={fell}, alarm t={first}, "
              f"alpha t={cross}, warning {warn if warn is not None else '-'}", flush=True)

        idx = np.arange(0, n, args.stride); tt = idx * tb.DT_DEFAULT
        fig = plt.figure(figsize=(17, 4.9))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.0, 1.25, 1.25], wspace=0.3)
        ax0, axL, ax1, ax2 = (fig.add_subplot(gs[j]) for j in range(4))
        col = "#c0392b" if fell else "#2980b9"

        ax0.set_xlim(-1.7, 1.7); ax0.set_ylim(-0.25, 2.35); ax0.set_aspect("equal")
        ax0.set_xticks([]); ax0.set_yticks([]); ax0.axhline(0, color="#2c3e50", lw=3)
        ghosts = [Polygon(corners(0.0), closed=True, fc="grey", ec="none", alpha=0.10)
                  for _ in range(args.n_probe)]
        for g in ghosts:
            ax0.add_patch(g)
        blk = Polygon(corners(0.0), closed=True, fc=col, ec="#2c3e50", lw=1.6, alpha=0.9)
        ax0.add_patch(blk)
        banner = ax0.text(0.5, 0.95, "", transform=ax0.transAxes, ha="center", fontsize=13,
                          fontweight="bold")
        info = ax0.text(0.02, 0.03, "", transform=ax0.transAxes, fontsize=8.5,
                        family="monospace")
        ax0.set_title("block now (solid) + every probe's\npredicted ENDING (ghosts)", fontsize=9)

        axL.set_title("the same probes in LATENT space\n(model only)", fontsize=9)
        axL.set_xlabel("PC1"); axL.set_ylabel("PC2")
        allp = proj(E0)
        axL.set_xlim(allp[:, 0].min() * 1.1, allp[:, 0].max() * 1.1)
        axL.set_ylim(allp[:, 1].min() * 1.1, allp[:, 1].max() * 1.1)
        axL.scatter(allp[:, 0], allp[:, 1], s=3, c="#bdc3c7", alpha=0.25, lw=0)
        for j in range(len(C)):
            axL.scatter([Cp[j, 0]], [Cp[j, 1]], s=170, marker="X", c=ACOL[j],
                        ec="k", lw=1.2, zorder=6, label=cname[j])
        pts = axL.scatter([], [], s=34, c=[], ec="k", lw=0.4, zorder=5)
        axL.legend(fontsize=7, loc="best", framealpha=0.9)

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
        ax1.set_title("GROUND TRUTH (not seen by the monitor)", fontsize=9)

        ax2.set_xlim(0, tt[-1]); ax2.set_ylim(-0.05, 1.25)
        ax2.set_xlabel("time"); ax2.set_ylabel("basin entropy S")
        ax2.axhline(np.log(2), color="grey", ls=":", lw=1)
        ax2.text(0.02, np.log(2), " ln2 = even 2-way split", fontsize=7, va="bottom")
        ax2.axhline(np.log(3), color="grey", ls=":", lw=1)
        ax2.text(0.02, np.log(3), " ln3 = even 3-way split", fontsize=7, va="bottom")
        ax2.axhline(0, color="#e67e22", ls="--", lw=1.5)
        ax2.plot(times * tb.DT_DEFAULT, S, "-", lw=0.8, color="#e67e22", alpha=0.25)
        lS, = ax2.plot([], [], "o-", lw=2, ms=3, color="#e67e22")
        ax2.set_title("monitor output — alarm when S > 0", fontsize=9)

        fig.suptitle(f"{name}   |   true margin {marg:.1%}"
                     + (f"   |   WARNING {warn} steps before the topple" if warn else ""),
                     fontsize=12)

        def update(k, blk=blk, ghosts=ghosts, banner=banner, info=info, pts=pts,
                   lth=lth, lS=lS, th=th, S=S, EE=EE, LL=LL, TH=TH,
                   cross=cross, first=first):
            j = idx[k]
            blk.set_xy(corners(th[j]))
            m = np.where(times <= j)[0]
            mi = m[-1] if len(m) else 0
            lab, thp, Ep = LL[mi], TH[mi], EE[mi]
            if lab is not None:
                for gi, g in enumerate(ghosts):
                    g.set_xy(corners(np.clip(thp[gi], -np.pi / 2, np.pi / 2)))
                    g.set_facecolor(ACOL[lab[gi]]); g.set_alpha(0.16)
                pts.set_offsets(proj(Ep)); pts.set_color([ACOL[l] for l in lab])
                votes = "/".join(str(int((lab == q).sum())) for q in range(len(C)))
            else:
                votes = "-"
            hot = S[mi] > 1e-9
            banner.set_text("⚠ NEAR BOUNDARY" if hot else "clear")
            banner.set_color("#c0392b" if hot else "#27ae60")
            info.set_text(f"t {tt[k]:6.2f}\nθ {th[j]:+6.3f}\nvotes {votes}\nS {S[mi]:6.3f}")
            lth.set_data(np.arange(j + 1) * tb.DT_DEFAULT, th[:j + 1])
            lS.set_data(times[m] * tb.DT_DEFAULT, S[m])
            return blk, *ghosts, banner, info, pts, lth, lS

        anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / args.fps, blit=False)
        out = OUT / f"live_{i}_{name.split()[0]}.mp4"
        anim.save(out, writer=FFMpegWriter(fps=args.fps, bitrate=2800))
        plt.close(fig)
        print(f"   -> {out}", flush=True)


if __name__ == "__main__":
    main()
