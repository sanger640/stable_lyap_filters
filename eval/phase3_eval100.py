"""100-episode evaluation of the refined monitor, plus 10 stratified demo videos.

Refinements over the earlier runs, both from analysing the five-case demo:

  * SCORE AT t=0, the decision point -- before the action executes. No latching. Latching
    converted a single noisy frame into an episode-level false positive (case D was correctly
    silent at t=0 and only alarmed at t=15).
  * ALARM ON k >= 2 DISSENTING PROBES, not S > 0. One dissenter out of 32 is a 3% rate, which
    with 32 draws is consistent with a true rate near zero -- it is the sampling floor. This is
    still calibration-free: a statement about the probe sample, not a threshold in latent units.

The 10 videos are STRATIFIED across verdict types, not sampled at random and not cherry-picked:
the point is to show every failure mode, including the blind spot where a decisive topple draws
no alarm because every probe agrees it falls.
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
from run_phase3_monitor import ACT_DIM, OBS_DIM, T_OFF, T_ON, make_data, margin_of, rollout
from phase3_pipeline import find_attractors, settle_latents       # noqa: E402
from phase3_live_demo import Monitor, corners, ACOL               # noqa: E402
from run_phase3_block import auc                                  # noqa: E402


def render(path, title, act, marg, fell, S, times, EE, LL, TH, C, proj, Cp, allp,
           cname, n, stride, fps, ALPHA):
    traj, _ = tb.simulate(act, dt=tb.DT_DEFAULT)
    th = traj[:, 0]
    cross = int(np.argmax(np.abs(th) >= ALPHA)) if (np.abs(th) >= ALPHA).any() else None
    idx = np.arange(0, n, stride); tt = idx * tb.DT_DEFAULT
    col = "#c0392b" if fell else "#2980b9"
    fig = plt.figure(figsize=(17, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.0, 1.25, 1.25], wspace=0.3)
    ax0, axL, ax1, ax2 = (fig.add_subplot(gs[j]) for j in range(4))

    ax0.set_xlim(-1.7, 1.7); ax0.set_ylim(-0.25, 2.35); ax0.set_aspect("equal")
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.axhline(0, color="#2c3e50", lw=3)
    ghosts = [Polygon(corners(0.0), closed=True, fc="grey", ec="none", alpha=0.10)
              for _ in range(len(LL[0]) if LL[0] is not None else 32)]
    for g in ghosts:
        ax0.add_patch(g)
    blk = Polygon(corners(0.0), closed=True, fc=col, ec="#2c3e50", lw=1.6, alpha=0.9)
    ax0.add_patch(blk)
    banner = ax0.text(0.5, 0.95, "", transform=ax0.transAxes, ha="center", fontsize=13,
                      fontweight="bold")
    info = ax0.text(0.02, 0.03, "", transform=ax0.transAxes, fontsize=8.5, family="monospace")
    ax0.set_title("block now (solid) + each probe's\npredicted ENDING (ghosts)", fontsize=9)

    axL.set_title("the same probes in LATENT space\n(model only)", fontsize=9)
    axL.set_xlabel("PC1"); axL.set_ylabel("PC2")
    axL.set_xlim(allp[:, 0].min() * 1.1, allp[:, 0].max() * 1.1)
    axL.set_ylim(allp[:, 1].min() * 1.1, allp[:, 1].max() * 1.1)
    axL.scatter(allp[:, 0], allp[:, 1], s=3, c="#bdc3c7", alpha=0.25, lw=0)
    for j in range(len(C)):
        axL.scatter([Cp[j, 0]], [Cp[j, 1]], s=170, marker="X", c=ACOL[j], ec="k", lw=1.2,
                    zorder=6, label=cname[j])
    pts = axL.scatter([], [], s=34, c=[], ec="k", lw=0.4, zorder=5)
    axL.legend(fontsize=7, loc="best", framealpha=0.9)

    ax1.set_xlim(0, tt[-1]); ax1.set_ylim(-1.75, 1.75)
    ax1.set_xlabel("time"); ax1.set_ylabel("tilt θ")
    for s_ in (+1, -1):
        ax1.axhline(s_ * ALPHA, color="k", ls="--", lw=0.9)
    ax1.plot(np.arange(n) * tb.DT_DEFAULT, th[:n], "-", lw=0.8, color=col, alpha=0.25)
    lth, = ax1.plot([], [], "-", lw=2, color=col)
    if cross is not None:
        ax1.axvline(cross * tb.DT_DEFAULT, color="#c0392b", lw=1.2, ls=":")
    ax1.set_title("GROUND TRUTH (not seen by the monitor)", fontsize=9)

    ax2.set_xlim(0, tt[-1]); ax2.set_ylim(-0.05, 1.25)
    ax2.set_xlabel("time"); ax2.set_ylabel("basin entropy S")
    for v, lb in ((np.log(2), " ln2"), (np.log(3), " ln3")):
        ax2.axhline(v, color="grey", ls=":", lw=1)
        ax2.text(0.02, v, lb, fontsize=7, va="bottom")
    ax2.plot(times * tb.DT_DEFAULT, S, "-", lw=0.8, color="#e67e22", alpha=0.25)
    lS, = ax2.plot([], [], "o-", lw=2, ms=3, color="#e67e22")
    ax2.set_title("basin entropy over time", fontsize=9)
    fig.suptitle(title, fontsize=12)

    def update(k):
        j = idx[k]
        blk.set_xy(corners(th[j]))
        m = np.where(times <= j)[0]; mi = m[-1] if len(m) else 0
        lab, thp, Ep = LL[mi], TH[mi], EE[mi]
        if lab is not None:
            for gi, g in enumerate(ghosts):
                g.set_xy(corners(np.clip(thp[gi], -np.pi / 2, np.pi / 2)))
                g.set_facecolor(ACOL[lab[gi]]); g.set_alpha(0.16)
            pts.set_offsets(proj(Ep)); pts.set_color([ACOL[l] for l in lab])
            v = [int((lab == q).sum()) for q in range(len(C))]
            votes = "/".join(map(str, v)); diss = len(lab) - max(v)
        else:
            votes, diss = "-", 0
        hot = diss >= 2
        banner.set_text("⚠ NEAR BOUNDARY" if hot else "clear")
        banner.set_color("#c0392b" if hot else "#27ae60")
        info.set_text(f"t {tt[k]:6.2f}\nθ {th[j]:+6.3f}\nvotes {votes}\n"
                      f"dissent {diss}\nS {S[mi]:6.3f}")
        lth.set_data(np.arange(j + 1) * tb.DT_DEFAULT, th[:j + 1])
        lS.set_data(times[m] * tb.DT_DEFAULT, S[m])
        return blk, *ghosts, banner, info, pts, lth, lS

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=2800))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--settle", type=int, default=150)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--near", type=float, default=0.10)
    ap.add_argument("--n-videos", type=int, default=10)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    n, st, ALPHA = args.steps, args.settle, tb.ALPHA_DEFAULT
    OUT = ROOT / "results" / "phase3" / "eval100"
    OUT.mkdir(parents=True, exist_ok=True)

    thr = tb.topple_threshold(n, T_ON, T_OFF)
    _, _, mu, sd, amu, asd = make_data(40, thr, n, seed=0)
    model = ShPLRNN(d=4, H=128, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
    model.load_state_dict(torch.load(ROOT / "results/phase3" / f"monitor_model_T{n}.pt"))
    s0n = ((torch.zeros(1, OBS_DIM).double() - mu) / sd)
    rng = np.random.default_rng(21)

    print("discovering attractors ...", flush=True)
    off = np.stack([tb.random_push(rng, n, thr) for _ in range(600)])
    E0, step = settle_latents(model, off, mu, sd, amu, asd, s0n, n, 400)
    scale = np.linalg.norm(E0 - E0.mean(0), axis=1).mean()
    C, _, _, _ = find_attractors(E0[step < 0.01 * scale], scale)
    basis = np.linalg.svd(E0 - E0.mean(0), full_matrices=False)[2][:2]
    proj = lambda X: (X - E0.mean(0)) @ basis.T                    # noqa: E731
    with torch.no_grad():
        cth = (model.observe(torch.from_numpy(C).double())[:, 0] * sd[0] + mu[0]).numpy()
    order = np.argsort(cth)
    cname = {order[0]: "falls left", order[1]: "stays up", order[2]: "falls right"}
    print(f"  {len(C)} attractors\n", flush=True)

    mon = Monitor(model, C, mu, sd, amu, asd, thr, st, args.eps, args.n_probe, rng)
    erng = np.random.default_rng(777)
    acts = np.stack([tb.random_push(erng, n, thr) for _ in range(args.n_episodes)])

    print(f"scoring {args.n_episodes} episodes at t=0 ...", flush=True)
    rows = []
    for i, a in enumerate(acts):
        S0, _, lab0, _ = mon.score(np.array([0.0, 0.0]), a)
        v = np.array([(lab0 == j).sum() for j in range(len(C))])
        k = int(args.n_probe - v.max())
        m = margin_of(a); f = bool(tb.simulate(a, dt=tb.DT_DEFAULT)[1][-1])
        rows.append(dict(i=i, margin=m, fell=f, S=S0, dissent=k,
                         near=bool(m < args.near), alarm=bool(k >= args.k_min)))
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{args.n_episodes}", flush=True)

    near = np.array([r["near"] for r in rows]); alarm = np.array([r["alarm"] for r in rows])
    fell = np.array([r["fell"] for r in rows]); K = np.array([r["dissent"] for r in rows])
    tp = int((alarm & near).sum()); fp = int((alarm & ~near).sum())
    fn = int((~alarm & near).sum()); tn = int((~alarm & ~near).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    blind = int((fell & ~near).sum())
    blind_missed = int((fell & ~near & ~alarm).sum())

    print(f"\n===== {args.n_episodes} EPISODES, alarm = k >= {args.k_min} of {args.n_probe},"
          f" scored at t=0, no latching =====")
    print(f"  ground truth: {int(near.sum())} near-boundary (margin < {args.near:.0%}), "
          f"{int(fell.sum())} topple")
    print(f"\n  {'':>14}{'near':>8}{'far':>8}")
    print(f"  {'ALARM':>14}{tp:>8}{fp:>8}")
    print(f"  {'no alarm':>14}{fn:>8}{tn:>8}")
    print(f"\n  precision {pr:.3f}   recall {rc:.3f}   F1 {f1:.3f}   "
          f"accuracy {(alarm == near).mean():.3f}")
    print(f"  AUC of the dissent count: {auc(K, near):.3f}")
    print(f"\n  BLIND SPOT: {blind} episodes topple while FAR from the boundary; "
          f"{blind_missed} of them draw no alarm")
    json.dump({"args": vars(args), "precision": pr, "recall": rc, "f1": f1,
               "auc": auc(K, near), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
               "n_near": int(near.sum()), "n_fell": int(fell.sum()),
               "blind_spot": blind, "blind_missed": blind_missed, "rows": rows},
              open(OUT / "results.json", "w"), indent=1, default=float)

    # ---- 10 stratified videos: every verdict type represented ---------------------------
    def verdict(r):
        if r["alarm"] and r["near"]:
            return "TP"
        if r["alarm"] and not r["near"]:
            return "FP"
        if not r["alarm"] and r["near"]:
            return "FN"
        return "BLIND" if r["fell"] else "TN"
    for r in rows:
        r["v"] = verdict(r)
    quota = {"TP": 3, "FP": 2, "FN": 2, "BLIND": 2, "TN": 1}
    chosen = []
    for v, q in quota.items():
        pool = [r for r in rows if r["v"] == v]
        chosen += pool[:q]
    chosen = chosen[:args.n_videos]
    print(f"\nrendering {len(chosen)} stratified videos "
          f"({ {v: sum(1 for r in chosen if r['v'] == v) for v in quota} })", flush=True)

    times = np.arange(0, n, args.every)
    allp = proj(E0); Cp = proj(C)
    for r in chosen:
        a = acts[r["i"]]
        traj, _ = tb.simulate(a, dt=tb.DT_DEFAULT)
        S, EE, LL, TH = [], [], [], []
        for t in times:
            s_, e_, l_, t_ = mon.score(traj[t], a[t:])
            S.append(s_); EE.append(e_); LL.append(l_); TH.append(t_)
        title = (f"[{r['v']}]  margin {r['margin']:.1%}  |  topples={r['fell']}  |  "
                 f"dissent {r['dissent']}/{args.n_probe} at t=0  ->  "
                 f"{'ALARM' if r['alarm'] else 'clear'}")
        p = OUT / f"ep{r['i']:03d}_{r['v']}.mp4"
        render(p, title, a, r["margin"], r["fell"], np.array(S), times, EE, LL, TH,
               C, proj, Cp, allp, cname, n, args.stride, args.fps, ALPHA)
        print(f"   {r['v']:>5}  margin {r['margin']:>6.1%}  -> {p.name}", flush=True)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
