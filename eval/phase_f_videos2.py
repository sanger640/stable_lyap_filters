"""Demo videos of the DINO-WM monitor -- watchable version.

The first attempt produced 1.3-second clips: 8 scored timesteps at 6 fps. Useless. The monitor only
updates 8 times per episode, but the BLOCK moves continuously, and rendering a frame costs ~10 ms.
So the camera panel runs at full temporal resolution while the monitor panels HOLD between scores --
which is also what a real runtime monitor looks like.

Scores are cached to .npz, so re-rendering never re-runs the model.

Four panels:
  1. what the monitor sees -- the 224x224 camera frame, its only input
  2. the 32 probes' predicted SETTLED endings, decoded to tilts, coloured by attractor
  3. those endings in the latent PCA plane, with the discovered centroids
  4. dissent count k over time, with a moving cursor, against the k>=2 rule
"""
import argparse, json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation       # noqa: E402
from matplotlib.patches import Polygon                             # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval"), "/home/sanger/wksp/dino_wm"]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_c_train import Predictor, NUM_HIST                      # noqa: E402
from phase_f_monitor import build_centroids, chunk_margin, STRIDE  # noqa: E402

OUT = ROOT / "results" / "phase_c"
VID = ROOT / "results" / "phase_f_videos"
SCORES = OUT / "phase_f_video_scores.npz"
ACOL = ["#8e44ad", "#27ae60", "#d35400"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--settle", type=int, default=40)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--pca", type=int, default=8)
    ap.add_argument("--times", type=int, nargs="+",
                    default=[3, 6, 9, 12, 15, 18, 21, 24])
    ap.add_argument("--sim-stride", type=int, default=2, help="render every Nth sim step")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    dev = "cuda"; VID.mkdir(parents=True, exist_ok=True); t0 = time.time()

    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    meta = np.load(OUT / "meta.npz"); A = meta["actions"]; S = meta["states"]
    p = np.load(OUT / "phase_c_eval.npz")
    mu, sd, amu, asd = (float(p[k]) for k in ("mu", "sd", "amu", "asd"))
    cache = np.load(OUT / "phase_e_endings_H20_n300.npz")
    E_all = cache[f"predictor_gtf_warm.pt_{args.settle}"].astype(np.float32)
    C, pmean, V, k_raw, k_keep, _, _ = build_centroids(E_all, args.pca)
    assign = ((((E_all - pmean) @ V.T)[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    V2 = np.linalg.svd((E_all - E_all.mean(0))[::2], full_matrices=False)[2][:2]
    Emean = E_all.mean(0)

    rng = np.random.default_rng(0); thr = tb.topple_threshold(450, 10, 60)
    acts, lights = [], []
    for _ in range(Z.shape[0]):
        a = tb.random_push(rng, 450, thr); tb.simulate(a, dt=tb.DT_DEFAULT)
        acts.append(a); lights.append(br.sample_lighting(rng))

    rows = json.load(open(OUT / "phase_f_results.json"))["rows"]
    picks = []
    for want, cond, n in (("TP", lambda r: r["m_min"] < 0.23 and r["k_max"] >= 2, 4),
                          ("FN", lambda r: r["m_min"] < 0.23 and r["k_max"] < 2, 2),
                          ("TN", lambda r: r["m_min"] >= 0.23 and r["k_max"] < 2, 3),
                          ("FP", lambda r: r["m_min"] >= 0.23 and r["k_max"] >= 2, 1)):
        picks += [(want, r) for r in sorted([x for x in rows if cond(x)],
                                            key=lambda x: x["m_min"])[:n]]
    ts = [t for t in args.times if t + args.H <= Z.shape[1]]

    if SCORES.exists() and not args.rescore:
        dat = dict(np.load(SCORES, allow_pickle=True))
        print(f"loaded cached scores ({SCORES.name})", flush=True)
    else:
        print(f"scoring {len(picks)} episodes x {len(ts)} times ...", flush=True)
        model = Predictor().to(dev)
        model.load_state_dict(torch.load(OUT / "predictor_gtf_warm.pt")); model.eval()
        ntr = 120
        Xr = np.asarray(Z[:ntr], np.float32).reshape(ntr * Z.shape[1], -1)
        yr = S[:ntr, :, 0].reshape(-1); xm = Xr.mean(0); Xc = Xr - xm
        Vk = np.linalg.svd(Xc[::3], full_matrices=False)[2][:256]
        P_ = Xc @ Vk.T
        Wg = np.linalg.solve(P_.T @ P_ + 1e-2 * np.eye(256), P_.T @ (yr - yr.mean()))
        ym = yr.mean(); del Xr, Xc, P_
        eps_rng = np.random.default_rng(777); dat = {}
        for tag, row in picks:
            i = row["i"]; st, _ = tb.simulate(acts[i], dt=tb.DT_DEFAULT)
            KS, LB, TH, XP, MS = [], [], [], [], []
            for t in ts:
                ch = A[i, t:t + args.H, 0].astype(np.float64)
                z = eps_rng.standard_normal((args.n_probe, 1))
                Pp = np.concatenate([ch[None] * (1.0 + args.eps * z),
                                     np.zeros((args.n_probe, args.settle))], axis=1)
                ac = torch.from_numpy(((Pp[:, :, None] - amu) / asd).astype(np.float32)).to(dev)
                ctx = np.asarray(Z[i, t - NUM_HIST + 1:t + 1], np.float32)
                win = ((torch.from_numpy(ctx).to(dev) - mu) / sd)[None].expand(
                    args.n_probe, -1, -1, -1).contiguous()
                with torch.no_grad():
                    for j in range(Pp.shape[1]):
                        aw = ac[:, max(0, j - NUM_HIST + 1):j + 1]
                        if aw.shape[1] < NUM_HIST:
                            aw = torch.cat([aw[:, :1].expand(-1, NUM_HIST-aw.shape[1], -1), aw], 1)
                        win = torch.cat([win[:, 1:], model(win, aw)[:, -1:]], 1)
                Ee = win[:, -1].reshape(args.n_probe, -1).float().cpu().numpy() * sd + mu
                X = (Ee - pmean) @ V.T
                lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
                KS.append(int(args.n_probe - max((lab == j).sum() for j in range(len(C)))))
                LB.append(lab); TH.append(((Ee - xm) @ Vk.T) @ Wg + ym)
                XP.append((Ee - Emean) @ V2.T)
                MS.append(chunk_margin(st[t*STRIDE], acts[i][t*STRIDE:(t+args.H)*STRIDE],
                                       args.settle * STRIDE))
            dat[f"{i}_k"] = np.array(KS); dat[f"{i}_lab"] = np.array(LB)
            dat[f"{i}_th"] = np.array(TH); dat[f"{i}_xp"] = np.array(XP)
            dat[f"{i}_m"] = np.array(MS)
            print(f"  ep{i} scored ({(time.time()-t0)/60:.1f} min)", flush=True)
        np.savez(SCORES, **dat); print(f"  cached -> {SCORES.name}", flush=True)

    # ---------------------------------------------------------------- render
    frames_idx = np.arange(0, 450, args.sim_stride)
    for tag, row in picks:
        i = row["i"]
        KS, LB, TH = dat[f"{i}_k"], dat[f"{i}_lab"], dat[f"{i}_th"]
        XP, MS = dat[f"{i}_xp"], dat[f"{i}_m"]
        st, _ = tb.simulate(acts[i], dt=tb.DT_DEFAULT)
        bg = br.prepare(lights[i], seed=i)
        cam = np.stack([br.render(st[f, 0], lights[i], bg=bg) for f in frames_idx])
        tsec = np.array(ts) * tb.DT_DEFAULT * STRIDE
        Xall = XP.reshape(-1, 2)

        fig = plt.figure(figsize=(15.5, 4.5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.3)
        ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[c]) for c in range(4))
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title("what the monitor sees (its only input)", fontsize=9)
        im = ax0.imshow(cam[0])
        ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-0.3, 2.4); ax1.set_aspect("equal")
        ax1.set_xticks([]); ax1.set_yticks([]); ax1.axhline(0, color="#2c3e50", lw=3)
        ax1.set_title("32 probes → predicted SETTLED ending", fontsize=9)
        # thin edges so 32 near-identical ghosts read as a stack rather than one blob
        ghosts = [Polygon(br.corners(0.0), closed=True, fc="grey", ec="white", lw=0.4,
                          alpha=0.20) for _ in range(len(TH[0]))]
        for g in ghosts:
            ax1.add_patch(g)
        info = ax1.text(0.02, 0.97, "", transform=ax1.transAxes, va="top", fontsize=8,
                        family="monospace")
        ax2.set_title("probe endings in latent PCA", fontsize=9)
        ax2.set_xticks([]); ax2.set_yticks([])
        cen2 = (np.stack([E_all[assign == j].mean(0) for j in range(len(C))
                          if (assign == j).any()]) - Emean) @ V2.T
        both = np.vstack([Xall, cen2])
        pad = 0.18 * (both.max(0) - both.min(0) + 1e-9)
        ax2.set_xlim(both[:,0].min()-pad[0], both[:,0].max()+pad[0])
        ax2.set_ylim(both[:,1].min()-pad[1], both[:,1].max()+pad[1])
        for q, cc in enumerate(cen2):                      # the discovered attractors
            ax2.scatter(*cc, marker="X", s=200, c=ACOL[q % 3], edgecolors="k",
                        linewidths=1.2, zorder=5)
        pts = ax2.scatter([], [], s=34, zorder=6)
        ax3.set_xlim(0, 450 * tb.DT_DEFAULT); ax3.set_ylim(-0.5, max(4, KS.max() + 1))
        ax3.axhline(2, color="#e67e22", ls="--", lw=1.5)
        ax3.text(0.02, 2.08, " alarm: k ≥ 2", color="#e67e22", fontsize=8)
        ax3.axvspan(tsec[-1], 450 * tb.DT_DEFAULT, color="grey", alpha=0.13, lw=0)
        ax3.text(tsec[-1] + 0.1, ax3.get_ylim()[1] * 0.55, "monitor horizon\nends here",
                 fontsize=7, color="#555")
        ax3.plot(tsec, KS, "o-", lw=1, color="#c0392b", alpha=0.25)
        kl, = ax3.plot([], [], "o-", lw=2.2, color="#c0392b")
        cur = ax3.axvline(0, color="#2c3e50", lw=1, alpha=0.6)
        ax3.set_xlabel("time (s)"); ax3.set_ylabel("dissent count k")
        ax3.set_title("monitor output", fontsize=9)
        verdict = ax3.text(0.5, 0.9, "", transform=ax3.transAxes, ha="center",
                           fontsize=13, fontweight="bold")
        fig.suptitle(f"[{tag}] episode {i} — DINO-WM monitor — min chunk margin "
                     f"{MS.min():.1%},  peak k {KS.max()}/{len(TH[0])} "
                     f"(Phase F saw {row['k_max']} at 3 scoring times)", fontsize=12)

        def update(f):
            simf = frames_idx[f]
            im.set_data(cam[f])
            j = int(np.searchsorted(np.array(ts) * STRIDE, simf, side="right") - 1)
            if j < 0:
                for g in ghosts:
                    g.set_alpha(0.0)
                pts.set_offsets(np.empty((0, 2))); info.set_text("warming up…")
                kl.set_data([], []); verdict.set_text("")
            else:
                for g, th_, l_ in zip(ghosts, TH[j], LB[j]):
                    g.set_alpha(0.22)
                    g.set_xy(br.corners(float(np.clip(th_, -np.pi/2, np.pi/2))))
                    g.set_facecolor(ACOL[l_ % 3])
                pts.set_offsets(XP[j]); pts.set_color([ACOL[l % 3] for l in LB[j]])
                votes = "/".join(str(int((LB[j] == q).sum())) for q in range(len(C)))
                info.set_text(f"t {simf*tb.DT_DEFAULT:5.2f}s\nvotes {votes}\n"
                              f"dissent {KS[j]}\nchunk margin {MS[j]:.1%}")
                kl.set_data(tsec[:j+1], KS[:j+1])
                hot = KS[j] >= 2
                verdict.set_text("⚠ NEAR BOUNDARY" if hot else "clear")
                verdict.set_color("#c0392b" if hot else "#27ae60")
            cur.set_xdata([simf * tb.DT_DEFAULT] * 2)
            return [im, *ghosts, pts, info, kl, cur, verdict]

        anim = FuncAnimation(fig, update, frames=len(frames_idx),
                             interval=1000/args.fps, blit=False)
        path = VID / f"ep{i:03d}_{tag}.mp4"
        anim.save(path, writer=FFMpegWriter(fps=args.fps, bitrate=3000))
        plt.close(fig)
        print(f"  -> {path.name}  {len(frames_idx)/args.fps:.1f}s "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"\n({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
