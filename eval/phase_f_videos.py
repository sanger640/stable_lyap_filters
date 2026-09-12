"""Demo videos of the DINO-WM monitor running.

Four panels per frame:
  1. THE CAMERA VIEW the monitor actually sees -- the rendered 224x224 frame. This is the whole
     point of the image swap: no theta, no omega, just pixels.
  2. THE PROBE ENSEMBLE -- each of the 32 perturbed action sequences rolled to its settled ending,
     decoded back to a tilt and drawn as a ghost block, coloured by which attractor it landed in.
     When the ghosts disagree, the action is near a boundary.
  3. THE LATENT -- probe endings in the PCA-2 plane with the discovered attractor centroids.
  4. DISSENT COUNT k over time, against the k>=2 alarm rule.

Everything in panels 2-4 comes from the learned model alone; panel 1 is what it was given.
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
from phase_c_train import Predictor, NUM_HIST, D                   # noqa: E402
from phase_f_monitor import build_centroids, chunk_margin, STRIDE  # noqa: E402

OUT = ROOT / "results" / "phase_c"
VID = ROOT / "results" / "phase_f_videos"
ACOL = ["#8e44ad", "#27ae60", "#d35400"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--settle", type=int, default=40)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--pca", type=int, default=8)
    ap.add_argument("--times", type=int, nargs="+", default=[3, 6, 9, 12, 15, 18, 21, 24])
    ap.add_argument("--n-videos", type=int, default=10)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    dev = "cuda"; VID.mkdir(parents=True, exist_ok=True); t0 = time.time()

    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    meta = np.load(OUT / "meta.npz"); A = meta["actions"]; S = meta["states"]
    p = np.load(OUT / "phase_c_eval.npz")
    mu, sd, amu, asd = (float(p[k]) for k in ("mu", "sd", "amu", "asd"))

    cache = np.load(OUT / "phase_e_endings_H20_n300.npz")
    E_all = cache[f"predictor_gtf_warm.pt_{args.settle}"].astype(np.float32)
    C, pmean, V, k_raw, k_keep, _, _ = build_centroids(E_all, args.pca)
    V2 = np.linalg.svd((E_all - E_all.mean(0))[::2], full_matrices=False)[2][:2]
    print(f"{k_keep} attractors (plateau {k_raw}, singletons dropped)", flush=True)

    model = Predictor().to(dev); model.load_state_dict(
        torch.load(OUT / "predictor_gtf_warm.pt")); model.eval()

    # theta readout, for turning a probe's ending latent back into a drawable tilt
    ntr = 120
    Xr = np.asarray(Z[:ntr], np.float32).reshape(ntr * Z.shape[1], -1)
    yr = S[:ntr, :, 0].reshape(-1); xm = Xr.mean(0); Xc = Xr - xm
    Vk = np.linalg.svd(Xc[::3], full_matrices=False)[2][:256]
    P_ = Xc @ Vk.T
    Wg = np.linalg.solve(P_.T @ P_ + 1e-2 * np.eye(256), P_.T @ (yr - yr.mean()))
    ymean = yr.mean()
    readout = lambda X: ((X - xm) @ Vk.T) @ Wg + ymean             # noqa: E731
    del Xr, Xc, P_

    rng = np.random.default_rng(0); thr = tb.topple_threshold(450, 10, 60)
    acts, lights = [], []
    for _ in range(Z.shape[0]):
        a = tb.random_push(rng, 450, thr); tb.simulate(a, dt=tb.DT_DEFAULT)
        acts.append(a); lights.append(br.sample_lighting(rng))

    rows = json.load(open(OUT / "phase_f_results.json"))["rows"]
    picks = []
    for want, cond in (("TP", lambda r: r["m_min"] < 0.23 and r["k_max"] >= 2),
                       ("FN", lambda r: r["m_min"] < 0.23 and r["k_max"] < 2),
                       ("TN", lambda r: r["m_min"] >= 0.23 and r["k_max"] < 2),
                       ("FP", lambda r: r["m_min"] >= 0.23 and r["k_max"] >= 2)):
        sel = sorted([r for r in rows if cond(r)], key=lambda r: r["m_min"])
        n = {"TP": 4, "FN": 2, "TN": 3, "FP": 1}[want]
        picks += [(want, r) for r in sel[:n]]
    picks = picks[:args.n_videos]
    print(f"rendering {len(picks)}: " + " ".join(f"{t}/ep{r['i']}" for t, r in picks), flush=True)

    eps_rng = np.random.default_rng(777)
    zero_pad = np.zeros((args.n_probe, args.settle))
    for tag, row in picks:
        i = row["i"]; ts = [t for t in args.times if t + args.H <= Z.shape[1]]
        st, _ = tb.simulate(acts[i], dt=tb.DT_DEFAULT)
        KS, LABS, THP, XP, MS = [], [], [], [], []
        for t in ts:
            chunk = A[i, t:t + args.H, 0].astype(np.float64)
            z = eps_rng.standard_normal((args.n_probe, 1))
            Pp = np.concatenate([chunk[None] * (1.0 + args.eps * z), zero_pad], axis=1)
            ac = torch.from_numpy(((Pp[:, :, None] - amu) / asd).astype(np.float32)).to(dev)
            ctx = np.asarray(Z[i, t - NUM_HIST + 1:t + 1], np.float32)
            win = ((torch.from_numpy(ctx).to(dev) - mu) / sd)[None].expand(
                args.n_probe, -1, -1, -1).contiguous()
            with torch.no_grad():
                for j in range(Pp.shape[1]):
                    aw = ac[:, max(0, j - NUM_HIST + 1):j + 1]
                    if aw.shape[1] < NUM_HIST:
                        aw = torch.cat([aw[:, :1].expand(-1, NUM_HIST - aw.shape[1], -1), aw], 1)
                    win = torch.cat([win[:, 1:], model(win, aw)[:, -1:]], 1)
            Ee = win[:, -1].reshape(args.n_probe, -1).float().cpu().numpy() * sd + mu
            X = (Ee - pmean) @ V.T
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            KS.append(int(args.n_probe - max((lab == j).sum() for j in range(len(C)))))
            LABS.append(lab); THP.append(readout(Ee)); XP.append((Ee - E_all.mean(0)) @ V2.T)
            MS.append(chunk_margin(st[t * STRIDE], acts[i][t * STRIDE:(t + args.H) * STRIDE],
                                   args.settle * STRIDE))
        KS = np.array(KS); MS = np.array(MS)

        bg = br.prepare(lights[i], seed=i)
        Xall = np.concatenate(XP); cen2 = (C @ np.linalg.pinv(V.T) - E_all.mean(0)) @ V2.T \
            if False else np.stack([Xall[np.concatenate(LABS) == j].mean(0)
                                    for j in range(len(C)) if (np.concatenate(LABS) == j).any()])

        fig = plt.figure(figsize=(15.5, 4.4))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.28)
        ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[c]) for c in range(4))
        ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_title("what the monitor sees", fontsize=10)
        im = ax0.imshow(br.render(st[ts[0] * STRIDE, 0], lights[i], bg=bg))

        ax1.set_xlim(-1.9, 1.9); ax1.set_ylim(-0.3, 2.3); ax1.set_aspect("equal")
        ax1.set_xticks([]); ax1.set_yticks([]); ax1.axhline(0, color="#2c3e50", lw=3)
        ax1.set_title("32 probes: predicted ENDING", fontsize=10)
        ghosts = [Polygon(br.corners(0.0), closed=True, fc="grey", ec="none", alpha=0.22)
                  for _ in range(args.n_probe)]
        for g in ghosts:
            ax1.add_patch(g)
        info = ax1.text(0.02, 0.97, "", transform=ax1.transAxes, va="top", fontsize=8,
                        family="monospace")

        ax2.set_title("probe endings in latent PCA", fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlim(Xall[:, 0].min() * 1.15, Xall[:, 0].max() * 1.15)
        ax2.set_ylim(Xall[:, 1].min() * 1.15, Xall[:, 1].max() * 1.15)
        ax2.scatter(cen2[:, 0], cen2[:, 1], marker="X", s=180, c="k", zorder=5)
        pts = ax2.scatter([], [], s=26)

        ax3.set_xlim(min(ts) * tb.DT_DEFAULT * STRIDE, max(ts) * tb.DT_DEFAULT * STRIDE)
        ax3.set_ylim(-0.5, max(4, KS.max() + 1))
        ax3.axhline(2, color="#e67e22", ls="--", lw=1.5)
        ax3.text(0.02, 2.05, " alarm: k >= 2", color="#e67e22", fontsize=8)
        ax3.set_xlabel("time (s)"); ax3.set_ylabel("dissent count k")
        ax3.set_title("monitor output", fontsize=10)
        kl, = ax3.plot([], [], "o-", lw=2, color="#c0392b")
        verdict = ax3.text(0.5, 0.92, "", transform=ax3.transAxes, ha="center",
                           fontsize=13, fontweight="bold")

        fig.suptitle(f"[{tag}] episode {i} — DINO-WM monitor — "
                     f"min chunk margin {row['m_min']:.1%},  peak k {row['k_max']}/32",
                     fontsize=12)

        def update(f):
            t = ts[f]
            im.set_data(br.render(st[t * STRIDE, 0], lights[i], bg=bg))
            for g, th_, l_ in zip(ghosts, THP[f], LABS[f]):
                g.set_xy(br.corners(float(np.clip(th_, -np.pi / 2, np.pi / 2))))
                g.set_facecolor(ACOL[l_ % 3])
            pts.set_offsets(XP[f]); pts.set_color([ACOL[l % 3] for l in LABS[f]])
            votes = "/".join(str(int((LABS[f] == j).sum())) for j in range(len(C)))
            info.set_text(f"t {t*tb.DT_DEFAULT*STRIDE:5.2f}s\nvotes {votes}\n"
                          f"dissent {KS[f]}\nchunk margin {MS[f]:.1%}")
            kl.set_data(np.array(ts[:f+1]) * tb.DT_DEFAULT * STRIDE, KS[:f+1])
            hot = KS[f] >= 2
            verdict.set_text("⚠ NEAR BOUNDARY" if hot else "clear")
            verdict.set_color("#c0392b" if hot else "#27ae60")
            return [im, *ghosts, pts, info, kl, verdict]

        anim = FuncAnimation(fig, update, frames=len(ts), interval=1000 / args.fps, blit=False)
        path = VID / f"ep{i:03d}_{tag}.mp4"
        anim.save(path, writer=FFMpegWriter(fps=args.fps, bitrate=2800))
        plt.close(fig)
        print(f"  -> {path.name}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"\n({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
