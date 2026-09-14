"""How short can the settle tail be? Tested on the toy, where the truth is exact.

The Jenga world model has NEVER seen a held pose -- measured over 6771 steps of demo data, the
longest run with <1 mm of EE motion is ONE step. So a 40-step zero-action tail is pure
extrapolation there. Before designing around that, find out how much tail the method actually
needs, on a system where we can check.

The label is always the FULLY SETTLED outcome (tail = 40, by which point 0% of blocks are still
mid-fall). That is the honest target: the monitor exists to flag proximity to the FINAL outcome, so
a short tail has to recover that, not merely describe the half-fallen state it can see.

tail = 0 means "read the latent at H with no settling at all" -- MONITOR.md warns this destroys the
attractor structure, and this measures whether that warning is real.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "src" / "models"), str(ROOT / "eval")]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_c_train import Predictor, NUM_HIST                      # noqa: E402
from phase_d_settle import basin, STRIDE                           # noqa: E402
from phase_e_attractors import pdist, plateau, agreement           # noqa: E402
from phase_f_monitor import build_centroids                        # noqa: E402

OUT = ROOT / "results" / "phase_c"
H, N = 20, 300


def main():
    t0 = time.time()
    cache = np.load(OUT / "phase_e_endings_H20_n300.npz")
    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    meta = np.load(OUT / "meta.npz"); A = meta["actions"]
    p = np.load(OUT / "phase_c_eval.npz")
    mu, sd, amu, asd = (float(p[k]) for k in ("mu", "sd", "amu", "asd"))
    idx = np.linspace(0, Z.shape[0] - 1, N).astype(int)

    rng = np.random.default_rng(0); thr = tb.topple_threshold(450, 10, 60)
    acts = []
    for _ in range(Z.shape[0]):
        a = tb.random_push(rng, 450, thr); tb.simulate(a, dt=tb.DT_DEFAULT)
        br.sample_lighting(rng); acts.append(a)

    # LABEL = the fully settled outcome, for every tail length
    truth = basin(np.array([tb.simulate(
        np.concatenate([acts[i][:H * STRIDE], np.zeros(40 * STRIDE)]),
        dt=tb.DT_DEFAULT)[0][-1, 0] for i in idx]))
    print(f"label (fully settled): {[int((truth==j).sum()) for j in range(3)]}"
          f"   majority baseline {max(np.bincount(truth))/len(truth):.1%}\n")

    ends = {s: cache[f"predictor_gtf_warm.pt_{s}"].astype(np.float32) for s in (10, 20, 40)}

    # tail = 0 is not cached: roll to H and stop
    print("rolling tail=0 (no settling at all) ...", flush=True)
    dev = "cuda"
    m = Predictor().to(dev); m.load_state_dict(torch.load(OUT / "predictor_gtf_warm.pt")); m.eval()
    E0 = []
    with torch.no_grad():
        for i in idx:
            win = ((torch.from_numpy(np.asarray(Z[i, :NUM_HIST], np.float32)).to(dev) - mu) / sd)[None]
            aa = (torch.from_numpy(A[i]).to(dev) - amu) / asd
            for t in range(NUM_HIST, H):
                win = torch.cat([win[:, 1:], m(win[:, -NUM_HIST:], aa[None, t-NUM_HIST:t])[:, -1:]], 1)
            E0.append(win[0, -1].reshape(-1).float().cpu().numpy() * sd + mu)
    ends[0] = np.stack(E0)
    print(f"  done ({time.time()-t0:.0f}s)\n", flush=True)

    print(f"{'tail':>6}{'sep ratio':>11}{'plateau k':>11}{'after singles':>15}"
          f"{'agreement':>11}   verdict")
    print("-" * 74)
    for s in (0, 10, 20, 40):
        E = ends[s]
        try:
            C, pmean, V, k_raw, k_keep, width, _ = build_centroids(E, 8)
        except ValueError as ex:
            print(f"{s:>6}{'--':>11}{'--':>11}{'--':>15}{'--':>11}   FRAGMENTED: {ex}")
            continue
        X = (E - pmean) @ V.T
        D = pdist(X); scale = float(np.linalg.norm(X - X.mean(0), axis=1).mean())
        iu = np.triu_indices(len(X), 1); pw = D[iu] / scale
        same = truth[iu[0]] == truth[iu[1]]
        sep = np.median(pw[~same]) / np.median(pw[same])
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        ag = agreement(lab, truth)
        ok = "OK" if (k_keep == 3 and ag > 0.80) else ("count wrong" if k_keep != 3 else "weak")
        print(f"{s:>6}{sep:>11.3f}{k_raw:>11}{k_keep:>15}{ag:>11.1%}   {ok}")
    print("\nlabel is the FULLY SETTLED outcome in every row, so a short tail has to")
    print("RECOVER the final structure, not just describe what it can see.")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
