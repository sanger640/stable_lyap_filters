"""Phase D, part 2: is the settle tail CONTRACTIVE? The proper test.

My first attempt compared basin agreement before vs after the tail and called the drop
"not contractive". That was invalid: at H the true basins are [2,55,3] -- 92% simply upright, so
guessing "upright" scores 91.7% -- while after the tail they are [11,35,14]. The tail is where the
outcome is decided, so agreement was always going to fall. The comparison measured task difficulty,
not contraction.

Contraction has a definition: do two nearby trajectories converge? So compare the MODEL's rollout
against the ENCODED TRUTH through the same tail,

    d(t) = || z_model(t) - z_true(t) ||

and ask whether d shrinks (the tail absorbs error, so Phase C's tracking error is survivable) or
grows (it amplifies, and no amount of tracking accuracy would have helped).

The ground-truth zero-action continuation is not in the corpus -- that rolled the TRUE action for
all 45 frames -- so it has to be rendered and encoded here.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "src" / "models"),
                str(ROOT / "eval")]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_c_train import Predictor, NUM_HIST                      # noqa: E402
from phase_b_probe import encode                                   # noqa: E402
from phase_d_settle import true_actions, STRIDE                    # noqa: E402

OUT = ROOT / "results" / "phase_c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=60)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    meta = np.load(OUT / "meta.npz"); A = meta["actions"]
    N, T = Z.shape[:2]; n_tr = N - args.n_test
    p = np.load(OUT / "phase_c_eval.npz")
    mu, sd, amu, asd = (float(p[k]) for k in ("mu", "sd", "amu", "asd"))

    # regenerate actions AND lighting in the data script's order, so renders match the corpus
    rng = np.random.default_rng(0)
    thr = tb.topple_threshold(450, 10, 60)
    acts, lights = [], []
    for _ in range(N):
        a = tb.random_push(rng, 450, thr); tb.simulate(a, dt=tb.DT_DEFAULT)
        acts.append(a); lights.append(br.sample_lighting(rng))

    print("rendering + encoding the TRUE zero-action continuation ...", flush=True)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(dev).eval()
    Ztrue = np.zeros((args.n_test, args.L, 256 * 384), np.float32)
    for k, i in enumerate(range(n_tr, N)):
        a = np.concatenate([acts[i][:args.H * STRIDE], np.zeros(args.L * STRIDE)])
        st, _ = tb.simulate(a, dt=tb.DT_DEFAULT)
        idx = args.H * STRIDE + (np.arange(args.L) + 1) * STRIDE
        bg = br.prepare(lights[i], seed=i)
        fr = np.stack([br.render(th, lights[i], bg=bg) for th in st[idx, 0]])
        Ztrue[k] = encode(fr, dino, dev).reshape(args.L, -1)
    print(f"  {args.n_test*args.L} frames ({time.time()-t0:.0f}s)\n", flush=True)

    scale = float(np.linalg.norm(Ztrue[:, -1] - Ztrue[:, -1].mean(0), axis=1).mean())
    zero_a = (0.0 - amu) / asd

    for tag, ckpt in (("single-step (dino_wm recipe)", "predictor.pt"),
                      ("GTF warm-started", "predictor_gtf_warm.pt")):
        if not (OUT / ckpt).exists():
            continue
        model = Predictor().to(dev); model.load_state_dict(torch.load(OUT / ckpt)); model.eval()
        D = np.zeros((args.n_test, args.L))
        with torch.no_grad():
            for k, i in enumerate(range(n_tr, N)):
                z = ((torch.from_numpy(np.asarray(Z[i, :NUM_HIST], np.float32)).to(dev)
                      - mu) / sd)[None]
                aa = (torch.from_numpy(A[i]).to(dev) - amu) / asd
                for t in range(NUM_HIST, args.H):
                    z = torch.cat([z, model(z[:, -NUM_HIST:], aa[None, t-NUM_HIST:t])[:, -1:]], 1)
                az = torch.full((1, NUM_HIST, 1), zero_a, device=dev, dtype=z.dtype)
                for j in range(args.L):
                    nxt = model(z[:, -NUM_HIST:], az)[:, -1:]
                    z = torch.cat([z, nxt], 1)
                    zp = nxt[0, 0].float().cpu().numpy().ravel() * sd + mu
                    D[k, j] = np.linalg.norm(zp - Ztrue[k, j])
        print(f"{'='*64}\n{tag}\n{'='*64}")
        print("  || z_model(t) - z_true(t) ||  as % of ending scale")
        js = [0, 1, 2, 4, 7, 11, 19, 29, args.L - 1]
        print("   step: " + "".join(f"{j:>8}" for j in js))
        print("   med : " + "".join(f"{100*np.median(D[:, j])/scale:>8.1f}" for j in js))
        r = D[:, -1] / np.maximum(D[:, 0], 1e-9)
        print(f"   ratio d_end/d_start: median {np.median(r):.3f}   "
              f"contractive on {100*(r<1).mean():.0f}% of episodes")
        print(f"   -> {'CONTRACTIVE' if np.median(r) < 1 else 'EXPANSIVE'}"
              f" (median ratio {'<' if np.median(r)<1 else '>'} 1)\n")
        np.savez(OUT / f"phase_d_contract_{ckpt.replace('.pt','')}.npz", D=D, scale=scale)
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
