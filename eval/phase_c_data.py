"""Phase C data: render + DINOv2-encode the 600-trajectory corpus at stride 10.

Stride 10 comes from Phase B: omega is recoverable at R^2 0.955, it is the cheapest option to
render, and it shrinks rollout+settle from 350 steps to 35 -- which is what makes Phase D (does an
autoregressive ViT settle?) a reasonable question to ask at all.

Lighting stays VARIED, per episode. That is the whole point: a linear probe could not handle it
from 32 episodes, and Phase C is the test of whether a nonlinear predictor can from 600. If
fidelity misses the bar, narrowing the lighting range is the first knob to turn.

Actions are averaged over the 10 sim steps each frame spans, so one action accompanies one frame.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval")]
import tipping_block as tb                                         # noqa: E402
import block_render as br                                          # noqa: E402
from phase_b_probe import encode                                   # noqa: E402
from run_phase3_monitor import T_ON, T_OFF                         # noqa: E402

OUT = ROOT / "results" / "phase_c"
OUT.mkdir(parents=True, exist_ok=True)
N_TRAJ, STEPS, STRIDE = 600, 450, 10
T = STEPS // STRIDE                                                 # 45 frames per episode
NP, D = 256, 384

if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(0)
    thr = tb.topple_threshold(STEPS, T_ON, T_OFF)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           verbose=False).to(dev).eval()

    Z = np.lib.format.open_memmap(OUT / "latents.npy", mode="w+",
                                  dtype=np.float16, shape=(N_TRAJ, T, NP, D))
    S = np.zeros((N_TRAJ, T, 2), np.float32)                        # (theta, omega) ground truth
    A = np.zeros((N_TRAJ, T, 1), np.float32)                        # mean force over the 10 steps
    print(f"{N_TRAJ} episodes x {T} frames at stride {STRIDE}", flush=True)

    for i in range(N_TRAJ):
        act = tb.random_push(rng, STEPS, thr)
        st, _ = tb.simulate(act, dt=tb.DT_DEFAULT)
        light = br.sample_lighting(rng)
        bg = br.prepare(light, seed=i)
        idx = np.arange(T) * STRIDE
        frames = np.stack([br.render(th, light, bg=bg) for th in st[idx, 0]])
        Z[i] = encode(frames, model, dev).astype(np.float16)
        S[i] = st[idx]
        A[i] = act[:T * STRIDE].reshape(T, STRIDE).mean(1)[:, None]
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{N_TRAJ}  {el/60:.1f} min  (eta {el/(i+1)*(N_TRAJ-i-1)/60:.1f} min)",
                  flush=True)

    Z.flush()
    np.savez(OUT / "meta.npz", states=S, actions=A, threshold=thr, stride=STRIDE)
    print(f"\n-> {OUT/'latents.npy'}  {Z.nbytes/1e9:.1f} GB")
    print(f"   theta std {S[...,0].std():.3f}  omega std {S[...,1].std():.3f}  "
          f"topple rate {(np.abs(S[:,-1,0])>1.4).mean():.1%}")
    print(f"({time.time()-t0:.0f}s)")
