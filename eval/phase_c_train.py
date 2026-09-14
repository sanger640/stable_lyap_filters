"""Phase C: train the DINO-WM predictor on the block, and check rollout fidelity.

THE ONLY TRAINING STEP IN THE PLAN. DINOv2 stays frozen; the ViT learns one thing -- given the last
`num_hist` latents and an action, what is the next latent.

Architecture is dino_wm's, imported rather than reimplemented (PLAN_DINOWM: the point is to
de-risk the Jenga code path, and a clean-room version would test something else):

    models.vit.ViTPredictor   depth 6, heads 16, mlp_dim 2048, dropout 0.1   [conf/predictor]
    concat_dim = 1, num_action_repeat = 1, num_hist = 3, num_pred = 1        [conf/train.yaml]
    token = [visual_384 | action_emb_10]  -> dim 394, tiled across patches

Two deliberate deviations, both noted rather than silent:
  * No proprio. The toy has no robot, and feeding theta would hand the model the state -- exactly
    the shortcut the image swap exists to remove.
  * The loss covers the VISUAL dims only. dino_wm includes the action dims, but the rollout
    overwrites them with the true next action anyway (`replace_actions_from_z`), so gradient there
    is noise.

Latents are scaled by a single GLOBAL scalar, not per-dimension. Per-dim standardisation would
distort Euclidean geometry, and Phase E clusters by `argmin ||E - C||^2` -- the geometry has to
survive intact.

ACCEPTANCE: theta RMSE <= 0.265 rad over an autoregressive rollout on held-out episodes -- parity
with the shPLRNN, which sees the exact state. Not superiority; it should lose. The question is
whether it is close enough to carry the monitor.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "src" / "models"),
                str(ROOT / "eval")]
from dinowm_vit import ViTPredictor        # vendored; see src/models/                                # noqa: E402

OUT = ROOT / "results" / "phase_c"
NP, D, ACT_EMB, NUM_HIST = 256, 384, 10, 3


class Predictor(nn.Module):
    """dino_wm's ViTPredictor plus the action embedding, at concat_dim=1."""

    def __init__(self, depth=6, heads=16, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.act_emb = nn.Sequential(nn.Linear(1, ACT_EMB), nn.GELU(),
                                     nn.Linear(ACT_EMB, ACT_EMB))
        self.vit = ViTPredictor(num_patches=NP, num_frames=NUM_HIST, dim=D + ACT_EMB,
                                depth=depth, heads=heads, mlp_dim=mlp_dim,
                                pool="mean", dropout=dropout)

    def tokens(self, z_vis, act):
        """(b,t,NP,D) + (b,t,1) -> (b,t,NP,D+ACT_EMB)"""
        a = self.act_emb(act)[:, :, None].expand(-1, -1, z_vis.shape[2], -1)
        return torch.cat([z_vis, a], dim=-1)

    def forward(self, z_vis, act):
        x = self.tokens(z_vis, act)
        b, t, p, d = x.shape
        y = self.vit(x.reshape(b, t * p, d)).reshape(b, t, p, d)
        return y[..., :D]                       # predicted VISUAL latents only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--warm", type=int, default=NUM_HIST)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    meta = np.load(OUT / "meta.npz")
    S, A = meta["states"], meta["actions"]
    N, T = Z.shape[:2]
    n_tr = N - args.n_test
    print(f"{N} episodes x {T} frames; train {n_tr} / test {args.n_test}", flush=True)

    sub = np.asarray(Z[:40], np.float32)
    mu, sd = float(sub.mean()), float(sub.std())        # GLOBAL scalars: geometry preserved
    amu, asd = float(A[:n_tr].mean()), float(A[:n_tr].std())
    print(f"latent scale mu {mu:.4f} sd {sd:.4f}", flush=True)
    del sub

    model = Predictor().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    print(f"predictor {n_par/1e6:.1f}M params", flush=True)

    rng = np.random.default_rng(0)

    def batch(n, lo, hi):
        ti = rng.integers(lo, hi, n)
        t0_ = rng.integers(0, T - NUM_HIST, n)          # need NUM_HIST+1 frames
        z = np.stack([Z[a, b:b + NUM_HIST + 1] for a, b in zip(ti, t0_)]).astype(np.float32)
        a = np.stack([A[a, b:b + NUM_HIST + 1] for a, b in zip(ti, t0_)])
        z = (torch.from_numpy(z).to(dev) - mu) / sd
        a = (torch.from_numpy(a).to(dev) - amu) / asd
        return z, a

    print("\ntraining ...", flush=True)
    model.train()
    for step in range(args.steps):
        z, a = batch(args.batch, 0, n_tr)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred = model(z[:, :NUM_HIST], a[:, :NUM_HIST])
            loss = ((pred.float() - z[:, 1:]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % max(1, args.steps // 12) == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  loss {float(loss):.5f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save(model.state_dict(), OUT / "predictor.pt")

    # ---- theta readout: ridge on TRUE train latents, then applied to PREDICTED latents
    print("\nfitting theta readout on true train latents ...", flush=True)
    ntr_r = 120
    Xr = np.asarray(Z[:ntr_r], np.float32).reshape(ntr_r * T, -1)
    yr = S[:ntr_r, :, 0].reshape(-1)
    xm = Xr.mean(0); Xc = Xr - xm
    U, sv, Vt = np.linalg.svd(Xc[::3], full_matrices=False)
    k = 256
    Vk = Vt[:k]
    P = Xc @ Vk.T
    W = np.linalg.solve(P.T @ P + 1e-2 * np.eye(k), P.T @ (yr - yr.mean()))
    readout = lambda X: ((X - xm) @ Vk.T) @ W + yr.mean()           # noqa: E731
    fit_rmse = float(np.sqrt(((readout(Xr) - yr) ** 2).mean()))
    print(f"  readout fit RMSE on true latents: {fit_rmse:.4f} rad", flush=True)
    del Xr, Xc, P, U

    # ---- autoregressive rollout on held-out episodes
    print("\nrolling out held-out episodes ...", flush=True)
    model.eval()
    errs, per_step = [], []
    with torch.no_grad():
        for i in range(n_tr, N):
            z = (torch.from_numpy(np.asarray(Z[i, :args.warm], np.float32)).to(dev) - mu) / sd
            z = z[None]
            a_all = (torch.from_numpy(A[i]).to(dev) - amu) / asd
            preds = []
            for t in range(args.warm, T):
                nxt = model(z[:, -NUM_HIST:], a_all[None, t - NUM_HIST:t])[:, -1:]
                preds.append(nxt)
                z = torch.cat([z, nxt], 1)
            Zp = (torch.cat(preds, 1)[0].float().cpu().numpy() * sd + mu)
            th_pred = readout(Zp.reshape(len(Zp), -1))
            th_true = S[i, args.warm:, 0]
            errs.append(th_pred - th_true)
            per_step.append((th_pred - th_true) ** 2)
    E = np.concatenate(errs); PS = np.stack(per_step)
    rmse = float(np.sqrt((E ** 2).mean()))
    print(f"\n  theta RMSE over {args.n_test} held-out episodes, "
          f"{T-args.warm} autoregressive steps: {rmse:.4f} rad")
    print(f"  finite: {np.isfinite(E).all()}   max |err| {np.abs(E).max():.3f}")
    print("  RMSE by rollout step: " +
          " ".join(f"{np.sqrt(PS[:, j].mean()):.3f}"
                   for j in range(0, PS.shape[1], max(1, PS.shape[1] // 8))))
    print(f"\nACCEPTANCE (<= 0.265 rad, shPLRNN parity): "
          f"{'PASS' if rmse <= 0.265 else 'FAIL'}")
    np.savez(OUT / "phase_c_eval.npz", rmse=rmse, err=E, per_step=PS,
             mu=mu, sd=sd, amu=amu, asd=asd)
    print(f"({(time.time()-t0)/60:.1f} min total)")


if __name__ == "__main__":
    main()
