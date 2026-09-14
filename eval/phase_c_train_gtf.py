"""Phase C, retrained with a ROLLOUT-AWARE loss. The first attempt failed for a fixable reason.

Attempt 1: single-step, fully teacher-forced training (num_pred=1, copying dino_wm's config),
evaluated on a 42-step autoregressive rollout. One-step error was 0.040 rad -- at the theta
readout's own floor -- but it compounded to 0.577 by step 40, giving RMSE 0.320 against a 0.265
bar and only 75% basin agreement. The model had never seen its own output as input.

The shPLRNN it is being compared against did NOT have that handicap: it trained with
`gtf_rollout_loss_latent` over 40-step windows at alpha=0.02, i.e. nearly free-running. So the
comparison was unfair AND the remedy is known.

GTF blends the model's own prediction with the truth at every step:

    z_t  <-  alpha * z_true  +  (1 - alpha) * z_pred

alpha=0 is free-running (gradients explode over long rollouts), alpha=1 is pure teacher forcing
(what failed). gtf.py's heuristic is alpha ~ 1 - exp(-lambda_max * dt). At stride 10 dt_eff is
0.2 s with lambda_max ~ 1.0, giving **alpha ~ 0.18** -- an order of magnitude above the 0.02 that
was right at stride 1, because each frame now spans ten times as much time.

NOTE FOR JENGA: conf/train.yaml carries `num_pred: 1 # only supports 1`, so the real world model
has this same single-step recipe. Fine for short-horizon CEM planning, but it predicts the same
compounding failure when a settle tail demands a long rollout.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "src" / "models"),
                str(ROOT / "eval")]
from phase_c_train import Predictor, NUM_HIST, D                   # noqa: E402

OUT = ROOT / "results" / "phase_c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.18, help="GTF blend; 1 = teacher forcing")
    ap.add_argument("--roll", type=int, default=8, help="rollout length trained through")
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--warm", type=int, default=NUM_HIST)
    ap.add_argument("--tag", default="gtf")
    ap.add_argument("--init", default="predictor.pt",
                    help="warm-start checkpoint; '' trains from scratch")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    Z = np.load(OUT / "latents.npy", mmap_mode="r")
    meta = np.load(OUT / "meta.npz"); S, A = meta["states"], meta["actions"]
    N, T = Z.shape[:2]; n_tr = N - args.n_test
    prev = np.load(OUT / "phase_c_eval.npz")
    mu, sd, amu, asd = (float(prev[k]) for k in ("mu", "sd", "amu", "asd"))
    print(f"{N} eps x {T} frames; train {n_tr}/test {args.n_test}; "
          f"alpha={args.alpha} roll={args.roll}", flush=True)

    model = Predictor().to(dev)
    # Warm-start from the teacher-forced checkpoint. Training GTF from scratch collapsed to the
    # do-nothing trajectory (predicted basins [1,59,0] against true [15,28,17]): with only 8k
    # windows it had not learned the dynamics before being asked to survive its own errors, and
    # MSE over a rollout regresses to the conditional mean -- which near a bifurcation sits
    # BETWEEN the basins, i.e. upright. Teacher forcing first, rollout fine-tuning second.
    if args.init and (OUT / args.init).exists():
        model.load_state_dict(torch.load(OUT / args.init))
        print(f"warm-started from {args.init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    rng = np.random.default_rng(0)
    W = NUM_HIST + args.roll

    print("\ntraining (GTF rollout) ...", flush=True)
    model.train()
    for step in range(args.steps):
        ti = rng.integers(0, n_tr, args.batch)
        t0_ = rng.integers(0, T - W, args.batch)
        z = np.stack([Z[a, b:b + W] for a, b in zip(ti, t0_)]).astype(np.float32)
        a = np.stack([A[a, b:b + W] for a, b in zip(ti, t0_)])
        zt = (torch.from_numpy(z).to(dev) - mu) / sd
        at = (torch.from_numpy(a).to(dev) - amu) / asd

        ctx = zt[:, :NUM_HIST]                          # warm start from truth
        loss = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            for k in range(args.roll):
                # Gradient checkpointing: backprop through an unrolled 10-step ViT holds every
                # layer's activations for every step and OOMs a 7.5 GiB card. Recomputing the
                # forward during backward trades ~2x compute for roll-x memory, which is what
                # makes the rollout length a modelling choice rather than a VRAM budget.
                nxt = torch.utils.checkpoint.checkpoint(
                    model, ctx[:, -NUM_HIST:], at[:, k:k + NUM_HIST],
                    use_reentrant=False)[:, -1:]
                tgt = zt[:, NUM_HIST + k:NUM_HIST + k + 1]
                loss = loss + ((nxt.float() - tgt) ** 2).mean()
                blend = args.alpha * tgt + (1.0 - args.alpha) * nxt     # <-- GTF
                ctx = torch.cat([ctx, blend], 1)
            loss = loss / args.roll
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  loss {float(loss):.5f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save(model.state_dict(), OUT / f"predictor_{args.tag}.pt")

    print("\nfitting theta readout ...", flush=True)
    ntr_r = 120
    Xr = np.asarray(Z[:ntr_r], np.float32).reshape(ntr_r * T, -1)
    yr = S[:ntr_r, :, 0].reshape(-1)
    xm = Xr.mean(0); Xc = Xr - xm
    Vk = np.linalg.svd(Xc[::3], full_matrices=False)[2][:256]
    P = Xc @ Vk.T
    Wg = np.linalg.solve(P.T @ P + 1e-2 * np.eye(256), P.T @ (yr - yr.mean()))
    readout = lambda X: ((X - xm) @ Vk.T) @ Wg + yr.mean()          # noqa: E731
    del Xr, Xc, P

    print("rolling out held-out episodes ...", flush=True)
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(n_tr, N):
            z = ((torch.from_numpy(np.asarray(Z[i, :args.warm], np.float32)).to(dev) - mu)
                 / sd)[None]
            aa = (torch.from_numpy(A[i]).to(dev) - amu) / asd
            preds = []
            for t in range(args.warm, T):
                nxt = model(z[:, -NUM_HIST:], aa[None, t - NUM_HIST:t])[:, -1:]
                preds.append(nxt); z = torch.cat([z, nxt], 1)
            Zp = torch.cat(preds, 1)[0].float().cpu().numpy() * sd + mu
            errs.append(readout(Zp.reshape(len(Zp), -1)) - S[i, args.warm:, 0])
    E = np.stack(errs)
    rmse = float(np.sqrt((E ** 2).mean()))
    th_true = S[n_tr:, args.warm:, 0]; th_pred = E + th_true
    basin = lambda t: np.where(t < -1.4, 0, np.where(t > 1.4, 2, 1))    # noqa: E731
    bt, bp = basin(th_true[:, -1]), basin(th_pred[:, -1])
    agree = float((bt == bp).mean())

    print(f"\n  theta RMSE  {rmse:.4f} rad   (was 0.3201 single-step; bar 0.265)")
    print(f"  basin agree {agree:.1%}        (was 75.0%; shPLRNN 87.0%)")
    print(f"  true basins {[int((bt==j).sum()) for j in range(3)]}  "
          f"pred {[int((bp==j).sum()) for j in range(3)]}")
    print("  RMSE by step: " + " ".join(f"{np.sqrt((E[:, j]**2).mean()):.3f}"
                                        for j in range(0, E.shape[1], max(1, E.shape[1]//8))))
    print(f"\nACCEPTANCE (<= 0.265): {'PASS' if rmse <= 0.265 else 'FAIL'}")
    np.savez(OUT / f"phase_c_eval_{args.tag}.npz", rmse=rmse, err=E, agree=agree,
             mu=mu, sd=sd, amu=amu, asd=asd, alpha=args.alpha, roll=args.roll)
    print(f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
