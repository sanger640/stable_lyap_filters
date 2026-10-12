"""Phase 2 amendment 9: set-level energy-distance objective on deterministic rollouts from Gauss-TE starts.

Frozen protocol: results/jenga/phase2_generality_protocol.json (amendment_9). Arms C0 (lambda 0), SL_low, SL_high,
each fine-tuning US2_long (seed 0) for 7,500 steps.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_models as pm  # noqa: E402

OUT = pm.OUT
RESULT = ROOT / "results/jenga/phase2_setloss_result.json"
S_STATES, M_FUT, STEPS, LR = 8, 8, 7_500, 1.5e-4
RATIOS = {"C0": 0.0, "SL_low": 0.5, "SL_high": 5.0}


def energy_distance(x, y):
    """x, y (S, M, D) -> mean over S of energy distance / sqrt(D)."""
    import torch
    m = x.shape[1]
    off = 1.0 - torch.eye(m, device=x.device)
    xy = torch.cdist(x, y).mean((1, 2))
    xx = (torch.cdist(x, x) * off).sum((1, 2)) / (m * (m - 1))
    yy = (torch.cdist(y, y) * off).sum((1, 2)) / (m * (m - 1))
    return ((xy - 0.5 * xx - 0.5 * yy) / x.shape[-1] ** 0.5).mean()


def setup(seed=0):
    import torch
    import fast_train as ft
    import step3_world_model as wm
    states, names = pm.load_train()
    te = pm.GaussTE(torch.load(OUT / "Gauss_TE_phase2.pt"))
    modes = json.loads((OUT / "modes.json").read_text())
    rare = np.array([k in set(modes[n]) for n, s in zip(names, states) for k in range(len(s["z"]))])
    has_rare_f = np.array([bool(modes[n]) for n, s in zip(names, states) for _ in range(len(s["z"]))])
    pools = (np.flatnonzero(rare), np.flatnonzero(~rare & has_rare_f))
    state_rare = np.flatnonzero([bool(modes[n]) for n in names])
    first = np.cumsum([0] + [len(s["z"]) for s in states])                 # flat index of each state's first future
    counts = np.diff(first)
    data = ft.Packed(states)
    del states
    return te, pools, state_rare, first, counts, data


def rollout(model, start, act, keep_grad):
    """Deterministic 19-step rollout with checkpointing; returns latents at HOLD10 and HOLD30 (flattened)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    import step3_world_model as wm
    zeros_tau = torch.zeros(len(start), device="cuda")
    net = model.net

    def step_fn(inp, a):
        return net(inp, torch.zeros_like(inp), a, zeros_tau)
    cur, ends = start, {}
    for t in range(wm.STEPS):
        r = checkpoint(step_fn, cur, act[:, t], use_reentrant=False) if keep_grad else step_fn(cur, act[:, t])
        cur = cur + r.float() * model.stats.res_std
        if t in (wm.HOLD10, wm.HOLD30):
            ends[t] = cur.reshape(len(cur), -1)
    return ends[wm.HOLD10], ends[wm.HOLD30]


def train(arm, seed=0, log_every=250):
    import torch
    from torch import nn
    import fast_train as ft
    import step3_world_model as wm
    import step3b_models as sm
    path = OUT / f"model_{arm}.pt"
    if path.exists():
        return
    te, pools, state_rare, first, counts, data = setup(seed)
    model = wm.WorldModel.load(OUT / "model_US2_long.pt")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    n_states = len(counts)

    def set_batch():
        si = np.concatenate([rng.choice(state_rare, S_STATES // 2, replace=False),
                             rng.choice(n_states, S_STATES - S_STATES // 2, replace=False)])
        fut = np.concatenate([first[s] + rng.choice(counts[s], M_FUT, replace=counts[s] < M_FUT) for s in si])
        z0, _, dd, target, act = data.batch(torch.as_tensor(fut, device="cuda"))
        return z0, target, model._act(act)

    def main_batch():
        idx = torch.as_tensor(np.concatenate([rng.choice(pools[0], ft.BATCH // 2),
                                              rng.choice(pools[1], ft.BATCH - ft.BATCH // 2)]), device="cuda")
        z0, rend, dd, target, act = data.batch(idx)
        return z0, rend, target, model._act(act)
    wts = torch.ones(wm.STEPS, device="cuda"); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT

    def per_future_loss(z0, rend, target, act, keep_grad=True):
        from torch.utils.checkpoint import checkpoint
        use_g = torch.rand(len(z0), device="cuda") < 0.5
        with torch.no_grad():
            start = torch.where(use_g[:, None, None, None], te(z0), rend)
        zeros_tau = torch.zeros(len(z0), device="cuda")

        def step_fn(inp, a):
            return model.net(inp, torch.zeros_like(inp), a, zeros_tau)
        cur, loss = start, 0.0
        for t in range(wm.STEPS):
            inp = cur + torch.randn_like(cur) * sm.INPUT_NOISE * model.stats.res_std
            r = checkpoint(step_fn, inp, act[:, t], use_reentrant=False) if keep_grad else step_fn(inp, act[:, t])
            cur = cur + r.float() * model.stats.res_std
            loss = loss + wts[t] * ((cur - target[:, t]) ** 2).mean()
        return loss / wts.sum()

    def set_term(keep_grad=True):
        z0, target, act = set_batch()
        with torch.no_grad():
            start = te(z0)
        x10, x30 = rollout(model, start, act, keep_grad)
        y10 = target[:, wm.HOLD10].reshape(len(z0), -1); y30 = target[:, wm.HOLD30].reshape(len(z0), -1)
        sh = (S_STATES, M_FUT, -1)
        return 0.5 * (energy_distance(x10.reshape(sh), y10.reshape(sh)) + energy_distance(x30.reshape(sh), y30.reshape(sh)))
    # calibration (before any update)
    model.net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pf0 = float(np.mean([float(per_future_loss(*main_batch(), keep_grad=False)) for _ in range(20)]))
        st0 = float(np.mean([float(set_term(keep_grad=False)) for _ in range(20)]))
    lam = RATIOS[arm] * pf0 / st0
    log = [{"arm": arm, "steps": STEPS, "lr": LR, "initial_per_future_loss": pf0, "initial_set_term": st0, "lambda": lam}]
    print(json.dumps(log[0]), flush=True)
    opt = torch.optim.AdamW(model.net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    t0 = time.time()
    for step in range(STEPS):
        model.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pf = per_future_loss(*main_batch())
            st = set_term() if lam > 0 else torch.zeros((), device="cuda")
            total = pf + lam * st
        opt.zero_grad(set_to_none=True); total.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % log_every == 0:
            if lam == 0:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    st = set_term(keep_grad=False)
            log.append({"step": step + 1, "loss": float(pf), "set": float(st), "seconds": round(time.time() - t0, 1)})
            print(f"{arm} step {step + 1}: per-future {float(pf):.5f} set {float(st):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.net.eval()
    model.save(path)
    (OUT / f"model_{arm}_log.json").write_text(json.dumps(log) + "\n")
    print("trained", arm, flush=True)


def grade():
    import torch
    import step3_world_model as wm
    import step3b_models as sm
    from phase2_amend4 import alarm, score
    te = pm.GaussTE(torch.load(OUT / "Gauss_TE_phase2.pt"))
    arms = [a for a in ("C0", "SL_low", "SL_high") if (OUT / f"model_{a}.pt").exists()]
    models = {a: wm.WorldModel.load(OUT / f"model_{a}.pt") for a in arms + ["US2_long"]}
    files = sorted((OUT / "h2").glob("*.npz"))
    fork, fam, ends, errs, spread = [], [], {a: [] for a in models}, {a: [] for a in models}, {a: [] for a in models}
    with torch.no_grad():
        for n_state, path in enumerate(files):
            with np.load(path) as f:
                k = int(f["harm"].sum())
                z0 = torch.as_tensor(f["z0"], device="cuda").float()
                g = torch.Generator(device="cuda").manual_seed(pm.SEEDS["monitor"] + n_state)
                starts = te(z0[None].expand(64, -1, -1, -1).contiguous(), g)
                acts = np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, 8)
                real30 = f["z"][:, 1].reshape(64, -1).astype(np.float64)
            fork.append(2 <= k <= 62); fam.append(path.stem.split("_")[0])
            for a, m in models.items():
                e10, e30 = sm.roll(m, starts, acts)
                ends[a].append((e10, e30))
                errs[a].extend(np.linalg.norm(e30 - real30, axis=1) / np.sqrt(real30.shape[1] // 3))
                if 2 <= k <= 62:
                    spread[a].append(float(e30.std(0).sum() / real30.std(0).sum()))
            torch.cuda.empty_cache()
    fork, fam = np.array(fork), np.array(fam)
    out = {"readout": "BOTH"}
    for a in models:
        tau = float(np.mean(errs[a]))
        al = np.array([alarm(e10, e30, "BOTH", tau) for e10, e30 in ends[a]])
        out[a] = {"tau": tau, "all": score(al, fork), "F1": score(al[fam == "F1"], fork[fam == "F1"]),
                  "F2": score(al[fam == "F2"], fork[fam == "F2"]), "fork_spread_ratio_median": float(np.median(spread[a]))}
        log = OUT / f"model_{a}_log.json"
        if log.exists() and a != "US2_long":
            out[a]["calibration"] = json.loads(log.read_text())[0]
    sl = [a for a in ("SL_low", "SL_high") if a in out]
    if "C0" in out and sl:
        best = max(sl, key=lambda a: out[a]["all"]["score"])
        out["best_set_arm"] = best
        out["set_loss_helps"] = bool(out[best]["all"]["score"] >= out["C0"]["all"]["score"] + 0.10)
        out["passes_in_family_bar_on_H2"] = bool(out[best]["all"]["caught"] >= 0.80 and out[best]["all"]["false_alarm"] <= 0.20)
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "grade"))
    parser.add_argument("--arm", choices=tuple(RATIOS))
    args = parser.parse_args()
    if not json.loads((ROOT / "results/jenga/phase2_generality_protocol.json").read_text())["amendment_9"]["status"].startswith("frozen"):
        raise SystemExit("amendment 9 not frozen")
    train(args.arm) if args.stage == "train" else grade()


if __name__ == "__main__":
    main()
