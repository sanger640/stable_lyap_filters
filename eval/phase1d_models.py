"""Branching-WM Phase 1d: start proprioception (P), training rebalanced toward toppling futures (R), both (PR).

Frozen protocol: results/jenga/phase1d_protocol.json. Base = cell E (eval/phase1b_models.train_ic recipe).
Deployable inputs only: start proprioception (gripper xyz, closed flag) and the commanded chunk; never the robot's
actual future proprioception.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase1_models as p1  # noqa: E402
import phase1b_models as p1b  # noqa: E402
import step3_world_model as wm  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase1d_protocol.json"
OUT = ROOT / "results/program_b/phase1d"
RESULT = ROOT / "results/jenga/phase1d_result.json"
DEVICE, BATCH, STEPS = "cuda", 32, 30_000
ARMS = {"P": (True, False), "R": (False, True), "PR": (True, True)}
LEAN_DEG = 10.0


def augment(actions, start, proprio):
    """actions (K, 38, 4) -> per model step (K, 19, 8) or, with proprio, (K, 19, 18)."""
    a = np.asarray(actions, np.float32).reshape(len(actions), wm.STEPS, 2, 4)
    if not proprio:
        return a.reshape(len(actions), wm.STEPS, 8)
    g = np.asarray(start, np.float32)[-4:]                                     # gripper xyz, closed flag
    rel = (a[..., :3] - g[:3]).reshape(len(actions), wm.STEPS, 6)
    return np.concatenate([a.reshape(len(actions), wm.STEPS, 8),
                           np.broadcast_to(g, (len(actions), wm.STEPS, 4)), rel], -1).astype(np.float32)


def topples(traces, tilt):
    from fresh_fragility_panel import topple_count
    return np.array([topple_count(t[None], tilt) > 0 for t in traces])


def load(folders, proprio):
    states = []
    for folder in folders:
        for path in sorted(Path(folder).glob("ep*.npz")):
            if path.name.endswith(".tmp.npz"):
                continue
            start = np.load(path.with_name(path.stem + ".start.npy"))
            with np.load(path) as f:
                n = len(f["z"])
                states.append({"z0": torch.from_numpy(f["z0"]), "z": torch.from_numpy(f["z"]),
                               "z0k": torch.from_numpy(f["z0k"]),
                               "actions": torch.from_numpy(augment(f["actions"], start, proprio)),
                               "d": torch.from_numpy(np.concatenate([f["d_pos"].reshape(n, 9), f["d_rot"].reshape(n, 9)], 1)).float(),
                               "top": topples(f["traces"], f["start_tilt"])})
    return states


def train(name, proprio, rebalance, seed=0):
    import step3b_models as sm
    path = OUT / f"model_{name}.pt"
    if path.exists():
        return wm.WorldModel.load(path)
    torch.manual_seed(seed); np.random.seed(seed)
    states = load(p1b.FOLDERS_ENRICHED, proprio)
    g = sm.G.load(sm.DATA / "G.pt")
    rows = [(si, k) for si, s in enumerate(states) for k in range(len(s["z"]))]
    top = np.array([bool(states[si]["top"][k]) for si, k in rows])
    idx_top, idx_stay = np.flatnonzero(top), np.flatnonzero(~top)
    seq = torch.cat([torch.cat([s["z0k"][:, None], s["z"]], 1) for s in states[:400]])
    pairs = torch.cat([s["actions"] for s in states[:400]])
    stats = wm.Stats(seq, pairs)
    model = wm.WorldModel("deterministic", stats, pairs.shape[-1])
    opt = torch.optim.AdamW(model.net.parameters(), lr=wm.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    wts = torch.ones(wm.STEPS, device=DEVICE); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT
    log = [{"toppling_future_share_in_data": float(top.mean())}]
    for step in range(STEPS):
        if rebalance:
            sel = np.concatenate([np.random.choice(idx_top, BATCH // 2), np.random.choice(idx_stay, BATCH // 2)])
        else:
            sel = np.random.randint(len(rows), size=BATCH)
        pick = [rows[i] for i in sel]
        z0 = torch.stack([states[i]["z0"] for i, _ in pick]).to(DEVICE).float()
        rend = torch.stack([states[i]["z0k"][k] for i, k in pick]).to(DEVICE).float()
        dd = torch.stack([states[i]["d"][k] for i, k in pick]).to(DEVICE)
        target = torch.stack([states[i]["z"][k] for i, k in pick]).to(DEVICE).float()
        act = model._act(torch.stack([states[i]["actions"][k] for i, k in pick]).to(DEVICE))
        use_g = torch.rand(BATCH, device=DEVICE) < 0.5
        start = rend.clone()
        if use_g.any():
            start[use_g] = g(z0[use_g], dd[use_g])
        model.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, loss = start, 0.0
            for t in range(wm.STEPS):
                inp = cur + torch.randn_like(cur) * sm.INPUT_NOISE * stats.res_std
                r = model.net(inp, torch.zeros_like(inp), act[:, t], torch.zeros(BATCH, device=DEVICE))
                cur = cur + r.float() * stats.res_std
                loss = loss + wts[t] * ((cur - target[:, t]) ** 2).mean()
            loss = loss / wts.sum()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            log.append({"step": step + 1, "loss": float(loss)})
            print(f"{name} step {step + 1}: loss {float(loss):.5f}", flush=True)
    model.net.eval()
    model.save(path)
    (OUT / f"model_{name}_log.json").write_text(json.dumps(log) + "\n")
    return model


@torch.no_grad()
def floor(model, proprio):
    """Mean hold-30 prediction error on step3b/val (exactly recovered states only), G starts, executed actions."""
    import step3b_models as sm
    gop = sm.G.load(sm.DATA / "G.pt")
    inexact = set(json.loads((sm.DATA / "val_recovery.json").read_text())["inexact"])
    errs = []
    for path in sorted((sm.DATA / "val").glob("ep*.npz")):
        if path.name.endswith(".tmp.npz") or path.stem in inexact:
            continue
        start = np.load(path.with_name(path.stem + ".start.npy"))
        with np.load(path) as f:
            n = len(f["z"])
            d = np.concatenate([f["d_pos"].reshape(n, 9), f["d_rot"].reshape(n, 9)], 1)
            z0 = torch.as_tensor(f["z0"], device=DEVICE).float()
            pred = sm.roll(model, gop(z0[None].expand(n, -1, -1, -1), torch.as_tensor(d).float()),
                           augment(f["actions"], start, proprio))[1]
            real = f["z"][:, wm.HOLD30].reshape(n, -1).astype(np.float64)
            errs.extend(np.linalg.norm(pred - real, axis=1) / np.sqrt(real.shape[1] // 3))
    return float(np.mean(errs)), len(errs)


@torch.no_grad()
def evaluate(model, proprio, tau):
    import step3b_models as sm
    from counterfactual_monitor import persistent_mode_alarm
    from outcome_modes import two_mode_test
    gop = sm.G.load(sm.DATA / "G.pt")
    out = []
    for path in sorted((p1.OUT / "h").glob("ep*.npz")):
        start = np.load(path.with_name(path.stem + ".start.npy"))
        with np.load(path) as f:
            from fresh_fragility_panel import topple_count
            k = topple_count(f["traces"], f["start_tilt"])
            d = np.concatenate([f["d_pos"].reshape(64, 9), f["d_rot"].reshape(64, 9)], 1)
            z0 = torch.as_tensor(f["z0"], device=DEVICE).float()
            e10, e30 = sm.roll(model, gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d).float()),
                               augment(f["actions"], start, proprio))
            r0 = bool(persistent_mode_alarm(e10, e30).alarm)
            sep = float(two_mode_test(e30, 0.0).separation_rms_mm)
            out.append({"state": path.stem, "fork": bool(2 <= k <= 62), "lean": bool(np.max(f["start_tilt"]) >= LEAN_DEG),
                        "R0": r0, "R1": bool(r0 and sep > tau)})
        torch.cuda.empty_cache()
    fork = np.array([r["fork"] for r in out]); lean = np.array([r["lean"] for r in out])

    def g(name):
        a = np.array([r[name] for r in out])
        return {"caught": float((a & fork).sum() / fork.sum()), "false_alarm": float((a & ~fork).sum() / (~fork).sum()),
                "forks_caught": int((a & fork).sum()), "false_alarms": int((a & ~fork).sum()),
                "lean_forks_caught": f"{int((a & fork & lean).sum())}/{int((fork & lean).sum())}",
                "contact_forks_caught": f"{int((a & fork & ~lean).sum())}/{int((fork & ~lean).sum())}"}
    return {"R0": g("R0"), "R1": g("R1"), "tau": tau}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=tuple(ARMS) + ("E_reference",))
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    if args.arm == "E_reference":
        model, proprio = wm.WorldModel.load(p1.OUT / "model_E.pt"), False
    else:
        proprio, rebalance = ARMS[args.arm]
        model = train(args.arm, proprio, rebalance)
    tau, n_val = floor(model, proprio)
    results[args.arm] = evaluate(model, proprio, tau) | {"validation_futures": n_val}
    if args.arm != "E_reference":
        log = [r for r in json.loads((OUT / f"model_{args.arm}_log.json").read_text()) if "step" in r]
        last = np.mean([r["loss"] for r in log if STEPS - 3000 < r["step"] <= STEPS])
        before = np.mean([r["loss"] for r in log if STEPS - 6000 < r["step"] <= STEPS - 3000])
        results[args.arm]["possibly_undertrained"] = bool(last < before - 0.02 * abs(before))   # Phase-1 addendum-1 rule
    scored = [a for a in ("E_reference", "P", "R", "PR") if a in results]
    results["ranking_R0"] = sorted(scored, key=lambda a: results[a]["R0"]["caught"] - results[a]["R0"]["false_alarm"], reverse=True)
    best = results[results["ranking_R0"][0]]["R0"]
    results["gate_for_panel_6"] = bool(best["caught"] >= 0.80 and best["false_alarm"] <= 0.20)
    RESULT.write_text(json.dumps(results, indent=1) + "\n")
    print(args.arm, json.dumps(results[args.arm]), flush=True)


if __name__ == "__main__":
    main()
