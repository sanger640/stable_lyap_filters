"""Phase 2 stage P2-0: object-free start samplers (G*, Gauss) in place of the 3-block G operator, with US.

Frozen protocol: results/jenga/phase2_generality_protocol.json (P2_0). G(z0, d) needs per-block pose noise for three
named blocks; G*(z0, eps) and Gauss(z0, eps) need only the image latent and noise.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase1_models as p1  # noqa: E402
import phase1b_models as p1b  # noqa: E402
import phase1d_models as p1d  # noqa: E402
import step3_world_model as wm  # noqa: E402
import step3b_models as sm  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase2_generality_protocol.json"
OUT = ROOT / "results/program_b/phase2_p0"
RESULT = ROOT / "results/jenga/phase2_p0_result.json"
DEVICE, EPS_DIM, STEPS, BATCH, M = "cuda", 32, 8_000, 16, 8
EVAL_SEED = 20261130


class StarNet(sm.PerturbNet):
    """G's network with the 18-number pose input replaced by EPS_DIM Gaussian noise."""

    def __init__(self):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(EPS_DIM, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())


def energy_score(x, y):
    """x (B, m, D) model samples, y (B, k, D) real samples -> mean energy score."""
    xy = torch.cdist(x, y).mean((1, 2))
    xx = torch.cdist(x, x).sum((1, 2)) / (x.shape[1] * (x.shape[1] - 1))
    return (xy - 0.5 * xx).mean()


def train_gstar(states, seed=0):
    path = OUT / "Gstar.pt"
    net = StarNet().to(DEVICE)
    if path.exists():
        net.load_state_dict(torch.load(path, map_location=DEVICE)); return net.eval()
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    z0 = torch.stack([s["z0"] for s in states]).to(DEVICE)
    z0k = [s["z0k"].to(DEVICE) for s in states]
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    log = []
    net.train()
    for step in range(STEPS):
        si = rng.choice(len(states), BATCH, replace=False)
        real = torch.stack([z0k[i][torch.as_tensor(rng.choice(len(z0k[i]), M, replace=False), device=DEVICE)]
                            for i in si]).float()                                            # (B, M, 28, 28, 8)
        base = z0[torch.as_tensor(si, device=DEVICE)].float()
        eps = torch.randn(BATCH * M, EPS_DIM, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(base.repeat_interleave(M, 0), eps)
        loss = energy_score(pred.float().reshape(BATCH, M, -1), real.reshape(BATCH, M, -1))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 250 == 0:
            log.append({"step": step + 1, "es": float(loss)})
            print(f"Gstar step {step + 1}: energy score {float(loss):.3f}", flush=True)
    torch.save(net.state_dict(), path)
    (OUT / "Gstar_log.json").write_text(json.dumps(log) + "\n")
    return net.eval()


def fit_gauss(states):
    path = OUT / "Gauss.pt"
    if path.exists():
        return torch.load(path)
    delta = torch.cat([(s["z0k"].float() - s["z0"][None].float()).reshape(-1, 28 * 28 * 8) for s in states])
    mean = delta.mean(0)
    _, sv, v = torch.linalg.svd((delta - mean).to(DEVICE), full_matrices=False)
    saved = {"mean": mean, "basis": v[:sm.GAUSS_DIMS].cpu(), "sd": (sv[:sm.GAUSS_DIMS] / math.sqrt(len(delta) - 1)).cpu()}
    torch.save(saved, path)
    return saved


def sampler(name, net=None, gauss=None):
    """-> starts(z0 (28,28,8) on DEVICE, n, seed) -> (n, 28, 28, 8)."""
    @torch.no_grad()
    def gstar(z0, n, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        return net(z0[None].expand(n, -1, -1, -1).float(), torch.randn(n, EPS_DIM, device=DEVICE, generator=g)).float()

    @torch.no_grad()
    def gs(z0, n, seed):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        eps = torch.randn(n, sm.GAUSS_DIMS, device=DEVICE, generator=g) * gauss["sd"].to(DEVICE)
        return z0[None].float() + (gauss["mean"].to(DEVICE) + eps @ gauss["basis"].to(DEVICE)).reshape(n, 28, 28, 8)
    return {"Gstar": gstar, "Gauss": gs}[name]


@torch.no_grad()
def floor_and_fit(model, starts):
    """US floor (mean hold-30 error on step3b/val, exact states) with sampler starts; energy score and spread ratio
    of the sampler's starts vs the real rendered starts."""
    inexact = set(json.loads((sm.DATA / "val_recovery.json").read_text())["inexact"])
    errs, es, spread = [], [], []
    for n_state, path in enumerate(sorted((sm.DATA / "val").glob("ep*.npz"))):
        if path.name.endswith(".tmp.npz") or path.stem in inexact:
            continue
        with np.load(path) as f:
            n = len(f["z"])
            z0 = torch.as_tensor(f["z0"], device=DEVICE).float()
            s = starts(z0, n, EVAL_SEED + 100_000 + n_state)
            real0 = torch.as_tensor(f["z0k"], device=DEVICE).float()
            es.append(float(energy_score(s.reshape(1, n, -1), real0.reshape(1, n, -1))))
            spread.append(float(s.reshape(n, -1).std(0).norm() / real0.reshape(n, -1).std(0).norm().clamp_min(1e-9)))
            pred = sm.roll(model, s, p1d.augment(f["actions"], None, False))[1]
            real = f["z"][:, wm.HOLD30].reshape(n, -1).astype(np.float64)
            errs.extend(np.linalg.norm(pred - real, axis=1) / np.sqrt(real.shape[1] // 3))
    return float(np.mean(errs)), {"val_energy_score": float(np.mean(es)), "val_spread_ratio_median": float(np.median(spread))}


@torch.no_grad()
def evaluate(model, starts, tau):
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from outcome_modes import two_mode_test
    out = []
    for n_state, path in enumerate(sorted((p1.OUT / "h").glob("ep*.npz"))):
        with np.load(path) as f:
            k = topple_count(f["traces"], f["start_tilt"])
            z0 = torch.as_tensor(f["z0"], device=DEVICE).float()
            e10, e30 = sm.roll(model, starts(z0, 64, EVAL_SEED + n_state), p1d.augment(f["actions"], None, False))
            r0 = bool(persistent_mode_alarm(e10, e30).alarm)
            sep = float(two_mode_test(e30, 0.0).separation_rms_mm)
            out.append({"state": path.stem, "fork": bool(2 <= k <= 62), "lean": bool(np.max(f["start_tilt"]) >= p1d.LEAN_DEG),
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
    parser.add_argument("stage", choices=("A",))
    parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["P2_0"]["status"].startswith("frozen"):
        raise SystemExit("P2-0 not frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    states = p1d.load(p1b.FOLDERS_ENRICHED, proprio=False)
    net, gauss = train_gstar(states), fit_gauss(states)
    del states
    model = wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt")
    us = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())["US"]
    score = lambda r: r["R0"]["caught"] - r["R0"]["false_alarm"]
    results = {"US_with_G": {"R0": us["R0"], "R1": us["R1"], "tau": us["tau"], "score": round(score(us), 4)}}
    for name in ("Gstar", "Gauss"):
        starts = sampler(name, net, gauss)
        tau, fit = floor_and_fit(model, starts)
        r = evaluate(model, starts, tau) | fit
        r["score"] = round(score(r), 4)
        r["passes_within_0.03"] = bool(score(r) >= score(us) - 0.03)
        results[f"US_with_{name}"] = r
        RESULT.write_text(json.dumps(results, indent=1) + "\n")
        print(name, json.dumps(r), flush=True)
    passing = [n for n in ("Gstar", "Gauss") if results[f"US_with_{n}"]["passes_within_0.03"]]
    results["selected_sampler"] = max(passing, key=lambda n: results[f"US_with_{n}"]["score"]) if passing else None
    results["step_B_needed"] = not passing
    RESULT.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({k: results[k] for k in ("selected_sampler", "step_B_needed")}), flush=True)


if __name__ == "__main__":
    main()
