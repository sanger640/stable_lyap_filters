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


def fit_te(states, radius=4):
    """P2-0b: stationary (translation-equivariant) Gaussian field fitted to (z0k - z0): spectral density per frequency
    (8x8), from the circular cross-covariance truncated to |lag| <= radius and radially averaged."""
    path = OUT / "Gauss_TE.pt"
    if path.exists():
        return torch.load(path)
    P = torch.zeros(28, 28, 8, 8, dtype=torch.complex128, device=DEVICE); n = 0
    for s in states:
        d = (s["z0k"].float() - s["z0"][None].float()).to(DEVICE).permute(0, 3, 1, 2).double()      # (K, 8, 28, 28)
        f = torch.fft.fft2(d).permute(0, 2, 3, 1)                                                  # (K, 28, 28, 8)
        P += torch.einsum("kabi,kabj->abij", f, f.conj()); n += len(d)
    P /= n * 28 * 28
    c = torch.fft.ifft2(P.permute(2, 3, 0, 1)).real.permute(2, 3, 0, 1)                            # cov(lag) (28,28,8,8)
    lag = torch.fft.fftfreq(28, 1 / 28, device=DEVICE).double()
    r = torch.sqrt(lag[:, None] ** 2 + lag[None, :] ** 2)
    rb = torch.round(r).long()
    iso = torch.zeros_like(c)
    for b in range(radius + 1):
        m = rb == b
        iso[m] = c[m].mean(0)
    iso = 0.5 * (iso + iso.transpose(-1, -2).flip(0, 1).roll((1, 1), (0, 1)))      # c(-l) = c(l)^T
    S = torch.fft.fft2(iso.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
    S = 0.5 * (S + S.conj().transpose(-1, -2))
    w, v = torch.linalg.eigh(S)
    root = v @ torch.diag_embed(w.clamp_min(0).sqrt().to(v.dtype)) @ v.conj().transpose(-1, -2)
    saved = {"root": root.cpu(), "lag0_cov": c[0, 0].cpu(), "iso_lag0_cov": iso[0, 0].cpu()}
    torch.save(saved, path)
    return saved


def fit_mask(states):
    path = OUT / "Gauss_TE_mask.pt"
    if path.exists():
        return torch.load(path)
    z = torch.cat([s["z0"][None].float() for s in states]).reshape(-1, 8)
    m = z.median(0).values
    dist = (z - m).norm(dim=1)
    saved = {"m": m, "q5": torch.quantile(dist[::7], 0.05), "q95": torch.quantile(dist[::7], 0.95)}
    torch.save(saved, path)
    return saved


def sampler(name, net=None, gauss=None, te=None, mask=None):
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
    @torch.no_grad()
    def gte(z0, n, seed, masked=False):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        w = torch.randn(n, 28, 28, 8, device=DEVICE, generator=g).double()
        f = torch.fft.fft2(w.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)                              # (n,28,28,8)
        x = torch.einsum("abij,nabj->nabi", te["root"].to(DEVICE), f)
        x = torch.fft.ifft2(x.permute(0, 3, 1, 2)).real.permute(0, 2, 3, 1).float()
        if masked:
            a = (((z0.float() - mask["m"].to(DEVICE)).norm(dim=-1) - mask["q5"]) / (mask["q95"] - mask["q5"])).clamp(0, 1)
            x = x * a[None, :, :, None]
        return z0[None].float() + x
    return {"Gstar": gstar, "Gauss": gs, "Gauss_TE": gte,
            "Gauss_TE_masked": lambda z0, n, seed: gte(z0, n, seed, True)}[name]


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
    parser.add_argument("stage", choices=("A", "b0"))
    args = parser.parse_args()
    if args.stage == "b0":
        return main_b0()
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


def main_b0():
    """P2-0b (amendment 1): location-independent samplers Gauss_TE and Gauss_TE_masked, evaluation-only swap with US."""
    if not json.loads(PROTOCOL.read_text())["amendment_1"]["status"].startswith("frozen"):
        raise SystemExit("amendment 1 not frozen")
    states = p1d.load(p1b.FOLDERS_ENRICHED, proprio=False)
    te, mask = fit_te(states), fit_mask(states)
    del states
    check = sampler("Gauss_TE", te=te)(torch.zeros(28, 28, 8, device=DEVICE), 4000, 1)
    emp = torch.cov(check.reshape(-1, 8).T.double()).cpu()
    model = wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt")
    us = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())["US"]
    score = lambda r: r["R0"]["caught"] - r["R0"]["false_alarm"]
    res = {"US_with_G_score": round(score(us), 4),
           "US_with_Gauss_global": json.loads(RESULT.read_text())["US_with_Gauss"],
           "TE_sampler_check": {"target_lag0_var_trace": float(te["iso_lag0_cov"].trace()),
                                "sampled_lag0_var_trace": float(emp.trace())}}
    out = ROOT / "results/jenga/phase2_p0b_result.json"
    for name in ("Gauss_TE", "Gauss_TE_masked"):
        starts = sampler(name, te=te, mask=mask)
        tau, fit = floor_and_fit(model, starts)
        r = evaluate(model, starts, tau) | fit
        r["score"] = round(score(r), 4)
        r["passes_within_0.03"] = bool(score(r) >= score(us) - 0.03)
        res[f"US_with_{name}"] = r
        out.write_text(json.dumps(res, indent=1) + "\n")
        print(name, json.dumps(r), flush=True)
    passing = [n for n in ("Gauss_TE", "Gauss_TE_masked") if res[f"US_with_{n}"]["passes_within_0.03"]]
    res["phase2_sampler"] = (max(passing, key=lambda n: res[f"US_with_{n}"]["score"]) if passing
                             else "Gauss_TE_masked (no candidate passed; shortfall reported)")
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(res["phase2_sampler"], flush=True)


if __name__ == "__main__":
    main()
