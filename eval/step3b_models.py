"""Step 3 iteration 2: perturbation operator G, Gaussian start baseline, unrolled deterministic latent
dynamics (+ three ablations), and the pre-panel check P1 on held-out TRAIN D0 states.

Frozen protocol: results/jenga/step3b_uncertain_start_protocol.json. Data: eval/step3b_data.py.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import step3_world_model as wm  # noqa: E402

PROTOCOL = ROOT / "results/jenga/step3b_uncertain_start_protocol.json"
DATA = ROOT / "results/program_b/step3b"
P1_RESULT = ROOT / "results/jenga/step3b_p1_result.json"
DEVICE = "cuda"
G_EPOCHS, DYN_EPOCHS, DYN_BATCH, INPUT_NOISE, HOLD_WEIGHT, GAUSS_DIMS = 40, 30, 32, 0.1, 2.0, 64
VARIANTS = {"full": dict(fix1=True, fix2=True, fix3=True), "A_no_fix1": dict(fix1=False, fix2=True, fix3=True),
            "A_no_fix2": dict(fix1=True, fix2=False, fix3=True), "A_no_fix3": dict(fix1=True, fix2=True, fix3=False)}


def load(split):
    keys = ("z0", "z0k", "z", "d_pos", "d_rot", "actions")
    out = {k: [] for k in keys}
    for path in sorted((DATA / split).glob("ep*.npz")):
        with np.load(path) as f:
            for k in keys:
                out[k].append(f[k])
    out = {k: torch.from_numpy(np.stack(v)) for k, v in out.items()}
    out["d"] = torch.cat([out.pop("d_pos").reshape(*out["z0k"].shape[:2], 9),
                          out.pop("d_rot").reshape(*out["z0k"].shape[:2], 9)], -1).float()
    return out                                    # z0 (S,..), z0k (S,K,..), z (S,K,19,..), d (S,K,18), actions (S,K,38,A)


# ------------------------------------------------------------------------------------ G
class PerturbNet(nn.Module):
    def __init__(self, width=96, blocks=4):
        super().__init__()
        cond = 128
        self.cond = nn.Sequential(nn.Linear(18, cond), nn.GELU(), nn.Linear(cond, cond), nn.GELU())
        self.stem = nn.Conv2d(8 + 2, width, 3, padding=1)
        self.blocks = nn.ModuleList(wm.FiLMBlock(width, cond) for _ in range(blocks))
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.GELU(), nn.Conv2d(width, 8, 3, padding=1))

    def forward(self, z, d):
        b, h, w, _ = z.shape
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=z.device), torch.linspace(-1, 1, w, device=z.device),
                                indexing="ij")
        f = self.stem(torch.cat([z.permute(0, 3, 1, 2), torch.stack([xs, ys])[None].expand(b, -1, -1, -1)], 1))
        c = self.cond(d)
        for block in self.blocks:
            f = block(f, c)
        return z + self.out(f).permute(0, 2, 3, 1)


class G:
    def __init__(self, d_mean, d_std):
        self.net, self.d_mean, self.d_std = PerturbNet().to(DEVICE), d_mean.to(DEVICE), d_std.to(DEVICE)

    @torch.no_grad()
    def __call__(self, z0, d):
        self.net.eval()
        return self.net(z0, (d.to(DEVICE) - self.d_mean) / self.d_std)

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "d_mean": self.d_mean.cpu(), "d_std": self.d_std.cpu()}, path)

    @classmethod
    def load(cls, path):
        s = torch.load(path, map_location=DEVICE)
        g = cls(s["d_mean"], s["d_std"]); g.net.load_state_dict(s["net"]); g.net.eval()
        return g


def train_g(train, seed=0):
    path = DATA / "G.pt"
    if path.exists():
        return G.load(path)
    torch.manual_seed(seed)
    s, k = train["z0k"].shape[:2]
    d = train["d"].reshape(-1, 18)
    g = G(d.mean(0), d.std(0).clamp_min(1e-9))
    z0 = train["z0"].to(DEVICE); z0k = train["z0k"].to(DEVICE); dd = train["d"].to(DEVICE)
    total, batch = s * k, 128
    opt = torch.optim.AdamW(g.net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, G_EPOCHS * math.ceil(total / batch))
    g.net.train()
    for epoch in range(G_EPOCHS):
        run = 0.0
        for idx in torch.randperm(total, device=DEVICE).split(batch):
            si, ki = idx // k, idx % k
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = g.net(z0[si].float(), (dd[si, ki] - g.d_mean) / g.d_std)
            loss = ((pred.float() - z0k[si, ki].float()) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"G epoch {epoch + 1}: mse {run / total:.5f}", flush=True)
    g.save(path)
    return g


def fit_gaussian(train):
    path = DATA / "G_gauss.pt"
    if path.exists():
        return torch.load(path)
    delta = (train["z0k"].float() - train["z0"][:, None].float()).reshape(-1, 28 * 28 * 8).to(DEVICE)
    mean = delta.mean(0)
    u, s, v = torch.linalg.svd(delta - mean, full_matrices=False)
    basis = v[:GAUSS_DIMS].cpu(); sd = (s[:GAUSS_DIMS] / math.sqrt(len(delta) - 1)).cpu()
    saved = {"mean": mean.cpu(), "basis": basis, "sd": sd}
    torch.save(saved, path)
    return saved


def gauss_starts(gauss, z0, n, generator):
    eps = torch.randn(n, GAUSS_DIMS, device=DEVICE, generator=generator) * gauss["sd"].to(DEVICE)
    delta = gauss["mean"].to(DEVICE) + eps @ gauss["basis"].to(DEVICE)
    return z0[None] + delta.reshape(n, 28, 28, 8)


# ------------------------------------------------------------------------------- dynamics
def make_stats(train):
    seq = torch.cat([train["z0k"][:, :, None], train["z"]], 2)
    pairs = train["actions"].reshape(*train["actions"].shape[:2], wm.STEPS, -1)
    return wm.Stats(seq, pairs)


def train_dynamics(name, train, g, seed=0):
    path = DATA / f"dyn_{name}.pt"
    if path.exists():
        return wm.WorldModel.load(path)
    cfg = VARIANTS[name]
    torch.manual_seed(seed); np.random.seed(seed)
    stats = make_stats(train)
    pairs = train["actions"].reshape(*train["actions"].shape[:2], wm.STEPS, -1)
    model = wm.WorldModel("deterministic", stats, pairs.shape[-1])
    z0, z0k, z = train["z0"].to(DEVICE), train["z0k"].to(DEVICE), train["z"].to(DEVICE)
    dd, pairs = train["d"].to(DEVICE), pairs.to(DEVICE)
    s, k = z0k.shape[:2]
    total = s * k
    weights = torch.ones(wm.STEPS, device=DEVICE); weights[[wm.HOLD10, wm.HOLD30]] = HOLD_WEIGHT
    steps = DYN_EPOCHS * math.ceil(total / DYN_BATCH)
    opt = torch.optim.AdamW(model.net.parameters(), lr=wm.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    res_std = stats.res_std
    log = []
    for epoch in range(DYN_EPOCHS):
        run, count = 0.0, 0
        for idx in torch.randperm(total, device=DEVICE).split(DYN_BATCH):
            si, ki = idx // k, idx % k
            if not cfg["fix1"]:
                start = z0[si].float()
            elif cfg["fix3"]:
                use_g = torch.rand(len(idx), device=DEVICE) < 0.5
                start = z0k[si, ki].float()
                if use_g.any():
                    start[use_g] = g(z0[si[use_g]].float(), dd[si[use_g], ki[use_g]])
            else:
                start = z0k[si, ki].float()
            act = model._act(pairs[si, ki])                                   # (B, 19, 2A)
            target = z[si, ki].float()                                        # (B, 19, 28, 28, 8)
            model.net.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if cfg["fix2"]:
                    cur, loss = start, 0.0
                    for t in range(wm.STEPS):
                        inp = cur + torch.randn_like(cur) * INPUT_NOISE * res_std
                        r = model.net(inp, torch.zeros_like(inp), act[:, t], torch.zeros(len(idx), device=DEVICE))
                        cur = cur + r.float() * res_std
                        loss = loss + weights[t] * ((cur - target[:, t]) ** 2).mean()
                    loss = loss / weights.sum()
                else:                                                         # one-step teacher forcing
                    inputs = torch.cat([start[:, None], target[:, :-1]], 1).reshape(-1, 28, 28, 8)
                    a = act.reshape(-1, act.shape[-1])
                    r = model.net(inputs, torch.zeros_like(inputs), a, torch.zeros(len(inputs), device=DEVICE))
                    loss = ((((target.reshape(-1, 28, 28, 8) - inputs) / res_std) - r.float()) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item(); count += 1
        log.append({"epoch": epoch + 1, "loss": run / count})
        print(f"{name} epoch {epoch + 1}: loss {run / count:.5f}", flush=True)
    model.net.eval()
    model.save(path)
    (DATA / f"dyn_{name}_log.json").write_text(json.dumps(log) + "\n")
    return model


# ------------------------------------------------------------------------------ rollouts
@torch.no_grad()
def roll(model, starts, actions):
    """starts (64, 28, 28, 8) on DEVICE; actions (64, 38, A) -> endings at hold 10 / 30, each (64, 6272)."""
    pairs = torch.as_tensor(actions, device=DEVICE).reshape(len(starts), wm.STEPS, -1)
    z, ends = starts.float(), {}
    for t in range(wm.STEPS):
        z = model.step(z, pairs[:, t])
        if t in (wm.HOLD10, wm.HOLD30):
            ends[t] = z.reshape(len(z), -1).double().cpu().numpy()
    return ends[wm.HOLD10], ends[wm.HOLD30]


def p1_check():
    import fragility_grading_dev as fg
    from step3_diagnostics import describe
    train = load("train")
    g = train_g(train)
    gauss = fit_gaussian(train)
    models = {name: train_dynamics(name, train, g) for name in VARIANTS}
    d0 = load("d0")
    gen = torch.Generator(device=DEVICE).manual_seed(20261073)
    variants = {"full_G": ("full", "G"), "full_gauss": ("full", "gauss"), "full_oracle": ("full", "rendered"),
                "full_noG": ("full", "z0"), "A_no_fix1_G": ("A_no_fix1", "G"), "A_no_fix2_G": ("A_no_fix2", "G"),
                "A_no_fix3_G": ("A_no_fix3", "G")}
    rows, g_err = [], []
    for n in range(len(d0["z0"])):
        z0 = d0["z0"][n].to(DEVICE).float()
        real = d0["z"][n].double().numpy()
        row = {"real": describe(real[:, 0].reshape(64, -1), real[:, 1].reshape(64, -1))}
        g_starts = g(z0[None].expand(64, -1, -1, -1), d0["d"][n])
        rendered = d0["z0k"][n].to(DEVICE).float()
        g_err.append(float(((g_starts - rendered) ** 2).mean() / ((rendered - z0) ** 2).mean()))
        for name, (model, start) in variants.items():
            starts = {"G": g_starts, "rendered": rendered, "z0": z0[None].expand(64, -1, -1, -1),
                      "gauss": None}[start]
            if start == "gauss":
                starts = gauss_starts(gauss, z0, 64, gen)
            row[name] = describe(*roll(models[model], starts, d0["actions"][n].numpy()))
        rows.append(row)
    real_alarm = [r["real"]["alarm"] for r in rows]
    summary = {}
    for name in ["real"] + list(variants):
        alarms = [r[name]["alarm"] for r in rows]
        summary[name] = {"alarms": int(sum(alarms)), "kappa_vs_real_latent": fg.kappa(alarms, real_alarm),
                         "median_spread_ratio_to_real": float(np.median([r[name]["spread"] / max(r["real"]["spread"], 1e-9)
                                                                         for r in rows])),
                         "median_pc1_share_on_real_forks": float(np.median([r[name]["pc1_share"] for r in rows
                                                                            if r["real"]["alarm"]]))}
    p1 = bool(summary["full_G"]["kappa_vs_real_latent"] >= 0.60)
    result = {"d0_states": len(rows), "G_relative_error_vs_perturbation": float(np.median(g_err)),
              "summary": summary, "P1_pass": p1}
    P1_RESULT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    p1_check()


if __name__ == "__main__":
    main()
