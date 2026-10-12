"""Phase 2 amendment 10: hybrid world model = frozen deterministic stream (US2_xlong) + residual flow-matching head.

Frozen protocol: results/jenga/phase2_generality_protocol.json (amendment_10). Foresight-Diffusion-style decoupling:
the deterministic model predicts where things go; a FlowWM-recipe flow head (wide projection head, noise-shifted
timesteps) samples the residual between real futures and that prediction at hold 10 / hold 30.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_models as pm  # noqa: E402

OUT = pm.OUT
RESULT = ROOT / "results/jenga/phase2_hybrid_result.json"
STEPS, BATCH, LR, WARMUP, ALPHA, EULER = 20_000, 128, 3e-4, 500, 8.0, 20
MONITOR_SEED = 20261250


def shift(t, alpha=ALPHA):
    """Esser et al. timestep shift toward noise (t = 0 noise, t = 1 data in our convention)."""
    s = 1.0 - t                                   # noise level
    s = alpha * s / (1 + (alpha - 1) * s)
    return 1.0 - s


def build_head(action_dim, width=384, depth=4, heads=6):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(width, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
            self.n2 = nn.LayerNorm(width, elementwise_affine=False)
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
            self.ada = nn.Linear(width, 6 * width)
            nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

        def forward(self, x, c):
            s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
            h = self.n1(x) * (1 + s1) + b1
            x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
            h = self.n2(x) * (1 + s2) + b2
            return x + g2 * self.mlp(h)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(64 + 32 + 64, width)
            self.pos = nn.Parameter(torch.zeros(1, 196, width)); nn.init.trunc_normal_(self.pos, std=0.02)
            self.tmb = nn.Sequential(nn.Linear(64, width), nn.GELU(), nn.Linear(width, width))
            self.amb = nn.Sequential(nn.Linear(action_dim * 19, width), nn.GELU(), nn.Linear(width, width))
            self.blocks = nn.ModuleList(Block() for _ in range(depth))
            self.proj = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1024), nn.GELU(), nn.Linear(1024, 1024),
                                      nn.GELU(), nn.Linear(1024, 64))          # wide projection head (FlowWM)
            nn.init.zeros_(self.proj[-1].weight); nn.init.zeros_(self.proj[-1].bias)

        @staticmethod
        def tok(x, c):                                   # (B, 28, 28, c) -> (B, 196, 4c)
            b = len(x)
            return x.reshape(b, 14, 2, 14, 2, c).permute(0, 1, 3, 2, 4, 5).reshape(b, 196, 4 * c)

        def forward(self, r_t, t, z0, zdet, act):
            """r_t (B,2,28,28,8) noisy residual; t (B,); z0 (B,28,28,8); zdet (B,2,28,28,8); act (B,19,A) normalised."""
            b = len(r_t)
            rt = self.tok(r_t.permute(0, 2, 3, 1, 4).reshape(b, 28, 28, 16), 16)
            zd = self.tok(zdet.permute(0, 2, 3, 1, 4).reshape(b, 28, 28, 16), 16)
            x = self.inp(torch.cat([rt, self.tok(z0, 8), zd], -1)) + self.pos
            freqs = torch.exp(torch.arange(32, device=t.device) * (-math.log(1000.0) / 32))
            te = torch.cat([torch.sin(t[:, None] * 1000 * freqs), torch.cos(t[:, None] * 1000 * freqs)], -1)
            c = self.tmb(te) + self.amb(act.reshape(b, -1))
            for blk in self.blocks:
                x = blk(x, c)
            v = self.proj(x)                             # (B, 196, 64)
            v = v.reshape(b, 14, 14, 2, 2, 2, 8).permute(0, 5, 1, 3, 2, 4, 6).reshape(b, 2, 28, 28, 8)
            return v
    return Head()


def det_endpoints(det, z0, act_norm):
    """Frozen deterministic rollout from the true start; act_norm already standardised (B,19,A)."""
    import torch
    import step3_world_model as wm
    with torch.no_grad():
        cur, ends = z0, []
        zeros_tau = torch.zeros(len(z0), device=z0.device)
        for t in range(wm.STEPS):
            r = det.net(cur, torch.zeros_like(cur), act_norm[:, t], zeros_tau)
            cur = cur + r * det.stats.res_std
            if t in (wm.HOLD10, wm.HOLD30):
                ends.append(cur)
    return torch.stack(ends, 1)                          # (B, 2, 28, 28, 8)


def train(seed=0, log_every=250):
    import torch
    from torch import nn
    import fast_train as ft
    import step3_world_model as wm
    path = OUT / "model_HYB.pt"
    if path.exists():
        return
    states, names = pm.load_train()
    modes = json.loads((OUT / "modes.json").read_text())
    rare = np.array([k in set(modes[n]) for n, s in zip(names, states) for k in range(len(s["z"]))])
    has_rare = np.array([bool(modes[n]) for n, s in zip(names, states) for _ in range(len(s["z"]))])
    pools = (np.flatnonzero(rare), np.flatnonzero(~rare & has_rare))
    data = ft.Packed(states)
    action_dim = states[0]["actions"].shape[-1]
    del states
    det = wm.WorldModel.load(OUT / "model_US2_xlong.pt"); det.net.eval()
    for p in det.net.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    head = build_head(action_dim).cuda()
    params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / STEPS))))
    # residual scale (fixed from the first 20 batches; stored with the model)
    scales = []
    for _ in range(20):
        idx = torch.as_tensor(np.concatenate([rng.choice(pools[0], BATCH // 2), rng.choice(pools[1], BATCH // 2)]), device="cuda")
        z0, _, _, target, act = data.batch(idx)
        a = det._act(act)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zd = det_endpoints(det, z0, a).float()
        real = torch.stack([target[:, wm.HOLD10], target[:, wm.HOLD30]], 1)
        scales.append(float((real - zd).std()))
    rscale = float(np.mean(scales))
    log, t0 = [{"steps": STEPS, "batch": BATCH, "lr": LR, "alpha": ALPHA, "params": params, "residual_scale": rscale}], time.time()
    print(json.dumps(log[0]), flush=True)
    for step in range(STEPS):
        idx = torch.as_tensor(np.concatenate([rng.choice(pools[0], BATCH // 2), rng.choice(pools[1], BATCH // 2)]), device="cuda")
        z0, _, _, target, act = data.batch(idx)
        a = det._act(act)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zd = det_endpoints(det, z0, a).float()
        r = (torch.stack([target[:, wm.HOLD10], target[:, wm.HOLD30]], 1) - zd) / rscale
        noise = torch.randn_like(r)
        t = shift(torch.rand(BATCH, device="cuda"))
        rt = (1 - t)[:, None, None, None, None] * noise + t[:, None, None, None, None] * r
        head.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = head(rt, t, z0, zd / rscale, a).float()
        loss = ((v - (r - noise)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % log_every == 0:
            log.append({"step": step + 1, "loss": float(loss), "seconds": round(time.time() - t0, 1)})
            print(f"HYB step {step + 1}: flow loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state_dict": head.state_dict(), "action_dim": action_dim, "residual_scale": rscale}, path)
    (OUT / "model_HYB_log.json").write_text(json.dumps(log) + "\n")
    print("trained HYB", flush=True)


def sample(head, det, z0, act_norm, rscale, generator):
    """64 futures: deterministic endpoints + residual sampled with EULER steps on the shifted schedule."""
    import torch
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        zd = det_endpoints(det, z0, act_norm).float()
        x = torch.randn(zd.shape, device="cuda", generator=generator)
        grid = shift(torch.linspace(0, 1, EULER + 1, device="cuda"))
        for i in range(EULER):
            t = grid[i].expand(len(x))
            v = head(x, t, z0, zd / rscale, act_norm).float()
            x = x + (grid[i + 1] - grid[i]) * v
    return zd + x * rscale


def grade():
    import torch
    import step3_world_model as wm
    from phase2_amend4 import alarm, score
    s = torch.load(OUT / "model_HYB.pt", map_location="cuda")
    head = build_head(s["action_dim"]).cuda(); head.load_state_dict(s["state_dict"]); head.eval()
    det = wm.WorldModel.load(OUT / "model_US2_xlong.pt"); det.net.eval()
    fork, fam, al, spread = [], [], [], []
    for n_state, path in enumerate(sorted((OUT / "h2").glob("*.npz"))):
        with np.load(path) as f:
            k = int(f["harm"].sum())
            z0 = torch.as_tensor(f["z0"], device="cuda").float()[None].expand(64, -1, -1, -1).contiguous()
            act = det._act(torch.as_tensor(np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, -1), device="cuda"))
            real30 = f["z"][:, 1].reshape(64, -1).astype(np.float64)
        g = torch.Generator(device="cuda").manual_seed(MONITOR_SEED + n_state)
        fut = sample(head, det, z0, act, s["residual_scale"], g)
        e10 = fut[:, 0].reshape(64, -1).double().cpu().numpy(); e30 = fut[:, 1].reshape(64, -1).double().cpu().numpy()
        fork.append(2 <= k <= 62); fam.append(path.stem.split("_")[0])
        al.append(alarm(e10, e30, "BOTH"))
        if 2 <= k <= 62:
            spread.append(float(e30.std(0).sum() / real30.std(0).sum()))
        torch.cuda.empty_cache()
    fork, fam, al = np.array(fork), np.array(fam), np.array(al)
    out = {"readout": "BOTH", "HYB": {"all": score(al, fork), "F1": score(al[fam == "F1"], fork[fam == "F1"]),
                                      "F2": score(al[fam == "F2"], fork[fam == "F2"]),
                                      "fork_spread_ratio_median": float(np.median(spread))}}
    out["hybrid_helps"] = bool(out["HYB"]["all"]["score"] >= 0.276 + 0.05)
    out["passes_in_family_bar_on_H2"] = bool(out["HYB"]["all"]["caught"] >= 0.80 and out["HYB"]["all"]["false_alarm"] <= 0.20)
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "grade"))
    args = parser.parse_args()
    if not json.loads((ROOT / "results/jenga/phase2_generality_protocol.json").read_text())["amendment_10"]["status"].startswith("frozen"):
        raise SystemExit("amendment 10 not frozen")
    train() if args.stage == "train" else grade()


if __name__ == "__main__":
    main()
