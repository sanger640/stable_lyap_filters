"""Branching-WM Phase 1c, option 1: does a transformer (DINO-WM's ViT predictor) and/or a 3-frame history
improve the deterministic step model inside the initial-condition recipe (cell E)?

Arms: vit3 (DINO-WM ViTPredictor, 3-frame window) and conv3 (cell E's conv body, 3-frame window).
Frozen protocol: results/jenga/phase1c_branching_protocol.json. Evaluated on the Phase-1 held-out set H.
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

PROTOCOL = ROOT / "results/jenga/phase1c_branching_protocol.json"
OUT = ROOT / "results/program_b/phase1c"
RESULT = ROOT / "results/jenga/phase1c_selection.json"
DEVICE, FRAMES, STEPS, BATCH = "cuda", 3, 30_000, 32
DIM, DEPTH, HEADS, DIM_HEAD, MLP, PATCHES = 404, 6, 16, 64, 2048, 196     # DINO-WM checkpoint settings


class ViTHist(nn.Module):
    """DINO-WM ViTPredictor over a 3-latent window; each frame's tokens carry that frame's action."""

    def __init__(self, action_dim, c=8):
        super().__init__()
        from models.dinowm.vit import ViTPredictor
        self.embed = nn.Linear(4 * c, DIM - 20)
        self.act = nn.Sequential(nn.Linear(action_dim, 64), nn.GELU(), nn.Linear(64, 20))
        self.vit = ViTPredictor(num_patches=PATCHES, num_frames=FRAMES, dim=DIM, depth=DEPTH, heads=HEADS,
                                mlp_dim=MLP, dim_head=DIM_HEAD)
        self.head = nn.Linear(DIM, 4 * c)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, window, actions):                       # window (B,F,28,28,8); actions (B,F,A)
        b, f, h, w, c = window.shape
        tok = window.reshape(b, f, h // 2, 2, w // 2, 2, c).permute(0, 1, 2, 4, 3, 5, 6).reshape(b, f * PATCHES, 4 * c)
        a = self.act(actions)[:, :, None].expand(-1, -1, PATCHES, -1).reshape(b, f * PATCHES, 20)
        out = self.head(self.vit(torch.cat([self.embed(tok), a], -1))[:, -PATCHES:])
        return out.reshape(b, h // 2, w // 2, 2, 2, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)


class ConvHist(nn.Module):
    """Cell E's conv body (FiLM residual blocks) with the 3-latent window stacked along channels."""

    def __init__(self, action_dim, c=8):
        super().__init__()
        cond = 256
        self.cond = nn.Sequential(nn.Linear(FRAMES * action_dim, cond), nn.GELU(), nn.Linear(cond, cond), nn.GELU())
        self.stem = nn.Conv2d(FRAMES * c + 2, wm.WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList(wm.FiLMBlock(wm.WIDTH, cond) for _ in range(wm.BLOCKS))
        self.out = nn.Sequential(nn.GroupNorm(8, wm.WIDTH), nn.GELU(), nn.Conv2d(wm.WIDTH, c, 3, padding=1))

    def forward(self, window, actions):
        b, f, h, w, c = window.shape
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=window.device),
                                torch.linspace(-1, 1, w, device=window.device), indexing="ij")
        coords = torch.stack([xs, ys])[None].expand(b, -1, -1, -1)
        x = self.stem(torch.cat([window.permute(0, 1, 4, 2, 3).reshape(b, f * c, h, w), coords], 1))
        cnd = self.cond(actions.reshape(b, -1))
        for block in self.blocks:
            x = block(x, cnd)
        return self.out(x).permute(0, 2, 3, 1)


class HistModel:
    def __init__(self, arch, stats, action_dim):
        self.arch, self.stats = arch, stats
        self.net = (ViTHist if arch == "vit3" else ConvHist)(action_dim).to(DEVICE)

    def _act(self, a):
        return (a.float() - self.stats.a_mean.to(a.device)) / self.stats.a_std.to(a.device)

    def rollout(self, start, pairs, noise=0.0, train=False):
        """start (B,28,28,8); pairs (B,19,A) raw -> list of 19 predicted latents (window padded with start)."""
        acts = self._act(pairs)
        window = [start] * FRAMES
        act_window = [acts[:, 0]] * FRAMES
        outs = []
        for t in range(wm.STEPS):
            act_window = act_window[1:] + [acts[:, t]]
            w = torch.stack(window, 1)
            if noise:
                w = w + torch.randn_like(w) * noise * self.stats.res_std
            if train:
                from torch.utils.checkpoint import checkpoint
                r = checkpoint(self.net, w, torch.stack(act_window, 1), use_reentrant=False)
            else:
                r = self.net(w, torch.stack(act_window, 1))
            cur = window[-1] + r.float() * self.stats.res_std
            outs.append(cur)
            window = window[1:] + [cur]
        return outs

    def save(self, path):
        torch.save({"arch": self.arch, "net": self.net.state_dict(), "stats": self.stats.state(),
                    "action_dim": 8}, path)

    @classmethod
    def load(cls, path):
        s = torch.load(path, map_location=DEVICE, weights_only=False)
        m = cls(s["arch"], wm.Stats.load(s["stats"]), s["action_dim"]); m.net.load_state_dict(s["net"]); m.net.eval()
        return m


def train(arch, seed=0):
    import step3b_models as sm
    path = OUT / f"model_{arch}.pt"
    if path.exists():
        return HistModel.load(path)
    torch.manual_seed(seed); np.random.seed(seed)
    states = p1.load_states(p1b.FOLDERS_ENRICHED)
    g = sm.G.load(sm.DATA / "G.pt")
    rows = [(si, k) for si, s in enumerate(states) for k in range(len(s["z"]))]
    seq = torch.cat([torch.cat([s["z0k"][:, None], s["z"]], 1) for s in states[:400]])
    pairs = torch.cat([s["actions"].reshape(len(s["z"]), wm.STEPS, -1) for s in states[:400]])
    stats = wm.Stats(seq, pairs)
    model = HistModel(arch, stats, pairs.shape[-1])
    opt = torch.optim.AdamW(model.net.parameters(), lr=wm.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    wts = torch.ones(wm.STEPS, device=DEVICE); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT
    log = []
    for step in range(STEPS):
        pick = [rows[i] for i in np.random.randint(len(rows), size=BATCH)]
        z0 = torch.stack([states[i]["z0"] for i, _ in pick]).to(DEVICE).float()
        rend = torch.stack([states[i]["z0k"][k] for i, k in pick]).to(DEVICE).float()
        dd = torch.stack([states[i]["d"][k] for i, k in pick]).to(DEVICE)
        target = torch.stack([states[i]["z"][k] for i, k in pick]).to(DEVICE).float()
        act = torch.stack([states[i]["actions"][k] for i, k in pick]).to(DEVICE).reshape(BATCH, wm.STEPS, -1)
        use_g = torch.rand(BATCH, device=DEVICE) < 0.5
        start = rend.clone()
        if use_g.any():
            start[use_g] = g(z0[use_g], dd[use_g])
        model.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = model.rollout(start, act, noise=sm.INPUT_NOISE, train=True)
            loss = sum(wts[t] * ((outs[t] - target[:, t]) ** 2).mean() for t in range(wm.STEPS)) / wts.sum()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            log.append({"step": step + 1, "loss": float(loss)})
            print(f"{arch} step {step + 1}: loss {float(loss):.5f}", flush=True)
    model.net.eval()
    model.save(path)
    (OUT / f"model_{arch}_log.json").write_text(json.dumps(log) + "\n")
    return model


def evaluate(arch, model, rows):
    import step3b_models as sm
    gop = sm.G.load(sm.DATA / "G.pt")

    @torch.no_grad()
    def endings(cell, m, row, seed):
        z0 = torch.as_tensor(row["z0"], device=DEVICE).float()
        starts = gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(row["d"]).float())
        pairs = torch.as_tensor(row["actions"], device=DEVICE).reshape(64, wm.STEPS, -1)
        outs = m.rollout(starts, pairs)
        return (outs[wm.HOLD10].reshape(64, -1).double().cpu().numpy(),
                outs[wm.HOLD30].reshape(64, -1).double().cpu().numpy())

    original = p1.model_endings
    p1.model_endings = endings
    try:
        return p1.evaluate(arch, model, rows)
    finally:
        p1.model_endings = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", choices=("vit3", "conv3"))
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    p1b.OUT = OUT
    model = train(args.arch)
    results = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    results[args.arch] = evaluate(args.arch, model, p1.h_states()) | {
        "possibly_undertrained": p1b.undertrained(args.arch, STEPS)}
    results["E_reference"] = json.loads((ROOT / "results/jenga/phase1_selection.json").read_text())["E"]
    ranked = sorted([a for a in ("vit3", "conv3") if a in results] + ["E_reference"],
                    key=lambda a: p1.score(results[a]), reverse=True)
    results["ranking"] = ranked
    best = results[ranked[0]]
    results["gate_for_panel_6"] = bool(best["caught"] >= 0.80 and best["false_alarm"] <= 0.20)
    RESULT.write_text(json.dumps(results, indent=1) + "\n")
    print(args.arch, json.dumps({k: results[args.arch][k] for k in ("caught", "false_alarm")}), flush=True)


if __name__ == "__main__":
    main()
