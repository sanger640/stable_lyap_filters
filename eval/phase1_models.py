"""Branching-WM Phase 1: whole-clip world models (flow matching / energy score), the step-wise
deterministic recipe on enriched data, the action-sensitivity variant, and evaluation on the held-out
selection set H.

Frozen protocol: results/jenga/phase1_branching_protocol.json.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import step3_world_model as wm  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase1_branching_protocol.json"
OUT = ROOT / "results/program_b/phase1"
CURRENT = ROOT / "results/program_b/step3b/train"
RESULT = ROOT / "results/jenga/phase1_selection.json"
DEVICE = "cuda"
T, H, W, C = 19, 28, 28, 8
WIDTH, BLOCKS, STEPS, BATCH, LR, EULER = 96, 8, 30_000, 32, 3e-4, 16
ENERGY_FUTURES, HOLD_WEIGHT = 4, 2.0                          # 4 futures x 2 samples = 8 samples per state
ENERGY_BATCH = 8                                              # addendum 1: compute-matched energy cells
CELLS = {"A": ("flow", "current"), "B": ("flow", "enriched"), "C": ("energy", "current"),
         "D": ("energy", "enriched"), "E": ("swdet", "enriched")}


# ------------------------------------------------------------------------------------- data
def load_states(folders):
    states = []
    for folder in folders:
        for path in sorted(Path(folder).glob("ep*.npz")):
            with np.load(path) as f:
                states.append({"z0": torch.from_numpy(f["z0"]), "z": torch.from_numpy(f["z"]),
                               "actions": torch.from_numpy(f["actions"]), "z0k": torch.from_numpy(f["z0k"]),
                               "d": torch.from_numpy(np.concatenate([f["d_pos"].reshape(len(f["z"]), 9),
                                                                     f["d_rot"].reshape(len(f["z"]), 9)], 1)).float()})
    return states


def dataset(which):
    folders = [CURRENT] if which == "current" else [CURRENT, OUT / "train"]
    return load_states(folders)


class Norm:
    """Residual (trajectory - start) scale and action statistics from the training set."""

    def __init__(self, states):
        res = torch.cat([(s["z"][:, :].float() - s["z0"].float()[None, None]).reshape(-1) for s in states[:400]])
        self.res_std = float(res.std())
        acts = torch.cat([s["actions"].reshape(-1, s["actions"].shape[-1]) for s in states]).float()
        self.a_mean, self.a_std = acts.mean(0), acts.std(0).clamp_min(1e-6)

    def state(self):
        return {"res_std": self.res_std, "a_mean": self.a_mean, "a_std": self.a_std}


# ------------------------------------------------------------------------------------ model
class STBlock(nn.Module):
    """Factorised space-time residual block with FiLM conditioning per time step."""

    def __init__(self, c, cond):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
        self.space = nn.Conv2d(c, c, 3, padding=1)
        self.time = nn.Conv1d(c, c, 3, padding=1)
        self.film = nn.Linear(cond, 2 * c)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, cond):                               # x (B, T, c, H, W); cond (B, T, cond)
        b, t, c, h, w = x.shape
        y = self.space(nn.functional.gelu(self.n1(x.reshape(b * t, c, h, w))))
        y = y.reshape(b, t, c, h, w).permute(0, 3, 4, 2, 1).reshape(b * h * w, c, t)
        y = self.time(y).reshape(b, h, w, c, t).permute(0, 4, 3, 1, 2)
        scale, shift = self.film(cond).reshape(b, t, 2 * c, 1, 1).chunk(2, 2)
        y = nn.functional.gelu(self.n2(y.reshape(b * t, c, h, w)).reshape(b, t, c, h, w) * (1 + scale) + shift)
        return x + self.out(y.reshape(b * t, c, h, w)).reshape(b, t, c, h, w)


class ClipNet(nn.Module):
    """(z0, actions, x, tau) -> per-step output over the whole clip (velocity or trajectory)."""

    def __init__(self, action_dim):
        super().__init__()
        cond = 192
        self.cond = nn.Sequential(nn.Linear(action_dim + 1 + 1, cond), nn.GELU(), nn.Linear(cond, cond), nn.GELU())
        self.stem = nn.Conv2d(2 * C + 2, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList(STBlock(WIDTH, cond) for _ in range(BLOCKS))
        self.head = nn.Sequential(nn.GroupNorm(8, WIDTH), nn.GELU(), nn.Conv2d(WIDTH, C, 3, padding=1))

    def forward(self, z0, actions, x, tau):                   # z0 (B,H,W,C); actions (B,T,A); x (B,T,H,W,C); tau (B,)
        b = len(z0)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=z0.device), torch.linspace(-1, 1, W, device=z0.device),
                                indexing="ij")
        coords = torch.stack([xs, ys])[None, None].expand(b, T, -1, -1, -1)
        start = z0.permute(0, 3, 1, 2)[:, None].expand(-1, T, -1, -1, -1)
        inp = torch.cat([start, x.permute(0, 1, 4, 2, 3), coords], 2).reshape(b * T, 2 * C + 2, H, W)
        f = self.stem(inp).reshape(b, T, WIDTH, H, W)
        t_idx = torch.linspace(0, 1, T, device=z0.device)[None, :, None].expand(b, -1, 1)
        cond = self.cond(torch.cat([actions, t_idx, tau[:, None, None].expand(-1, T, 1)], 2))
        for block in self.blocks:
            f = checkpoint(block, f, cond, use_reentrant=False) if self.training else block(f, cond)
        out = self.head(f.reshape(b * T, WIDTH, H, W)).reshape(b, T, C, H, W)
        return out.permute(0, 1, 3, 4, 2)


class ClipModel:
    def __init__(self, kind, norm, action_dim):
        self.kind, self.norm = kind, norm
        self.net = ClipNet(action_dim).to(DEVICE)

    def _act(self, actions):                                  # (B, 38, 4) -> (B, 19, 8) standardised
        a = (actions.float() - self.norm.a_mean.to(actions.device)) / self.norm.a_std.to(actions.device)
        return a.reshape(len(a), T, -1)

    @torch.no_grad()
    def sample(self, z0, actions, generator=None):
        """z0 (B,H,W,C), actions (B,38,4) -> trajectories (B,T,H,W,C)."""
        self.net.eval()
        a = self._act(actions)
        noise = torch.randn((len(z0), T, H, W, C), device=DEVICE, generator=generator)
        if self.kind == "energy":
            r = self.net(z0, a, noise, torch.zeros(len(z0), device=DEVICE))
        else:
            r = noise
            for k in range(EULER):
                r = r + self.net(z0, a, r, torch.full((len(z0),), k / EULER, device=DEVICE)) / EULER
        return z0[:, None] + r * self.norm.res_std

    def save(self, path):
        torch.save({"kind": self.kind, "net": self.net.state_dict(), "norm": self.norm.state(),
                    "action_dim": self.net.cond[0].in_features - 2}, path)

    @classmethod
    def load(cls, path):
        s = torch.load(path, map_location=DEVICE, weights_only=False)
        norm = Norm.__new__(Norm); norm.res_std, norm.a_mean, norm.a_std = (s["norm"][k] for k in ("res_std", "a_mean", "a_std"))
        m = cls(s["kind"], norm, s["action_dim"]); m.net.load_state_dict(s["net"]); m.net.eval()
        return m


def weights():
    w = torch.ones(T, device=DEVICE); w[[wm.HOLD10, wm.HOLD30]] = HOLD_WEIGHT
    return w / w.mean()


def train_clip(cell, kind, which, action_sensitivity=False, seed=0):
    path = OUT / f"model_{cell}.pt"
    if path.exists():
        return ClipModel.load(path)
    torch.manual_seed(seed); np.random.seed(seed)
    states = dataset(which)
    norm = Norm(states)
    model = ClipModel(kind, norm, 2 * states[0]["actions"].shape[-1])
    opt = torch.optim.AdamW(model.net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    w = weights()[None, :, None, None, None]
    log = []
    for step in range(STEPS):
        idx = np.random.randint(len(states), size=BATCH if kind == "flow" else ENERGY_BATCH)
        model.net.train()
        if kind == "flow":
            ks = [np.random.randint(len(states[i]["z"])) for i in idx]
            z0 = torch.stack([states[i]["z0"] for i in idx]).to(DEVICE).float()
            traj = torch.stack([states[i]["z"][k] for i, k in zip(idx, ks)]).to(DEVICE).float()
            acts = model._act(torch.stack([states[i]["actions"][k] for i, k in zip(idx, ks)]).to(DEVICE))
            x1 = (traj - z0[:, None]) / norm.res_std
            x0 = torch.randn_like(x1)
            tau = torch.rand(BATCH, device=DEVICE)
            xt = (1 - tau[:, None, None, None, None]) * x0 + tau[:, None, None, None, None] * x1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model.net(z0, acts, xt, tau)
            loss = (w * (v.float() - (x1 - x0)) ** 2).mean()
            if action_sensitivity:
                loss = loss + sensitivity_loss_flow(model, states, idx, norm)
        else:
            ks = [np.random.choice(len(states[i]["z"]), ENERGY_FUTURES, replace=False) for i in idx]
            z0 = torch.stack([states[i]["z0"] for i in idx]).to(DEVICE).float()
            z0r = z0.repeat_interleave(ENERGY_FUTURES, 0)
            real = torch.stack([states[i]["z"][k] for i, kk in zip(idx, ks) for k in kk]).to(DEVICE).float()
            acts = model._act(torch.stack([states[i]["actions"][k] for i, kk in zip(idx, ks) for k in kk]).to(DEVICE))
            y = (real - z0r[:, None]) / norm.res_std
            n = len(y)
            z2, a2 = torch.cat([z0r, z0r]), torch.cat([acts, acts])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = model.net(z2, a2, torch.randn((2 * n, T, H, W, C), device=DEVICE), torch.zeros(2 * n, device=DEVICE))
            x = x.float() * w
            yw = y * w
            d1 = (x[:n] - yw).flatten(1).norm(dim=1)
            d2 = (x[n:] - yw).flatten(1).norm(dim=1)
            dxx = (x[:n] - x[n:]).flatten(1).norm(dim=1)
            loss = ((d1 + d2) / 2 - 0.5 * dxx).mean() / math.sqrt(T * H * W * C)
            if action_sensitivity:
                loss = loss + sensitivity_loss_pairs(x[:n] / w, y, ENERGY_FUTURES)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            log.append({"step": step + 1, "loss": float(loss)})
            print(f"{cell} step {step + 1}: loss {float(loss):.5f}", flush=True)
    model.save(path)
    (OUT / f"model_{cell}_log.json").write_text(json.dumps(log) + "\n")
    return model


def sensitivity_loss_pairs(pred, real, group):
    """Predicted hold-30 differences between futures of the same state must match the real ones."""
    p = pred[:, wm.HOLD30].reshape(-1, group, H * W * C)
    r = real[:, wm.HOLD30].reshape(-1, group, H * W * C)
    dp = p[:, :, None] - p[:, None]
    dr = r[:, :, None] - r[:, None]
    return ((dp - dr) ** 2).mean()


def sensitivity_loss_flow(model, states, idx, norm):
    """For WC-flow: deterministic mean path (zero noise, one Euler step from 0) for two futures per state."""
    pairs = [np.random.choice(len(states[i]["z"]), 2, replace=False) for i in idx[:8]]
    z0 = torch.stack([states[i]["z0"] for i in idx[:8]]).to(DEVICE).float().repeat_interleave(2, 0)
    acts = model._act(torch.stack([states[i]["actions"][k] for i, kk in zip(idx[:8], pairs) for k in kk]).to(DEVICE))
    real = torch.stack([states[i]["z"][k] for i, kk in zip(idx[:8], pairs) for k in kk]).to(DEVICE).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mean_path = model.net(z0, acts, torch.zeros((len(z0), T, H, W, C), device=DEVICE), torch.zeros(len(z0), device=DEVICE))
    return sensitivity_loss_pairs(mean_path.float(), (real - z0[:, None]) / norm.res_std, 2)


def train_swdet(seed=0):
    """Cell E: the iteration-2 dyn_full recipe (own-start pairing, 19-step unrolled loss with input noise,
    50/50 G or rendered starts, hold weighting) on enriched data, 30,000 steps of 32 futures. Data stay in
    CPU memory (too large for the GPU); otherwise identical to step3b_models.train_dynamics."""
    import step3b_models as sm
    path = OUT / "model_E.pt"
    if path.exists():
        return wm.WorldModel.load(path)
    torch.manual_seed(seed); np.random.seed(seed)
    states = dataset("enriched")
    g = sm.G.load(sm.DATA / "G.pt")
    rows = [(si, k) for si, s in enumerate(states) for k in range(len(s["z"]))]
    seq = torch.cat([torch.cat([s["z0k"][:, None], s["z"]], 1) for s in states[:400]])
    pairs = torch.cat([s["actions"].reshape(len(s["z"]), wm.STEPS, -1) for s in states[:400]])
    stats = wm.Stats(seq, pairs)
    model = wm.WorldModel("deterministic", stats, pairs.shape[-1])
    opt = torch.optim.AdamW(model.net.parameters(), lr=wm.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    wts = torch.ones(wm.STEPS, device=DEVICE); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT
    log = []
    for step in range(STEPS):
        pick = [rows[i] for i in np.random.randint(len(rows), size=BATCH)]
        z0 = torch.stack([states[si]["z0"] for si, _ in pick]).to(DEVICE).float()
        rend = torch.stack([states[si]["z0k"][k] for si, k in pick]).to(DEVICE).float()
        dd = torch.stack([states[si]["d"][k] for si, k in pick]).to(DEVICE)
        target = torch.stack([states[si]["z"][k] for si, k in pick]).to(DEVICE).float()
        act = model._act(torch.stack([states[si]["actions"][k] for si, k in pick]).to(DEVICE).reshape(BATCH, wm.STEPS, -1))
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
            print(f"E step {step + 1}: loss {float(loss):.5f}", flush=True)
    model.net.eval()
    model.save(path)
    (OUT / "model_E_log.json").write_text(json.dumps(log) + "\n")
    return model


# ------------------------------------------------------------------------------- evaluation
def h_states():
    from fresh_fragility_panel import topple_count
    rows = []
    for path in sorted((OUT / "h").glob("ep*.npz")):
        with np.load(path) as f:
            rows.append({"name": path.stem, "fork": 2 <= topple_count(f["traces"], f["start_tilt"]) <= 62,
                         "z0": f["z0"], "z0k": f["z0k"], "z": f["z"], "actions": f["actions"],
                         "d": np.concatenate([f["d_pos"].reshape(64, 9), f["d_rot"].reshape(64, 9)], 1)})
    return rows


@torch.no_grad()
def model_endings(cell, model, row, seed):
    """64 futures of one H state -> (hold-10, hold-30) endings, each (64, 6272)."""
    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    acts = torch.as_tensor(row["actions"], device=DEVICE)
    if cell == "E":
        import step3b_models as sm
        gop = sm.G.load(sm.DATA / "G.pt")
        z0 = torch.as_tensor(row["z0"], device=DEVICE).float()
        starts = gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(row["d"]).float())
        return sm.roll(model, starts, row["actions"])
    z0 = torch.as_tensor(row["z0"], device=DEVICE).float()[None].expand(64, -1, -1, -1).contiguous()
    traj = torch.cat([model.sample(z0[i:i + 16], acts[i:i + 16], g) for i in range(0, 64, 16)])
    return (traj[:, wm.HOLD10].reshape(64, -1).double().cpu().numpy(),
            traj[:, wm.HOLD30].reshape(64, -1).double().cpu().numpy())


def evaluate(cell, model, rows):
    from counterfactual_monitor import persistent_mode_alarm
    from outcome_modes import two_mode_test
    alarms, spread_ratio, axis_share, side = [], [], [], []
    for n, row in enumerate(rows):
        e10, e30 = model_endings(cell, model, row, 20261085 + n)
        alarms.append(bool(persistent_mode_alarm(e10, e30).alarm))
        real = row["z"][:, 1].reshape(64, -1).astype(np.float64)
        mean = real.mean(0)
        sv = np.linalg.svd(real - mean, full_matrices=False)
        axis = sv[2][0]
        spread_ratio.append(np.mean(np.sum((e30 - e30.mean(0)) ** 2, 1)) / max(np.mean(np.sum((real - mean) ** 2, 1)), 1e-9))
        c = e30 - e30.mean(0); s2 = np.linalg.svd(c, compute_uv=False) ** 2
        axis_share.append(float(s2[0] / s2.sum()) if s2.sum() > 0 else 0.0)
        if row["fork"]:
            from fresh_fragility_panel import topple_count
            labels = two_mode_test(real, 0.0).labels
            pr, pm = (real - mean) @ axis, (e30 - mean) @ axis
            thr = (pr[labels == 0].mean() + pr[labels == 1].mean()) / 2
            side.append(float(np.mean((pr > thr) == (pm > thr))))
    fork = np.array([r["fork"] for r in rows]); a = np.array(alarms)
    return {"caught": float((a & fork).sum() / fork.sum()), "false_alarm": float((a & ~fork).sum() / (~fork).sum()),
            "alarms": int(a.sum()), "forks": int(fork.sum()), "states": len(rows),
            "median_spread_ratio": float(np.median(spread_ratio)), "median_main_axis_share": float(np.median(axis_share)),
            "mean_side_agreement_on_forks": float(np.mean(side))}


def undertrained(cell):
    """Addendum 1: mean loss over the last 3,000 steps more than 2% below the 3,000 before."""
    log = json.loads((OUT / f"model_{cell}_log.json").read_text())
    last = np.mean([r["loss"] for r in log if 27_000 < r["step"] <= 30_000])
    before = np.mean([r["loss"] for r in log if 24_000 < r["step"] <= 27_000])
    return bool(last < before - 0.02 * abs(before)), float(last), float(before)


def score(r):
    return r["caught"] - r["false_alarm"], r["mean_side_agreement_on_forks"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "select"))
    parser.add_argument("--cell", default=None)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    if args.stage == "train":
        kind, which = CELLS[args.cell]
        (train_swdet() if kind == "swdet" else train_clip(args.cell, kind, which))
        return
    rows = h_states()
    results = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    for cell, (kind, which) in CELLS.items():
        if cell in results or not (OUT / f"model_{cell}.pt").exists():
            continue
        model = wm.WorldModel.load(OUT / f"model_{cell}.pt") if kind == "swdet" else ClipModel.load(OUT / f"model_{cell}.pt")
        results[cell] = evaluate(cell, model, rows)
        flag, last, before = undertrained(cell)
        results[cell].update({"possibly_undertrained": flag, "loss_last_3k": last, "loss_prev_3k": before})
        print(cell, results[cell], flush=True)
        RESULT.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
