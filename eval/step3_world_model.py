"""Step 3: stochastic token world model (flow matching) and its deterministic control, plus the
world-model fragility monitor on fresh panel 4.

Frozen protocol: results/jenga/step3_token_world_model_protocol.json (+ addendum 1). Latents are the
standardised close-camera channel-PCA-8 grids (28 x 28 x 8) written by eval/step3_data.py. One model
step = 2 simulator steps; 19 steps reach trace index 37 (hold 30); step output 8 is hold 10 (index 17).
"""
import argparse
from datetime import datetime, timezone
import hashlib
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

PROTOCOL = ROOT / "results/jenga/step3_token_world_model_protocol.json"
DATA = ROOT / "results/program_b/step3"
PANEL4 = ROOT / "results/program_b/panel4"
RESULT = ROOT / "results/jenga/step3_token_world_model_result.json"
STEPS, HOLD10, HOLD30 = 19, 8, 18
WIDTH, BLOCKS, EULER = 128, 6, 16
EPOCHS, BATCH, LR = 60, 512, 3e-4
DEVICE = "cuda"


# -------------------------------------------------------------------------------------- data
def load_split(split):
    z0, z, act = [], [], []
    for path in sorted((DATA / split).glob("ep*.npz")):
        with np.load(path) as f:
            z0.append(f["z0"]); z.append(f["z"]); act.append(f["actions"])
    z0 = torch.from_numpy(np.stack(z0))                     # (S, 28, 28, 8) float16
    z = torch.from_numpy(np.stack(z))                       # (S, K, 19, 28, 28, 8)
    act = torch.from_numpy(np.stack(act))                   # (S, K, 38, A)
    seq = torch.cat([z0[:, None, None].expand(-1, z.shape[1], -1, -1, -1, -1), z], 2)   # (S, K, 20, ...)
    pairs = act.reshape(*act.shape[:2], STEPS, -1)          # (S, K, 19, 2A): actions of each model step
    return seq, pairs


class Stats:
    def __init__(self, seq, pairs):
        diff = (seq[:, :, 1:].float() - seq[:, :, :-1].float())
        self.res_std = diff.std().item()
        flat = pairs.reshape(-1, pairs.shape[-1]).float()
        self.a_mean, self.a_std = flat.mean(0), flat.std(0).clamp_min(1e-6)

    def state(self):
        return {"res_std": self.res_std, "a_mean": self.a_mean, "a_std": self.a_std}

    @classmethod
    def load(cls, d):
        s = cls.__new__(cls); s.res_std, s.a_mean, s.a_std = d["res_std"], d["a_mean"], d["a_std"]
        return s


# ------------------------------------------------------------------------------------- model
class FiLMBlock(nn.Module):
    def __init__(self, c, cond):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)
        self.film = nn.Linear(cond, 2 * c)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, cond):
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, 1)
        h = self.c1(nn.functional.gelu(self.n1(x)))
        h = nn.functional.gelu(self.n2(h) * (1 + scale) + shift)
        return x + self.c2(h)


class Transition(nn.Module):
    """f(z_t, noisy residual x_tau, actions, tau) -> velocity (stochastic) or residual (deterministic)."""

    def __init__(self, action_dim, channels=8):
        super().__init__()
        cond = 256
        self.cond = nn.Sequential(nn.Linear(action_dim + 1, cond), nn.GELU(), nn.Linear(cond, cond), nn.GELU())
        self.stem = nn.Conv2d(2 * channels + 2, WIDTH, 3, padding=1)
        self.blocks = nn.ModuleList(FiLMBlock(WIDTH, cond) for _ in range(BLOCKS))
        self.out = nn.Sequential(nn.GroupNorm(8, WIDTH), nn.GELU(), nn.Conv2d(WIDTH, channels, 3, padding=1))

    def forward(self, z, x, actions, tau):                 # z, x: (B, 28, 28, 8)
        b, h, w, _ = z.shape
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=z.device), torch.linspace(-1, 1, w, device=z.device),
                                indexing="ij")
        coords = torch.stack([xs, ys])[None].expand(b, -1, -1, -1)
        f = self.stem(torch.cat([z.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2), coords], 1))
        c = self.cond(torch.cat([actions, tau[:, None]], 1))
        for block in self.blocks:
            f = block(f, c)
        return self.out(f).permute(0, 2, 3, 1)


class WorldModel:
    def __init__(self, kind, stats, action_dim):
        self.kind, self.stats = kind, stats
        self.net = Transition(action_dim).to(DEVICE)

    def _act(self, a):
        return (a.float() - self.stats.a_mean.to(a.device)) / self.stats.a_std.to(a.device)

    @torch.no_grad()
    def step(self, z, actions, generator=None):
        """One model step for a batch; fresh noise per call (stochastic) or the mean (deterministic)."""
        a = self._act(actions)
        b = len(z)
        if self.kind == "deterministic":
            r = self.net(z, torch.zeros_like(z), a, torch.zeros(b, device=z.device))
        else:
            x = torch.randn(z.shape, device=z.device, generator=generator)
            for k in range(EULER):
                tau = torch.full((b,), k / EULER, device=z.device)
                x = x + self.net(z, x, a, tau) / EULER
            r = x
        return z + r * self.stats.res_std

    def loss(self, z, target, actions):
        a = self._act(actions)
        r = (target - z) / self.stats.res_std
        b = len(z)
        if self.kind == "deterministic":
            pred = self.net(z, torch.zeros_like(z), a, torch.zeros(b, device=z.device))
            return ((pred - r) ** 2).mean()
        x0 = torch.randn_like(r)
        tau = torch.rand(b, device=z.device)
        xt = (1 - tau[:, None, None, None]) * x0 + tau[:, None, None, None] * r
        return ((self.net(z, xt, a, tau) - (r - x0)) ** 2).mean()

    def save(self, path):
        torch.save({"kind": self.kind, "state_dict": self.net.state_dict(), "stats": self.stats.state(),
                    "action_dim": self.net.cond[0].in_features - 1}, path)

    @classmethod
    def load(cls, path):
        global WIDTH
        saved = torch.load(path, map_location=DEVICE, weights_only=False)
        default, WIDTH = WIDTH, int(saved["state_dict"]["stem.weight"].shape[0])   # wider nets (Phase 2 amendment 4)
        try:
            m = cls(saved["kind"], Stats.load(saved["stats"]), saved["action_dim"])
        finally:
            WIDTH = default
        m.net.load_state_dict(saved["state_dict"]); m.net.eval()
        return m


def train(kind, seed=0):
    path = DATA / f"wm_{kind}.pt"
    if path.exists():
        return WorldModel.load(path)
    torch.manual_seed(seed); np.random.seed(seed)
    seq, pairs = load_split("train")
    vseq, vpairs = load_split("val")
    stats = Stats(seq, pairs)
    model = WorldModel(kind, stats, pairs.shape[-1])
    seq, pairs = seq.to(DEVICE), pairs.to(DEVICE)
    s, k = seq.shape[:2]
    total = s * k * STEPS
    steps = EPOCHS * math.ceil(total / BATCH)
    opt = torch.optim.AdamW(model.net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    log = []
    for epoch in range(EPOCHS):
        order = torch.randperm(total, device=DEVICE)
        running = 0.0
        for n, first in enumerate(range(0, total, BATCH)):
            idx = order[first:first + BATCH]
            si, ki, ti = idx // (k * STEPS), (idx // STEPS) % k, idx % STEPS
            z, target, act = seq[si, ki, ti].float(), seq[si, ki, ti + 1].float(), pairs[si, ki, ti]
            if n % 2 == 1:                               # self-rollout input for half the batches
                keep = ti >= 1
                if keep.any():
                    model.net.eval()
                    prev = seq[si[keep], ki[keep], ti[keep] - 1].float()
                    z = z.clone()
                    z[keep] = model.step(prev, pairs[si[keep], ki[keep], ti[keep] - 1])
                    model.net.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(z, target, act)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
            opt.step(); sched.step()
            running += loss.item()
        val = validate(model, vseq, vpairs)
        log.append({"epoch": epoch + 1, "train_loss": running / (n + 1), **val})
        print(f"{kind} epoch {epoch + 1}: loss {running / (n + 1):.4f} val {val}", flush=True)
    model.net.eval()
    model.save(path)
    (DATA / f"wm_{kind}_log.json").write_text(json.dumps(log, indent=0) + "\n")
    return model


@torch.no_grad()
def validate(model, vseq, vpairs, states=40):
    """Teacher-forced loss and 19-step open-loop latent error on held-out states (reported only)."""
    model.net.eval()
    vs, vp = vseq[:states].to(DEVICE), vpairs[:states].to(DEVICE)
    s, k = vs.shape[:2]
    z = vs[:, :, :-1].reshape(-1, 28, 28, 8).float()
    target = vs[:, :, 1:].reshape(-1, 28, 28, 8).float()
    act = vp.reshape(-1, vp.shape[-1])
    tf = float(np.mean([model.loss(z[i:i + 1024], target[i:i + 1024], act[i:i + 1024]).item()
                        for i in range(0, len(z), 1024)]))
    cur = vs[:, :, 0].reshape(-1, 28, 28, 8).float()
    for t in range(STEPS):
        cur = model.step(cur, vp[:, :, t].reshape(-1, vp.shape[-1]))
    end_err = ((cur - vs[:, :, -1].reshape(-1, 28, 28, 8).float()) ** 2).mean().item()
    model.net.train()
    return {"val_loss": tf, "val_rollout_end_mse": end_err}


# ---------------------------------------------------------------------------------- monitor
@torch.no_grad()
def wm_rollouts(model, z0, actions, seed):
    """z0 (28,28,8); actions (64, 38, A) -> latent endings at hold 10 and hold 30, each (64, 6272)."""
    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    z = torch.as_tensor(z0, device=DEVICE).float()[None].expand(len(actions), -1, -1, -1).contiguous()
    pairs = torch.as_tensor(actions, device=DEVICE).reshape(len(actions), STEPS, -1)
    ends = {}
    for t in range(STEPS):
        z = model.step(z, pairs[:, t], generator=g)
        if t in (HOLD10, HOLD30):
            ends[t] = z.reshape(len(z), -1).double().cpu().numpy()
    return ends[HOLD10], ends[HOLD30]


def panel_actions(chunk, snippets):
    actions = []
    for noise in snippets:
        a = np.asarray(chunk, np.float32).copy()
        a[:, :3] -= noise
        actions.append(np.concatenate([a, np.repeat(np.asarray(chunk, np.float32)[-1:], 30, 0)]))
    return np.stack(actions)


def evaluate():
    import fragility_grading_dev as fg
    import uncertainty_monitor_dev as um
    import visual_v2_panel4 as v4
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings, wilson
    from step3_data import Latent
    panel, screen, _ = v4.screen_panel(1)
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    snippets = np.load(um.BENCH)["snippets"]
    latent = Latent()
    models = {"stochastic": WorldModel.load(DATA / "wm_stochastic.pt"),
              "deterministic": WorldModel.load(DATA / "wm_deterministic.pt")}
    dec = {m: [] for m in models}
    minority = {m: [] for m in models}
    truth_a, harm, truth_minority = [], [], []
    for n, (e, i) in enumerate(keys):
        with np.load(PANEL4 / f"ep{e}_s{i}.npz") as f:
            z0 = latent(f["close_frame"][None])[0].astype(np.float32)
            actions = panel_actions(f["chunk"], snippets)
            for name, model in models.items():
                early, late = wm_rollouts(model, z0, actions, seed=20261047 + n)
                res = persistent_mode_alarm(early, late)
                dec[name].append(bool(res.alarm)); minority[name].append(int(res.late["minority"]) / 64)
            ta, mino, _ = decide(*trace_endings(f["truth_A"], NON_ROBOT["jenga"]))
            truth_a.append(ta); truth_minority.append(mino / 64)
            harm.append(2 <= topple_count(f["truth_A"], np.array(screen[e][str(i)]["start_tilt"])) <= 62)
    truth_a, harm = np.array(truth_a), np.array(harm)
    graded = {m: fg.grade(dec[m], truth_a) for m in models}
    st = np.array(dec["stochastic"])
    hits = int((st & harm).sum())
    w1 = bool(harm.sum() and hits / harm.sum() >= 0.80 and (st & ~harm).sum() / max((~harm).sum(), 1) <= 0.20)
    w2 = bool(graded["stochastic"]["kappa"] >= 0.60)
    w3 = bool(graded["stochastic"]["kappa"] - graded["deterministic"]["kappa"] >= 0.15)
    fr = truth_a.astype(bool)
    fidelity = (float(np.corrcoef(np.array(minority["stochastic"])[fr], np.array(truth_minority)[fr])[0, 1])
                if fr.sum() > 2 and np.std(np.array(minority["stochastic"])[fr]) > 0 else None)
    d0 = json.loads((DATA / "d0_result.json").read_text())
    result = {
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel_states": len(keys), "truth_fragile_A": int(truth_a.sum()), "harm_fork_states": int(harm.sum()),
        "D0": d0, "graded_vs_truth_A": graded,
        "harm": {m: {"recall": int((np.array(dec[m]) & harm).sum()) / harm.sum() if harm.sum() else None,
                     "recall_wilson": wilson(int((np.array(dec[m]) & harm).sum()), int(harm.sum())),
                     "alarm_on_non_harm": float((np.array(dec[m]) & ~harm).sum() / max((~harm).sum(), 1))}
                 for m in models},
        "branching_fidelity_minority_corr_on_truth_fragile": fidelity,
        "gates": {"D0": d0["pass"], "W1_harm": w1, "W2_fragility": w2, "W3_stochasticity_matters": w3},
        "confirmed": bool(d0["pass"] and w1 and w2 and w3),
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("graded_vs_truth_A", "harm", "gates", "confirmed")}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "evaluate"))
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    if args.stage == "train":
        d0 = json.loads((DATA / "d0_result.json").read_text())
        if not d0["pass"]:
            raise SystemExit(f"D0 failed: {d0}")
        for kind in ("stochastic", "deterministic"):
            train(kind)
    else:
        evaluate()


if __name__ == "__main__":
    main()
