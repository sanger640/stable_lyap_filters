"""Fast trainer for the cell-E family (Phase 1e amendment 1): packed GPU-resident data (option 1), torch.compile
(option 2), batch 128 x 7,500 steps with sqrt-scaled learning rate (option 4, same number of samples as 32 x 30,000).

Same model, loss, start mixing (50/50 G or rendered starts), input noise and hold weighting as cell E; only batching,
step count, learning rate, compilation and per-step gradient checkpointing (memory only) differ.
"""
import json
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import step3_world_model as wm

DEVICE = "cuda"
BATCH, STEPS, LR = 128, 7_500, wm.LR * 2.0


class Packed:
    """All futures of a state list in flat GPU tensors."""

    def __init__(self, states):
        self.state_of = torch.tensor([si for si, s in enumerate(states) for _ in range(len(s["z"]))], device=DEVICE)
        self.z0 = torch.stack([s["z0"] for s in states]).to(DEVICE)                     # (S,28,28,8) fp16
        self.z0k = torch.cat([s["z0k"] for s in states]).to(DEVICE)                     # (F,28,28,8) fp16
        # FAST_TRAIN_Z_ON_CPU=1 keeps the large future tensor in pinned host memory (16 GB GPUs); same values, gathered per batch
        self.z_cpu = os.environ.get("FAST_TRAIN_Z_ON_CPU") == "1"
        z = torch.cat([s["z"] for s in states])                                         # (F,19,28,28,8) fp16
        self.z = z.pin_memory() if self.z_cpu else z.to(DEVICE)
        self.d = torch.cat([s["d"] for s in states]).to(DEVICE)                         # (F,18)
        self.a = torch.cat([s["actions"] for s in states]).to(DEVICE)                   # (F,19,A)

    def batch(self, idx):
        z = self.z[idx.cpu()].to(DEVICE, non_blocking=True) if self.z_cpu else self.z[idx]
        return (self.z0[self.state_of[idx]].float(), self.z0k[idx].float(), self.d[idx], z.float(), self.a[idx])


def train(name, out_dir, states, sampler, seed=0, steps=STEPS, compile_net=True, log_every=250, start_fn=None):
    """sampler(rng, batch) -> numpy indices into the flat future list. start_fn(z0) -> sampled starts replaces the
    3-block G operator in the 50/50 start mix (Phase 2: object-free Gauss-TE)."""
    import step3b_models as sm
    path = out_dir / f"model_{name}.pt"
    if path.exists():
        return wm.WorldModel.load(path)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    g = sm.G.load(sm.DATA / "G.pt")
    seq = torch.cat([torch.cat([s["z0k"][:, None], s["z"]], 1) for s in states[:400]])
    pairs = torch.cat([s["actions"] for s in states[:400]])
    stats = wm.Stats(seq, pairs)
    model = wm.WorldModel("deterministic", stats, pairs.shape[-1])
    data = Packed(states)
    net = torch.compile(model.net) if compile_net else model.net
    opt = torch.optim.AdamW(model.net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    wts = torch.ones(wm.STEPS, device=DEVICE); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT
    zeros_tau = torch.zeros(BATCH, device=DEVICE)
    log, t0 = [{"batch": BATCH, "steps": steps, "lr": LR}], time.time()

    def step_fn(inp, act_t):
        return net(inp, torch.zeros_like(inp), act_t, zeros_tau)
    for step in range(steps):
        idx = torch.as_tensor(sampler(rng, BATCH), device=DEVICE)
        z0, rend, dd, target, act = data.batch(idx)
        act = model._act(act)
        use_g = torch.rand(BATCH, device=DEVICE) < 0.5
        with torch.no_grad():
            sampled = g(z0, dd) if start_fn is None else start_fn(z0)
            start = torch.where(use_g[:, None, None, None], sampled, rend)
        model.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, loss = start, 0.0
            for t in range(wm.STEPS):
                inp = cur + torch.randn_like(cur) * sm.INPUT_NOISE * stats.res_std
                r = checkpoint(step_fn, inp, act[:, t], use_reentrant=False)
                cur = cur + r.float() * stats.res_std
                loss = loss + wts[t] * ((cur - target[:, t]) ** 2).mean()
            loss = loss / wts.sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % log_every == 0:
            log.append({"step": step + 1, "loss": float(loss), "seconds": round(time.time() - t0, 1)})
            print(f"{name} step {step + 1}: loss {float(loss):.5f} ({time.time() - t0:.0f}s)", flush=True)
    model.net.eval()
    model.save(path)
    (out_dir / f"model_{name}_log.json").write_text(json.dumps(log) + "\n")
    return model


def undertrained(log, steps=STEPS):
    """Phase-1 addendum-1 rule scaled to the step count (last 10% vs the 10% before)."""
    w = steps // 10
    rows = [r for r in log if "step" in r]
    last = np.mean([r["loss"] for r in rows if steps - w < r["step"] <= steps])
    before = np.mean([r["loss"] for r in rows if steps - 2 * w < r["step"] <= steps - w])
    return bool(last < before - 0.02 * abs(before))
