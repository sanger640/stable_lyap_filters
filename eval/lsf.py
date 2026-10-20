"""Latent safety filter (LSF) baseline on Jenga — docs/LATENT_SAFETY_FILTER.md (protocol fixed before this code).

Stages:
  data        (CPU) labelled latents from the training splits of closedloop_v1, cf_v1, cf_v2, oil_v1: every future
              step's latent (28x28x8), proprioception (current target xyz + gripper) and a sticky failure label
              (neighbour toppled, eval/fresh_fragility_panel rule); all failures + 3x as many safe latents (seeded);
              validation splits -> a held-out set. Also reset latents for imagination (z0 of every training state).
  classifier  (GPU) l(z, g): conv trunk + MLP, tanh output, hinge loss with margin DELTA (labels: safe +, failed -)
  value       (GPU) DDPG in the world model's imagination (model_OIL): critic V/Q, actor = safety policy (bounded target
              change), discounted reachability target y = (1-g) l + g min(l, Q'(z', pi'(z'))), g = 0.9999
usage: python3 eval/lsf.py data|classifier|value
"""
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = Path(os.environ.get("LSF_OUT", str(ROOT / "results/program_b/lsf_v1")))
SOURCES = os.environ.get("LSF_SOURCES", "closedloop_v1,cf_v1,cf_v2,oil_v1").split(",")
TAG = os.environ.get("LSF_TAG", "")                 # part files (data built where the sources live), merged by stage_merge
SAFE_PER_FAIL, SEED = 3, 20261020


def stage_merge():
    """Concatenate part files (LSF_TAGS=a,b) into the canonical train_/val_ arrays."""
    tags = os.environ["LSF_TAGS"].split(",")
    for split in ("train", "val"):
        for k in ("z", "prop", "fail", "reset_z", "reset_prop"):
            parts = [np.load(OUT / f"{split}_{k}_{t}.npy") for t in tags if (OUT / f"{split}_{k}_{t}.npy").exists()]
            np.save(OUT / f"{split}_{k}.npy", np.concatenate(parts))
        y = np.load(OUT / f"{split}_fail.npy")
        json.dump({"latents": int(len(y)), "failed": int(y.sum()), "resets": int(len(np.load(OUT / f"{split}_reset_z.npy"))), "tags": tags},
                  open(OUT / f"{split}_summary.json", "w")); print(split, len(y), int(y.sum()), flush=True)


def labels_for(path):
    """-> latents (K, 19, 28, 28, 8) fp16, proprio (K, 19, 4), fail (K, 19) bool (sticky), z0, start target."""
    from fresh_fragility_panel import neighbour_tilts, TOPPLE_DEG
    d = np.load(path)
    tr, fi, acts = d["traces"], d["frame_index"], d["actions"]
    elig = d["start_tilt"] < TOPPLE_DEG
    K = len(tr); fail = np.zeros((K, len(fi)), bool); prop = np.zeros((K, len(fi), 4), np.float32)
    for k in range(K):
        f = np.array([np.any((neighbour_tilts(tr[k][min(i, len(tr[k]) - 1)]) >= TOPPLE_DEG) & elig) for i in fi])
        fail[k] = np.maximum.accumulate(f)
        prop[k] = acts[k][np.minimum(fi, len(acts[k]) - 1)]           # the target commanded at that control step
    return d["z"], prop, fail, d["z0"], acts[:, 0]


def stage_data():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for split in ("train", "val"):
        Z, P, Y, R0, RP = [], [], [], [], []
        for src in SOURCES:
            files = [f for f in sorted(glob.glob(str(ROOT / f"results/program_b/{src}/{split}/ep*.npz"))) if not f.endswith(".tmp.npz")]
            for f in files:
                z, prop, fail, z0, a0 = labels_for(f)
                flat_fail = fail.reshape(-1)
                idx_f = np.flatnonzero(flat_fail); idx_s = np.flatnonzero(~flat_fail)
                keep_s = rng.choice(idx_s, min(len(idx_s), SAFE_PER_FAIL * max(len(idx_f), 1)), replace=False) if len(idx_s) else idx_s
                keep = np.concatenate([idx_f, keep_s])
                Z.append(z.reshape(-1, 28, 28, 8)[keep]); P.append(prop.reshape(-1, 4)[keep]); Y.append(flat_fail[keep])
                R0.append(z0); RP.append(a0[0])
            print(split, src, len(files), "files", flush=True)
        Z, P, Y = np.concatenate(Z), np.concatenate(P), np.concatenate(Y)
        sfx = f"_{TAG}" if TAG else ""
        np.save(OUT / f"{split}_z{sfx}.npy", Z.astype(np.float16)); np.save(OUT / f"{split}_prop{sfx}.npy", P); np.save(OUT / f"{split}_fail{sfx}.npy", Y)
        np.save(OUT / f"{split}_reset_z{sfx}.npy", np.stack(R0).astype(np.float16)); np.save(OUT / f"{split}_reset_prop{sfx}.npy", np.stack(RP))
        json.dump({"latents": int(len(Y)), "failed": int(Y.sum()), "resets": len(R0), "sources": SOURCES}, open(OUT / f"{split}_summary{sfx}.json", "w"))
        print(split, "latents", len(Y), "failed", int(Y.sum()), "resets", len(R0), flush=True)


# ------------------------------------------------------------------------------------------- networks
DELTA, GAMMA, TAU = 0.25, 0.9999, 0.005
DEV = os.environ.get("LSF_DEVICE", "cuda")
WM_PATH = ROOT / "results/program_b/oil_v1/model/model_OIL.pt"
MAX_STEP = 0.01                                  # m per control step (the policy's own range)


def nets():
    import torch
    from torch import nn

    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(8, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                   nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Flatten())        # 28 -> 4 x 4 x 64

        def forward(self, z):
            return self.c(z.float().permute(0, 3, 1, 2))

    def mlp(i, o, h=512, n=4):
        layers, d = [], i
        for _ in range(n):
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        return nn.Sequential(*layers, nn.Linear(d, o))

    class Classifier(nn.Module):                   # l(z, g) in (-1, 1)
        def __init__(self):
            super().__init__(); self.t = Trunk(); self.m = mlp(1024 + 4, 1, n=2)

        def forward(self, z, g):
            return torch.tanh(self.m(torch.cat([self.t(z), g], 1))).squeeze(1)

    class Critic(nn.Module):                       # Q(z, g, a)
        def __init__(self):
            super().__init__(); self.t = Trunk(); self.m = mlp(1024 + 4 + 8, 1)

        def forward(self, z, g, a):
            return self.m(torch.cat([self.t(z), g, a], 1)).squeeze(1)

    class Actor(nn.Module):                        # safety policy: bounded target change for the 2 control steps
        def __init__(self):
            super().__init__(); self.t = Trunk(); self.m = mlp(1024 + 4, 6)

        def forward(self, z, g):
            d = torch.tanh(self.m(torch.cat([self.t(z), g], 1))).view(-1, 2, 3) * MAX_STEP
            p1 = g[:, :3] + d[:, 0]; p2 = p1 + d[:, 1]
            return torch.cat([p1, g[:, 3:], p2, g[:, 3:]], 1)                       # (B, 8): xyz, grip, xyz, grip
    return Classifier, Critic, Actor


def stage_classifier(epochs=int(os.environ.get("LSF_EPOCHS", 20)), batch=512, limit=None):
    import torch
    Classifier, _, _ = nets()
    data = {s: (np.load(OUT / f"{s}_z.npy", mmap_mode="r"), np.load(OUT / f"{s}_prop.npy"), np.load(OUT / f"{s}_fail.npy")) for s in ("train", "val")}
    if limit:
        data = {s: tuple(x[:limit] for x in v) for s, v in data.items()}
    clf = Classifier().to(DEV); opt = torch.optim.AdamW(clf.parameters(), lr=1e-4)
    rng = np.random.default_rng(SEED); best, log = 1e9, []

    def loss_of(z, g, y):
        l = clf(z, g); safe, fail = ~y, y
        ls = torch.relu(DELTA - l[safe]).mean() if safe.any() else 0 * l.sum()
        lf = torch.relu(DELTA + l[fail]).mean() if fail.any() else 0 * l.sum()
        return ls + lf, l

    def T(a, i):
        return torch.as_tensor(np.ascontiguousarray(a[i]), device=DEV)
    for ep in range(epochs):
        clf.train(); idx = rng.permutation(len(data["train"][2]))
        for b in range(0, len(idx), batch):
            i = np.sort(idx[b:b + batch]); z, g, y = T(data["train"][0], i), T(data["train"][1], i), T(data["train"][2], i)
            loss, _ = loss_of(z, g, y); opt.zero_grad(); loss.backward(); opt.step()
        clf.eval(); vl, acc, n = 0.0, 0, 0
        with torch.no_grad():
            for b in range(0, len(data["val"][2]), 2048):
                i = np.arange(b, min(b + 2048, len(data["val"][2])))
                z, g, y = T(data["val"][0], i), T(data["val"][1], i), T(data["val"][2], i)
                lv, l = loss_of(z, g, y); vl += float(lv) * len(i); acc += int(((l < 0) == y).sum()); n += len(i)
        log.append({"epoch": ep, "val_hinge": vl / n, "val_acc": acc / n}); print(log[-1], flush=True)
        if vl / n < best:
            best = vl / n; torch.save(clf.state_dict(), OUT / "classifier.pt")
    json.dump(log, open(OUT / "classifier_log.json", "w"))


def stage_value(iters=int(os.environ.get("LSF_ITERS", 300)), envs=256, horizon=15, grad_steps=64, batch=512, cap=150_000):
    import copy
    import torch
    import step3_world_model as wm
    wm.DEVICE = DEV
    Classifier, Critic, Actor = nets()
    model = wm.WorldModel.load(WM_PATH); model.net.eval()
    clf = Classifier().to(DEV); clf.load_state_dict(torch.load(OUT / "classifier.pt", map_location=DEV)); clf.eval()
    Q, A = Critic().to(DEV), Actor().to(DEV); Qt, At = copy.deepcopy(Q), copy.deepcopy(A)
    oq, oa = torch.optim.AdamW(Q.parameters(), lr=1e-3), torch.optim.AdamW(A.parameters(), lr=1e-4)
    R_z = np.load(OUT / "train_reset_z.npy"); R_g = np.load(OUT / "train_reset_prop.npy")
    buf = {"z": torch.zeros(cap, 28, 28, 8, dtype=torch.float16, device=DEV), "g": torch.zeros(cap, 4, device=DEV),
           "a": torch.zeros(cap, 8, device=DEV), "z2": torch.zeros(cap, 28, 28, 8, dtype=torch.float16, device=DEV),
           "g2": torch.zeros(cap, 4, device=DEV), "l": torch.zeros(cap, device=DEV)}
    n, ptr, rng, log = 0, 0, np.random.default_rng(SEED), []
    for it in range(iters):
        i = rng.choice(len(R_z), envs); z = torch.as_tensor(R_z[i], device=DEV).float(); g = torch.as_tensor(R_g[i], device=DEV).float()
        with torch.no_grad():
            for t in range(horizon):                                   # imagined rollout of the (noisy) safety actor
                a = A(z, g); a[:, [0, 1, 2, 4, 5, 6]] += torch.randn(envs, 6, device=DEV) * 0.3 * MAX_STEP
                z2 = model.step(z, a).float(); g2 = torch.cat([a[:, 4:7], a[:, 7:8]], 1); l = clf(z, g)
                k = torch.arange(ptr, ptr + envs, device=DEV) % cap
                buf["z"][k], buf["g"][k], buf["a"][k], buf["z2"][k], buf["g2"][k], buf["l"][k] = z.half(), g, a, z2.half(), g2, l
                ptr = (ptr + envs) % cap; n = min(n + envs, cap); z, g = z2, g2
        for _ in range(grad_steps):
            j = torch.as_tensor(rng.integers(0, n, batch), device=DEV)
            zb, gb, ab, z2b, g2b, lb = (buf[k][j] for k in ("z", "g", "a", "z2", "g2", "l"))
            with torch.no_grad():
                y = (1 - GAMMA) * lb + GAMMA * torch.minimum(lb, Qt(z2b, g2b, At(z2b, g2b)))
            lq = ((Q(zb, gb, ab) - y) ** 2).mean(); oq.zero_grad(); lq.backward(); oq.step()
            la = -Q(zb, gb, A(zb, gb)).mean(); oa.zero_grad(); la.backward(); oa.step()
            with torch.no_grad():
                for p, pt in zip(Q.parameters(), Qt.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p)
                for p, pt in zip(A.parameters(), At.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p)
        if it % 10 == 0 or it == iters - 1:
            log.append({"iter": it, "critic_loss": float(lq), "actor_value": float(-la), "buffer": n}); print(log[-1], flush=True)
    torch.save({"critic": Q.state_dict(), "actor": A.state_dict()}, OUT / "value.pt")
    json.dump(log, open(OUT / "value_log.json", "w"))


if __name__ == "__main__":
    {"data": stage_data, "merge": stage_merge, "classifier": stage_classifier, "value": stage_value}[sys.argv[1]]()
