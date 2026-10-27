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
SOURCES = os.environ.get("LSF_SOURCES", "closedloop_v1,cf_v1,cf_v2,oil_v1").replace("+", ",").split(",")
TAG = os.environ.get("LSF_TAG", "")                 # part files (data built where the sources live), merged by stage_merge
SAFE_PER_FAIL, SEED = 3, 20261020


def stage_merge():
    """Concatenate part files (LSF_TAGS=a,b) into the canonical train_/val_ arrays."""
    tags = os.environ["LSF_TAGS"].replace("+", ",").split(",")
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
DELTA, TAU = 0.25, 0.005
GAMMA = float(os.environ.get("LSF_GAMMA", 0.9999))
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


def sac_nets():
    """UNISafe-style SAC (CMU-IntentLab/UNISafe reachability/policy/SAC_Reachability_Env.py): tanh-Gaussian actor over the
    NORMALISED target motion d in [-1, 1]^6 (2 control steps x xyz, x MAX_STEP), twin critics Q(z, g, d)."""
    import torch
    from torch import nn
    Classifier, _, _ = nets()

    def mlp(i, o, h=512, n=3):
        layers, d = [], i
        for _ in range(n):
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        return nn.Sequential(*layers, nn.Linear(d, o))

    def Trunk():
        return nn.Sequential(nn.Conv2d(8, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                             nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Flatten())

    class QNet(nn.Module):
        def __init__(self):
            super().__init__(); self.t = Trunk(); self.m = mlp(1024 + 4 + 6, 1)

        def forward(self, z, g, d):
            return self.m(torch.cat([self.t(z.float().permute(0, 3, 1, 2)), g, d], 1)).squeeze(1)

    class SACActor(nn.Module):
        def __init__(self):
            super().__init__(); self.t = Trunk(); self.m = mlp(1024 + 4, 12)

        def dist(self, z, g):
            mu, ls = self.m(torch.cat([self.t(z.float().permute(0, 3, 1, 2)), g], 1)).chunk(2, 1)
            return mu, ls.clamp(-5, 2)

        def sample(self, z, g):
            mu, ls = self.dist(z, g); u = mu + ls.exp() * torch.randn_like(mu); d = torch.tanh(u)
            logp = (-0.5 * ((u - mu) / ls.exp()) ** 2 - ls - 0.9189385).sum(1) - torch.log(1 - d.pow(2) + 1e-6).sum(1)
            return d, logp

        def mean(self, z, g):
            return torch.tanh(self.dist(z, g)[0])
    return QNet, SACActor


def plan_from_d(g, d):
    """normalised motion (B, 6) from the current target g (B, 4) -> absolute 2-step plan (B, 8) as the policy's format."""
    import torch
    d = d.view(-1, 2, 3) * MAX_STEP; p1 = g[:, :3] + d[:, 0]; p2 = p1 + d[:, 1]
    return torch.cat([p1, g[:, 3:], p2, g[:, 3:]], 1)


class ValueFn:
    """Uniform interface for the deployed filter / diagnostics: V(z, g) under the safety policy, act(z, g) -> plan (B, 8)."""
    def __init__(self, name, dev="cuda"):
        import torch
        w = torch.load(OUT / name, map_location=dev); self.sac = w.get("algo") == "sac"
        if self.sac:
            QNet, SACActor = sac_nets()
            self.q1, self.q2, self.a = QNet().to(dev).eval(), QNet().to(dev).eval(), SACActor().to(dev).eval()
            self.q1.load_state_dict(w["q1"]); self.q2.load_state_dict(w["q2"]); self.a.load_state_dict(w["actor"])
        else:
            _, Critic, Actor = nets(); self.q, self.a = Critic().to(dev).eval(), Actor().to(dev).eval()
            self.q.load_state_dict(w["critic"]); self.a.load_state_dict(w["actor"])

    def act(self, z, g):
        return plan_from_d(g, self.a.mean(z, g)) if self.sac else self.a(z, g)

    def V(self, z, g):
        import torch
        if self.sac:
            d = self.a.mean(z, g); return torch.minimum(self.q1(z, g, d), self.q2(z, g, d))
        return self.q(z, g, self.a(z, g))


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
           "g2": torch.zeros(cap, 4, device=DEV), "l": torch.zeros(cap, device=DEV), "done": torch.zeros(cap, dtype=torch.bool, device=DEV)}
    pess = os.environ.get("LSF_PESS") == "1"
    models = ensemble() if pess else None
    trunc = os.environ.get("LSF_TRUNC") == "1"          # UNISafe-style: the last imagined step is terminal (target = l)
    n, ptr, rng, log = 0, 0, np.random.default_rng(SEED), []
    g0, g1 = float(os.environ.get("LSF_GAMMA0", 0.9)), GAMMA        # gamma annealed toward 1 (reachability-RL practice)
    for it in range(iters):
        gam = 1 - (1 - g0) * ((1 - g1) / (1 - g0)) ** min(1.0, it / max(1, int(0.8 * iters)))
        i = rng.choice(len(R_z), envs); z = torch.as_tensor(R_z[i], device=DEV).float(); g = torch.as_tensor(R_g[i], device=DEV).float()
        with torch.no_grad():
            for t in range(horizon):                                   # imagined rollout of the (noisy) safety actor
                a = A(z, g); a[:, [0, 1, 2, 4, 5, 6]] += torch.randn(envs, 6, device=DEV) * 0.3 * MAX_STEP
                g2 = torch.cat([a[:, 4:7], a[:, 7:8]], 1)
                if pess:
                    z2, l, _ = pessimistic_step(models, clf, z, a, g2)               # margin of the pessimistic next state
                else:
                    z2 = model.step(z, a).float(); l = clf(z, g)
                k = torch.arange(ptr, ptr + envs, device=DEV) % cap
                buf["z"][k], buf["g"][k], buf["a"][k], buf["z2"][k], buf["g2"][k], buf["l"][k] = z.half(), g, a, z2.half(), g2, l
                buf["done"][k] = trunc and t == horizon - 1
                ptr = (ptr + envs) % cap; n = min(n + envs, cap); z, g = z2, g2
        for _ in range(grad_steps):
            j = torch.as_tensor(rng.integers(0, n, batch), device=DEV)
            zb, gb, ab, z2b, g2b, lb, db = (buf[k][j] for k in ("z", "g", "a", "z2", "g2", "l", "done"))
            with torch.no_grad():
                y = (1 - gam) * lb + gam * torch.minimum(lb, Qt(z2b, g2b, At(z2b, g2b)))
                y = torch.where(db, lb, y)
            lq = ((Q(zb, gb, ab) - y) ** 2).mean(); oq.zero_grad(); lq.backward(); oq.step()
            la = -Q(zb, gb, A(zb, gb)).mean(); oa.zero_grad(); la.backward(); oa.step()
            with torch.no_grad():
                for p, pt in zip(Q.parameters(), Qt.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p)
                for p, pt in zip(A.parameters(), At.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p)
        if it % 10 == 0 or it == iters - 1:
            log.append({"iter": it, "gamma": gam, "critic_loss": float(lq), "actor_value": float(-la), "buffer": n}); print(log[-1], flush=True)
    torch.save({"critic": Q.state_dict(), "actor": A.state_dict()}, OUT / os.environ.get("LSF_VALUE_NAME", "value.pt"))
    json.dump(log, open(OUT / "value_log.json", "w"))


def stage_ddpg_pub(envs=256, ep_len=10, batch=512, cap=40_000, upd_ratio=8):
    """Paper / author-code faithful latent reachability DDPG (docs/LSF_FIDELITY_AUDIT.md; Nakamura, Peters & Bajcsy, RSS
    2025, Table VI; CMU-IntentLab/latent-safety, dino branch run_training_ddpg-dinowm.py + franka-DINOwm.py):
      actor / critic 4 x 512 ReLU (after our conv trunk for the 28 x 28 x 8 patch latent), AdamW actor lr 1e-4, critic lr 1e-3
      (critic weight decay 1e-3 as the code), tau 0.005, batch 512, replay 40k, Gaussian exploration sigma 0.1 on the
      normalised action, one gradient step per 8 imagined transitions, imagined episodes of 10 steps from encoded data
      latents (DINO Franka env), deterministic transitions (DINO-WM: no sampling, no pessimism, no uncertainty margin),
      reward r = l(z', g') = classifier margin of the NEXT latent, target y = gamma min(r, Q'(z', pi'(z'))) + (1 - gamma) r,
      no terminal handling (PyHJ ignores done).
    LSF_PUB_MODE=paper: gamma 0.9999 constant (Table VI), 76,250 updates;
    LSF_PUB_MODE=code:  the dino-branch schedule: 1,250 critic-only updates at gamma 0 (warm-up epoch), then 75,000 at gamma 0.95.
    Not reproduced: the code's extra actor penalties on rotation / xyz magnitude (rotation does not exist in our action)."""
    import copy
    import torch
    import step3_world_model as wm
    wm.DEVICE = DEV
    torch.manual_seed(SEED)
    mode = os.environ.get("LSF_PUB_MODE", "paper")
    Classifier, Critic, Actor = nets()
    model = wm.WorldModel.load(WM_PATH); model.net.eval()
    clf = Classifier().to(DEV); clf.load_state_dict(torch.load(OUT / "classifier.pt", map_location=DEV)); clf.eval()
    Q, A = Critic().to(DEV), Actor().to(DEV); Qt, At = copy.deepcopy(Q), copy.deepcopy(A)
    oq = torch.optim.AdamW(Q.parameters(), lr=1e-3, weight_decay=1e-3); oa = torch.optim.AdamW(A.parameters(), lr=1e-4)
    R_z = np.load(OUT / "train_reset_z.npy"); R_g = np.load(OUT / "train_reset_prop.npy")
    buf = {"z": torch.zeros(cap, 28, 28, 8, dtype=torch.float16, device=DEV), "g": torch.zeros(cap, 4, device=DEV),
           "a": torch.zeros(cap, 8, device=DEV), "z2": torch.zeros(cap, 28, 28, 8, dtype=torch.float16, device=DEV),
           "g2": torch.zeros(cap, 4, device=DEV), "r": torch.zeros(cap, device=DEV)}
    sched = [(0.0, False)] * 1250 + [(0.95, True)] * 75000 if mode == "code" else [(0.9999, True)] * 76250
    n, ptr, rng, log, upd, t = 0, 0, np.random.default_rng(SEED), [], 0, ep_len
    while upd < len(sched):
        if t == ep_len:                                                # reset: encoded data latents
            i = rng.choice(len(R_z), envs); z = torch.as_tensor(R_z[i], device=DEV).float(); g = torch.as_tensor(R_g[i], device=DEV).float(); t = 0
        with torch.no_grad():
            a = A(z, g); d = torch.cat([a[:, 0:3] - g[:, :3], a[:, 4:7] - a[:, 0:3]], 1) / MAX_STEP
            d = (d + 0.1 * torch.randn_like(d)).clamp(-1, 1); a = plan_from_d(g, d)
            z2 = model.step(z, a).float(); g2 = torch.cat([a[:, 4:7], a[:, 7:8]], 1); r = clf(z2, g2)
            k = torch.arange(ptr, ptr + envs, device=DEV) % cap
            buf["z"][k], buf["g"][k], buf["a"][k], buf["z2"][k], buf["g2"][k], buf["r"][k] = z.half(), g, a, z2.half(), g2, r
            ptr = (ptr + envs) % cap; n = min(n + envs, cap); z, g, t = z2, g2, t + 1
        for _ in range(envs // upd_ratio):
            if upd >= len(sched):
                break
            gam, actor_on = sched[upd]
            j = torch.as_tensor(rng.integers(0, n, batch), device=DEV)
            zb, gb, ab, z2b, g2b, rb = (buf[x][j] for x in ("z", "g", "a", "z2", "g2", "r"))
            with torch.no_grad():
                y = gam * torch.minimum(rb, Qt(z2b, g2b, At(z2b, g2b))) + (1 - gam) * rb
            lq = ((Q(zb, gb, ab) - y) ** 2).mean(); oq.zero_grad(); lq.backward(); oq.step()
            la = torch.zeros(())
            if actor_on:
                la = -Q(zb, gb, A(zb, gb)).mean(); oa.zero_grad(); la.backward(); oa.step()
            with torch.no_grad():
                for p_, pt in zip(Q.parameters(), Qt.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p_)
                for p_, pt in zip(A.parameters(), At.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p_)
            upd += 1
            if upd % 2000 == 0:
                with torch.no_grad():
                    aa = A(zb, gb); dd = torch.cat([aa[:, 0:3] - gb[:, :3], aa[:, 4:7] - aa[:, 0:3]], 1) / MAX_STEP
                log.append({"update": upd, "gamma": gam, "critic_loss": float(lq), "actor_value": float(-la), "r_mean": float(rb.mean()),
                            "actor_saturated": float((dd.abs() > 0.95).float().mean())}); print(log[-1], flush=True)
    torch.save({"critic": Q.state_dict(), "actor": A.state_dict(), "algo": "ddpg_pub", "mode": mode}, OUT / os.environ.get("LSF_VALUE_NAME", "value_pub.pt"))
    json.dump(log, open(OUT / (os.environ.get("LSF_VALUE_NAME", "value_pub.pt")[:-3] + "_log.json"), "w"))


def ensemble():
    import step3_world_model as wm
    ms = [wm.WorldModel.load(WM_PATH)] + [wm.WorldModel.load(ROOT / f"results/program_b/ens_v1/model_ENS{e}.pt") for e in range(4)]
    for m in ms:
        m.net.eval()
    return ms


def disagreement(models, z, a):
    """Epistemic uncertainty of the transition (z, a): std over the 5 world models of the predicted next latent."""
    import torch
    p = torch.stack([m.step(z, a).float() for m in models]); return p[0], p.std(0).mean((1, 2, 3))


def load_clf(dev=None):
    """The margin l(z, g): LSF_CLF (default classifier.pt, the labelled failure classifier). A checkpoint saved with
    no_g (eval/lflsf.py margin net, label-free, results/jenga/lflsf_protocol.json) is state-only: its proprio input is zeroed."""
    import torch
    dev = dev or DEV
    Classifier, _, _ = nets(); w = torch.load(OUT / os.environ.get("LSF_CLF", "classifier.pt"), map_location=dev)
    net = Classifier().to(dev).eval()
    if isinstance(w, dict) and w.get("no_g"):
        net.load_state_dict(w["state"])
        class NoG(torch.nn.Module):
            def __init__(s):
                super().__init__(); s.net = net
            def forward(s, z, g):
                return s.net(z, torch.zeros_like(g))
        return NoG().eval()
    net.load_state_dict(w); return net


def pessimistic_step(models, clf, z, a, g2):
    """Original LSF (CMU-IntentLab/latent-safety, PyHJ/reach_rl_gym_envs/franka-wm.py): sample the world model N times and
    keep the MOST PESSIMISTIC transition (lowest safety margin). Our world model is deterministic, so the N samples are the
    5 ensemble members. -> next latent (B, ...), its margin l(z', g') and the disagreement u."""
    import torch
    p = torch.stack([m.step(z, a).float() for m in models])                        # (5, B, 28, 28, 8)
    lm = torch.stack([clf(p[k], g2) for k in range(len(models))])                   # (5, B)
    w = lm.argmin(0); idx = torch.arange(len(w), device=w.device)
    return p[w, idx], lm.min(0).values, p.std(0).mean((1, 2, 3))


ALEA = int(os.environ.get("LSF_ALEA", 0))           # audit 3 stage 2: N noisy transitions per imagined step, keep the worst
ALEA_SCALE = float(os.environ.get("LSF_ALEA_SCALE", 2.0))   # execution noise scale (the episodes' EXEC_NOISE_SCALE)


class Noise:
    """The monitor's noise model for the LSF's imagination (docs/LATENT_SAFETY_FILTER.md audit 3, stage 2): perception
    noise on a start latent (start operator G with a perturbation drawn from the measured error model, scale 1) and
    execution noise on a model step's 2 commands (rows of the measured snippets, x ALEA_SCALE)."""
    def __init__(self, seed=SEED, bank=4096):
        import torch
        import cem_filter as cf
        import step3b_models as sm
        import uncertainty_monitor_dev as um
        import visual_pipeline_panel3 as p3
        self.torch, self.G = torch, sm.G.load(cf.V3 / "G.pt")
        E = p3.ErrorModel()
        self.d = torch.as_tensor(np.stack([np.concatenate([(p * [1, 1, 0]).reshape(9), (r * [0, 0, 1]).reshape(9)])
                                           for p, r in (E.sample(seed, 7, j) for j in range(bank))]), dtype=torch.float32)
        self.snip = torch.as_tensor(np.load(um.BENCH)["snippets"], dtype=torch.float32, device=DEV) * ALEA_SCALE   # (64, 8, 3)
        self.rng = np.random.default_rng([seed, 11])

    def starts(self, z):
        """z (B, ...) -> perceived starts (B, ...), one perturbation each."""
        d = self.d[self.rng.integers(len(self.d), size=len(z))]
        return self.G(z, d).float()

    def actions(self, a, n):
        """a (B, 8) -> (n, B, 8): each model step's 2 commands minus 2 consecutive rows of a random snippet."""
        torch = self.torch; B = len(a)
        i = torch.as_tensor(self.rng.integers(64, size=(n, B)), device=a.device)
        r = torch.as_tensor(self.rng.integers(0, 7, size=(n, B)), device=a.device)
        e0, e1 = self.snip[i, r], self.snip[i, r + 1]                                    # (n, B, 3)
        A = a[None].repeat(n, 1, 1).clone(); A[..., 0:3] -= e0; A[..., 4:7] -= e1
        return A


def load_irr(dev=None):
    """IRRM (results/jenga/irrm_protocol.json): the learned irreversibility margin l_irr(z, g, d) of eval/irrm2.py
    (LSF_IRR_MARGIN, a file in OUT), d = the actor's normalised motion (B, 6)."""
    import torch
    dev = dev or DEV
    _, Critic, _ = nets(); w = torch.load(OUT / os.environ["LSF_IRR_MARGIN"], map_location=dev)
    net = Critic().to(dev).eval(); net.load_state_dict(w["state"])
    for p_ in net.parameters():
        p_.requires_grad_(False)
    return lambda z, g, d: torch.tanh(net(z.float(), g, torch.cat([d, torch.zeros(len(d), 2, device=d.device)], 1)))


def alea_step(model, clf, noise, z, a, g2, n=None):
    """Original LSF's pessimistic sampling with our noise model: n noisy transitions of (z, a), keep the one with the
    lowest classifier margin. -> next latent (B, ...), its margin."""
    import torch
    n = n or ALEA
    A = noise.actions(a, n); B = len(z)
    p = model.step(z.repeat(n, 1, 1, 1), A.reshape(n * B, 8)).float().view(n, B, *z.shape[1:])
    lm = clf(p.flatten(0, 1), g2.repeat(n, 1)).view(n, B)
    w = lm.argmin(0); idx = torch.arange(B, device=z.device)
    return p[w, idx], lm.min(0).values


def stage_sac(updates=int(os.environ.get("LSF_UPDATES", 60000)), envs=256, horizon=15, batch=512, cap=400_000):
    """UNISafe recipe in our world model's imagination. Margin l = min(failure margin of the classifier, uncertainty
    margin); uncertainty margin = clip(2 (thr - u) / thr, -1, 1), u = ensemble disagreement of the transition,
    thr = 95th percentile of u under the POLICY's own actions on training resets (data-calibrated OOD threshold, as
    UNISafe's ood_threshold). Twin critics, entropy term in the target, learned alpha (target entropy -6), gamma
    annealed 0.85 -> 0.9999 (UNISafe StepLRMargin: period 4000 updates, decay 0.95), episodes of 15 model steps
    (30 control steps) truncated as terminal (target = l), lr 1e-4, tau 0.005."""
    import copy
    import torch
    import step3_world_model as wm
    wm.DEVICE = DEV
    torch.manual_seed(SEED)
    QNet, SACActor = sac_nets()
    clf = load_clf()
    models = ensemble()
    q1, q2, A = QNet().to(DEV), QNet().to(DEV), SACActor().to(DEV); q1t, q2t = copy.deepcopy(q1), copy.deepcopy(q2)
    oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-4); oa = torch.optim.Adam(A.parameters(), lr=1e-4)
    log_alpha = torch.zeros(1, device=DEV, requires_grad=True); oal = torch.optim.Adam([log_alpha], lr=1e-4)
    R_z = np.load(OUT / "train_reset_z.npy"); R_g = np.load(OUT / "train_reset_prop.npy")
    rng = np.random.default_rng(SEED); use_uq = os.environ.get("LSF_UQ", "1") == "1"; pess = os.environ.get("LSF_PESS") == "1"
    noise = Noise() if ALEA else None
    irr = load_irr() if os.environ.get("LSF_IRR_MARGIN") else None           # IRRM: action-level irreversibility margin (off by default)
    mode = os.environ.get("LSF_MARGIN_MODE", "frag")                          # frag (default) | irr | union
    # OOD threshold from the policy's own actions (hold-out of nothing: training resets + the recorded policy targets)
    pol = np.load(OUT / "train_reset_plan.npy") if (OUT / "train_reset_plan.npy").exists() else None
    with torch.no_grad():
        i = rng.choice(len(R_z), 2048); z = torch.as_tensor(R_z[i], device=DEV).float(); g = torch.as_tensor(R_g[i], device=DEV).float()
        a = torch.cat([g, g], 1) if pol is None else torch.as_tensor(pol[i, :2].reshape(-1, 8), device=DEV).float()
        us = torch.cat([disagreement(models, z[k:k + 256], a[k:k + 256])[1] for k in range(0, 2048, 256)])
        thr = float(torch.quantile(us, 0.95))
    print("ood threshold", thr, "from", "policy" if pol is not None else "hold", flush=True)
    buf = {"z": torch.zeros(cap, 28, 28, 8, dtype=torch.float16, device=DEV), "g": torch.zeros(cap, 4, device=DEV),
           "d": torch.zeros(cap, 6, device=DEV), "z2": torch.zeros(cap, 28, 28, 8, dtype=torch.float16, device=DEV),
           "g2": torch.zeros(cap, 4, device=DEV), "l": torch.zeros(cap, device=DEV), "done": torch.zeros(cap, dtype=torch.bool, device=DEV)}
    n, ptr, upd, log = 0, 0, 0, []
    while upd < updates:
        i = rng.choice(len(R_z), envs); z = torch.as_tensor(R_z[i], device=DEV).float(); g = torch.as_tensor(R_g[i], device=DEV).float()
        if ALEA:
            with torch.no_grad():
                z = noise.starts(z)                                          # perceived start (perception noise)
        for t in range(horizon):
            with torch.no_grad():
                d, _ = A.sample(z, g); a = plan_from_d(g, d); g2 = torch.cat([a[:, 4:7], a[:, 7:8]], 1)
                if ALEA:                                                     # worst of ALEA noisy transitions; margin of z
                    z2, _ = alea_step(models[0], clf, noise, z, a, g2); u = disagreement(models, z, a)[1]; lf = clf(z, g)
                elif pess:
                    z2, lf, u = pessimistic_step(models, clf, z, a, g2)
                else:
                    (z2, u), lf = disagreement(models, z, a), clf(z, g)
                if irr is not None:                                          # l_irr(z, g, d) of THIS transition's action
                    li = irr(z, g, d)
                    lf = li if mode == "irr" else torch.minimum(lf, li)
                l = torch.minimum(lf, (2 * (thr - u) / thr).clamp(-1, 1)) if use_uq else lf
                k = torch.arange(ptr, ptr + envs, device=DEV) % cap
                buf["z"][k], buf["g"][k], buf["d"][k], buf["z2"][k], buf["g2"][k], buf["l"][k] = z.half(), g, d, z2.half(), g2, l
                buf["done"][k] = t == horizon - 1
                ptr = (ptr + envs) % cap; n = min(n + envs, cap); z, g = z2, g2
            for _ in range(4):                                               # 4 updates per imagined step of 256 envs
                gam = 0.9999 - (0.9999 - 0.85) * 0.95 ** (upd // 4000); alpha = float(log_alpha.exp())
                j = torch.as_tensor(rng.integers(0, n, batch), device=DEV)
                zb, gb, db, z2b, g2b, lb, dn = (buf[x][j] for x in ("z", "g", "d", "z2", "g2", "l", "done"))
                with torch.no_grad():
                    d2, lp2 = A.sample(z2b, g2b)
                    vn = torch.minimum(q1t(z2b, g2b, d2), q2t(z2b, g2b, d2)) - alpha * lp2
                    y = torch.where(dn, lb, gam * torch.minimum(lb, vn) + (1 - gam) * lb)
                lq = ((q1(zb, gb, db) - y) ** 2).mean() + ((q2(zb, gb, db) - y) ** 2).mean()
                oq.zero_grad(); lq.backward(); oq.step()
                dn_, lp = A.sample(zb, gb)
                la = (alpha * lp - torch.minimum(q1(zb, gb, dn_), q2(zb, gb, dn_))).mean()
                oa.zero_grad(); la.backward(); oa.step()
                lal = -(log_alpha * (lp.detach() - 6).mean()); oal.zero_grad(); lal.backward(); oal.step()
                with torch.no_grad():
                    for net, tgt in ((q1, q1t), (q2, q2t)):
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(1 - TAU).add_(TAU * p)
                upd += 1
                if upd % 2000 == 0:
                    with torch.no_grad():
                        sat = float((A.mean(zb, gb).abs() > 0.95).float().mean())
                    log.append({"update": upd, "gamma": gam, "alpha": alpha, "critic_loss": float(lq), "actor_loss": float(la),
                                "l_mean": float(lb.mean()), "unc_fail_share": float((l < 0).float().mean()), "actor_saturated": sat})
                    print(log[-1], flush=True)
    torch.save({"algo": "sac", "q1": q1.state_dict(), "q2": q2.state_dict(), "actor": A.state_dict(), "ood_threshold": thr, "uq": use_uq, "pess": pess,
                "alea": ALEA, "alea_scale": ALEA_SCALE, "margin_mode": mode, "irr_margin": os.environ.get("LSF_IRR_MARGIN")},
               OUT / os.environ.get("LSF_VALUE_NAME", "value_sac.pt"))
    json.dump(log, open(OUT / f"{os.environ.get('LSF_VALUE_NAME', 'value_sac.pt')[:-3]}_log.json", "w"))


if __name__ == "__main__":
    {"data": stage_data, "merge": stage_merge, "classifier": stage_classifier, "value": stage_value, "sac": stage_sac,
     "ddpg_pub": stage_ddpg_pub}[sys.argv[1]]()
