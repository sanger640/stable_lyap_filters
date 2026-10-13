"""Amendment 13: joint three-task model. Deterministic stream (US recipe) then residual flow head, both trained on
three-block + T2 + Phase-2 F1/F2 data with batches balanced 1/3 per task; graded on each task's selection set.

Stages (run on the 5080 machine): pack | det | head | grade
"""
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

OUT = ROOT / "results/program_b/joint"
PB = ROOT / "results/program_b"
TASKS = ("T1", "T2", "P2")
DET_STEPS, HEAD_STEPS = 67_500, 40_000
RESULT = ROOT / "results/jenga/joint_hybrid_result.json"


def sources():
    """(task, file paths, modes dict keyed by file NAME (folder/name to stay unique))."""
    key = lambda p: f"{Path(p).parent.name}/{Path(p).name}"
    def modes_of(path):
        return {key(k): v for k, v in json.loads(Path(path).read_text()).items()}
    t1 = [p for f in ("step3b/train", "phase1/train") for p in sorted((PB / f).glob("ep*.npz")) if not p.name.endswith(".tmp.npz")]
    t2 = [p for p in sorted((PB / "t2/train").glob("*.npz")) if not p.name.endswith(".tmp.npz")]
    p2 = [p for p in sorted((PB / "phase2/train").glob("*.npz")) if not p.name.endswith(".tmp.npz")]
    return [("T1", t1, modes_of(PB / "phase1e/modes.json")), ("T2", t2, modes_of(PB / "t2/modes.json")),
            ("P2", p2, modes_of(PB / "phase2/modes.json"))], key


def pack():
    import phase1d_models as p1d
    OUT.mkdir(parents=True, exist_ok=True)
    srcs, key = sources()
    files = [(t, p, m) for t, ps, m in srcs for p in ps]
    counts = []
    for _, p, _ in files:
        with np.load(p) as d:
            counts.append(len(d["z"]))
    F, S = int(sum(counts)), len(files)
    z = np.lib.format.open_memmap(OUT / "z.npy", "w+", np.float16, (F, 19, 28, 28, 8))
    z0k = np.lib.format.open_memmap(OUT / "z0k.npy", "w+", np.float16, (F, 28, 28, 8))
    act = np.lib.format.open_memmap(OUT / "actions.npy", "w+", np.float32, (F, 19, 8))
    z0 = np.zeros((S, 28, 28, 8), np.float16); state_of = np.zeros(F, np.int64); task = np.zeros(F, np.int8)
    rare, has_rare = np.zeros(F, bool), np.zeros(F, bool)
    i, missing = 0, 0
    for si, (t, p, modes) in enumerate(files):
        with np.load(p) as d:
            n = len(d["z"])
            z[i:i + n] = d["z"]; z0k[i:i + n] = d["z0k"]; act[i:i + n] = p1d.augment(d["actions"], None, False)
            z0[si] = d["z0"]
        m = modes.get(key(p))
        if m is None:
            missing += 1; m = []
        state_of[i:i + n] = si; task[i:i + n] = TASKS.index(t)
        if m:
            rare[i + np.array(m, int)] = True
        has_rare[i:i + n] = bool(m)
        i += n
        if si % 1000 == 0:
            print(f"pack {si}/{S}", flush=True)
    z.flush(); z0k.flush(); act.flush()
    for name, arr in (("z0", z0), ("state_of", state_of), ("task", task), ("rare", rare), ("has_rare", has_rare)):
        np.save(OUT / f"{name}.npy", arr)
    summary = {"states": S, "futures": F, "per_task_states": {t: sum(1 for x in files if x[0] == t) for t in TASKS},
               "missing_modes": missing}
    (OUT / "pack.json").write_text(json.dumps(summary) + "\n")
    fit_gauss_te()
    print(json.dumps(summary), flush=True)


def fit_gauss_te(radius=4, max_futures=60_000):
    """Joint Gauss-TE (same estimator as phase2_p0.fit_te), from a uniform subsample of futures."""
    import torch
    z0k = np.load(OUT / "z0k.npy", mmap_mode="r"); z0 = np.load(OUT / "z0.npy"); so = np.load(OUT / "state_of.npy")
    rng = np.random.default_rng(20261330)
    idx = np.sort(rng.choice(len(so), min(max_futures, len(so)), replace=False))
    P = torch.zeros(28, 28, 8, 8, dtype=torch.complex128, device="cuda"); n = 0
    for b in range(0, len(idx), 2048):
        i = idx[b:b + 2048]
        d = (torch.from_numpy(np.ascontiguousarray(z0k[i])).float() - torch.from_numpy(z0[so[i]]).float()).cuda().permute(0, 3, 1, 2).double()
        f = torch.fft.fft2(d).permute(0, 2, 3, 1)
        P += torch.einsum("kabi,kabj->abij", f, f.conj()); n += len(d)
    P /= n * 28 * 28
    c = torch.fft.ifft2(P.permute(2, 3, 0, 1)).real.permute(2, 3, 0, 1)
    lag = torch.fft.fftfreq(28, 1 / 28, device="cuda").double()
    rb = torch.round(torch.sqrt(lag[:, None] ** 2 + lag[None, :] ** 2)).long()
    iso = torch.zeros_like(c)
    for b in range(radius + 1):
        m = rb == b; iso[m] = c[m].mean(0)
    iso = 0.5 * (iso + iso.transpose(-1, -2).flip(0, 1).roll((1, 1), (0, 1)))
    S = torch.fft.fft2(iso.permute(2, 3, 0, 1)).permute(2, 3, 0, 1); S = 0.5 * (S + S.conj().transpose(-1, -2))
    w, v = torch.linalg.eigh(S)
    root = v @ torch.diag_embed(w.clamp_min(0).sqrt().to(v.dtype)) @ v.conj().transpose(-1, -2)
    torch.save({"root": root.cpu(), "lag0_cov": c[0, 0].cpu(), "iso_lag0_cov": iso[0, 0].cpu()}, OUT / "Gauss_TE_joint.pt")


class Data:
    def __init__(self):
        import torch
        self.z = np.load(OUT / "z.npy", mmap_mode="r"); self.z0k = np.load(OUT / "z0k.npy", mmap_mode="r")
        self.act = np.load(OUT / "actions.npy", mmap_mode="r")
        self.z0 = torch.from_numpy(np.load(OUT / "z0.npy")).cuda(); self.state_of = np.load(OUT / "state_of.npy")
        task, rare, has = np.load(OUT / "task.npy"), np.load(OUT / "rare.npy"), np.load(OUT / "has_rare.npy")
        self.pools = [(np.flatnonzero(rare & (task == t)), np.flatnonzero(~rare & has & (task == t))) for t in range(len(TASKS))]

    def sample(self, rng, batch):
        """Balanced 1/3 per task; within each task half rare-mode, half sibling majority (label-free US rule)."""
        parts, per = [], [batch // 3 + (1 if t < batch % 3 else 0) for t in range(3)]
        for (r, s), n in zip(self.pools, per):
            parts += [rng.choice(r, n // 2), rng.choice(s, n - n // 2)]
        return np.sort(np.concatenate(parts))

    def batch(self, i):
        import torch
        z0 = self.z0[torch.as_tensor(self.state_of[i], device="cuda")].float()
        return (z0, torch.from_numpy(np.ascontiguousarray(self.z0k[i])).cuda().float(),
                torch.from_numpy(np.ascontiguousarray(self.z[i])).cuda().float(),
                torch.from_numpy(np.ascontiguousarray(self.act[i])).cuda())


def det(seed=0, log_every=250):
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    import fast_train as ft
    import phase2_models as pm
    import step3_world_model as wm
    import step3b_models as sm
    path = OUT / "model_DET_joint.pt"
    if path.exists():
        return
    data = Data(); te = pm.GaussTE(torch.load(OUT / "Gauss_TE_joint.pt"))
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    so = data.state_of
    idx = np.concatenate([np.flatnonzero(np.load(OUT / "task.npy") == t)[:4000] for t in range(3)])   # stats: ~equal per task
    seq = torch.cat([torch.from_numpy(np.ascontiguousarray(data.z0k[idx]))[:, None].float(), torch.from_numpy(np.ascontiguousarray(data.z[idx])).float()], 1)
    stats = wm.Stats(seq, torch.from_numpy(np.ascontiguousarray(data.act[idx]))); del seq
    model = wm.WorldModel("deterministic", stats, 8)
    net = torch.compile(model.net)
    opt = torch.optim.AdamW(model.net.parameters(), lr=ft.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, DET_STEPS)
    wts = torch.ones(wm.STEPS, device="cuda"); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT
    zt = torch.zeros(ft.BATCH, device="cuda")
    log, t0 = [{"steps": DET_STEPS, "batch": ft.BATCH}], time.time()

    def step_fn(inp, a):
        return net(inp, torch.zeros_like(inp), a, zt)
    for step in range(DET_STEPS):
        z0, rend, target, act = data.batch(data.sample(rng, ft.BATCH))
        a = model._act(act)
        use_g = torch.rand(ft.BATCH, device="cuda") < 0.5
        with torch.no_grad():
            start = torch.where(use_g[:, None, None, None], te(z0), rend)
        model.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, loss = start, 0.0
            for t in range(wm.STEPS):
                inp = cur + torch.randn_like(cur) * sm.INPUT_NOISE * stats.res_std
                cur = cur + checkpoint(step_fn, inp, a[:, t], use_reentrant=False).float() * stats.res_std
                loss = loss + wts[t] * ((cur - target[:, t]) ** 2).mean()
            loss = loss / wts.sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % log_every == 0:
            log.append({"step": step + 1, "loss": float(loss), "seconds": round(time.time() - t0, 1)})
            print(f"DET_joint step {step + 1}: loss {float(loss):.5f} ({time.time() - t0:.0f}s)", flush=True)
    model.net.eval(); model.save(path)
    (OUT / "model_DET_joint_log.json").write_text(json.dumps(log) + "\n")
    print("trained DET_joint", flush=True)


def head(seed=0, log_every=250):
    import torch
    from torch import nn
    import phase2_hybrid as hy
    import step3_world_model as wm
    path = OUT / "model_HYB_joint.pt"
    if path.exists():
        return
    data = Data()
    det_m = wm.WorldModel.load(OUT / "model_DET_joint.pt"); det_m.net.eval()
    for p in det_m.net.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    h = hy.build_head(8).cuda()
    opt = torch.optim.AdamW(h.parameters(), lr=hy.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / hy.WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / HEAD_STEPS))))

    def resid():
        z0, _, target, act = data.batch(data.sample(rng, hy.BATCH))
        a = det_m._act(act)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zd = hy.det_endpoints(det_m, z0, a).float()
        return z0, a, zd, torch.stack([target[:, wm.HOLD10], target[:, wm.HOLD30]], 1) - zd
    rscale = float(np.mean([float(resid()[3].std()) for _ in range(20)]))
    log, t0 = [{"steps": HEAD_STEPS, "residual_scale": rscale}], time.time()
    for step in range(HEAD_STEPS):
        z0, a, zd, r = resid(); r = r / rscale
        noise = torch.randn_like(r); t = hy.shift(torch.rand(len(r), device="cuda"))
        rt = (1 - t)[:, None, None, None, None] * noise + t[:, None, None, None, None] * r
        h.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = h(rt, t, z0, zd / rscale, a).float()
        loss = ((v - (r - noise)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(h.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % log_every == 0:
            log.append({"step": step + 1, "loss": float(loss), "seconds": round(time.time() - t0, 1)})
            print(f"HYB_joint step {step + 1}: flow loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state_dict": h.state_dict(), "action_dim": 8, "residual_scale": rscale, "det": "DET_joint"}, path)
    (OUT / "model_HYB_joint_log.json").write_text(json.dumps(log) + "\n")
    print("trained HYB_joint", flush=True)


def grade():
    import torch
    import phase1d_models as p1d
    import phase2_hybrid as hy
    import phase2_models as pm
    import step3_world_model as wm
    import step3b_models as sm
    from fresh_fragility_panel import topple_count
    from phase2_amend4 import alarm, score
    det_m = wm.WorldModel.load(OUT / "model_DET_joint.pt"); det_m.net.eval()
    s = torch.load(OUT / "model_HYB_joint.pt", map_location="cuda")
    h = hy.build_head(8).cuda(); h.load_state_dict(s["state_dict"]); h.eval()
    te = pm.GaussTE(torch.load(OUT / "Gauss_TE_joint.pt"))
    sets = {"T1_H": (PB / "phase1/h", "ep*.npz"), "T2_H": (PB / "t2/h2", "*.npz"), "P2_H2": (PB / "phase2/h2", "*.npz")}
    out = {}
    for name, (folder, pat) in sets.items():
        fork, al_det, al_hyb = [], [], []
        for n, path in enumerate(sorted(folder.glob(pat))):
            with np.load(path) as f:
                if name == "T1_H":
                    k = topple_count(f["traces"], f["start_tilt"]); acts = p1d.augment(f["actions"], None, False)
                else:
                    k = int(f["harm"].sum()); acts = np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, 8)
                z0 = torch.as_tensor(f["z0"], device="cuda").float()
            fork.append(2 <= k <= 62)
            g = torch.Generator(device="cuda").manual_seed(pm.SEEDS["monitor"] + n)
            st = te(z0[None].expand(64, -1, -1, -1).contiguous(), g)
            e10, e30 = sm.roll(det_m, st, acts); al_det.append(alarm(e10, e30, "BOTH"))
            g2 = torch.Generator(device="cuda").manual_seed(hy.MONITOR_SEED + n)
            fut = hy.sample(h, det_m, z0[None].expand(64, -1, -1, -1).contiguous(), det_m._act(torch.as_tensor(acts, device="cuda")),
                            s["residual_scale"], g2)
            al_hyb.append(alarm(fut[:, 0].reshape(64, -1).double().cpu().numpy(), fut[:, 1].reshape(64, -1).double().cpu().numpy(), "BOTH"))
            torch.cuda.empty_cache()
        fork = np.array(fork)
        out[name] = {"DET_joint": score(np.array(al_det), fork), "HYB_joint": score(np.array(al_hyb), fork)}
        RESULT.write_text(json.dumps(out, indent=1) + "\n")
        print(name, json.dumps(out[name]), flush=True)


if __name__ == "__main__":
    {"pack": pack, "det": det, "head": head, "grade": grade}[sys.argv[1]]()
