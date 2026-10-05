"""Branching-WM Phase 1b arms on the initial-condition route, plus the object-centric test and the
hidden-cause theory check, evaluated on the Phase-1 held-out set H.

Frozen protocol: results/jenga/phase1b_branching_protocol.json.
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
import step3_world_model as wm  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase1b_branching_protocol.json"
OUT = ROOT / "results/program_b/phase1b"
RESULT = ROOT / "results/jenga/phase1b_selection.json"
DEVICE = "cuda"
FOLDERS_ENRICHED = [ROOT / "results/program_b/step3b/train", ROOT / "results/program_b/phase1/train"]
FOLDERS_MORE = FOLDERS_ENRICHED + [OUT / "train"]
LONG, SHORT, BATCH = 90_000, 30_000, 32


# ------------------------------------------------------- initial-condition latent recipe (cell E)
def train_ic(name, folders, steps, sensitivity=False, seed=0):
    """Cell-E recipe; with `sensitivity`, half of each batch is 16 states x 2 futures and the predicted
    hold-30 latent difference of each pair must match the real difference."""
    import step3b_models as sm
    path = OUT / f"model_{name}.pt"
    if path.exists():
        return wm.WorldModel.load(path)
    torch.manual_seed(seed); np.random.seed(seed)
    states = p1.load_states(folders)
    g = sm.G.load(sm.DATA / "G.pt")
    rows = [(si, k) for si, s in enumerate(states) for k in range(len(s["z"]))]
    seq = torch.cat([torch.cat([s["z0k"][:, None], s["z"]], 1) for s in states[:400]])
    pairs = torch.cat([s["actions"].reshape(len(s["z"]), wm.STEPS, -1) for s in states[:400]])
    stats = wm.Stats(seq, pairs)
    model = wm.WorldModel("deterministic", stats, pairs.shape[-1])
    opt = torch.optim.AdamW(model.net.parameters(), lr=wm.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    wts = torch.ones(wm.STEPS, device=DEVICE); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT
    log = []
    for step in range(steps):
        if sensitivity:
            si = np.random.randint(len(states), size=BATCH // 2)
            pick = [(i, int(k)) for i in si for k in np.random.choice(len(states[i]["z"]), 2, replace=False)]
        else:
            pick = [rows[i] for i in np.random.randint(len(rows), size=BATCH)]
        z0 = torch.stack([states[i]["z0"] for i, _ in pick]).to(DEVICE).float()
        rend = torch.stack([states[i]["z0k"][k] for i, k in pick]).to(DEVICE).float()
        dd = torch.stack([states[i]["d"][k] for i, k in pick]).to(DEVICE)
        target = torch.stack([states[i]["z"][k] for i, k in pick]).to(DEVICE).float()
        act = model._act(torch.stack([states[i]["actions"][k] for i, k in pick]).to(DEVICE).reshape(BATCH, wm.STEPS, -1))
        use_g = torch.rand(BATCH, device=DEVICE) < 0.5
        start = rend.clone()
        if use_g.any():
            start[use_g] = g(z0[use_g], dd[use_g])
        model.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, loss, hold30 = start, 0.0, None
            for t in range(wm.STEPS):
                inp = cur + torch.randn_like(cur) * sm.INPUT_NOISE * stats.res_std
                r = model.net(inp, torch.zeros_like(inp), act[:, t], torch.zeros(BATCH, device=DEVICE))
                cur = cur + r.float() * stats.res_std
                loss = loss + wts[t] * ((cur - target[:, t]) ** 2).mean()
                if t == wm.HOLD30:
                    hold30 = cur
            loss = loss / wts.sum()
            if sensitivity:
                dp = hold30[0::2] - hold30[1::2]
                dr = target[0::2, wm.HOLD30] - target[1::2, wm.HOLD30]
                loss = loss + ((dp - dr) ** 2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            log.append({"step": step + 1, "loss": float(loss)})
            print(f"{name} step {step + 1}: loss {float(loss):.5f}", flush=True)
    model.net.eval()
    model.save(path)
    (OUT / f"model_{name}_log.json").write_text(json.dumps(log) + "\n")
    return model


# ------------------------------------------------------------------- object-centric arm (c)
def rot6(six):
    c0 = six[..., [0, 2, 4]]; c1 = six[..., [1, 3, 5]]
    c0 = c0 / np.linalg.norm(c0, axis=-1, keepdims=True)
    c1 = c1 - (c0 * c1).sum(-1, keepdims=True) * c0
    c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    return np.stack([c0, c1, np.cross(c0, c1)], -1)


def perturb_state(start, d_pos, d_rot):
    """Shift the three block poses of a 61-D start state by a perception-error draw (as perturb_blocks)."""
    from scipy.spatial.transform import Rotation
    s = start.copy()
    s[:9] += d_pos.reshape(9)
    for b in range(3):
        r = rot6(s[9 + 6 * b:15 + 6 * b])
        r2 = Rotation.from_rotvec(d_rot[b]).as_matrix() @ r
        s[9 + 6 * b:15 + 6 * b] = np.array([r2[0, 0], r2[0, 1], r2[1, 0], r2[1, 1], r2[2, 0], r2[2, 1]])
    return s


def object_rows(folders):
    rows = []
    for folder in folders:
        for path in sorted(Path(folder).glob("ep*.npz")):
            start_file = path.with_name(path.stem + ".start.npy")
            if not start_file.exists() or "_s" not in path.stem:
                continue
            start = np.load(start_file)
            with np.load(path) as f:
                for k in range(len(f["traces"])):
                    rows.append((perturb_state(start, f["d_pos"][k], f["d_rot"][k]).astype(np.float32),
                                 f["actions"][k].astype(np.float32), f["traces"][k].astype(np.float32)))
    return rows


class ObjectDynamics(nn.Module):
    def __init__(self, dim=61, action=4, width=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + action, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, dim))

    def forward(self, x, a):
        return self.net(torch.cat([x, a], -1))


def train_object(seed=0):
    path = OUT / "model_c.pt"
    if path.exists():
        return torch.load(path, map_location=DEVICE, weights_only=False)
    import step3b_models as sm
    torch.manual_seed(seed); np.random.seed(seed)
    rows = object_rows(FOLDERS_ENRICHED)
    starts = torch.from_numpy(np.stack([r[0] for r in rows])); acts = torch.from_numpy(np.stack([r[1] for r in rows]))
    traces = torch.from_numpy(np.stack([r[2] for r in rows]))
    full = torch.cat([starts[:, None], traces], 1)
    s_mean, s_std = full.reshape(-1, 61).mean(0), full.reshape(-1, 61).std(0).clamp_min(1e-6)
    res_std = (full[:, 1:] - full[:, :-1]).reshape(-1, 61).std(0).clamp_min(1e-6)
    a_mean, a_std = acts.reshape(-1, 4).mean(0), acts.reshape(-1, 4).std(0).clamp_min(1e-6)
    stats = {k: v.to(DEVICE) for k, v in dict(s_mean=s_mean, s_std=s_std, res_std=res_std, a_mean=a_mean, a_std=a_std).items()}
    net = ObjectDynamics().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, SHORT)
    wts = torch.ones(38, device=DEVICE); wts[[17, 37]] = sm.HOLD_WEIGHT
    log = []
    for step in range(SHORT):
        idx = torch.randint(len(rows), (BATCH,))
        cur, a, tgt = starts[idx].to(DEVICE), acts[idx].to(DEVICE), traces[idx].to(DEVICE)
        loss = 0.0
        for t in range(38):
            inp = (cur + torch.randn_like(cur) * sm.INPUT_NOISE * stats["res_std"] - stats["s_mean"]) / stats["s_std"]
            cur = cur + net(inp, (a[:, t] - stats["a_mean"]) / stats["a_std"]) * stats["res_std"]
            loss = loss + wts[t] * (((cur - tgt[:, t]) / stats["s_std"]) ** 2).mean()
        loss = loss / wts.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            log.append({"step": step + 1, "loss": float(loss)})
            print(f"c step {step + 1}: loss {float(loss):.5f}", flush=True)
    saved = {"net": net, "stats": stats}
    torch.save(saved, path)
    (OUT / "model_c_log.json").write_text(json.dumps(log) + "\n")
    return saved


@torch.no_grad()
def object_traces(saved, start, d_pos, d_rot, actions):
    net, st = saved["net"].eval(), saved["stats"]
    cur = torch.as_tensor(np.stack([perturb_state(start, d_pos[k], d_rot[k]) for k in range(64)]), device=DEVICE).float()
    a = torch.as_tensor(actions, device=DEVICE).float()
    out = []
    for t in range(38):
        cur = cur + net((cur - st["s_mean"]) / st["s_std"], (a[:, t] - st["a_mean"]) / st["a_std"]) * st["res_std"]
        out.append(cur)
    return torch.stack(out, 1).cpu().numpy()


def evaluate_object(saved):
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings
    from fresh_fragility_panel import topple_count
    alarms, fork = [], []
    for path in sorted((p1.OUT / "h").glob("ep*.npz")):
        start = np.load(path.with_name(path.stem + ".start.npy"))
        with np.load(path) as f:
            fork.append(2 <= topple_count(f["traces"], f["start_tilt"]) <= 62)
            pred = object_traces(saved, start, f["d_pos"], f["d_rot"], f["actions"])
            alarms.append(decide(*trace_endings(pred, NON_ROBOT["jenga"]))[0])
    a, y = np.array(alarms), np.array(fork)
    return {"caught": float((a & y).sum() / y.sum()), "false_alarm": float((a & ~y).sum() / (~y).sum()),
            "alarms": int(a.sum()), "forks": int(y.sum()), "states": len(y)}


# --------------------------------------------------------------- hidden-cause theory check (d)
def train_d(seed=0):
    """Phase-1 whole-clip flow, conditioned on each future's perturbed start latent z0k instead of z0."""
    path = OUT / "model_d.pt"
    if path.exists():
        return p1.ClipModel.load(path)
    torch.manual_seed(seed); np.random.seed(seed)
    states = p1.load_states(FOLDERS_ENRICHED)
    norm = p1.Norm(states)
    model = p1.ClipModel("flow", norm, 2 * states[0]["actions"].shape[-1])
    opt = torch.optim.AdamW(model.net.parameters(), lr=p1.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, SHORT)
    w = p1.weights()[None, :, None, None, None]
    log = []
    for step in range(SHORT):
        idx = np.random.randint(len(states), size=BATCH)
        ks = [np.random.randint(len(states[i]["z"])) for i in idx]
        z0 = torch.stack([states[i]["z0k"][k] for i, k in zip(idx, ks)]).to(DEVICE).float()
        traj = torch.stack([states[i]["z"][k] for i, k in zip(idx, ks)]).to(DEVICE).float()
        acts = model._act(torch.stack([states[i]["actions"][k] for i, k in zip(idx, ks)]).to(DEVICE))
        x1 = (traj - z0[:, None]) / norm.res_std
        x0 = torch.randn_like(x1)
        tau = torch.rand(BATCH, device=DEVICE)
        xt = (1 - tau[:, None, None, None, None]) * x0 + tau[:, None, None, None, None] * x1
        model.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model.net(z0, acts, xt, tau)
        loss = (w * (v.float() - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.net.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            log.append({"step": step + 1, "loss": float(loss)})
            print(f"d step {step + 1}: loss {float(loss):.5f}", flush=True)
    model.save(path)
    (OUT / "model_d_log.json").write_text(json.dumps(log) + "\n")
    return model


def evaluate_d(model, rows):
    """As p1.evaluate, but each rollout is conditioned on its G start (the deployable z0k proxy)."""
    import step3b_models as sm
    gop = sm.G.load(sm.DATA / "G.pt")

    @torch.no_grad()
    def endings(cell, m, row, seed):
        g = torch.Generator(device=DEVICE).manual_seed(int(seed))
        z0 = torch.as_tensor(row["z0"], device=DEVICE).float()
        starts = gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(row["d"]).float())
        acts = torch.as_tensor(row["actions"], device=DEVICE)
        traj = torch.cat([m.sample(starts[i:i + 16], acts[i:i + 16], g) for i in range(0, 64, 16)])
        return (traj[:, wm.HOLD10].reshape(64, -1).double().cpu().numpy(),
                traj[:, wm.HOLD30].reshape(64, -1).double().cpu().numpy())

    original = p1.model_endings
    p1.model_endings = endings
    try:
        return p1.evaluate("d", model, rows)
    finally:
        p1.model_endings = original


# ------------------------------------------------------------------------------- driver
def undertrained(name, steps):
    log = json.loads((OUT / f"model_{name}_log.json").read_text())
    last = np.mean([r["loss"] for r in log if steps - 3000 < r["step"] <= steps])
    before = np.mean([r["loss"] for r in log if steps - 6000 < r["step"] <= steps - 3000])
    return bool(last < before - 0.02 * abs(before))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("a1", "a2", "b", "c", "d", "combined"))
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    results = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    rows = p1.h_states()
    e_ref = json.loads((ROOT / "results/jenga/phase1_selection.json").read_text())["E"]
    if args.arm == "c":
        saved = train_object()
        results["c"] = evaluate_object(saved) | {"possibly_undertrained": undertrained("c", SHORT),
                                                 "note": "privileged object-centric start state; not eligible for panel 6"}
    elif args.arm == "d":
        results["d"] = evaluate_d(train_d(), rows) | {"possibly_undertrained": undertrained("d", SHORT)}
    elif args.arm in ("a1", "a2", "b"):
        folders, steps, sens = {"a1": (FOLDERS_ENRICHED, LONG, False), "a2": (FOLDERS_MORE, LONG, False),
                                "b": (FOLDERS_ENRICHED, SHORT, True)}[args.arm]
        model = train_ic(args.arm, folders, steps, sens)
        results[args.arm] = p1.evaluate("E", model, rows) | {"possibly_undertrained": undertrained(args.arm, steps)}
    else:
        beats = {a: p1.score(results[a]) > p1.score(e_ref) for a in ("a1", "a2", "b")}
        results["combined_rule"] = beats
        if any(beats.values()):
            folders = FOLDERS_MORE if p1.score(results["a2"]) > p1.score(results["a1"]) else FOLDERS_ENRICHED
            model = train_ic("combined", folders, LONG, sensitivity=beats["b"])
            results["combined"] = p1.evaluate("E", model, rows) | {"possibly_undertrained": undertrained("combined", LONG)}
        eligible = [a for a in ("a1", "a2", "b", "d", "combined") if a in results and isinstance(results[a], dict)]
        best = max(eligible, key=lambda a: p1.score(results[a]))
        results["selected"] = best
        results["gate_for_panel_6"] = bool(results[best]["caught"] >= 0.80 and results[best]["false_alarm"] <= 0.20)
        results["E_reference"] = e_ref
    RESULT.write_text(json.dumps(results, indent=1) + "\n")
    print(args.arm, json.dumps({k: v for k, v in results.items() if k == args.arm or k in ("selected", "gate_for_panel_6")}),
          flush=True)


if __name__ == "__main__":
    main()
