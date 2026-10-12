"""Phase 2 amendment 8: transformer predictor (T) and transformer + auxiliary future-pixel head (T_pix).

Frozen protocol: results/jenga/phase2_generality_protocol.json (amendment_8).
  frames  re-render the training futures' close-camera frames at hold 10 / hold 30 (112x112) with the identical
          seeds and order as eval/phase2_data.py; traces verified against the stored ones
  train   --arm T | T_pix   (US2_long recipe with the transformer predictor)
  grade   H2, BOTH readout, own floor, beside the conv arms
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_data as pdata  # noqa: E402
import phase2_models as pm  # noqa: E402
import phase2_scenes as ps  # noqa: E402

OUT = pm.OUT
FRAMES = OUT / "train_frames"
RESULT = ROOT / "results/jenga/phase2_transformer_result.json"
PIX_INDEX = (17, 37)                  # trace indices of hold 10 / hold 30 (model steps HOLD10, HOLD30)
PIX_WEIGHT = 0.01


# ------------------------------------------------------------------------------------------- frames
def frames_job(job):
    import traceback
    try:
        return job[0], job[1], _frames(*job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _frames(family, seed, roles):
    """Mirror of phase2_data._render (same chain, seeds and future order); renders instead of encoding."""
    import mujoco
    import uncertainty_monitor_dev as um
    from PIL import Image
    from jenga_render_frames import close_camera
    scene, sim, acts, starts = pdata.prepare(family, seed)
    snippets = np.load(um.BENCH)["snippets"]
    pos_cov, rot_cov = ps.pooled_error_model()
    n_obj = len(scene["objects"])
    renderer, camera = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco)
    worst = 0.0
    try:
        for index, act in enumerate(acts):
            if index in roles:
                path = FRAMES / f"{family}_{seed}_s{index}.npz"
                snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy()
                chunk = acts[index:index + 8]
                n = pdata.FUTURES[roles[index]]
                frames, traces = [], []
                for k in range(n):
                    key = ({"F1": 1, "F2": 2}[family], seed, index, k)
                    noise = snippets[np.random.default_rng([pdata.SEEDS["futures"], *key]).integers(len(snippets))]
                    err = ps.draw_error(n_obj, pos_cov, rot_cov, pdata.SEEDS["futures"], *key)
                    sim.restore(snap); sim.perturb(*err)
                    renderer.update_scene(sim.data, camera=camera); renderer.render()   # as the original (start render)
                    a = np.asarray(chunk, np.float32).copy(); a[:, :3] -= noise
                    tr, fr = [], []
                    for t, c in enumerate(list(a) + [chunk[-1]] * 30):
                        sim.execute(c); tr.append(sim.object_state())
                        if t in pdata.FRAME_INDEX:
                            renderer.update_scene(sim.data, camera=camera); img = renderer.render().copy()
                            if t in PIX_INDEX:
                                fr.append(np.asarray(Image.fromarray(img).resize((112, 112), Image.BILINEAR)))
                    traces.append(np.stack(tr)); frames.append(np.stack(fr))
                with np.load(OUT / "train" / f"{family}_{seed}_s{index}.npz") as f:
                    worst = max(worst, float(np.abs(np.stack(traces).astype(np.float32) - f["traces"]).max()))
                np.savez_compressed(path.with_suffix(".tmp.npz"), frames=np.stack(frames))
                path.with_suffix(".tmp.npz").replace(path)
                sim.restore(snap); sim.data.qacc_warmstart[:] = warm
                if index == max(roles):
                    break
            sim.execute(act)
    finally:
        renderer.close()
    return worst


def render_frames(workers):
    FRAMES.mkdir(parents=True, exist_ok=True)
    sel = json.loads((OUT / "selection.json").read_text())["roles"]
    jobs = []
    for key, roles in sel.items():
        fam, seed = key.split(":")
        roles = {int(i): r for i, r in roles.items()}
        if not all((FRAMES / f"{fam}_{seed}_s{i}.npz").exists() for i in roles):
            jobs.append((fam, int(seed), roles))
    res = pdata.pool_run(frames_job, jobs, workers, "frames")
    worst = max(res.values()) if res else 0.0
    (OUT / "train_frames_check.json").write_text(json.dumps({"max_abs_trace_difference": worst, "jobs": len(res)}) + "\n")
    print("max abs trace difference", worst, flush=True)


# ------------------------------------------------------------------------------------------- model
def build_net(action_dim, width=384, depth=6, heads=6, pix=False):
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
            nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)          # AdaLN-Zero

        def forward(self, x, c):
            s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
            h = self.n1(x) * (1 + s1) + b1
            x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
            h = self.n2(x) * (1 + s2) + b2
            return x + g2 * self.mlp(h)

    class TransNet(nn.Module):
        """Same interface as step3_world_model.Transition (deterministic use): f(z, x, actions, tau) -> residual."""

        def __init__(self):
            super().__init__()
            self.cfg = {"width": width, "depth": depth, "heads": heads, "pix": pix}
            self.inp = nn.Linear(32, width)
            self.pos = nn.Parameter(torch.zeros(1, 196, width)); nn.init.trunc_normal_(self.pos, std=0.02)
            self.reg = nn.Parameter(torch.zeros(1, 4, width)); nn.init.trunc_normal_(self.reg, std=0.02)
            self.cond = nn.Sequential(nn.Linear(action_dim + 1, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
            self.blocks = nn.ModuleList(Block() for _ in range(depth))
            self.norm = nn.LayerNorm(width, elementwise_affine=False)
            self.out = nn.Linear(width, 32)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
            self.pix = nn.Linear(width, 192) if pix else None
            self.cond_in_features = action_dim + 1

        def tokens(self, z, actions, tau):
            b = len(z)
            t = z.reshape(b, 14, 2, 14, 2, 8).permute(0, 1, 3, 2, 4, 5).reshape(b, 196, 32)
            x = torch.cat([self.reg.expand(b, -1, -1), self.inp(t) + self.pos], 1)
            c = self.cond(torch.cat([actions, tau[:, None]], 1))
            for blk in self.blocks:
                x = blk(x, c)
            return self.norm(x)[:, 4:]

        def forward(self, z, x, actions, tau, return_tokens=False):
            h = self.tokens(z, actions, tau)
            b = len(z)
            r = self.out(h).reshape(b, 14, 14, 2, 2, 8).permute(0, 1, 3, 2, 4, 5).reshape(b, 28, 28, 8)
            return (r, h) if return_tokens else r
    return TransNet()


def make_model(stats, action_dim, pix):
    import step3_world_model as wm
    m = wm.WorldModel("deterministic", stats, action_dim)
    m.net = build_net(action_dim, pix=pix).to("cuda")
    return m


def save_model(m, path):
    import torch
    torch.save({"arch": "transformer", "cfg": m.net.cfg, "state_dict": m.net.state_dict(), "stats": m.stats.state(),
                "action_dim": m.net.cond_in_features - 1}, path)


def load_model(path):
    import torch
    import step3_world_model as wm
    s = torch.load(path, map_location="cuda", weights_only=False)
    m = wm.WorldModel("deterministic", wm.Stats.load(s["stats"]), s["action_dim"])
    m.net = build_net(s["action_dim"], width=s["cfg"]["width"], depth=s["cfg"]["depth"], heads=s["cfg"]["heads"],
                      pix=s["cfg"]["pix"]).cuda()
    m.net.load_state_dict(s["state_dict"]); m.net.eval()
    return m


def pixel_targets(frames):
    """(B, 112, 112, 3) uint8 -> (B, 196, 192) per-patch normalised RGB."""
    import torch
    x = frames.float() / 255.0
    b = len(x)
    p = x.reshape(b, 14, 8, 14, 8, 3).permute(0, 1, 3, 2, 4, 5).reshape(b, 196, 192)
    return (p - p.mean(-1, keepdim=True)) / (p.std(-1, keepdim=True) + 1e-6)


# ------------------------------------------------------------------------------------------- training
def train(arm, steps=22_500, seed=0, log_every=250):
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    import fast_train as ft
    import step3_world_model as wm
    import step3b_models as sm
    path = OUT / f"model_{arm}.pt"
    if path.exists():
        return
    pix = arm == "T_pix"
    states, names = pm.load_train()
    te = pm.GaussTE(torch.load(OUT / "Gauss_TE_phase2.pt"))
    modes = json.loads((OUT / "modes.json").read_text())
    rare = np.array([k in set(modes[n]) for n, s in zip(names, states) for k in range(len(s["z"]))])
    has_rare = np.array([bool(modes[n]) for n, s in zip(names, states) for _ in range(len(s["z"]))])
    pools = (np.flatnonzero(rare), np.flatnonzero(~rare & has_rare))
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    seq = torch.cat([torch.cat([s["z0k"][:, None], s["z"]], 1) for s in states[:400]])
    pairs = torch.cat([s["actions"] for s in states[:400]])
    model = make_model(wm.Stats(seq, pairs), pairs.shape[-1], pix)
    data = ft.Packed(states)
    pix_all = None
    if pix:
        pix_all = torch.cat([torch.from_numpy(np.load(FRAMES / Path(n).name)["frames"]) for n in names])   # (F, 2, 112, 112, 3)
        pix_all = pix_all.pin_memory()
    del states
    params = sum(p.numel() for p in model.net.parameters())
    opt = torch.optim.AdamW(model.net.parameters(), lr=ft.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    wts = torch.ones(wm.STEPS, device="cuda"); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT
    zeros_tau = torch.zeros(ft.BATCH, device="cuda")
    log, t0 = [{"batch": ft.BATCH, "steps": steps, "lr": ft.LR, "params": params, "arm": arm}], time.time()
    net = model.net

    def step_fn(inp, act_t):
        r, h = net(inp, torch.zeros_like(inp), act_t, zeros_tau, return_tokens=True)
        return r, h
    for step in range(steps):
        idx_np = np.concatenate([rng.choice(pools[0], ft.BATCH // 2), rng.choice(pools[1], ft.BATCH - ft.BATCH // 2)])
        idx = torch.as_tensor(idx_np, device="cuda")
        z0, rend, dd, target, act = data.batch(idx)
        act = model._act(act)
        use_g = torch.rand(ft.BATCH, device="cuda") < 0.5
        with torch.no_grad():
            start = torch.where(use_g[:, None, None, None], te(z0), rend)
        ptgt = None
        if pix:
            ptgt = pixel_targets(pix_all[torch.as_tensor(idx_np)].to("cuda", non_blocking=True).reshape(-1, 112, 112, 3)).reshape(ft.BATCH, 2, 196, 192)
        net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, loss, ploss = start, 0.0, 0.0
            for t in range(wm.STEPS):
                inp = cur + torch.randn_like(cur) * sm.INPUT_NOISE * model.stats.res_std
                r, h = checkpoint(step_fn, inp, act[:, t], use_reentrant=False)
                cur = cur + r.float() * model.stats.res_std
                loss = loss + wts[t] * ((cur - target[:, t]) ** 2).mean()
                if pix and t in (wm.HOLD10, wm.HOLD30):
                    ploss = ploss + ((net.pix(h).float() - ptgt[:, 0 if t == wm.HOLD10 else 1]) ** 2).mean()
            loss = loss / wts.sum()
            total = loss + PIX_WEIGHT * ploss if pix else loss
        opt.zero_grad(set_to_none=True); total.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % log_every == 0:
            row = {"step": step + 1, "loss": float(loss), "seconds": round(time.time() - t0, 1)}
            if pix:
                row["pix_loss"] = float(ploss)
            log.append(row)
            print(f"{arm} step {step + 1}: loss {float(loss):.5f}" + (f" pix {float(ploss):.4f}" if pix else "")
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    net.eval()
    save_model(model, path)
    (OUT / f"model_{arm}_log.json").write_text(json.dumps(log) + "\n")
    print("trained", arm, flush=True)


# ------------------------------------------------------------------------------------------- grading
def grade():
    import torch
    import fast_train as ft
    import step3_world_model as wm
    import step3b_models as sm
    from phase2_amend4 import alarm, score
    te = pm.GaussTE(torch.load(OUT / "Gauss_TE_phase2.pt"))
    models = {a: load_model(OUT / f"model_{a}.pt") for a in ("T", "T_pix") if (OUT / f"model_{a}.pt").exists()}
    models.update({a: wm.WorldModel.load(OUT / f"model_{a}.pt") for a in ("US2_long", "US2_long_s1", "US2_xlong")})
    files = sorted((OUT / "h2").glob("*.npz"))
    fork, fam, ends, errs, spread = [], [], {a: [] for a in models}, {a: [] for a in models}, {a: [] for a in models}
    with torch.no_grad():
        for n_state, path in enumerate(files):
            with np.load(path) as f:
                k = int(f["harm"].sum())
                z0 = torch.as_tensor(f["z0"], device="cuda").float()
                g = torch.Generator(device="cuda").manual_seed(pm.SEEDS["monitor"] + n_state)
                starts = te(z0[None].expand(64, -1, -1, -1).contiguous(), g)
                acts = np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, 8)
                real30 = f["z"][:, 1].reshape(64, -1).astype(np.float64)
            fork.append(2 <= k <= 62); fam.append(path.stem.split("_")[0])
            for a, m in models.items():
                e10, e30 = sm.roll(m, starts, acts)
                ends[a].append((e10, e30))
                errs[a].extend(np.linalg.norm(e30 - real30, axis=1) / np.sqrt(real30.shape[1] // 3))
                if 2 <= k <= 62:
                    spread[a].append(float(e30.std(0).sum() / real30.std(0).sum()))
            torch.cuda.empty_cache()
    fork, fam = np.array(fork), np.array(fam)
    out = {"readout": "BOTH"}
    for a in models:
        tau = float(np.mean(errs[a]))
        al = np.array([alarm(e10, e30, "BOTH", tau) for e10, e30 in ends[a]])
        out[a] = {"tau": tau, "all": score(al, fork), "F1": score(al[fam == "F1"], fork[fam == "F1"]),
                  "F2": score(al[fam == "F2"], fork[fam == "F2"]), "fork_spread_ratio_median": float(np.median(spread[a]))}
        log = OUT / f"model_{a}_log.json"
        if log.exists():
            L = json.loads(log.read_text())
            out[a]["final_loss"] = [r["loss"] for r in L if "step" in r][-1]
            out[a]["possibly_undertrained"] = ft.undertrained(L, L[0]["steps"])
    best = max([a for a in ("T", "T_pix") if a in out], key=lambda a: out[a]["all"]["score"], default=None)
    if best:
        out["best_transformer"] = best
        out["transformer_helps"] = bool(out[best]["all"]["score"] > 0.276 + 0.05)
        out["passes_in_family_bar_on_H2"] = bool(out[best]["all"]["caught"] >= 0.80 and out[best]["all"]["false_alarm"] <= 0.20)
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("frames", "train", "grade"))
    parser.add_argument("--arm", choices=("T", "T_pix"))
    parser.add_argument("--steps", type=int, default=22_500)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if not json.loads((ROOT / "results/jenga/phase2_generality_protocol.json").read_text())["amendment_8"]["status"].startswith("frozen"):
        raise SystemExit("amendment 8 not frozen")
    if args.stage == "frames":
        render_frames(args.workers)
    elif args.stage == "train":
        train(args.arm, steps=args.steps)
    else:
        grade()


if __name__ == "__main__":
    main()
