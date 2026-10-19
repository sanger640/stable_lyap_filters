"""SafeLIBERO world model (L2) and offline check (L3), same recipe as Jenga, run in the Jenga env (python3, GPU).

Stages (docs/LIBERO_PLAN.md, L2 / L3 specs fixed before any encoded data was looked at):
  pca     channel PCA 8 of frozen DINOv2-392 tokens, fitted on nominal frames of training dev episodes
  encode  every L2 file -> latents (nominal trajectory + branched futures); evalonly obstacle labels per future
  train   deterministic residual FiLM transition (eval/fast_train.py, unchanged), 10 model steps = 10-step chunk + 10
          hold env steps; starts = the true start latent (z0k = z0); 50% branch futures (half of them label-free
          rare-mode futures), 50% windows of the nominal trajectory
  ens     4 members fine-tuned from the base model with seeds 0-3 (as eval/ens_train.py)
  l3      offline check on held-out dev episodes (ep % 10 == 0): fork barrier h and regime change o of the policy's
          chunk, imagined from the start latent under execution noise scaled by SCALES; AUROC vs collision / damage
          in the real futures (labels used only here)

usage: python3 libero/wm_libero.py {pca|encode|train|ens K|l3}
"""
import io
import json
import os
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "libero"), str(ROOT / "eval"), str(ROOT / "src")]
DATA = ROOT / "results/libero/l2_data"
OUT = ROOT / os.environ.get("LIBERO_WM", "results/libero/wm")
ENC = OUT / "enc"
STEPS, HOLD10, HOLD30 = 10, 4, 9            # model steps; end of the chunk (10 env steps), end of the hold
SCALES = (0.5, 1.0, 2.0, 3.0)
N_FUT, WIN_EVERY = 32, 5
H_CAP = float(np.log2(SCALES[-1]))


def held_out(name):
    return int(name.split("_e")[-1][:2]) % 10 == 0


def scenario(name):
    return "_".join(name.split("_")[:4])


def frames_of(jpgs):
    from PIL import Image
    return np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in jpgs])


# ------------------------------------------------------------------------------------------- encoder
def stage_pca():
    import torch
    from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
    from jenga_vjepa_encode import dino_tokens
    from phase2_encoder_diag import fit_channel_pca
    path = OUT / "channel_pca8.pt"
    if path.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    enc = load_world_model(DEFAULT_CHECKPOINT, "cuda").encoder
    files = [f for f in sorted(DATA.glob("*.npz")) if not held_out(f.stem)]
    rng = np.random.default_rng(20261020)
    pick = [files[i] for i in sorted(rng.choice(len(files), min(120, len(files)), replace=False))]
    toks = []
    for f in pick:
        nf = np.load(f, allow_pickle=True)["nominal_frames"]
        idx = sorted(rng.choice(len(nf), min(4, len(nf)), replace=False))
        t = dino_tokens(enc, frames_of(nf[idx]), "cuda").reshape(-1, 384)
        toks.append(t[torch.as_tensor(rng.choice(len(t), 400, replace=False), device="cuda")])
    toks = torch.cat(toks)
    mean, basis, zm, zs = fit_channel_pca(toks)
    torch.save({"mean": mean.cpu(), "basis": basis.cpu(), "z_mean": zm.cpu(), "z_std": zs.cpu(), "files": [f.name for f in pick],
                "tokens": len(toks)}, path)
    print("pca done", len(toks), "tokens from", len(pick), "files", flush=True)


class Latent:
    def __init__(self):
        import torch
        from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
        s = torch.load(OUT / "channel_pca8.pt")
        self.mean, self.basis, self.zm, self.zs = (s[k].cuda() for k in ("mean", "basis", "z_mean", "z_std"))
        self.enc = load_world_model(DEFAULT_CHECKPOINT, "cuda").encoder

    def __call__(self, frames):
        from jenga_vjepa_encode import dino_tokens
        z = ((dino_tokens(self.enc, frames, "cuda") - self.mean) @ self.basis - self.zm) / self.zs
        return z.reshape(len(z), 28, 28, 8).cpu().numpy().astype(np.float16)


def active_obstacle(objects, poses0):
    """The benchmark's active obstacle (policy_eval.py rule) from the first nominal poses (evaluation only)."""
    for i, n in enumerate(objects):
        p = poses0[i, :3]
        if "obstacle" in n and p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            return i
    return None


def tilt(q):
    x, y = q[..., 0], q[..., 1]
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def stage_encode():
    ENC.mkdir(parents=True, exist_ok=True)
    lat = None
    for f in sorted(DATA.glob("*.npz")):
        out = ENC / f.name
        if out.exists() or time.time() - f.stat().st_mtime < 120:          # skip files the collector may still be writing
            continue
        if lat is None:
            lat = Latent()
        d = np.load(f, allow_pickle=True)
        objs = list(d["objects"]); ob = active_obstacle(objs, d["evalonly_nominal_poses"][0])
        nz = lat(frames_of(d["nominal_frames"]))
        NP = d["evalonly_nominal_poses"]                                                 # obstacle moved before each branch?
        pre = np.array([np.nan if ob is None else 1000 * np.abs(NP[:b["t"] // 2 + 1, ob, :3] - NP[0, ob, :3]).sum(-1).max()
                        for b in d["branches"]], np.float32)
        bz, ba, bt, beef, btheta, lab = [], [], [], [], [], []
        for b in d["branches"]:
            fr = [frames_of(fu["frames"]) for fu in b["futures"]]
            z = lat(np.concatenate(fr)).reshape(len(fr), STEPS + 1, 28, 28, 8)
            bz.append(z); bt.append(b["t"]); beef.append(b["eef"])
            ba.append(np.stack([fu["actions"].reshape(STEPS, 14) for fu in b["futures"]]))
            btheta.append(np.stack([fu["theta"] for fu in b["futures"]]))
            if ob is None:
                lab.append(np.full((len(fr), 2), np.nan))
            else:
                P = np.stack([fu["evalonly_poses"][:, ob] for fu in b["futures"]])           # (8, 11, 7)
                disp = np.abs(P[:, :, :3] - P[:, :1, :3]).sum(-1).max(1) * 1000
                dtilt = np.abs(tilt(P[:, :, 3:]) - tilt(P[:, :1, 3:])).max(1)
                lab.append(np.stack([disp, dtilt], 1))
        np.savez(out.with_suffix(".tmp.npz"), nominal_z=nz, nominal_actions=d["nominal_actions"], nominal_eef=d["nominal_eef"],
                 branch_z=np.array(bz, np.float16).reshape(len(bz), 8, STEPS + 1, 28, 28, 8),
                 branch_actions=np.array(ba, np.float32).reshape(len(bz), 8, STEPS, 14), branch_t=np.array(bt),
                 branch_eef=np.array(beef, np.float32).reshape(len(bz), 9), branch_theta=np.array(btheta, np.float32).reshape(len(bz), 8, 4),
                 evalonly_disp_mm_tilt_deg=np.array(lab, np.float32).reshape(len(bz), 8, 2),
                 evalonly_obstacle=-1 if ob is None else ob, success=d["success"], evalonly_pre_disp_mm=pre)
        out.with_suffix(".tmp.npz").replace(out)
        print("encoded", f.name, len(bz), flush=True)


# ------------------------------------------------------------------------------------------- training
def _patch_wm():
    import step3_world_model as wm
    wm.STEPS, wm.HOLD10, wm.HOLD30 = STEPS, HOLD10, HOLD30


def load_states(split):
    """split 'train' / 'val' -> fast_train states + pool ids (0 branch future, 1 nominal window) and names."""
    import torch
    states, kinds, names = [], [], []
    for f in sorted(ENC.glob("*.npz")):
        if f.name.endswith(".tmp.npz") or held_out(f.stem) != (split == "val"):
            continue
        d = np.load(f)
        for bi in range(len(d["branch_z"])):
            z = torch.from_numpy(d["branch_z"][bi])                                      # (8, 11, 28, 28, 8)
            states.append({"z0": z[0, 0], "z0k": z[:, 0], "z": z[:, 1:], "actions": torch.from_numpy(d["branch_actions"][bi]),
                           "d": torch.zeros(len(z), 18)})
            kinds.append(0); names.append(f"{f.stem}:b{bi}")
        nz, na = d["nominal_z"], d["nominal_actions"]
        for s in range(0, len(nz) - STEPS - 1, WIN_EVERY):                              # windows of the nominal trajectory
            a = na[2 * s:2 * (s + STEPS)]
            if len(a) < 2 * STEPS:
                break
            z = torch.from_numpy(nz[s:s + STEPS + 1])[None]
            states.append({"z0": z[0, 0], "z0k": z[:, 0], "z": z[:, 1:], "actions": torch.from_numpy(a.reshape(1, STEPS, 14).astype(np.float32)),
                           "d": torch.zeros(1, 18)})
            kinds.append(1); names.append(f"{f.stem}:n{s}")
    return states, np.array(kinds), names


def _sampler(states, kinds, names):
    import phase1e_models as p1e
    br = [i for i, k in enumerate(kinds) if k == 0]
    modes = p1e.detect_modes([states[i] for i in br], [names[i] for i in br])
    flat = np.cumsum([0] + [len(s["z"]) for s in states])
    rare, other, nom = [], [], []
    for i, s in enumerate(states):
        ids = range(flat[i], flat[i + 1])
        if kinds[i] == 1:
            nom += ids
        else:
            r = set(modes[names[i]])
            rare += [flat[i] + k for k in r]; other += [flat[i] + k for k in range(len(s["z"])) if k not in r]
    pools = [np.array(p) for p in (rare, other, nom)]
    if not len(pools[0]):
        pools[0] = pools[1]
    (OUT / "modes_summary.json").write_text(json.dumps({"branch_states": len(br), "with_rare_mode": sum(bool(v) for v in modes.values()),
                                                        "rare_futures": len(rare), "other_futures": len(other), "nominal_windows": len(nom)}) + "\n")
    q = lambda b: (b // 4, b // 4, b - 2 * (b // 4))
    return lambda rng, b: np.concatenate([rng.choice(p, n) for p, n in zip(pools, q(b))])


def stage_train():
    import fast_train as ft
    _patch_wm()
    states, kinds, names = load_states("train")
    print("train states", len(states), "branch", int((kinds == 0).sum()), "nominal windows", int((kinds == 1).sum()), flush=True)
    ft.train("L", OUT, states, _sampler(states, kinds, names), seed=0, start_fn=lambda z0: z0, compile_net=False)
    print("val loss", val_loss(OUT / "model_L.pt"), flush=True)


def stage_ens(k):
    import fast_train as ft
    _patch_wm()
    states, kinds, names = load_states("train")
    ft.train(f"ENS{k}", OUT, states, _sampler(states, kinds, names), seed=k, steps=3000, init=OUT / "model_L.pt",
             lr=ft.LR * 0.3, start_fn=lambda z0: z0, compile_net=False)


def val_loss(path):
    import torch
    import step3_world_model as wm
    _patch_wm()
    model = wm.WorldModel.load(path)
    states, kinds, _ = load_states("val")
    out = {0: [], 1: []}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s, k in zip(states, kinds):
            z = s["z0k"].cuda().float(); a = s["actions"].cuda(); tgt = s["z"].cuda().float()
            for t in range(STEPS):
                z = model.step(z, a[:, t]).float()
            out[k].append(float(((z - tgt[:, -1]) ** 2).mean()))
    return {"branch_end_mse": float(np.mean(out[0])), "nominal_end_mse": float(np.mean(out[1])), "n": [len(out[0]), len(out[1])]}


# ------------------------------------------------------------------------------------------- monitor (latent only)
def hand_mask(cam, eef, deltas, links, radius_px=36, corridor_px=24):
    """(28, 28) True on the robot's own region (proprioception only): discs around the projected hand path (eef now +
    cumulative chunk deltas, 5 cm per unit), capsules along the current arm (links 0-7 + hand, libero/arm_links.py)
    and from the wrist to each planned hand point (the forearm follows the hand)."""
    from camera_dump import project
    pts = np.vstack([eef[None], eef[None] + 0.05 * np.cumsum(deltas[:, :3], 0)])
    uv = project(cam, pts); arm = project(cam, np.asarray(links))
    c = (np.arange(28) + 0.5) * cam["res"] / 28
    U, V = np.meshgrid(c, c)
    m = np.zeros((28, 28), bool)

    def capsule(p, q, r):
        d = q - p; L2 = max(float(d @ d), 1e-9)
        s_ = np.clip(((U - p[0]) * d[0] + (V - p[1]) * d[1]) / L2, 0, 1)
        return (U - p[0] - s_ * d[0]) ** 2 + (V - p[1] - s_ * d[1]) ** 2 <= r ** 2
    for u, v in uv:
        m |= (U - u) ** 2 + (V - v) ** 2 <= radius_px ** 2
        m |= capsule(arm[-3], np.array([u, v]), corridor_px)
    for p, q in zip(arm[:-1], arm[1:]):
        m |= capsule(p, q, corridor_px)
    return m


def score_plans(model, z0, plans, eef, links, cam, n, seed, return_e30=False):
    """Fork barrier h and regime change o for C candidate chunks (C, 10, 7) under ONE world model.
    Futures: chunk + 10 hold steps (zero deltas, the chunk's last gripper command), execution noise x s with common
    random numbers (same draws for every candidate and scale), start = the true start latent.
    h = log2 of the smallest s in SCALES whose hold-10 and hold-30 endings split (top scale skipped: it never changes h),
    capped at log2 3; o = max over futures (s = 1) of the mean squared latent change outside the robot mask."""
    import torch
    import cem_filter as cf
    import g_transform as G
    plans = np.asarray(plans, np.float64); C = len(plans)
    rng = np.random.default_rng(seed)
    noise = np.stack([G.exec_noise(rng, plans.shape[1]) for _ in range(n)])            # (n, 10, 7)
    split = np.zeros((C, len(SCALES) - 1), bool); o = np.zeros(C); e30_1 = None
    masks = torch.as_tensor(np.stack([~hand_mask(cam, eef, p, links) for p in plans]), device="cuda").float()
    for si, s in enumerate(SCALES[:-1]):
        acts = plans[:, None] + s * noise[None]                                          # (C, n, 10, 7)
        hold = np.zeros((C, n, STEPS * 2 - plans.shape[1], 7)); hold[..., 6] = plans[:, None, -1:, 6]
        acts = np.concatenate([acts, hold], 2); acts[..., :6] = np.clip(acts[..., :6], -1, 1)
        a = torch.as_tensor(acts.reshape(C * n, STEPS, 14), dtype=torch.float32, device="cuda")
        z = z0.float()[None].expand(C * n, -1, -1, -1).contiguous(); e = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(STEPS):
                z = model.step(z, a[:, t]).float()
                if t in (HOLD10, HOLD30):
                    e[t] = z.reshape(C, n, 28, 28, 8)
        ok10, _, _ = cf.split_test_gpu(e[HOLD10].reshape(C, n, -1)); ok30, _, _ = cf.split_test_gpu(e[HOLD30].reshape(C, n, -1))
        split[:, si] = (ok10 & ok30).cpu().numpy()
        if s == 1.0:
            diff = (e[HOLD30] - z0.float()[None, None]).pow(2).mean(-1)                 # (C, n, 28, 28)
            o = (((diff * masks[:, None]).sum((2, 3)) / masks.sum((1, 2))[:, None].clamp_min(1)).max(1).values).cpu().numpy()
            e30_1 = e[HOLD30] if return_e30 else None
    h = np.full(C, H_CAP)
    for c in range(C):
        hit = np.flatnonzero(split[c])
        if len(hit):
            h[c] = float(np.log2(SCALES[hit[0]]))
    return (h, o, e30_1, masks) if return_e30 else (h, o)


def scores(models, z0, chunk, eef, links, cam, seed):
    """One chunk, several models (ensemble = every member must pass, so h = min and o = max over members);
    u = ensemble disagreement outside the robot mask (variance over members of the s = 1 endings)."""
    hs, os_, es = [], [], []
    for m in models:
        h, o, e30, masks = score_plans(m, z0, chunk[None], eef, links, cam, N_FUT, seed, return_e30=True)
        hs.append(h[0]); os_.append(o[0]); es.append(e30[0])
    u = float("nan")
    if len(models) > 1:
        import torch
        v = torch.stack(es).var(0).mean(-1)                                               # (n, 28, 28)
        u = float(((v * masks[0]).sum((1, 2)) / masks[0].sum().clamp_min(1)).mean())
    return float(min(hs)), float(max(os_)), u


def real_o(z0, zend, eef, chunk, links, cam):
    """o on the REAL encoded futures (ceiling for the score, no world model)."""
    m = ~hand_mask(cam, eef, chunk, links)
    diff = ((zend.astype(np.float32) - z0.astype(np.float32)[None]) ** 2).mean(-1)
    return float((diff * m).sum((1, 2)).max() / max(m.sum(), 1))


def auroc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); s = np.asarray(score, float)
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def stage_calib():
    """Label-free O_MAX = 99th percentile of o for the policy's own chunks (f0) at training-dev branch points (base
    model, 64 futures) on a random 600 of them, as Jenga's calibration; also the label-free trigger rate it implies."""
    import torch
    import step3_world_model as wm
    _patch_wm()
    cam = json.loads((ROOT / "results/libero/camera.json").read_text())
    base = wm.WorldModel.load(OUT / "model_L.pt"); rows = []
    pts = [(f, bi) for f in sorted(ENC.glob("*.npz")) if not f.name.endswith(".tmp.npz") and not held_out(f.stem)
           for bi in range(len(np.load(f)["branch_t"]))]
    pick = np.random.default_rng(20261023).choice(len(pts), min(600, len(pts)), replace=False)   # random 600 decisions
    by = {}
    for i in sorted(pick):
        by.setdefault(pts[i][0], []).append(pts[i][1])
    for f, bis in by.items():
        d = np.load(f); arm = np.load(ROOT / "results/libero/arm" / (f.stem + ".npy"))
        for bi in bis:
            chunk = d["branch_actions"][bi][0].reshape(STEPS * 2, 7)[:10].astype(np.float64)
            hold = np.zeros_like(chunk); hold[:, 6] = chunk[0, 6]
            z0 = torch.as_tensor(d["branch_z"][bi][0, 0], device="cuda")
            seed = int(np.random.default_rng([20261022, len(rows)]).integers(2 ** 31))
            h, o = score_plans(base, z0, [chunk, hold], d["branch_eef"][bi][:3], arm[int(d["branch_t"][bi]) // 2],
                               cam[scenario(f.stem)], 64, seed)
            rows.append([float(h[0]), float(o[0]), float(h[1])])
    r = np.array(rows); o_max = float(np.quantile(r[:, 1], 0.99))
    trig = ~((r[:, 0] >= np.minimum(r[:, 2], H_CAP)) & (r[:, 1] <= o_max))
    res = {"O_MAX": o_max, "decisions": len(r), "label_free_trigger_rate": float(trig.mean()),
           "share_h_below_hold_or_cap": float((r[:, 0] < np.minimum(r[:, 2], H_CAP)).mean()), "h_hold_mean": float(r[:, 2].mean())}
    (OUT / "calibration.json").write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res), flush=True)


def stage_l3():
    import torch
    import step3_world_model as wm
    _patch_wm()
    cam = json.loads((ROOT / "results/libero/camera.json").read_text())
    base = wm.WorldModel.load(OUT / "model_L.pt")
    ens = [wm.WorldModel.load(OUT / f"model_ENS{k}.pt") for k in range(4) if (OUT / f"model_ENS{k}.pt").exists()]
    rows = []
    for f in sorted(ENC.glob("*.npz")):
        if f.name.endswith(".tmp.npz") or not held_out(f.stem):
            continue
        d = np.load(f)
        if int(d["evalonly_obstacle"]) < 0:
            continue
        for bi in range(len(d["branch_z"])):
            z = d["branch_z"][bi]; acts = d["branch_actions"][bi].reshape(-1, STEPS * 2, 7)
            chunk = acts[0, :10]                                                          # f0 = the policy's chunk as-is
            eef = d["branch_eef"][bi][:3]; c = cam[scenario(f.stem)]
            links = np.load(ROOT / "results/libero/arm" / (f.stem + ".npy"))[int(d["branch_t"][bi]) // 2]
            z0 = torch.as_tensor(z[0, 0], device="cuda")
            seed = int(np.random.default_rng([20261021, bi, len(rows)]).integers(2 ** 31))
            h, o, _ = scores([base], z0, chunk, eef, links, c, seed)
            r = {"file": f.stem, "branch": bi, "h": h, "o": o}
            if ens:
                he, oe, ue = scores(ens, z0, chunk, eef, links, c, seed)
                r.update({"h_ens": he, "o_ens": oe, "u_ens": ue})
            r["o_real_f0"] = real_o(z[0, 0], z[:1, -1], eef, chunk, links, c)
            lab = d["evalonly_disp_mm_tilt_deg"][bi]
            r["coll_f0"] = bool(lab[0, 0] > 1.0); r["dmg_f0"] = bool(lab[0, 0] >= 50 or lab[0, 1] >= 45)
            r["coll_f012"] = float(np.mean(lab[:3, 0] > 1.0)); r["pre_disp_mm"] = float(d["evalonly_pre_disp_mm"][bi])
            rows.append(r)
    (OUT / "l3_rows.json").write_text(json.dumps(rows) + "\n")
    res = {}
    for pop, sel in (("undisturbed", [r for r in rows if r["pre_disp_mm"] <= 1.0]), ("all", rows)):
        q = {"n_branches": len(sel), "collision_rate_f0": float(np.mean([r["coll_f0"] for r in sel])),
             "damage_rate_f0": float(np.mean([r["dmg_f0"] for r in sel]))}
        for key, sign in (("o", 1), ("h", -1), ("o_ens", 1), ("h_ens", -1), ("u_ens", 1), ("o_real_f0", 1)):
            if key in rows[0]:
                for lab in ("coll_f0", "dmg_f0"):
                    q[f"AUROC_{key}_{lab}"] = auroc([sign * r[key] for r in sel], [r[lab] for r in sel])
        res[pop] = q
    prim = res["undisturbed"]["AUROC_h_coll_f0"]            # fork score primary (user decision 2026-10-19, before results)
    res["gate_primary_AUROC"] = prim; res["gate_pass"] = bool(prim >= 0.7)
    (ROOT / "results/libero/l3_result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    st = sys.argv[1]
    {"pca": stage_pca, "encode": stage_encode, "train": stage_train, "calib": stage_calib, "l3": stage_l3}.get(st, lambda: stage_ens(int(sys.argv[2])))()
