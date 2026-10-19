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
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "libero"), str(ROOT / "eval"), str(ROOT / "src")]
DATA = ROOT / "results/libero/l2_data"
OUT = ROOT / "results/libero/wm"
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
        if out.exists():
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
                 branch_z=np.array(bz, np.float16).reshape(len(bz), -1, STEPS + 1, 28, 28, 8),
                 branch_actions=np.array(ba, np.float32).reshape(len(bz), -1, STEPS, 14), branch_t=np.array(bt),
                 branch_eef=np.array(beef, np.float32).reshape(len(bz), -1), branch_theta=np.array(btheta, np.float32).reshape(len(bz), -1, 4),
                 evalonly_disp_mm_tilt_deg=np.array(lab, np.float32).reshape(len(bz), -1, 2),
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
def hand_mask(cam, eef, deltas, radius_px=36, corridor_px=30):
    """(28, 28) True on the robot's own region: discs around the projected hand path (eef now + cumulative chunk deltas,
    5 cm per unit) and a corridor from the projected robot base to each path point (the arm)."""
    from camera_dump import project
    pts = np.vstack([eef[None], eef[None] + 0.05 * np.cumsum(deltas[:, :3], 0)])
    uv = project(cam, pts); base = project(cam, np.asarray(cam["robot_base"])[None])[0]
    c = (np.arange(28) + 0.5) * cam["res"] / 28
    U, V = np.meshgrid(c, c)
    m = np.zeros((28, 28), bool)
    for u, v in uv:
        m |= (U - u) ** 2 + (V - v) ** 2 <= radius_px ** 2
        d = np.array([u, v]) - base; L2 = max(float(d @ d), 1e-9)
        s = np.clip(((U - base[0]) * d[0] + (V - base[1]) * d[1]) / L2, 0, 1)
        m |= (U - base[0] - s * d[0]) ** 2 + (V - base[1] - s * d[1]) ** 2 <= corridor_px ** 2
    return m


def imagine(models, z0, chunk_deltas, gripper, rng_seed, n=N_FUT):
    """Imagined endings under execution noise scaled by s: dict s -> (e10, e30) (n, D) for the mean model and the
    per-member endings at scale 1 (for disagreement). chunk_deltas (10, 7) -> 10 chunk steps + 10 hold."""
    import torch
    import g_transform as G
    out = {}
    rng = np.random.default_rng(rng_seed)
    noise = np.stack([G.exec_noise(rng, len(chunk_deltas)) for _ in range(n)])         # common random numbers across s
    hold = np.tile(np.r_[np.zeros(6), gripper], (STEPS * 2 - len(chunk_deltas), 1))
    for s in SCALES:
        acts = np.stack([np.vstack([chunk_deltas + s * noise[j], hold]) for j in range(n)])
        acts[..., :6] = np.clip(acts[..., :6], -1, 1)
        a = torch.as_tensor(acts.reshape(n, STEPS, 14), dtype=torch.float32, device="cuda")
        ends = []
        for m in models:
            z = z0.float()[None].expand(n, -1, -1, -1).contiguous(); e = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(STEPS):
                    z = m.step(z, a[:, t]).float()
                    if t in (HOLD10, HOLD30):
                        e[t] = z.clone()
            ends.append(e)
        out[s] = ({t: torch.stack([e[t] for e in ends]).mean(0) for t in (HOLD10, HOLD30)}, ends)
    return out


def scores(models, z0, chunk, eef, cam, seed):
    """-> h (fork barrier, log2 of the smallest splitting scale, capped), o (max over futures of the mean squared latent
    change outside the hand mask, at scale 1), u (ensemble disagreement outside the hand at scale 1)."""
    import torch
    import cem_filter as cf
    im = imagine(models, z0, chunk, chunk[-1, 6], seed)
    h = H_CAP
    for s in SCALES[:-1]:
        e = im[s][0]
        ok10, _, _ = cf.split_test_gpu(e[HOLD10].reshape(1, N_FUT, -1)); ok30, _, _ = cf.split_test_gpu(e[HOLD30].reshape(1, N_FUT, -1))
        if bool(ok10[0] & ok30[0]):
            h = float(np.log2(s)); break
    m = torch.as_tensor(~hand_mask(cam, eef, chunk), device="cuda").float()
    e30 = im[1.0][0][HOLD30]
    diff = (e30 - z0.float()[None]).pow(2).mean(-1)
    o = float(((diff * m).sum((1, 2)) / m.sum().clamp_min(1)).max())
    mem = torch.stack([e[HOLD30] for e in im[1.0][1]])                                    # (M, n, 28, 28, 8)
    u = float(((mem.var(0).mean(-1) * m).sum((1, 2)) / m.sum().clamp_min(1)).mean()) if len(models) > 1 else float("nan")
    return h, o, u


def real_o(z0, zend, eef, chunk, cam):
    """o on the REAL encoded futures (ceiling for the score, no world model)."""
    m = ~hand_mask(cam, eef, chunk)
    diff = ((zend.astype(np.float32) - z0.astype(np.float32)[None]) ** 2).mean(-1)
    return float((diff * m).sum((1, 2)).max() / max(m.sum(), 1))


def auroc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); s = np.asarray(score, float)
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


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
            z0 = torch.as_tensor(z[0, 0], device="cuda")
            seed = int(np.random.default_rng([20261021, bi, len(rows)]).integers(2 ** 31))
            h, o, _ = scores([base], z0, chunk, eef, c, seed)
            r = {"file": f.stem, "branch": bi, "h": h, "o": o}
            if ens:
                he, oe, ue = scores(ens, z0, chunk, eef, c, seed)
                r.update({"h_ens": he, "o_ens": oe, "u_ens": ue})
            r["o_real_f0"] = real_o(z[0, 0], z[:1, -1], eef, chunk, c)
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
    prim = res["undisturbed"].get("AUROC_o_ens_coll_f0", res["undisturbed"].get("AUROC_o_coll_f0"))
    res["gate_primary_AUROC"] = prim; res["gate_pass"] = bool(prim >= 0.7)
    (ROOT / "results/libero/l3_result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    st = sys.argv[1]
    {"pca": stage_pca, "encode": stage_encode, "train": stage_train, "l3": stage_l3}.get(st, lambda: stage_ens(int(sys.argv[2])))()
