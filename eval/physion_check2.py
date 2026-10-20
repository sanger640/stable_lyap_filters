"""Physion due diligence, check 2 (docs/BENCHMARK_DD.md, spec fixed before data): does our fork test detect the real
Dominoes forks found in check 1?

Stages (Jenga env, python3, GPU):
  pca     channel PCA 8 of frozen DINOv2-392 tokens on training frames
  encode  training trials, test clean renders, Gauss-TE variants -> latents (every 10th frame, frame 0 = start)
  gte     object-free Gauss-TE field fitted on z(noisy frame 0) - z(clean frame 0) (eval/phase2_p0.fit_te)
  train   passive world model (eval/fast_train.py unchanged; zero actions), 20 model steps = frames 10..200;
          starts 50/50 Gauss-TE or the rendered start
  eval    64 Gauss-TE starts at s in (0.5, 1, 2, 3) per test scene -> roll out -> two-mode split at step 10 and 20
          (both must split, as Jenga) -> h; AUROC(-h, real fork) and alarm rates vs check-1 labels

usage: python3 eval/physion_check2.py {pca|encode|gte|train|eval}
"""
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
PH = Path.home() / "wksp/physion/dd"
OUT = ROOT / "results/physion_check2"
STEPS, MID, END = 20, 9, 19
SCALES = (0.5, 1.0, 2.0, 3.0)
H_CAP = float(np.log2(SCALES[-1]))
CFGS = ["pilot_dominoes_1mid_J025R45_boxroom", "pilot_dominoes_2mid_J025R30_tdwroom",
        "pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom", "pilot_dominoes_2mid_J020R15_d3chairs_o1plants_tdwroom"]


def frames21(path):
    f = np.load(path)["frames"]
    if len(f) < STEPS + 1:
        f = np.concatenate([f, np.repeat(f[-1:], STEPS + 1 - len(f), 0)])
    return f[:STEPS + 1]


def patch_wm():
    import step3_world_model as wm
    wm.STEPS, wm.HOLD10, wm.HOLD30 = STEPS, MID, END


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


def stage_pca():
    import torch
    from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
    from jenga_vjepa_encode import dino_tokens
    from phase2_encoder_diag import fit_channel_pca
    OUT.mkdir(parents=True, exist_ok=True)
    enc = load_world_model(DEFAULT_CHECKPOINT, "cuda").encoder
    files = sorted(glob.glob(str(PH / "train/*/c*/*.npz")))
    rng = np.random.default_rng(1); toks = []
    for f in [files[i] for i in rng.choice(len(files), min(120, len(files)), replace=False)]:
        fr = frames21(f)[rng.choice(STEPS + 1, 4, replace=False)]
        t = dino_tokens(enc, fr, "cuda").reshape(-1, 384)
        toks.append(t[torch.as_tensor(rng.choice(len(t), 400, replace=False), device="cuda")])
    mean, basis, zm, zs = fit_channel_pca(torch.cat(toks))
    torch.save({"mean": mean.cpu(), "basis": basis.cpu(), "z_mean": zm.cpu(), "z_std": zs.cpu()}, OUT / "channel_pca8.pt")
    print("pca done", flush=True)


def stage_encode():
    lat = Latent()
    jobs = {"train": sorted(glob.glob(str(PH / "train/*/c*/*.npz"))),
            "test_old": sorted(glob.glob(str(PH / "test_clean_old/*/*.npz"))),
            "test_new": sorted(glob.glob(str(PH / "test_clean_new/*/*.npz"))),
            "gte": sorted(glob.glob(str(PH / "gte/*/n*/*.npz")))}
    for name, files in jobs.items():
        z = np.stack([lat(frames21(f)) for f in files]) if files else np.zeros((0, STEPS + 1, 28, 28, 8), np.float16)
        np.savez(OUT / f"z_{name}.npz", z=z, files=np.array([str(Path(f).relative_to(PH)) for f in files]))
        print("encoded", name, z.shape, flush=True)


def stage_gte():
    import torch
    import phase2_p0 as p0
    p0.OUT = OUT
    tr = np.load(OUT / "z_train.npz"); g = np.load(OUT / "z_gte.npz")
    idx = {f: i for i, f in enumerate(tr["files"])}
    states = []
    for c in CFGS:
        for t in range(10):
            clean = f"train/{c}/c0/{t:04d}.npz"
            noisy = [i for i, f in enumerate(g["files"]) if f.startswith(f"gte/{c}/n") and f.endswith(f"/{t:04d}.npz")]
            if clean in idx and noisy:
                states.append({"z0": torch.from_numpy(tr["z"][idx[clean], 0]), "z0k": torch.from_numpy(g["z"][noisy, 0])})
    fitted = p0.fit_te(states)
    d = np.concatenate([(s["z0k"].float() - s["z0"][None].float()).numpy().ravel() for s in states])
    print("gte fitted on", len(states), "scenes; latent noise rms", float(np.sqrt((d ** 2).mean())), flush=True)


def gte_sampler():
    import torch
    from phase2_models import GaussTE
    return GaussTE(torch.load(OUT / "Gauss_TE.pt"))


def stage_train():
    import torch
    import fast_train as ft
    patch_wm()
    tr = np.load(OUT / "z_train.npz")["z"]
    keep = [i for i in range(len(tr)) if i % 10 != 0]                     # every 10th trial held out (validation)
    states = [{"z0": torch.from_numpy(tr[i, 0]), "z0k": torch.from_numpy(tr[i, :1]), "z": torch.from_numpy(tr[i, 1:][None]),
               "actions": torch.zeros(1, STEPS, 1), "d": torch.zeros(1, 18)} for i in keep]
    g = gte_sampler()
    ft.train("P", OUT, states, lambda rng, b: rng.integers(0, len(states), b), seed=0, start_fn=lambda z0: g(z0),
             compile_net=False, steps=3000)
    print("trained on", len(states), "trials", flush=True)


def labels(kind):
    """check-1 style real fork labels (computed by ~/wksp/physion/dd/labels.py in the TDW env, which has h5py):
    (config, trial) -> contact per noisy run (16)."""
    lab = json.loads((PH / "labels.json").read_text())
    return {(k.split("|")[1], k.split("|")[2]): v for k, v in lab.items() if k.startswith(kind + "|")}


def stage_eval():
    import torch
    import cem_filter as cf
    import step3_world_model as wm
    from wm_libero import auroc
    patch_wm()
    model = wm.WorldModel.load(OUT / "model_P.pt"); g = gte_sampler(); rows = []
    for kind in ("test_old", "test_new"):
        d = np.load(OUT / f"z_{kind}.npz"); lab = labels(kind)
        for zi, f in zip(d["z"], d["files"]):
            parts = Path(f).parts; c, trial = parts[1], Path(f).stem
            runs = lab.get((c, trial))
            if not runs or len(runs) < 16:
                continue
            z0 = torch.as_tensor(zi[0], device="cuda").float(); split = []
            for si, s in enumerate(SCALES[:-1]):
                torch.manual_seed(1000 + si)
                st = z0[None] + s * (g(z0[None].expand(64, -1, -1, -1)) - z0[None])
                z = st; e = {}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for t in range(STEPS):
                        z = model.step(z, torch.zeros(64, 1, device="cuda")).float()
                        if t in (MID, END):
                            e[t] = z.reshape(1, 64, -1)
                ok1, _, _ = cf.split_test_gpu(e[MID]); ok2, _, _ = cf.split_test_gpu(e[END])
                split.append(bool(ok1[0] & ok2[0]))
            h = next((float(np.log2(s)) for s, sp in zip(SCALES, split) if sp), H_CAP)
            n = int(sum(runs)); rows.append({"kind": kind, "config": c, "trial": trial, "contacts_of_16": n,
                                             "real_fork": 2 <= n <= 14, "h": h})
    (OUT / "eval_rows.json").write_text(json.dumps(rows))
    y = np.array([r["real_fork"] for r in rows]); h = np.array([r["h"] for r in rows]); alarm = h < H_CAP
    res = {"scenes": len(rows), "real_forks": int(y.sum()), "AUROC_minus_h": auroc(-h, y),
           "alarm_rate_forks": float(alarm[y].mean()) if y.any() else None,
           "alarm_rate_nonforks": float(alarm[~y].mean()) if (~y).any() else None,
           "h_values": {str(v): int((h == v).sum()) for v in np.unique(h)}}
    res["per_config"] = {}                                  # reporting only (occluder / distractor configs vs not)
    for c in CFGS:
        m = np.array([r["config"] == c for r in rows])
        if m.any():
            yc, hc = y[m], h[m]
            res["per_config"][c] = {"scenes": int(m.sum()), "real_forks": int(yc.sum()),
                                    "AUROC_minus_h": auroc(-hc, yc) if 0 < yc.sum() < len(yc) else None,
                                    "alarms_forks": int((hc < H_CAP)[yc].sum()), "alarms_nonforks": int((hc < H_CAP)[~yc].sum())}
    res["gate_pass"] = bool(res["AUROC_minus_h"] >= 0.7 and res["alarm_rate_forks"] >= 2 * max(res["alarm_rate_nonforks"], 1e-9))
    (OUT / "check2_result.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    {"pca": stage_pca, "encode": stage_encode, "gte": stage_gte, "train": stage_train, "eval": stage_eval}[sys.argv[1]]()
