"""SafeManip world model + imagined fork test (docs/BENCHMARK_DD.md, check 2b-WM spec + amendment 1), one model per view.
Same recipe as libero/wm_libero.py: frozen DINOv2-392 tokens -> channel PCA 8 -> 28x28x8 latent; deterministic residual
FiLM transition (eval/fast_train.py unchanged). Differences (amendment 1): target latents have robot patches (> 30% robot
pixels) zeroed; imagined futures use the real branching's execution-noise model; endings = end of chunk / end of hold.

Model step = 2 control steps. Model input per step (all known at imagination time): the 2 policy actions (2 x 12), the
start proprioception (14) and the commanded end-effector path so far (cumulative arm deltas x 0.05 m, 3) -> 41 dims.

Stages (VIEW = monitor | wrist, env SM_VIEW):
  pca    channel PCA 8 fitted on start frames of training points (robot patches excluded)
  encode train / eval npz -> latents (masked) + model inputs + labels
  train  base model; ens K: 4 members fine-tuned from it (as Jenga / LIBERO)
  check  imagined fork score h on the held-out eval points vs real forks (2..30 of 32 noisy replays bad) -> gate
usage: SM_VIEW=monitor python3 eval/safemanip_wm.py {pca|encode|train|ens K|check}
"""
import json
import os
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
VIEW = os.environ.get("SM_VIEW", "monitor"); VI = {"monitor": 0, "wrist": 1}[VIEW]
DATA = Path(os.environ.get("SM_DATA", str(Path.home() / "wksp/safemanip/wm_data")))   # <DATA>/train/s*/, <DATA>/eval/s*/
OUT = Path(os.environ.get("SM_OUT", str(ROOT / f"results/safemanip/wm_{VIEW}")))
ENC = OUT / "enc"
STEPS, END_CHUNK, END_HOLD = 15, 9, 14          # model steps (30 control steps); indices of the two endings
SCALES = (0.5, 1.0, 2.0, 3.0)
H_CAP = float(np.log2(SCALES[-1]))
N_IMAG = 32


def files(split):
    return sorted(f for f in (DATA / split).glob("s*/*.npz") if not f.name.endswith(".tmp.npz"))


def patch_keep(mask):
    """(..., 256, 256) robot mask -> (..., 28, 28) True where the patch has <= 30% robot pixels."""
    e = np.linspace(0, 256, 29).astype(int)
    frac = np.stack([np.stack([mask[..., e[i]:e[i + 1], e[j]:e[j + 1]].mean((-1, -2)) for j in range(28)], -1) for i in range(28)], -2)
    return frac <= 0.3


def unpack(m):
    return np.unpackbits(m, axis=-1)[..., :256].astype(bool)


def model_inputs(actions, p0):
    """actions (K, 30, 12), start proprio (K, 14) -> (K, 15, 41)."""
    K = len(actions)
    a2 = actions.reshape(K, STEPS, 24)
    path = 0.05 * np.cumsum(actions[:, :, :3], 1)[:, 1::2]                             # after each model step
    return np.concatenate([a2, np.repeat(p0[:, None], STEPS, 1), path], -1).astype(np.float32)


class Latent:
    def __init__(self):
        import torch
        from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
        s = torch.load(OUT / "channel_pca8.pt")
        self.mean, self.basis, self.zm, self.zs = (s[k].cuda() for k in ("mean", "basis", "z_mean", "z_std"))
        self.enc = load_world_model(DEFAULT_CHECKPOINT, "cuda").encoder

    def __call__(self, frames, keep):
        from jenga_vjepa_encode import dino_tokens
        z = ((dino_tokens(self.enc, frames, "cuda") - self.mean) @ self.basis - self.zm) / self.zs
        z = z.reshape(len(z), 28, 28, 8).cpu().numpy()
        return (z * keep[..., None]).astype(np.float16)                                  # robot patches zeroed


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
    fs = files("train"); rng = np.random.default_rng(20261021)
    toks = []
    for f in [fs[i] for i in sorted(rng.choice(len(fs), min(150, len(fs)), replace=False))]:
        d = np.load(f)
        fr = d["frames"][0, 0, VI][None]; keep = patch_keep(unpack(d["masks"][0, 0, VI]))
        t = dino_tokens(enc, fr, "cuda").reshape(28, 28, 384)[torch.as_tensor(keep, device="cuda")]
        toks.append(t[torch.as_tensor(rng.choice(len(t), min(400, len(t)), replace=False), device="cuda")])
    toks = torch.cat(toks); mean, basis, zm, zs = fit_channel_pca(toks)
    torch.save({"mean": mean.cpu(), "basis": basis.cpu(), "z_mean": zm.cpu(), "z_std": zs.cpu(), "tokens": len(toks)}, path)
    print("pca done", len(toks), flush=True)


def stage_encode():
    lat = None
    for split in ("train", "eval"):
        (ENC / split).mkdir(parents=True, exist_ok=True)
        for f in files(split):
            out = ENC / split / f"{f.parent.name}_{f.name}"
            if out.exists() or time.time() - f.stat().st_mtime < 120:      # skip files the collector may still be writing
                continue
            lat = lat or Latent()
            d = np.load(f); F = d["frames"][:, :, VI]; M = unpack(d["masks"][:, :, VI])        # (K, T, 256, 256, 3)
            K, T = F.shape[:2]
            z = lat(F.reshape(K * T, 256, 256, 3), patch_keep(M.reshape(K * T, 256, 256))).reshape(K, T, 28, 28, 8)
            np.savez(out.with_suffix(".tmp.npz"), z=z, inputs=model_inputs(d["actions"], d["proprio"][:, 0]))
            out.with_suffix(".tmp.npz").replace(out)
            print("encoded", split, out.name, flush=True)


def _patch_wm():
    import step3_world_model as wm
    wm.STEPS, wm.HOLD10, wm.HOLD30 = STEPS, END_CHUNK, END_HOLD


def load_states(split="train"):
    import torch
    states = []
    for f in sorted((ENC / split).glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        d = np.load(f); z = torch.from_numpy(d["z"])                                    # (5, 16, 28, 28, 8)
        states.append({"z0": z[0, 0], "z0k": z[:, 0], "z": z[:, 1:], "actions": torch.from_numpy(d["inputs"]), "d": torch.zeros(len(z), 18)})
    return states


def stage_train():
    import fast_train as ft
    _patch_wm()
    st = load_states(); flat = sum(len(s["z"]) for s in st)
    print("train points", len(st), "futures", flat, flush=True)
    ft.train("SM", OUT, st, lambda rng, b: rng.choice(flat, b), seed=0, start_fn=lambda z0: z0, compile_net=False)


def stage_ens(k):
    import fast_train as ft
    _patch_wm()
    st = load_states(); flat = sum(len(s["z"]) for s in st)
    ft.train(f"ENS{k}", OUT, st, lambda rng, b: rng.choice(flat, b), seed=k, steps=3000, init=OUT / "model_SM.pt",
             lr=ft.LR * 0.3, start_fn=lambda z0: z0, compile_net=False)


def exec_noise(rng, n, T):
    """The real branching's arm noise (safemanip_fork_check.branch): 3-tap smoothed Gaussian, 5% / (5/3)% on the 6 arm dims."""
    e = rng.normal(size=(n, T + 2, 6)) * np.r_[[0.05] * 3, [0.05 / 3] * 3]
    return (e[:, :-2] + e[:, 1:-1] + e[:, 2:]) / 3


def stage_check():
    """Imagined fork score h at the held-out eval points vs real forks. h = log2 of the smallest s whose end-of-chunk and
    end-of-hold imagined endings both split (two-mode split test), capped at log2 3 (as Jenga / LIBERO)."""
    import torch
    import cem_filter as cf
    import step3_world_model as wm
    from wm_libero import auroc
    _patch_wm()
    model = wm.WorldModel.load(OUT / "model_SM.pt")
    rows = {}
    for jf in (DATA / "eval").glob("s*/*.json"):
        for r in json.loads(jf.read_text())["rows"]:
            rows[f"{jf.parent.name}_{r['task']}_s{jf.parent.name[1:]}_e{r['episode']}_t{r['t']}.npz"] = r
    rng = np.random.default_rng(20261021)
    res = []
    for f in sorted((ENC / "eval").glob("*.npz")):
        r = rows.get(f.name)
        if r is None:
            continue
        d = np.load(f); src = np.load(DATA / "eval" / f.name.split("_", 1)[0] / f.name.split("_", 1)[1])
        base = src["actions"][0]; p0 = src["proprio"][0, 0]                                 # exact replay's actions (chunk + hold)
        z0 = torch.as_tensor(d["z"][0, 0], device="cuda").float()
        noise = exec_noise(rng, N_IMAG, 30); split = []
        for s in SCALES[:-1]:
            acts = np.repeat(base[None], N_IMAG, 0); acts[:, :20, :6] = np.clip(acts[:, :20, :6] + s * noise[:, :20], -1, 1)
            inp = torch.as_tensor(model_inputs(acts, np.repeat(p0[None], N_IMAG, 0)), device="cuda")
            z = z0[None].expand(N_IMAG, -1, -1, -1).contiguous(); e = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(STEPS):
                    z = model.step(z, inp[:, t]).float()
                    if t in (END_CHUNK, END_HOLD):
                        e[t] = z.reshape(1, N_IMAG, -1)
            ok1, _, _ = cf.split_test_gpu(e[END_CHUNK]); ok2, _, _ = cf.split_test_gpu(e[END_HOLD])
            split.append(bool(ok1[0]) and bool(ok2[0]))
        hit = np.flatnonzero(split); h = float(np.log2(SCALES[hit[0]])) if len(hit) else H_CAP
        res.append({"file": f.name, "h": h, "split_s1": split[1], "n_bad": r["n_bad"], "fork": 2 <= r["n_bad"] <= 30})
    fork = np.array([x["fork"] for x in res]); h = np.array([x["h"] for x in res]); sp = np.array([x["split_s1"] for x in res])
    out = {"view": VIEW, "points": len(res), "real_forks": int(fork.sum()),
           "AUROC_minus_h_vs_fork": auroc(-h, fork), "imagined_split_rate_s1_forks": float(sp[fork].mean()),
           "imagined_split_rate_s1_nonforks": float(sp[~fork].mean())}
    out["gate_pass"] = bool(out["AUROC_minus_h_vs_fork"] >= 0.7 and out["imagined_split_rate_s1_forks"] >= 2 * max(out["imagined_split_rate_s1_nonforks"], 1e-9))
    (OUT / "check_rows.json").write_text(json.dumps(res)); (OUT / "check_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    st = sys.argv[1]
    {"pca": stage_pca, "encode": stage_encode, "train": stage_train, "check": stage_check}.get(st, lambda: stage_ens(int(sys.argv[2])))()
