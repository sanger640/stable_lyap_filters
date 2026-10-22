"""Pillar task world model + imagined fork test (docs/PILLAR_TASK.md "World-model check"), the SafeManip 2b-WM recipe
(eval/safemanip_wm.py) on the pillar task's side camera.

Data (label-free for training): the closed-loop ExpertPolicy (2 mm perception error, pillar 8 mm, 2x execution noise).
  data train : seeds 3000-3099, every 2nd decision: exact + 4 noisy replays (fresh 2x Jenga snippets on the 8-action
               chunk) of chunk + 22 hold = 30 actions; frames (256 px, cam_fixed) at the start and after every 2 actions
               (16 frames), robot masks, the 30 executed targets, start proprio.
  data eval  : seeds 4000-4023 (held out), every decision: exact + 32 noisy replays; frames at start / end of chunk / end
               of hold; per-replay generic hold label vs the exact replay (BAD = cube > 2 cm lower or pillar > 45 deg
               more tilted; GOOD = both within 1 cm); real fork = >= 2 BAD and >= 2 GOOD.
Model: frozen DINOv2 tokens -> channel PCA 8 (fitted on training start frames, robot patches excluded) -> 28x28x8;
deterministic residual FiLM transition (eval/fast_train.py), model step = 2 actions; input per step = the 2 targets
relative to the start gripper position (x10) + gripper commands + start proprio (12 dims). Robot patches zeroed.
Check: imagined fork score h (smallest noise scale s in {0.5, 1, 2} of the real branching noise at which the 32 imagined
endings split at the end of the chunk AND of the hold; cap log2 3) vs real forks. Gate (as SafeManip): AUROC(-h, fork)
>= 0.7 AND imagined split rate at s = 1 on forks >= 2x non-forks.
usage: python3 eval/pillar_wm.py data train|eval [procs] ; pca ; encode ; train ; ens K ; check
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PILLAR_HALF_W", "0.008"); os.environ.setdefault("PILLAR_PERC", "0.002")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
DATA = Path(os.environ.get("PW_DATA", str(Path.home() / "wksp/pillar_wm_data")))
OUT = ROOT / "results/pillar/wm"
ENC = OUT / "enc"
H, HOLD, PX = 8, 22, 256
STEPS, END_CHUNK, END_HOLD = 15, 3, 14          # model steps (2 actions each); endings after 8 and 30 actions
SCALES = (0.5, 1.0, 2.0, 3.0)
H_CAP = float(np.log2(SCALES[-1]))
N_IMAG = 32
SEEDS = {"train": range(3000, 3100), "eval": range(4000, 4024)}


# ------------------------------------------------------------------------------------------------ data
class Cam:
    def __init__(self, sim):
        import mujoco
        self.mj, self.sim = mujoco, sim
        self.r = mujoco.Renderer(sim.model, height=PX, width=PX); self.s = mujoco.Renderer(sim.model, height=PX, width=PX)
        self.s.enable_segmentation_rendering()
        m = sim.model
        keep = {m.body(n).id for n in ("world", "table", "cube", "pillar", "cam_target")}
        self.robot_geom = np.array([m.geom_bodyid[g] not in keep for g in range(m.ngeom)], bool)

    def __call__(self):
        self.r.update_scene(self.sim.data, camera="cam_fixed"); f = self.r.render().copy()
        self.s.update_scene(self.sim.data, camera="cam_fixed"); sg = self.s.render()
        gid = sg[..., 0]; isg = sg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM)
        mask = isg & (gid >= 0) & self.robot_geom[np.clip(gid, 0, len(self.robot_geom) - 1)]
        return f, mask


def proprio(sim):
    return np.r_[sim.data.site_xpos[sim.site], 1.0 if sim.gripper < 50 else -1.0].astype(np.float32)


def data_job(args):
    split, seed = args
    import copy
    import pillar_task as P
    import uncertainty_monitor_dev as um
    out = DATA / split / f"s{seed}"
    if (out / "done.json").exists():
        return seed
    out.mkdir(parents=True, exist_ok=True)
    sim = P.PillarSim(); P.reset(sim, seed); pol = P.ExpertPolicy(sim, P.PERC, seed); cam = Cam(sim)
    snippets = np.load(um.BENCH)["snippets"]; rng = np.random.default_rng([20261502, seed]); brng = np.random.default_rng([11, seed])
    K = 5 if split == "train" else 33
    for k in range(30):
        plan = pol.plan(); chunk = plan[:H].copy()
        if split == "eval" or k % 2 == 0:
            snap = sim.snapshot(); pst = copy.deepcopy((pol.rng.bit_generator.state, pol.opened_on_pillar))
            frames, masks, acts, p0s, ends = [], [], [], [], []
            for b in range(K):
                sim.restore(snap)
                nz = np.zeros((H, 3)) if b == 0 else snippets[brng.integers(len(snippets))] * 2.0
                p0s.append(proprio(sim)); f, m = cam(); fr, mk = [f], [m]
                seq = [np.r_[chunk[j, :3] - nz[j], chunk[j, 3]] for j in range(H)] + [np.r_[chunk[-1, :3] - nz[-1], chunk[-1, 3]]] * HOLD
                for i, a in enumerate(seq):
                    sim.execute(np.asarray(a, np.float32))
                    if (split == "train" and i % 2 == 1) or (split == "eval" and i in (H - 1, H + HOLD - 1)):
                        f, m = cam(); fr.append(f); mk.append(m)
                frames.append(np.stack(fr)); masks.append(np.stack(mk)); acts.append(np.stack(seq).astype(np.float32))
                ends.append((P.body_pos(sim, "cube"), P.body_pos(sim, "pillar"), float(sim.block_diagnostics()["tilt"][1])))
            sim.restore(snap); pol.rng.bit_generator.state = pst[0]; pol.opened_on_pillar = pst[1]
            row = {"seed": seed, "t": k}
            if split == "eval":
                (c_ex, p_ex, t_ex) = ends[0]
                bad = [bool(c[2] < c_ex[2] - 0.02 or tl > t_ex + 45) for c, pp, tl in ends[1:]]
                good = [bool(np.linalg.norm(c - c_ex) < 0.01 and np.linalg.norm(pp - p_ex) < 0.01) for c, pp, tl in ends[1:]]
                row.update(n_bad=int(sum(bad)), n_good=int(sum(good)), fork=bool(sum(bad) >= 2 and sum(good) >= 2), bad_replays=bad)
            np.savez_compressed(out / f"t{k}.tmp.npz", frames=np.stack(frames).astype(np.uint8), masks=np.packbits(np.stack(masks), axis=-1),
                                actions=np.stack(acts), proprio=np.stack(p0s), row=json.dumps(row))
            (out / f"t{k}.tmp.npz").replace(out / f"t{k}.npz")
        nz = snippets[rng.integers(len(snippets))] * 2.0                       # the main episode continues
        for j, a in enumerate(chunk):
            a = a.copy(); a[:3] -= nz[j]; sim.execute(a)
        if pol.opened_on_pillar and sim.data.site_xpos[sim.site][2] > P.body_pos(sim, "pillar")[2] + 0.12:
            break
    (out / "done.json").write_text(json.dumps({"seed": seed, "chunks": k + 1}))
    return seed


def stage_data(split, procs=16):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(procs)) as pool:
        for n, s in enumerate(pool.imap_unordered(data_job, [(split, s) for s in SEEDS[split]]), 1):
            if n % 10 == 0:
                print("data", split, n, flush=True)
    print("data done", split, flush=True)


# ------------------------------------------------------------------------------------------------ model
def files(split):
    return sorted(f for f in (DATA / split).glob("s*/t*.npz") if not f.name.endswith(".tmp.npz"))


def patch_keep(mask):
    e = np.linspace(0, PX, 29).astype(int)
    frac = np.stack([np.stack([mask[..., e[i]:e[i + 1], e[j]:e[j + 1]].mean((-1, -2)) for j in range(28)], -1) for i in range(28)], -2)
    return frac <= 0.3


def unpack(m):
    return np.unpackbits(m, axis=-1)[..., :PX].astype(bool)


def model_inputs(actions, p0):
    """actions (K, 30, 4) absolute targets, start proprio (K, 4) -> (K, 15, 12): per model step the 2 targets relative to
    the start gripper position (x10) with their gripper commands, plus the start proprio (position x10 relative to the
    table centre, gripper)."""
    K = len(actions)
    rel = actions.copy(); rel[..., :3] = (actions[..., :3] - p0[:, None, :3]) * 10
    p = p0.copy(); p[:, :3] = (p0[:, :3] - np.array([0.5, 0.0, 0.5])) * 10
    return np.concatenate([rel.reshape(K, STEPS, 8), np.repeat(p[:, None], STEPS, 1)], -1).astype(np.float32)


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
        return (z * keep[..., None]).astype(np.float16)


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
    fs = files("train"); rng = np.random.default_rng(20261022); toks = []
    for f in [fs[i] for i in sorted(rng.choice(len(fs), min(200, len(fs)), replace=False))]:
        d = np.load(f); fr = d["frames"][0, 0][None]; keep = patch_keep(unpack(d["masks"][0, 0]))
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
            if out.exists():
                continue
            lat = lat or Latent()
            d = np.load(f); F = d["frames"]; M = unpack(d["masks"]); K, T = F.shape[:2]
            z = lat(F.reshape(K * T, PX, PX, 3), patch_keep(M.reshape(K * T, PX, PX))).reshape(K, T, 28, 28, 8)
            np.savez(out.with_suffix(".tmp.npz"), z=z, inputs=model_inputs(d["actions"], d["proprio"]), row=d["row"])
            out.with_suffix(".tmp.npz").replace(out)
        print("encoded", split, flush=True)


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
    ft.train("PW", OUT, st, lambda rng, b: rng.choice(flat, b), seed=0, start_fn=lambda z0: z0, compile_net=False)


def stage_ens(k):
    import fast_train as ft
    _patch_wm()
    st = load_states(); flat = sum(len(s["z"]) for s in st)
    ft.train(f"ENS{k}", OUT, st, lambda rng, b: rng.choice(flat, b), seed=k, steps=3000, init=OUT / "model_PW.pt",
             lr=ft.LR * 0.3, start_fn=lambda z0: z0, compile_net=False)


def stage_check():
    """Imagined fork score h at the held-out eval points vs real forks (gate as SafeManip 2b-WM)."""
    import torch
    import cem_filter as cf
    import step3_world_model as wm
    import uncertainty_monitor_dev as um
    from wm_libero import auroc
    _patch_wm()
    model = wm.WorldModel.load(OUT / "model_PW.pt")
    snippets = np.load(um.BENCH)["snippets"]; rng = np.random.default_rng(20261022); res = []
    for f in sorted((ENC / "eval").glob("*.npz")):
        d = np.load(f); r = json.loads(str(d["row"])); src = np.load(DATA / "eval" / f.name.split("_", 1)[0] / f.name.split("_", 1)[1])
        base = src["actions"][0]; p0 = src["proprio"][0]
        z0 = torch.as_tensor(d["z"][0, 0], device="cuda").float(); split = []
        nz = snippets[rng.integers(len(snippets), size=N_IMAG)] * 2.0                    # the real branching's noise (s = 1)
        for s in SCALES[:-1]:
            acts = np.repeat(base[None], N_IMAG, 0); acts[:, :H, :3] -= s * nz; acts[:, H:, :3] -= s * nz[:, -1:]
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
        res.append({"file": f.name, "h": h, "split_s1": split[1], "n_bad": r["n_bad"], "fork": r["fork"]})
    fork = np.array([x["fork"] for x in res]); h = np.array([x["h"] for x in res]); sp = np.array([x["split_s1"] for x in res])
    out = {"points": len(res), "real_forks": int(fork.sum()), "AUROC_minus_h_vs_fork": auroc(-h, fork),
           "imagined_split_rate_s1_forks": float(sp[fork].mean()) if fork.any() else None,
           "imagined_split_rate_s1_nonforks": float(sp[~fork].mean())}
    out["gate_pass"] = bool(fork.any() and out["AUROC_minus_h_vs_fork"] >= 0.7
                            and out["imagined_split_rate_s1_forks"] >= 2 * max(out["imagined_split_rate_s1_nonforks"], 1e-9))
    (OUT / "check_rows.json").write_text(json.dumps(res)); (OUT / "check_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "data":
        stage_data(sys.argv[2], *(sys.argv[3:4] or []))
    elif st == "ens":
        stage_ens(int(sys.argv[2]))
    else:
        {"pca": stage_pca, "encode": stage_encode, "train": stage_train, "check": stage_check}[st]()
