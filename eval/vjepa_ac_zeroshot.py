"""V-JEPA 2-AC zero-shot vs US2_xlong on H2, from the same rendered perturbed starts.

Frozen protocol: results/jenga/vjepa_ac_zeroshot_protocol.json.
  render  (CPU pool) per H2 state: 64 perturbed starts (monitor draws, seed 20261230) and the 64 real futures (H2 truth
          draws, snippet j) rendered after 18 and 36 commands; commanded targets/gripper, start end-effector pose
  eval    (GPU) encode with V-JEPA 2-AC's ViT-g (256) and DINOv2 -> PCA 8 (392); roll both models; BOTH readout;
          ceilings on real futures in each space
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "external/vjepa2")]

import phase2_data as pdata  # noqa: E402
import phase2_models as pm  # noqa: E402
import phase2_scenes as ps  # noqa: E402

OUT = ROOT / "results/program_b/vjepa_ac"
RESULT = ROOT / "results/jenga/vjepa_ac_zeroshot_result.json"
MONITOR_SEED = 20261230
STEP, N_STEPS = 3, 12                    # V-JEPA step = 3 commands; frames after 3, 6, ..., 36 commands
HOLD10_K, HOLD30_K = 5, 11               # predictor step indices whose frames come after 18 and 36 commands
RENDER_AT = (17, 35)                     # command indices (0-based) after which real futures are rendered


def render_job(job):
    import traceback
    try:
        return job[0], job[1], _render(*job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _render(family, seed, index):
    import mujoco
    import uncertainty_monitor_dev as um
    from jenga_render_frames import close_camera
    from scipy.spatial.transform import Rotation
    path = OUT / "frames" / f"{family}_{seed}_s{index}.npz"
    if path.exists():
        return index
    scene, sim, acts, starts = pdata.prepare(family, seed)
    snippets = np.load(um.BENCH)["snippets"]
    pos_cov, rot_cov = ps.pooled_error_model()
    n_obj = len(scene["objects"])
    renderer, camera = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco)

    def render():
        renderer.update_scene(sim.data, camera=camera); return renderer.render().copy()
    try:
        for i, act in enumerate(acts):
            if i == index:
                snap = sim.snapshot()
                key = ({"F1": 1, "F2": 2}[family], seed, index)
                true_start = render()
                pose = np.r_[sim.data.site_xpos[sim.site],
                             Rotation.from_matrix(sim.data.site_xmat[sim.site].reshape(3, 3)).as_euler("xyz"),
                             1.0 if sim.gripper < 50 else 0.0]
                prev_target = sim.target.copy()
                starts_img = []
                for j in range(64):
                    sim.restore(snap)
                    sim.perturb(*ps.draw_error(n_obj, pos_cov, rot_cov, MONITOR_SEED, *key, j))
                    starts_img.append(render())
                chunk = acts[index:index + 8]
                futures, executed = [], []
                for j in range(64):
                    sim.restore(snap)
                    sim.perturb(*ps.draw_error(n_obj, pos_cov, rot_cov, pm.SEEDS["h2_futures"], *key, j))
                    a = np.asarray(chunk, np.float32).copy(); a[:, :3] -= snippets[j]
                    ex = list(a) + [chunk[-1]] * 30
                    fr = []
                    for t, c in enumerate(ex):
                        sim.execute(c)
                        if t in RENDER_AT:
                            fr.append(render())
                    futures.append(np.stack(fr)); executed.append(np.asarray(ex, np.float32))
                with np.load(pm.OUT / "h2" / f"{family}_{seed}_s{index}.npz") as h:
                    harm = h["harm"]
                np.savez_compressed(path.with_suffix(".tmp.npz"), true_start=true_start, starts=np.stack(starts_img),
                                    futures=np.stack(futures), executed=np.stack(executed), pose=pose,
                                    prev_target=prev_target, prev_closed=pose[-1], harm=harm)
                path.with_suffix(".tmp.npz").replace(path)
                break
            sim.execute(act)
    finally:
        renderer.close()
    return index


def render_all(workers):
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in sorted((pm.OUT / "h2").glob("*.npz")):
        fam, seed, s = p.stem.split("_")
        jobs.append((fam, int(seed), int(s[1:])))
    pdata.pool_run(render_job, jobs, workers, "render")


def vjepa_actions(executed, prev_target, prev_closed):
    """(64, 38, 4) executed commands -> (64, 12, 7) V-JEPA actions (xyz delta, 0 rotation, closedness delta)."""
    n = len(executed)
    acts = np.zeros((n, N_STEPS, 7), np.float32)
    for j in range(n):
        target, closed = prev_target.copy(), float(prev_closed)
        for k in range(N_STEPS):
            seg = executed[j, k * STEP:(k + 1) * STEP]
            new_t = seg[-1, :3]
            new_c = closed
            for c in seg:
                new_c = 1.0 if c[3] > 0.9 else (0.0 if c[3] < -0.9 else new_c)
            acts[j, k, :3] = new_t - target; acts[j, k, 6] = new_c - closed
            target, closed = new_t, new_c
    return acts


def evaluate():
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import functional as TF
    import src.hub.backbones as bb
    import step3_data as sd
    import step3_world_model as wm
    import step3b_models as sm
    from notebooks.utils.mpc_utils import compute_new_pose
    from phase2_amend4 import alarm, score
    bb.VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
    encoder, predictor = bb.vjepa2_ac_vit_giant(pretrained=True)
    encoder, predictor = encoder.cuda().eval(), predictor.cuda().eval()
    tpf = (256 // encoder.patch_size) ** 2
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1, 1)

    def vj_encode(frames):
        out = []
        for i in range(0, len(frames), 16):
            x = torch.as_tensor(frames[i:i + 16], device="cuda").permute(0, 3, 1, 2).float() / 255
            x = TF.resize(x, [256, 256], antialias=True)[:, :, None].repeat(1, 1, 2, 1, 1)
            x = (x - mean) / std
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = encoder(x).float()
            out.append(F.layer_norm(h, (h.size(-1),)))
        return torch.cat(out)                                           # (N, 256, 1408)

    latent = sd.Latent()
    ours = wm.WorldModel.load(pm.OUT / "model_US2_xlong.pt")
    rows = []
    files = sorted((OUT / "frames").glob("*.npz"))
    for n_state, path in enumerate(files):
        with np.load(path) as f:
            starts, futures, executed = f["starts"], f["futures"], f["executed"]
            pose, prev_t, prev_c, harm = f["pose"], f["prev_target"], f["prev_closed"], f["harm"]
        k = int(harm.sum()); fork = 2 <= k <= 62
        acts_v = torch.as_tensor(vjepa_actions(executed, prev_t, prev_c), device="cuda")
        with torch.inference_mode():
            z0 = vj_encode(starts)                                       # (64, 256, D)
            s = torch.as_tensor(np.tile(pose, (64, 1))[:, None], device="cuda", dtype=torch.float32)
            z_ctx, s_ctx, preds = z0, s, []
            for kk in range(N_STEPS):
                a_ctx = acts_v[:, :kk + 1]
                outs = []
                for b in range(0, 64, 16):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o = predictor(z_ctx[b:b + 16], a_ctx[b:b + 16], s_ctx[b:b + 16])[:, -tpf:].float()
                    outs.append(F.layer_norm(o, (o.size(-1),)))
                z_new = torch.cat(outs)
                preds.append(z_new)
                s_new = compute_new_pose(s_ctx[:, -1:], acts_v[:, kk:kk + 1])
                z_ctx = torch.cat([z_ctx, z_new], 1); s_ctx = torch.cat([s_ctx, s_new.to(s_ctx.dtype)], 1)
            v10 = preds[HOLD10_K].reshape(64, -1).double().cpu().numpy()
            v30 = preds[HOLD30_K].reshape(64, -1).double().cpu().numpy()
            real = vj_encode(futures.reshape(-1, 384, 384, 3)).reshape(64, 2, -1).double().cpu().numpy()
            # our model from the same rendered starts (DINOv2 -> PCA 8)
            zs = torch.as_tensor(latent(starts), device="cuda").float()
            acts_o = executed.reshape(64, wm.STEPS, 8)
            o10, o30 = sm.roll(ours, zs, acts_o)
            real_o = latent(futures.reshape(-1, 384, 384, 3)).reshape(64, 2, -1).astype(np.float64)
        sp = lambda m, r: float(m.std(0).sum() / max(r.std(0).sum(), 1e-12))
        rows.append({"state": path.stem, "fork": fork,
                     "vjepa": alarm(v10, v30, "BOTH"), "vjepa_real": alarm(real[:, 0], real[:, 1], "BOTH"),
                     "ours": alarm(o10, o30, "BOTH"), "ours_real": alarm(real_o[:, 0], real_o[:, 1], "BOTH"),
                     "vjepa_spread": sp(v30, real[:, 1]), "ours_spread": sp(o30, real_o[:, 1])})
        torch.cuda.empty_cache()
        if (n_state + 1) % 20 == 0:
            print(f"eval {n_state + 1}/{len(files)}", flush=True)
    fork = np.array([r["fork"] for r in rows])
    out = {"states": len(rows), "forks": int(fork.sum())}
    for c in ("vjepa", "vjepa_real", "ours", "ours_real"):
        out[c] = score(np.array([r[c] for r in rows]), fork)
    for c in ("vjepa_spread", "ours_spread"):
        out[c + "_fork_median"] = float(np.median([r[c] for r in rows if r["fork"]]))
    out["rows"] = rows
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("render", "eval"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    if not json.loads((ROOT / "results/jenga/vjepa_ac_zeroshot_protocol.json").read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    render_all(args.workers) if args.stage == "render" else evaluate()


if __name__ == "__main__":
    main()
