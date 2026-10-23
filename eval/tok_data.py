"""Full-token counterfactual data for the DINO-WM world model (docs/FILTER_DEEP_DIVE.md step 2).

Same states, paired perturbations and execution noise as eval/cf_data.py, but:
- frames stored as FULL DINOv2 ViT-S/14 patch tokens at 196 px (14x14x384; int8 with a per-token scale), the DINO-WM
  input, instead of the 8-channel PCA;
- proprioception per frame (gripper site xyz, gripper closed +1 / open -1);
- V plans per state = the policy's plan + V-1 task-agnostic random EDITS of it, drawn to cover the filter's search space:
  smooth xyz offsets (as cf_data.nudge), time-warps (slow down / pause / speed back up), and both combined.
usage (cluster array): TOK_OUT=... CF_RUN=... python3 eval/tok_data.py   (CF_TASK=k/N from py_array.sbatch)
"""
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import cf_data as cfd  # noqa: E402

OUT = ROOT / os.environ.get("TOK_OUT", "results/program_b/tok_v1")
cfd.OUT = OUT
V = int(os.environ.get("TOK_V", 4))
PX = 196                                     # DINO-WM input size (14x14 patches; bundled predictor pos table = 3 x 196)


def warp(plan, rng):
    """Random monotone time-warp of steps 1..15: tau(t) = clip(v * max(t - k, 0), 0, 15), v ~ U(0.3, 1.2), pause k ~ U(0, 4)."""
    p = np.asarray(plan, np.float32); v, k = rng.uniform(0.3, 1.2), rng.uniform(0, 4)
    tau = np.clip(v * np.maximum(np.arange(16) - k, 0), 0, 15); i = np.clip(np.floor(tau).astype(int), 0, 14); w = (tau - i)[:, None]
    q = p.copy(); q[:, :3] = p[i, :3] * (1 - w) + p[i + 1, :3] * w
    return q


def edit(plan, rng):
    u = rng.random()
    if u < 0.4:
        return cfd.nudge(plan, rng)
    if u < 0.7:
        return warp(plan, rng)
    return cfd.nudge(warp(plan, rng), rng)


class Tokens:
    def __init__(self):
        import torch
        from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
        self.torch, self.enc = torch, load_world_model(DEFAULT_CHECKPOINT, "cuda").encoder

    def __call__(self, frames):
        """(N, H, W, 3) uint8 -> int8 tokens (N, 196, 384), scales (N, 196) float16."""
        torch = self.torch; out_q, out_s = [], []
        from torchvision.transforms import functional as TF
        for a in range(0, len(frames), 32):
            x = torch.as_tensor(frames[a:a + 32], device="cuda").permute(0, 3, 1, 2).float() / 255
            x = (TF.resize(x, [PX, PX], antialias=True) - 0.5) / 0.5
            with torch.inference_mode():
                t = self.enc.forward(x).float()                                     # (B, 196, 384)
            s = t.abs().amax(-1).clamp_min(1e-6) / 127
            out_q.append((t / s[..., None]).round().clamp(-127, 127).to(torch.int8).cpu().numpy()); out_s.append(s.half().cpu().numpy())
        return np.concatenate(out_q), np.concatenate(out_s)


def _job(args):
    (e, c), split, state, ctrl, target, gripper, tilt, plan = args
    import mujoco
    import program_b_reference as pr
    import step3_data as sd
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import lookahead_check as la
    from jenga_render_frames import close_camera
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    path = OUT / split / f"ep{e}_s{c}.npz"
    if path.exists():
        return "exists"
    g = globals()
    if "_W" not in g:
        sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="tok_"))))
        g["_W"] = (sim, mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), Tokens())
    sim, r, cam, tok = g["_W"]
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    rng = np.random.default_rng([cfd.SEED, 7, e, c])
    plans = [np.asarray(plan, np.float32)] + [edit(plan, rng) for _ in range(V - 1)]
    snap = (state, ctrl, target, float(gripper))

    def prop():
        return np.r_[sim.data.site_xpos[sim.site], float(sim.gripper)].astype(np.float32)          # raw gripper ctrl, as the bundle
    sim.restore(snap); r.update_scene(sim.data, camera=cam); true_start = r.render().copy(); prop0 = prop()
    starts, frames, props, acts, which = [], [], [], [], []
    for p_ in range(cfd.P):
        j = int(np.random.default_rng([cfd.SEED, e, c, p_]).integers(64))
        for v, pl in enumerate(plans):
            sim.restore(snap); pr.perturb_blocks(sim, *errors.sample(cfd.SEED, e, c, p_))
            r.update_scene(sim.data, camera=cam); starts.append(r.render().copy()); pp = [prop()]
            cmds = la.horizon_actions(pl, "H15", la.noise_for(snippets, j) * la.EXEC_SCALE)
            for t, a in enumerate(cmds):
                sim.execute(a)
                if t in sd.FRAME_INDEX:
                    r.update_scene(sim.data, camera=cam); frames.append(r.render().copy()); pp.append(prop())
            props.append(np.stack(pp)); acts.append(cmds); which.append((p_, v))
    K = len(acts); nF = len(sd.FRAME_INDEX)
    q, s = tok(np.concatenate([true_start[None], np.stack(starts), np.stack(frames)]))
    np.savez(path.with_suffix(".tmp.npz"), q0=q[0], s0=s[0], q_start=q[1:1 + K], s_start=s[1:1 + K],
             q=q[1 + K:].reshape(K, nF, 196, 384), s=s[1 + K:].reshape(K, nF, 196), prop0=prop0, props=np.stack(props),
             actions=np.stack(acts), start_tilt=tilt, frame_index=np.array(sd.FRAME_INDEX), pair=np.array(which),
             plans=np.stack(plans))
    path.with_suffix(".tmp.npz").replace(path)
    return "ok"


def job(args):
    import traceback
    try:
        return args[0], _job(args), None
    except Exception:
        return args[0], None, traceback.format_exc()


cfd._job, cfd.job = _job, job          # cf_data.main's pool pickles tok_data.job, so spawned workers import this module

if __name__ == "__main__":
    cfd.main()
