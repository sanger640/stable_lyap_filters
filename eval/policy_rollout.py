"""Closed-loop three-block episodes with the user's diffusion policy (sanger640/diffusion_policy fork, checkpoint
data/jenga/policy/latest.ckpt), in the deterministic sim (DirectJengaSim, 0.1 s per command).

Policy I/O as in the original sim client (vendor/panda_express_sim.tar: panda_express/sim.py SimDualCamera, get_state):
camera_1 = cam_wrist, camera_2 = cam_fixed, rendered at 480 x 640 and resized to 240 x 320 (as the training frames), CHW in [0, 1]; agent_pos = [gripper site xyz,
+1 closed / -1 open]; 2 observation steps; the policy plans 16 commands and 8 are executed before re-planning.
Commands are [x, y, z, gripper] (> 0.9 close, < -0.9 open), exactly what DirectJengaSim.execute takes.

Run with /home/corey/wksp/dp_venv/bin/python (system packages + diffusers/hydra/dill). Stage `baseline`: N episodes on
fresh layouts (RESET_BASE + i), no monitor, no filter -> success / topple rates and a few videos.
"""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
DP_DIR = Path(os.environ.get("DIFFUSION_POLICY_DIR", "/home/corey/wksp/diffusion_policy"))
CKPT = ROOT / "data/jenga/policy/latest.ckpt"
OUT = ROOT / "results/program_b/policy_rollouts"
RESET_BASE = 15000                     # layouts never used before (panels 10000/11000, filter 12000, video 14000)
MAX_CHUNKS = 40                        # 32 s
LIFT_M = 0.05                          # success: middle block lifted >= 5 cm above its start ...
TOPPLE_DEG = 45.0                      # ... and no neighbour tilted >= 45 deg at any point


def load_policy(steps=8):
    import dill
    import hydra
    import torch
    sys.path.insert(0, str(DP_DIR))
    payload = torch.load(open(CKPT, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    pol = hydra.utils.instantiate(cfg.policy)
    pol.load_state_dict(payload["state_dicts"]["ema_model" if cfg.training.use_ema else "model"])
    pol.cuda().eval(); pol.num_inference_steps = steps
    return pol


class Cameras:
    def __init__(self, sim):
        import mujoco
        self.sim, self.r = sim, mujoco.Renderer(sim.model, height=480, width=640)

    def obs(self):
        """Rendered at 480 x 640 and resized with cv2 to 320 x 240, as the training frames (create_dual_zarr*.py)."""
        import cv2
        self.r.update_scene(self.sim.data, camera="cam_wrist"); wrist = cv2.resize(self.r.render(), (320, 240))
        self.r.update_scene(self.sim.data, camera="cam_fixed"); fixed = cv2.resize(self.r.render(), (320, 240))
        closed = 1.0 if self.sim.gripper < 50 else -1.0
        return wrist, fixed, np.r_[self.sim.data.site_xpos[self.sim.site], closed].astype(np.float32)

    def close(self):
        self.r.close()


def policy_chunk(pol, hist):
    import torch
    w = np.stack([h[0] for h in hist[-2:]]); f = np.stack([h[1] for h in hist[-2:]]); s = np.stack([h[2] for h in hist[-2:]])
    t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2)[None].float().cuda() / 255.0
    with torch.no_grad():
        r = pol.predict_action({"camera_1": t(w), "camera_2": t(f), "agent_pos": torch.from_numpy(s)[None].cuda()})
    return r["action"][0].cpu().numpy()                                  # (8, 4)


def episode(pol, i, video=False, chunk_hook=None):
    """One closed-loop episode. chunk_hook(sim, chunk, k) -> chunk to execute (monitor / filter go here)."""
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="pol_"))))
    cams = Cameras(sim)
    try:
        sim.reset(RESET_BASE + i)
        z0 = None                                   # resting height, read after the first chunk (blocks spawn in the air)
        hist = [cams.obs()]; hist.append(hist[0])
        frames, chunks, peak_neigh, max_lift = [], [], 0.0, 0.0
        for k in range(MAX_CHUNKS):
            chunk = policy_chunk(pol, hist)
            if chunk_hook is not None:
                chunk = chunk_hook(sim, chunk, k)
            chunks.append(chunk)
            for a in chunk:
                sim.execute(a)
                o = cams.obs(); hist.append(o)
                if video:
                    frames.append(np.concatenate([o[1], o[0]], 1))
                if z0 is None:
                    continue
                tilt = sim.block_diagnostics()["tilt"]
                peak_neigh = max(peak_neigh, float(np.max(tilt[1:])))
                max_lift = max(max_lift, float(sim.data.xpos[sim.tracked_block_ids[0]][2] - z0))
            if z0 is None:
                z0 = float(sim.data.xpos[sim.tracked_block_ids[0]][2])
            if max_lift >= LIFT_M:
                break
        res = {"episode": i, "chunks": k + 1, "lifted": max_lift >= LIFT_M, "max_lift_mm": round(max_lift * 1000, 1),
               "neighbour_toppled": peak_neigh >= TOPPLE_DEG, "peak_neighbour_tilt": round(peak_neigh, 1)}
        res["success"] = res["lifted"] and not res["neighbour_toppled"]
        return res, frames, np.stack(chunks)
    finally:
        cams.close(); sim.close()


def baseline(n=20, videos=4):
    import imageio_ffmpeg
    OUT.mkdir(parents=True, exist_ok=True)
    pol = load_policy()
    rows = []
    for i in range(n):
        res, frames, chunks = episode(pol, i, video=i < videos)
        rows.append(res); print(res, flush=True)
        np.save(OUT / f"chunks_{i}.npy", chunks)
        if frames:
            w = imageio_ffmpeg.write_frames(str(OUT / f"baseline_{i}.mp4"), (640, 240), fps=10, codec="libx264", quality=8,
                                            macro_block_size=1)
            w.send(None)
            for fr in frames:
                w.send(np.ascontiguousarray(fr))
            w.close()
    out = {"episodes": n, "success": sum(r["success"] for r in rows), "lifted": sum(r["lifted"] for r in rows),
           "neighbour_toppled": sum(r["neighbour_toppled"] for r in rows), "rows": rows}
    (ROOT / "results/jenga/policy_baseline.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    {"baseline": baseline}[sys.argv[1]]()
