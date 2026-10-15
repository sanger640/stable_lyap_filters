"""Fork-boundary barrier for a CBF-style safety filter, purely in the world model's latent space (no object state).

Safe set: the current regime (the scene the robot sees now, assumed safe). Boundary: the fork, where imagined futures
split. Two latent quantities per candidate plan:

  barrier h   distance to the fork in units of uncertainty: perception and execution error are scaled by s in SCALES and
              h = log2 of the smallest scale at which the 64 imagined futures split (BOTH test), capped at log2(max scale)
              when they never split; h < 0 means they already split at less than the measured uncertainty.
  regime r    share of the 64 imagined futures whose latent outside the robot's own hand path differs from the current
              latent by more than a threshold: "the world changed somewhere I am not touching". The hand path is the
              planned gripper positions projected into the monitor's close camera (robot proprioception + camera
              geometry; no object information).
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
SCALES = (0.5, 1.0, 2.0, 3.0)
IMG, GRID = 384, 28


def close_camera_matrix():
    """World -> pixel projection of eval/jenga_render_frames.CLOSE_CAMERA (MuJoCo free camera, fovy 45 deg, 384 px)."""
    from jenga_render_frames import CLOSE_CAMERA as C
    az, el = np.radians(C["azimuth"]), np.radians(C["elevation"])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = np.asarray(C["lookat"]) - C["distance"] * fwd
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = (IMG / 2) / np.tan(np.radians(45.0) / 2)
    return pos, fwd, right, up, f


def project(points):
    """(N, 3) world points -> (N, 2) pixel (u right, v down) in the 384 x 384 close-camera image."""
    pos, fwd, right, up, f = close_camera_matrix()
    d = np.asarray(points, float) - pos
    z = d @ fwd
    return np.stack([IMG / 2 + f * (d @ right) / z, IMG / 2 - f * (d @ up) / z], 1)


def hand_mask(points, radius_px=32, corridor_px=42):
    """Latent-grid mask (28, 28) of the robot's own region (True): patches within radius_px of the projected hand path,
    plus a vertical corridor from the top of the image down to each path point (the arm enters from the top in the
    close camera)."""
    uv = project(points)
    c = (np.arange(GRID) + 0.5) * IMG / GRID
    U, V = np.meshgrid(c, c)                                    # U = column centre, V = row centre
    m = np.zeros((GRID, GRID), bool)
    for u, v in uv:
        m |= (U - u) ** 2 + (V - v) ** 2 <= radius_px ** 2
        m |= (np.abs(U - u) <= corridor_px) & (V <= v)
    return m


def plan_points(sim, chunk):
    """Gripper site now plus the planned targets (the hand path), world coordinates."""
    return np.vstack([sim.data.site_xpos[sim.site][None], np.asarray(chunk)[:, :3]])


class Barrier:
    def __init__(self, mon):
        self.mon = mon

    def starts_scaled(self, frame, key_seed, scale):
        """64 perceived starts with the perception error scaled (planar, as the monitor)."""
        torch = self.mon.torch
        z0 = torch.as_tensor(self.mon.latent(frame[None])[0], device="cuda").float()
        d = np.stack([np.concatenate([(p * [1, 1, 0] * scale).reshape(9), (r * [0, 0, 1] * scale).reshape(9)])
                      for p, r in (self.mon.errors.sample(key_seed, 0, 0, j) for j in range(64))])
        return z0, self.mon.g(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d, dtype=torch.float32))

    def evaluate(self, sim, chunks, key_seed=0):
        """chunks (C, 8, 4) -> dict per candidate: h (barrier), splits per scale, r (regime-change share), per-future
        outside-hand change at scale 1 (for calibration)."""
        import cem_filter as cf
        torch, mon = self.mon.torch, self.mon
        frame = mon.frame(sim)
        C = len(chunks)
        split = np.zeros((C, len(SCALES)), bool)
        outside = None
        for si, s in enumerate(SCALES):
            z0, st = self.starts_scaled(frame, key_seed, s)
            snip = mon.snippets * s
            keep = mon.snippets
            mon.snippets = snip
            try:
                e10, e30 = mon.endpoints(st, np.stack(chunks))
            finally:
                mon.snippets = keep
            ok10, _, _ = cf.split_test_gpu(e10); ok30, _, _ = cf.split_test_gpu(e30)
            split[:, si] = (ok10 & ok30).cpu().numpy()
            if s == 1.0:
                outside = []
                for c in range(C):
                    m = torch.as_tensor(~hand_mask(plan_points(sim, chunks[c])), device="cuda")
                    diff = (e30[c].reshape(64, GRID, GRID, 8) - z0[None]).pow(2).mean(-1)      # (64, 28, 28)
                    outside.append(((diff * m).sum((1, 2)) / m.sum().clamp_min(1)).cpu().numpy())
                outside = np.stack(outside)
        h = np.full(C, np.log2(SCALES[-1]))
        for c in range(C):
            hit = np.flatnonzero(split[c])
            if len(hit):
                h[c] = np.log2(SCALES[hit[0]])
        return {"h": h, "split": split, "outside": outside}


def offline_check(n_max=None):
    """On the saved decision states with ground truth (policy-alone arm): does the barrier fall and does the regime score
    rise before real topples? Labels are used only to evaluate."""
    import os
    import tempfile
    import cem_filter as cf
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    R = ROOT / "results/program_b/policy_rollouts"
    rows = json.loads((R / "paired_monitor_truth.json").read_text())
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="cbf_"))))
    mon = cf.Monitor(seed=20261560); bar = Barrier(mon)
    cache = {}
    out = []
    for n, r in enumerate(rows[:n_max] if n_max else rows):
        e, c = r["episode"], r["chunk_index"]
        if e not in cache:
            cache[e] = dict(np.load(R / "paired_monitor" / f"ep{e:03d}.npz"))
        S = cache[e]; i = int(np.flatnonzero(S["chunk_index"] == c)[0])
        sim.restore((S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i])))
        res = bar.evaluate(sim, S["nominal"][i][None], key_seed=20261561 + n)
        out.append({"episode": e, "chunk": c, "risk": r["nominal"]["topples"] / 64, "alarm": bool(r["alarm"]),
                    "h": float(res["h"][0]), "split": res["split"][0].tolist(),
                    "outside_mean": float(res["outside"][0].mean()), "outside_q90": float(np.quantile(res["outside"][0], 0.9))})
        if n % 50 == 0:
            print("check", n, len(rows), flush=True)
    (ROOT / "results/jenga/cbf_barrier_offline.json").write_text(json.dumps(out) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    offline_check(int(sys.argv[1]) if len(sys.argv) > 1 else None)
