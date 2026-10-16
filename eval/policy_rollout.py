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


def annotate(fr, info, k):
    """Banner on a video frame: what the monitor / filter decided for this chunk."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(fr); d = ImageDraw.Draw(im)
    if info.get("abstained"):
        txt, col = f"chunk {k}: FLAGGED -> no safe nearby action, HOLD", (200, 60, 40)
    elif info.get("intervened"):
        off = np.abs(np.asarray(info.get("offset_mm", [0] * 6))).max()
        txt, col = f"chunk {k}: FLAGGED -> filter shifted action (max {off:.1f} mm)", (215, 130, 20)
    elif info.get("alarm"):
        txt, col = f"chunk {k}: monitor ALARM (logging only)", (200, 60, 40)
    else:
        txt, col = f"chunk {k}: OK", (30, 140, 80)
    d.rectangle([0, 0, 640, 18], fill=col); d.text((6, 3), txt, fill=(255, 255, 255))
    return np.asarray(im)


def films(mode="cem", n=12):
    """Filmed closed-loop episodes with the monitor/filter decision on every frame (settled start, 60 s, seeded,
    execution noise on): results/program_b/policy_rollouts/films_<mode>/."""
    import imageio_ffmpeg
    import cem_filter as cf
    vdir = OUT / f"films_{mode}"; vdir.mkdir(parents=True, exist_ok=True)
    pol, mon, rng = load_policy(), cf.Monitor(), np.random.default_rng(7)

    def hook(sim, chunk, k):
        if mode == "monitor":
            a, sec, _ = mon.check(sim, chunk); return chunk, {"alarm": a, "seconds": round(sec, 3)}
        return cf.cem_filter(mon, sim, chunk, rng=rng)
    rows = []
    for i in range(n):
        res, frames, _ = episode(pol, i, video=True, chunk_hook=hook, exec_noise=True, max_chunks=75, settle=True, seed=1000 + i)
        L = res.pop("log"); res["interventions"] = sum(bool(c.get("intervened") or c.get("alarm")) for c in L)
        rows.append(res); print(mode, res, flush=True)
        tag = "toppled" if res["neighbour_toppled"] else ("ok" if res["success"] else "fail")
        w = imageio_ffmpeg.write_frames(str(vdir / f"ep{i:02d}_{tag}_{res['interventions']}flags.mp4"), (640, 240), fps=10,
                                        codec="libx264", quality=7, macro_block_size=1)
        w.send(None)
        for fr in frames:
            w.send(np.ascontiguousarray(fr))
        w.close()
    (vdir / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")


PLAN_START = 1          # diffusion_policy executes action_pred[To-1 : To-1+8] with To = 2 observation steps


def policy_chunk(pol, hist, full=False, n=1):
    """The policy's next 8 commands; full=True: its whole 16-step plan; n > 1: n independent plans (n, 16, 4)."""
    import torch
    w = np.stack([h[0] for h in hist[-2:]]); f = np.stack([h[1] for h in hist[-2:]]); s = np.stack([h[2] for h in hist[-2:]])
    t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2)[None].float().cuda().expand(n, -1, -1, -1, -1) / 255.0
    with torch.no_grad():
        r = pol.predict_action({"camera_1": t(w), "camera_2": t(f), "agent_pos": torch.from_numpy(s)[None].cuda().expand(n, -1, -1)})
    if n > 1:
        return r["action_pred"].cpu().numpy()                             # (n, 16, 4)
    return r["action_pred"][0].cpu().numpy() if full else r["action"][0].cpu().numpy()


HARD = {"gap_shrink_mm": (8.0, 8.0), "yaw_deg": 6.0}   # stress layouts: neighbours moved toward the middle (default gaps 20 mm)


def harden(sim, i, shrink=None, lean=None):
    """Stress layout for episode i: move each neighbour toward the middle block by `shrink` mm and twist it by up to
    HARD['yaw_deg']; then forward. Deterministic in i."""
    rng = np.random.default_rng([20261510, i])
    m, d = sim.model, sim.data
    mid_y = d.xpos[m.body("block_middle").id][1]
    for name in ("block_left", "block_right"):
        adr = m.jnt_qposadr[m.body_jntadr[m.body(name).id]]
        sgn = np.sign(d.qpos[adr + 1] - mid_y)
        g = (shrink if shrink is not None else rng.uniform(*HARD["gap_shrink_mm"])) / 1000
        d.qpos[adr + 1] -= sgn * g
        yaw = np.radians(rng.uniform(-HARD["yaw_deg"], HARD["yaw_deg"]))
        q = np.zeros(4); sim.mj.mju_mulQuat(q, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), d.qpos[adr + 3:adr + 7].copy())
        d.qpos[adr + 3:adr + 7] = q
        lean_deg = lean if lean is not None else float(os.environ.get("HARD_LEAN", 0))
        if lean_deg:                                 # tip the top toward the middle block (about the world x axis)
            a = np.radians(rng.uniform(0.6, 1.0) * lean_deg) * sgn
            q2 = np.zeros(4); sim.mj.mju_mulQuat(q2, np.array([np.cos(a / 2), np.sin(a / 2), 0, 0]), d.qpos[adr + 3:adr + 7].copy())
            d.qpos[adr + 3:adr + 7] = q2
            d.qpos[adr + 2] += 0.002                 # lift slightly; it settles onto its edge during the settle period
    fr = float(os.environ.get("HARD_FRICTION", 0)); ms = float(os.environ.get("HARD_MASS", 0))
    if fr or ms:                                     # stickier blocks / lighter neighbours: the lift drags neighbours more
        for name in ("block_middle", "block_left", "block_right"):
            b = m.body(name).id
            if fr:
                m.geom_friction[np.flatnonzero(m.geom_bodyid == b), 0] = fr
            if ms and name != "block_middle":
                if not hasattr(sim, "_mass0"):
                    sim._mass0 = {}
                sim._mass0.setdefault(b, (float(m.body_mass[b]), m.body_inertia[b].copy()))
                m.body_mass[b] = sim._mass0[b][0] * ms; m.body_inertia[b] = sim._mass0[b][1] * ms
    sim.mj.mj_forward(m, d)


def episode(pol, i, video=False, chunk_hook=None, exec_noise=False, max_chunks=None, settle=False, seed=None, run_full=False,
            hard=None, snaps=None):
    """One closed-loop episode. chunk_hook(sim, chunk, k) -> (chunk to execute or None = hold, info) (monitor / filter
    go here). exec_noise: each executed chunk gets a measured execution-noise snippet (as every simulated future)."""
    import uncertainty_monitor_dev as um_
    snippets = np.load(um_.BENCH)["snippets"]; nrng = np.random.default_rng([20261502, i])
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="pol_"))))
    cams = Cameras(sim)
    try:
        sim.reset(RESET_BASE + i)
        if hard is not None:
            harden(sim, i, hard)
        if seed is not None:                         # paired runs: the policy's denoising noise is seeded per episode
            import torch
            torch.manual_seed(seed)
        if settle:                                   # let the blocks land (they spawn in the air) before the policy starts
            for _ in range(10):
                sim.execute(np.r_[sim.target, 0.0].astype(np.float32))
        z0 = None                                   # resting height, read after the first chunk (blocks spawn in the air)
        hist = [cams.obs()]; hist.append(hist[0])
        if settle:
            z0 = float(sim.data.xpos[sim.tracked_block_ids[0]][2])
        frames, chunks, peak_neigh, max_lift, log, lift_t = [], [], 0.0, 0.0, [], None
        for k in range(max_chunks or MAX_CHUNKS):
            plan = policy_chunk(pol, hist, full=True)                       # (16, 4) action_pred; steps 1..8 are executed
            chunk = plan[PLAN_START:PLAN_START + 8].copy()                   # identical to the policy's own "action" output
            nominal = chunk.copy()
            info = {}
            if snaps is not None and z0 is not None:
                snap_k = sim.snapshot(); tilt_k = sim.block_diagnostics()["tilt"][1:].copy()
            if chunk_hook is not None and z0 is not None:
                if getattr(chunk_hook, "wants_context", False):
                    chunk, info = chunk_hook(sim, chunk, k, plan=plan, hist=hist, pol=pol)
                else:
                    chunk, info = chunk_hook(sim, chunk, k)
                if chunk is None:                                            # abstain: hold for one chunk, re-plan
                    chunk = np.repeat(np.r_[sim.target, 0.0][None], 8, 0).astype(np.float32)
            chunks.append(chunk)
            if snaps is not None and z0 is not None:
                snaps.append({"chunk_index": k, "state": snap_k[0], "ctrl": snap_k[1], "target": snap_k[2], "gripper": snap_k[3],
                              "start_tilt": tilt_k, "nominal": nominal, "executed": np.asarray(chunk, np.float32),
                              "plan16": np.asarray(plan, np.float32)})
            noise = snippets[nrng.integers(len(snippets))] * float(os.environ.get("EXEC_NOISE_SCALE", 1.0)) if exec_noise else np.zeros((8, 3))
            peak_before = peak_neigh
            for j, a in enumerate(chunk):
                a = np.asarray(a, np.float32).copy(); a[:3] -= noise[j]
                sim.execute(a)
                o = cams.obs(); hist.append(o)
                if video:
                    fr = np.concatenate([o[1], o[0]], 1)
                    if chunk_hook is not None and info:
                        fr = annotate(fr, info, k)
                    frames.append(fr)
                if z0 is None:
                    continue
                tilt = sim.block_diagnostics()["tilt"]
                peak_neigh = max(peak_neigh, float(np.max(tilt[1:])))
                max_lift = max(max_lift, float(sim.data.xpos[sim.tracked_block_ids[0]][2] - z0))
            if z0 is None:
                z0 = float(sim.data.xpos[sim.tracked_block_ids[0]][2])
            log.append({**{k_: v for k_, v in info.items() if k_ != "starts"}, "chunk": k,
                        "topple_started": bool(peak_before < TOPPLE_DEG <= peak_neigh)})
            if lift_t is None and max_lift >= 0.03:
                lift_t = round((k + 1) * 0.8, 1)
            if max_lift >= LIFT_M and not run_full:
                break
        res = {"episode": i, "chunks": k + 1, "lifted": max_lift >= LIFT_M, "max_lift_mm": round(max_lift * 1000, 1),
               "neighbour_toppled": peak_neigh >= TOPPLE_DEG, "peak_neighbour_tilt": round(peak_neigh, 1)}
        res["success"] = res["lifted"] and not res["neighbour_toppled"]
        res["log"] = log; res["seconds_to_3cm"] = lift_t
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


def run(mode, n=100, start=0):
    """mode: none | monitor (log alarms, no intervention) | abstain (hold on alarm) | cem. Execution noise on."""
    import cem_filter as cf
    OUT.mkdir(parents=True, exist_ok=True)
    pol = load_policy()
    mon = cf.Monitor() if mode != "none" else None
    rng = np.random.default_rng(20261503)
    path = ROOT / f"results/jenga/policy_{mode}.jsonl"
    done = {json.loads(l)["episode"] for l in path.read_text().splitlines()} if path.exists() else set()
    per_ep = os.environ.get("PAIRED_PER_EP") == "1"              # cluster jobs: one result file per episode (no shared appends)
    if per_ep:
        done |= {int(q.stem[2:]) for q in sdir.glob("ep*.json")}

    def hook(sim, chunk, k):
        if mode == "monitor" or mode == "abstain":
            alarm, sec, _ = mon.check(sim, chunk)
            return (None if (mode == "abstain" and alarm) else chunk), {"alarm": alarm, "seconds": round(sec, 3)}
        out, info = cf.cem_filter(mon, sim, chunk, rng=rng)
        return out, info
    for i in range(start, start + n):
        if i in done:
            continue
        res, _, _ = episode(pol, i, chunk_hook=None if mode == "none" else hook, exec_noise=True)
        if per_ep:
            tmp = sdir / f"ep{i:03d}.json.tmp"; tmp.write_text(json.dumps(res) + "\n"); tmp.rename(sdir / f"ep{i:03d}.json")
        else:
            with open(path, "a") as f:
                f.write(json.dumps(res) + "\n")
        print(mode, {k: v for k, v in res.items() if k != "log"}, flush=True)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    summ = {"mode": mode, "episodes": len(rows), "success": sum(r["success"] for r in rows),
            "lifted": sum(r["lifted"] for r in rows), "neighbour_toppled": sum(r["neighbour_toppled"] for r in rows)}
    if mode != "none":
        L = [c for r in rows for c in r["log"]]
        summ.update({"decisions": len(L), "alarms": sum(bool(c.get("alarm") or c.get("intervened")) for c in L),
                     "abstained": sum(bool(c.get("abstained")) for c in L),
                     "decision_seconds_median": float(np.median([c["seconds"] for c in L if "seconds" in c])) if L else None})
    (ROOT / f"results/jenga/policy_{mode}_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ), flush=True)


def paired(mode, n=100, shrink=None):
    if mode == "cbfopt":                       # the single CBF optimisation filter: its own world model + calibration
        os.environ["CBF_FIX"] = "opt"
        for k in ("OPT_WM", "OPT_CALIB"):
            if os.environ.get(k):
                os.environ["CBF_WM" if k == "OPT_WM" else "CBF_CALIB"] = os.environ[k]
        import cbf_filter as cbff
        cbff.CFG["FIX"] = "opt"
        if os.environ.get("OPT_CALIB"):
            cbff.CALIB = Path(os.environ["OPT_CALIB"]) if os.path.isabs(os.environ["OPT_CALIB"]) else ROOT / os.environ["OPT_CALIB"]
        return _paired("cbfopt", "cbf", n, shrink)
    return _paired(mode, mode, n, shrink)


def _paired(name, mode, n=100, shrink=None):
    """Paired stress test: stress layouts, settled start, policy seeded per episode (4000 + i), execution noise on, 48 s.
    mode: monitor (alarms logged, actions untouched = the policy-alone baseline) | abstain (hold on alarm) | cem.
    Every decision state is saved for per-decision ground truth (eval/loop_truth.py)."""
    import cem_filter as cf
    shrink = shrink if shrink is not None else float(os.environ.get("HARD_SHRINK", 10))
    tag = f"paired_{name}" + ("" if int(os.environ.get("CEM_VERSION", 1)) == 1 or mode != "cem" else f"_v{os.environ['CEM_VERSION']}")
    tag += os.environ.get("PAIRED_TAG", "")                       # e.g. "_test" or a new experiment name
    path = ROOT / f"results/jenga/{tag}.jsonl"; sdir = OUT / tag; sdir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["episode"] for l in path.read_text().splitlines()} if path.exists() else set()
    per_ep = os.environ.get("PAIRED_PER_EP") == "1"              # cluster jobs: one result file per episode (no shared appends)
    if per_ep:
        done |= {int(q.stem[2:]) for q in sdir.glob("ep*.json")}
    pol, rng = load_policy(), np.random.default_rng(20261511)
    mon = cf.Monitor() if mode != "record" else None
    if mode == "cbf":
        import cbf_filter as cbff
        cbf = cbff.CBF(mon)

    def hook(sim, chunk, k, **ctx):
        if mode == "cbf":
            return cbf.filter(sim, chunk, k, **ctx)
        if mode in ("monitor", "abstain"):
            a, sec, _ = mon.check(sim, chunk)
            return (None if (mode == "abstain" and a) else chunk), {"alarm": a, "seconds": round(sec, 3)}
        return cf.cem_filter(mon, sim, chunk, rng=rng)
    shard, nshard = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))   # parallel workers: episode i % nshard == shard
    off = int(os.environ.get("EP_OFFSET", 0))                     # test episodes use a fresh index range
    for i in range(off, off + n):
        if i in done or i % nshard != shard:
            continue
        if os.environ.get("PER_EPISODE_SEEDS", "1") == "1":   # results independent of how episodes are split over workers
            rng = np.random.default_rng([20261511, i])
            if mon is not None:
                mon.rng = np.random.default_rng([20261501, i])
        snaps = []
        hook.wants_context = (mode == "cbf")
        res, _, _ = episode(pol, i, chunk_hook=None if mode == "record" else hook, exec_noise=True, max_chunks=60, settle=True,
                            seed=4000 + i, hard=None if os.environ.get("HARD_OFF") else shrink,
                            snaps=snaps)
        info = {c["chunk"]: c for c in res["log"]}
        if snaps:
            arr = {k: np.stack([s_[k] for s_ in snaps]) for k in snaps[0]}
            arr["episode"] = np.full(len(snaps), i)
            for key in ("alarm", "intervened", "abstained"):
                arr[key] = np.array([bool(info.get(int(c), {}).get(key, False)) for c in arr["chunk_index"]])
            np.savez_compressed(sdir / f"ep{i:03d}.npz", **arr)
        res["shrink_mm"] = shrink
        if per_ep:
            tmp = sdir / f"ep{i:03d}.json.tmp"; tmp.write_text(json.dumps(res) + "\n"); tmp.rename(sdir / f"ep{i:03d}.json")
        else:
            with open(path, "a") as f:
                f.write(json.dumps(res) + "\n")
        print(tag, {k: v for k, v in res.items() if k != "log"}, flush=True)


def calibrate(shrink, n=20):
    """Topple rate of the policy alone on stress layouts with a given gap reduction (seeded, settled, noise on)."""
    pol = load_policy(); rows = []
    for i in range(n):
        res, _, _ = episode(pol, i, exec_noise=True, max_chunks=60, settle=True, seed=3000 + i, hard=shrink)
        res.pop("log"); rows.append(res)
    out = {"shrink_mm": shrink, "episodes": n, "toppled": sum(r["neighbour_toppled"] for r in rows),
           "lifted_3cm": sum(r["seconds_to_3cm"] is not None for r in rows)}
    print(json.dumps(out), flush=True)
    return out


def diagnose(n=20):
    """Evaluation check: 60 s limit, policy starts after the blocks settle, seeded, no execution noise; every episode
    filmed (fixed | wrist camera); records time to 3 cm and to 5 cm lift."""
    import imageio_ffmpeg
    vdir = OUT / "diagnose"; vdir.mkdir(parents=True, exist_ok=True)
    pol = load_policy(); rows = []
    for i in range(n):
        res, frames, _ = episode(pol, i, video=True, max_chunks=75, settle=True, seed=1000 + i, run_full=True)
        res.pop("log"); rows.append(res); print(res, flush=True)
        w = imageio_ffmpeg.write_frames(str(vdir / f"ep{i:02d}_{'ok' if res['success'] else 'fail'}.mp4"), (640, 240), fps=10,
                                        codec="libx264", quality=7, macro_block_size=1)
        w.send(None)
        for fr in frames:
            w.send(np.ascontiguousarray(fr))
        w.close()
    out = {"episodes": n, "limit_s": 60, "settled_start": True, "exec_noise": False,
           "lift_3cm": sum(r["seconds_to_3cm"] is not None for r in rows), "lift_5cm": sum(r["lifted"] for r in rows),
           "success_5cm": sum(r["success"] for r in rows), "toppled": sum(r["neighbour_toppled"] for r in rows),
           "seconds_to_3cm": [r["seconds_to_3cm"] for r in rows], "rows": rows}
    (ROOT / "results/jenga/policy_diagnose.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "diagnose":
        diagnose()
    elif sys.argv[1] == "paired":
        paired(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 100)
    elif sys.argv[1] == "calibrate":
        calibrate(float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    elif sys.argv[1] == "films":
        films(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif sys.argv[1] == "baseline":
        baseline()
    else:
        run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100)
