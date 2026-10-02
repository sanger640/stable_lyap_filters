"""Visual pipeline v2 on fresh panel 3: Step-1 replication and the fully visual (token-read) aware monitor.

Frozen protocol: results/jenga/visual_pipeline_protocol.json. Physics stays the exact simulator.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

PROTOCOL = ROOT / "results/jenga/visual_pipeline_panel3_protocol.json"
PANEL3 = ROOT / "results/program_b/visual_panel3"
OUT = ROOT / "results/program_b/visual_panel3"
RESULT = ROOT / "results/jenga/visual_pipeline_panel3_result.json"
RESET_BASE = 7000
EPISODES = 80
SELECT_SEED = 20261006
ERROR_MODEL = ROOT / "results/jenga/visual_perception_error_model.json"
HOLD_FRAMES = (17, 37)                      # trace indices of hold 10 and hold 30
SEEDS = {"hat": 20261030, "aware": 20261031, "v2_hyp": 20261032, "v2_probe": 20261033,
         "truth_A": 20261034, "truth_B": 20261035}


def screen_panel(workers):
    """Screen 80 seeded episodes (seed + 7000) exactly like panels 1-2 and select the panel."""
    import fresh_fragility_panel as fp
    from concurrent.futures import ProcessPoolExecutor
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    screen_file = PANEL3 / "screen.json"
    if not screen_file.exists():
        replay = JengaReplay(DEFAULT_LMDB); ids = list(replay.episode_ids); replay.close()
        episodes = sorted(np.random.default_rng(SELECT_SEED).choice(ids, EPISODES, replace=False).tolist(), key=int)
        fp.RESET_BASE = RESET_BASE
        with ProcessPoolExecutor(workers, initializer=_set_base) as pool:
            screen = {e: s for e, s in pool.map(fp.chain_episode, [(e, set(), str(PANEL3)) for e in episodes])}
        screen_file.write_text(json.dumps(screen) + "\n")
    screen = json.loads(screen_file.read_text())
    states = [(e, int(i), v["topples"]) for e, rows in screen.items() for i, v in rows.items()]
    forks = [s for s in states if 2 <= s[2] <= 62]
    safe = [s for s in states if s[2] == 0]
    pick = np.random.default_rng(SELECT_SEED + 1)
    forks = [forks[i] for i in sorted(pick.choice(len(forks), min(60, len(forks)), replace=False))]
    safe = [safe[i] for i in sorted(pick.choice(len(safe), min(100, len(safe)), replace=False))]
    return {(e, i): ("topple_fork" if t >= 2 else "safe") for e, i, t in forks + safe}, screen, len(states)


def _set_base():
    import fresh_fragility_panel as fp
    fp.RESET_BASE = RESET_BASE


class ErrorModel:
    def __init__(self):
        e = json.loads(ERROR_MODEL.read_text())
        self.pos_cov = np.asarray(e["position_covariance_m2"])
        self.rot_cov = [np.asarray(c) for c in e["rotation_covariance_rad2_per_block"]]

    def sample(self, seed, *key):
        rng = np.random.default_rng([seed, *key])
        pos = rng.multivariate_normal(np.zeros(9), self.pos_cov).reshape(3, 3)
        rot = np.stack([rng.multivariate_normal(np.zeros(3), c) for c in self.rot_cov])
        return pos, rot


def delta_to(sim, poses):
    """Perturbation (position deltas, rotation vectors) that moves the blocks to `poses` (3, 7 wxyz)."""
    from scipy.spatial.transform import Rotation
    from jenga_short_held_tails import BLOCK_NAMES
    m, d = sim.model, sim.data
    pos, rot = [], []
    for k, name in enumerate(BLOCK_NAMES):
        adr = m.jnt_qposadr[m.body_jntadr[m.body(name).id]]
        w, x, y, z = d.qpos[adr + 3:adr + 7]
        target = Rotation.from_quat(poses[k, [4, 5, 6, 3]])
        current = Rotation.from_quat([x, y, z, w])
        pos.append(poses[k, :3] - d.qpos[adr:adr + 3])
        rot.append((target * current.inv()).as_rotvec())
    return np.asarray(pos), np.asarray(rot)


def rollout_render(sim, snapshot, chunk, noise, perturbations, render):
    import program_b_reference as pr
    from jenga_state_data import step_state
    sim.restore(snapshot)
    for position, rotation in perturbations:
        pr.perturb_blocks(sim, position, rotation)
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise
    states, frames = [], []
    for k, action in enumerate(list(actions) + [chunk[-1]] * 30):
        sim.execute(action)
        states.append(step_state(sim))
        if render and k in HOLD_FRAMES:
            frames.append(sim.render())
    return np.stack(states), (np.stack(frames) if frames else None)


def worker(job):
    """Run one episode; return (episode, None) or (episode, traceback text) so errors surface."""
    import traceback
    try:
        return job[0], _worker(job)
    except Exception:
        return job[0], traceback.format_exc()


def _worker(job):
    episode_id, selected, out = job
    import mujoco
    import uncertainty_monitor_dev as um
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim
    from visual_perception_model import Perceiver
    from jenga_render_frames import close_camera
    perceive = Perceiver()
    errors = ErrorModel()
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    starts = set(chunk_starts(len(actions)))
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="vispipe_"))))
    close = mujoco.Renderer(sim.model, height=384, width=384)
    last = max(selected)
    try:
        sim.reset(int(episode_id) + RESET_BASE)
        for index, action in enumerate(actions):
            if index in starts and index + 8 <= len(actions):
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                target = index in selected
                exact = [rollout_render(sim, snapshot, chunk, n, [], target) for n in snippets]
                warm_end = sim.data.qacc_warmstart.copy()
                if target:
                    key = (int(episode_id) + RESET_BASE, index)
                    sim.restore(snapshot)
                    close.update_scene(sim.data, camera=close_camera(mujoco))
                    hat_poses = perceive(close.render()[None])[0]
                    true_pose = np.concatenate([sim.data.xpos[sim.tracked_block_ids],
                                                sim.data.xquat[sim.tracked_block_ids]], 1)
                    hat = delta_to(sim, hat_poses)
                    out_traces = {"exact": np.stack([t for t, _ in exact]).astype(np.float32)}
                    frames = {"exact": np.stack([f for _, f in exact])}
                    out_traces["naive"] = np.stack([rollout_render(sim, snapshot, chunk, n, [hat], False)[0]
                                                    for n in snippets]).astype(np.float32)
                    aware = [rollout_render(sim, snapshot, chunk, n,
                                            [hat, errors.sample(SEEDS["aware"], *key, j)], True)
                             for j, n in enumerate(snippets)]
                    out_traces["aware"] = np.stack([t for t, _ in aware]).astype(np.float32)
                    frames["aware"] = np.stack([f for _, f in aware])
                    for m in range(5):
                        d_m = errors.sample(SEEDS["v2_hyp"], *key, m)
                        out_traces[f"v2h{m}"] = np.stack([rollout_render(
                            sim, snapshot, chunk, n, [hat, d_m, errors.sample(SEEDS["v2_probe"], *key, m, j)],
                            False)[0] for j, n in enumerate(snippets)]).astype(np.float32)
                    for name in ("truth_A", "truth_B"):
                        out_traces[name] = np.stack([rollout_render(
                            sim, snapshot, chunk, n, [errors.sample(SEEDS[name], *key, j)], False)[0]
                            for j, n in enumerate(snippets)]).astype(np.float32)
                    np.savez_compressed(Path(out) / f"ep{episode_id}_s{index}.npz", hat_pose=hat_poses,
                                        true_pose=true_pose, **out_traces)
                    np.savez(Path(out) / f"ep{episode_id}_s{index}_frames.npz", **frames)
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm_end
                if index == last:
                    break
            sim.execute(action)
    finally:
        close.close()
        sim.close()
    return None


def token_decisions(frames_file, encoder, device="cuda"):
    """Split rule on flattened DINO tokens of rendered endings (hold 10, hold 30)."""
    import torch
    from torchvision.transforms import functional as TF
    from counterfactual_monitor import persistent_mode_alarm
    out = {}
    with np.load(frames_file) as f:
        for cond in f.files:
            frames = f[cond]                                        # (64, 2, H, W, 3)
            flat = []
            for t in range(2):
                x = torch.as_tensor(frames[:, t], device=device).permute(0, 3, 1, 2).float() / 255
                x = (TF.resize(x, [196, 196], antialias=True) - 0.5) / 0.5
                with torch.inference_mode():
                    flat.append(encoder.forward(x).float().reshape(len(x), -1).cpu().numpy())
            out[cond] = bool(persistent_mode_alarm(flat[0], flat[1]).alarm)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    PANEL3.mkdir(parents=True, exist_ok=True)
    panel, screen, screened = screen_panel(args.workers)
    print(f'screened {screened} states; panel {len(panel)}', flush=True)
    by_episode = {}
    for e, i in panel:
        by_episode.setdefault(e, set()).add(i)
    todo = [(e, idx, str(OUT)) for e, idx in sorted(by_episode.items(), key=lambda x: int(x[0]))
            if not all((OUT / f"ep{e}_s{i}.npz").exists() for i in idx)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        failures = []
        for number, (e, error) in enumerate(pool.imap_unordered(worker, todo), 1):
            print(f"{number}/{len(todo)} ep{e}" + (" FAILED" if error else ""), flush=True)
            if error:
                failures.append(e); print(error, flush=True)
        if failures:
            raise SystemExit(f"failed episodes: {failures}")

    import torch
    import fragility_grading_dev as fg
    from fresh_fragility_panel import topple_count
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings, wilson
    from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
    encoder = load_world_model(DEFAULT_CHECKPOINT, "cuda").encoder
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    conds = ("exact", "naive", "aware", "truth_A", "truth_B")
    dec = {c: [] for c in conds}
    dec["aware_v2"], tok, harm, perc = [], {"exact": [], "aware": []}, [], []
    for e, i in keys:
        with np.load(OUT / f"ep{e}_s{i}.npz") as f:
            for c in conds:
                dec[c].append(decide(*trace_endings(f[c], NON_ROBOT["jenga"]))[0])
            votes = [decide(*trace_endings(f[f"v2h{m}"], NON_ROBOT["jenga"]))[0] for m in range(5)]
            dec["aware_v2"].append(sum(votes) >= 3)
            start_tilt = np.array(screen[e][str(i)]["start_tilt"])
            t = topple_count(f["truth_A"], start_tilt)
            harm.append(2 <= t <= 62)
            perc.append(1000 * np.linalg.norm(f["hat_pose"][:, :3] - f["true_pose"][:, :3], axis=1))
        td = token_decisions(OUT / f"ep{e}_s{i}_frames.npz", encoder)
        for c in tok:
            tok[c].append(td[c])
    truth_a, truth_b = np.array(dec["truth_A"]), np.array(dec["truth_B"])
    ceiling = fg.grade(truth_b, truth_a)
    graded = {c: fg.grade(dec[c], truth_a) for c in ("exact", "naive", "aware", "aware_v2")}
    harm = np.array(harm)
    v2 = np.array(dec["aware_v2"])
    hits = int((v2 & harm).sum())
    g1 = bool(graded["aware_v2"]["kappa"] >= 0.8 * ceiling["kappa"] and graded["aware_v2"]["recall"] >= 0.80
              and graded["aware_v2"]["alarm_on_robust"] <= ceiling["alarm_on_robust"] + 0.10)
    g2 = bool(graded["aware_v2"]["kappa"] - graded["naive"]["kappa"] >= 0.15)
    g3 = bool(harm.sum() and hits / harm.sum() >= 0.80)
    tok_aware = np.array(tok["aware"])
    s2_harm = int((tok_aware & harm).sum()) / harm.sum() if harm.sum() else None
    s2_nonharm = float((tok_aware & ~harm).sum() / max((~harm).sum(), 1))
    s2_kappa = fg.kappa(tok_aware, truth_a)
    s2 = bool(s2_harm is not None and s2_harm >= 0.80 and s2_nonharm <= 0.20 and s2_kappa >= 0.60)
    graded["aware_token"] = fg.grade(tok_aware, truth_a)
    token = {c: {"kappa_vs_state": fg.kappa(tok[c], dec[c]), "agreement": float(np.mean(np.array(tok[c]) == np.array(dec[c]))),
                 "token_alarms": int(np.sum(tok[c])), "state_alarms": int(np.sum(dec[c]))} for c in tok}
    gate_t = bool(all(token[c]["kappa_vs_state"] >= 0.70 for c in token))
    perc = np.array(perc)
    result = {
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "panel_states": len(keys), "truth_fragile_A": int(truth_a.sum()),
        "perception_error_mm_per_block": {"mean": perc.mean(0).tolist(), "p90": np.percentile(perc, 90, axis=0).tolist()},
        "ceiling_A_vs_B": ceiling, "graded_vs_truth_A": graded,
        "harm": {"harm_fork_states": int(harm.sum()), "aware_v2_recall": hits / harm.sum() if harm.sum() else None,
                 "aware_v2_recall_wilson": wilson(hits, int(harm.sum()))},
        "token": token,
        "visual_monitor": {"harm_recall": s2_harm, "alarm_on_non_harm": s2_nonharm, "kappa_vs_truth_A": s2_kappa},
        "gates": {"G1_calibration": g1, "G2_beats_naive": g2, "G3_harm": g3, "S2_visual_monitor": s2,
                  "token_vs_state_kappa_ge_0.70_descriptive": gate_t},
        "screened_states": screened,
        "S1_replicated": bool(g1 and g2 and g3), "S2_confirmed": s2,
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("perception_error_mm_per_block", "ceiling_A_vs_B", "graded_vs_truth_A",
                                            "harm", "visual_monitor", "token", "gates")}, indent=1))


if __name__ == "__main__":
    main()
