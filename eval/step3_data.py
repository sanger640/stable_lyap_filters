"""Step 3 data: channel-PCA-8 latent, TRAIN screen, selection, rendered futures, gate D0.

Frozen protocol: results/jenga/step3_token_world_model_protocol.json. Every future starts from the TRUE
TRAIN state plus a draw from the measured perception error model (and a bench execution-noise snippet),
so the spread of futures for one image is the robot's own uncertainty. Frames are close-camera renders
encoded with frozen DINOv2 (392) and projected onto the fixed channel PCA inside the workers.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

PROTOCOL = ROOT / "results/jenga/step3_token_world_model_protocol.json"
OUT = ROOT / "results/program_b/step3"
PCA_FILE = OUT / "channel_pca8.pt"
ENCODED_CLOSE = ROOT / "results/program_b/vjepa/enc_dinov2_392_close"
SEEDS = {"screen": 20261045, "select": 20261040, "futures": 20261041, "d0": 20261042, "d0_futures": 20261046}
K_FUTURES, K_SCREEN, D0_FUTURES = 8, 8, 64
TRAIN_STATES, VAL_STATES, DISAGREE_CAP = 1500, 150, 750
FRAME_INDEX = tuple(range(1, 38, 2))            # trace indices 1, 3, ..., 37 (hold 10 = 17, hold 30 = 37)
D0_INDEX = (17, 37)


# ------------------------------------------------------------------------------------- latent
def fit_pca():
    """Channel PCA 8 on the training split of the close-camera TRAIN tokens (perception split)."""
    import torch
    import jenga_spatial_readout as sr
    from token_compression_test import ChannelPCA
    if PCA_FILE.exists():
        return
    grids, _, ids = sr.load(ENCODED_CLOSE)
    unique = sorted(set(ids)); val = set(unique[:max(1, int(len(unique) * 0.12))])
    train = grids[np.array([i not in val for i in ids])]
    pca = ChannelPCA(8).fit(train)
    z = pca.transform(train)
    torch.save({"mean": pca.mean, "basis": pca.basis, "z_mean": torch.from_numpy(z.mean((0, 1, 2))),
                "z_std": torch.from_numpy(z.std((0, 1, 2)))}, PCA_FILE)


class Latent:
    """close-camera frames (N, 384, 384, 3) uint8 -> standardised (N, 28, 28, 8) float16."""

    def __init__(self, device="cuda"):
        import torch
        from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
        saved = torch.load(PCA_FILE, map_location=device)
        self.mean, self.basis = saved["mean"].to(device), saved["basis"].to(device)
        self.z_mean, self.z_std = saved["z_mean"].to(device), saved["z_std"].to(device)
        self.encoder = load_world_model(DEFAULT_CHECKPOINT, device).encoder
        self.device = device

    def __call__(self, frames):
        from jenga_vjepa_encode import dino_tokens
        tokens = dino_tokens(self.encoder, np.asarray(frames), self.device)          # (N, 784, 384)
        z = ((tokens - self.mean) @ self.basis - self.z_mean) / self.z_std
        return z.reshape(len(z), 28, 28, 8).cpu().numpy().astype(np.float16)


# ------------------------------------------------------------------------------- simulation
def train_chains():
    """(episode, seed) pairs of the V1 TRAIN layouts, with the held-out episode set."""
    episodes = sorted({p.name.split("_seed")[0][2:] for p in ENCODED_CLOSE.glob("ep*_seed*.npz")}, key=int)
    held = set(episodes[::10])
    return [(e, s) for e in episodes for s in range(100, 110)], held


def future(sim, snapshot, chunk, noise, perturbation, render_at=(), renderer=None, camera=None):
    """One future from the true state + perturbation; returns (61-D trace, executed actions, frames)."""
    import program_b_reference as pr
    from jenga_state_data import step_state
    sim.restore(snapshot)
    pr.perturb_blocks(sim, *perturbation)
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise
    executed = list(actions) + [chunk[-1]] * 30
    states, frames = [], []
    for k, action in enumerate(executed):
        sim.execute(action)
        states.append(step_state(sim))
        if k in render_at:
            renderer.update_scene(sim.data, camera=camera)
            frames.append(renderer.render().copy())
    return np.stack(states), np.asarray(executed, np.float32), frames


def chain_job(job):
    import traceback
    try:
        return job[0], job[1], _chain_job(job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _chain_job(job):
    """Replay one TRAIN chain (plain replay: warm start restored after every probe set).
    mode 'screen': 8 state-only futures at every chunk start -> topple counts.
    mode 'render': at the selected starts, the encoded true start frame plus K futures (train / val)
    or 64 hold-frame futures (d0)."""
    episode_id, seed, mode, selected, out = job
    import mujoco
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import topple_count
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim
    errors = p3.ErrorModel()
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    starts = [s for s in chunk_starts(len(actions)) if s + 8 <= len(actions)]
    targets = starts if mode == "screen" else sorted(selected)
    if not targets:
        return {}
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="step3_"))))
    renderer = camera = latent = None
    if mode == "render":
        renderer = mujoco.Renderer(sim.model, height=384, width=384)
        camera = close_camera(mujoco)
        latent = Latent()
    result = {}
    try:
        sim.reset(int(seed))
        for index, action in enumerate(actions):
            if index in targets:
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                warm0 = sim.data.qacc_warmstart.copy()
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                key = (int(episode_id), int(seed), int(index))
                if mode == "screen":
                    traces = []
                    for k in range(K_SCREEN):
                        rng = np.random.default_rng([SEEDS["screen"], *key, k])
                        traces.append(future(sim, snapshot, chunk, snippets[rng.integers(len(snippets))],
                                             errors.sample(SEEDS["screen"], *key, k))[0])
                    result[index] = {"topples": int(topple_count(traces, start_tilt)),
                                     "start_tilt": start_tilt.tolist()}
                else:
                    for role in selected[index]:
                        sim.restore(snapshot)
                        renderer.update_scene(sim.data, camera=camera)
                        start_frame = renderer.render().copy()
                        if role == "d0":
                            n, seed_name, at = D0_FUTURES, "d0_futures", D0_INDEX
                        else:
                            n, seed_name, at = K_FUTURES, "futures", FRAME_INDEX
                        traces, executed, frames = [], [], []
                        for k in range(n):
                            noise = snippets[k] if role == "d0" else snippets[
                                np.random.default_rng([SEEDS[seed_name], *key, k]).integers(len(snippets))]
                            t, a, f = future(sim, snapshot, chunk, noise, errors.sample(SEEDS[seed_name], *key, k),
                                             at, renderer, camera)
                            traces.append(t); executed.append(a); frames.extend(f)
                        z = latent(np.concatenate([start_frame[None], np.stack(frames)]))
                        path = Path(out) / role / f"ep{episode_id}_seed{seed}_s{index}.npz"
                        np.savez(path.with_suffix(".tmp.npz"), z0=z[0], z=z[1:].reshape(n, len(at), 28, 28, 8),
                                 traces=np.stack(traces).astype(np.float32), actions=np.stack(executed),
                                 start_frame=start_frame, start_tilt=start_tilt, frame_index=np.array(at))
                        path.with_suffix(".tmp.npz").replace(path)
                        result.setdefault(index, []).append(role)
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm0
                if index == max(targets):
                    break
            sim.execute(action)
    finally:
        if renderer is not None:
            renderer.close()
        sim.close()
    return result


def run_pool(jobs, workers, label):
    ctx = mp.get_context("spawn")
    results, failures = {}, []
    with ctx.Pool(workers) as pool:
        for n, (e, s, res, err) in enumerate(pool.imap_unordered(chain_job, jobs), 1):
            if err:
                failures.append((e, s)); print(err, flush=True)
            else:
                results[f"{e}:{s}"] = res
            if n % 10 == 0 or n == len(jobs):
                print(f"{label} {n}/{len(jobs)}", flush=True)
    if failures:
        raise SystemExit(f"{label} failures: {failures}")
    return results


def select(screen, held):
    rng = np.random.default_rng(SEEDS["select"])
    rows = [(k.split(":")[0], int(k.split(":")[1]), int(i), v["topples"]) for k, r in screen.items()
            for i, v in r.items()]

    def pick(pool, total, cap):
        disagree = [r for r in pool if 0 < r[3] < K_SCREEN]
        other = [r for r in pool if not 0 < r[3] < K_SCREEN]
        a = [disagree[i] for i in rng.permutation(len(disagree))[:cap]]
        b = [other[i] for i in rng.permutation(len(other))[:total - len(a)]]
        return a, b

    train_a, train_b = pick([r for r in rows if r[0] not in held], TRAIN_STATES, DISAGREE_CAP)
    held_rows = [r for r in rows if r[0] in held]
    val_a, val_b = pick(held_rows, VAL_STATES, VAL_STATES // 2)
    # D0 draws from ALL held-out states, independently of validation (it renders its own 64 futures and
    # trains nothing); the held-out episodes hold too few disagreeing states for disjoint sets.
    d0_rng = np.random.default_rng(SEEDS["d0"])
    dis = [r for r in held_rows if 0 < r[3] < K_SCREEN]; oth = [r for r in held_rows if not 0 < r[3] < K_SCREEN]
    d0 = [dis[i] for i in d0_rng.permutation(len(dis))[:30]] + [oth[i] for i in d0_rng.permutation(len(oth))[:30]]
    roles = {}
    for name, group in (("train", train_a + train_b), ("val", val_a + val_b), ("d0", d0)):
        for e, s, i, _ in group:
            roles.setdefault((e, s), {}).setdefault(i, []).append(name)
    summary = {"train": len(train_a + train_b), "train_disagree": len(train_a), "val": len(val_a + val_b),
               "val_disagree": len(val_a), "d0": len(d0), "d0_disagree": min(30, len(dis)),
               "screened_states": len(rows), "screened_disagree": sum(0 < r[3] < K_SCREEN for r in rows)}
    return roles, summary


def d0_gate():
    import fragility_grading_dev as fg
    from counterfactual_monitor import persistent_mode_alarm
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings
    lat, st = [], []
    for path in sorted((OUT / "d0").glob("ep*.npz")):
        with np.load(path) as f:
            z = f["z"].astype(np.float64)                                    # (64, 2, 28, 28, 8)
            lat.append(bool(persistent_mode_alarm(z[:, 0].reshape(64, -1), z[:, 1].reshape(64, -1)).alarm))
            st.append(decide(*trace_endings(f["traces"], NON_ROBOT["jenga"]))[0])
    kappa = fg.kappa(lat, st)
    result = {"states": len(lat), "latent_alarms": int(sum(lat)), "state_alarms": int(sum(st)),
              "agreement": float(np.mean(np.array(lat) == np.array(st))), "kappa": kappa, "pass": bool(kappa >= 0.70)}
    (OUT / "d0_result.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    for d in ("train", "val", "d0"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    fit_pca()
    print("pca ready", flush=True)
    chains, held = train_chains()
    screen_file = OUT / "screen.json"
    if not screen_file.exists():
        screen = run_pool([(e, s, "screen", None, str(OUT)) for e, s in chains], args.workers, "screen")
        screen_file.write_text(json.dumps(screen) + "\n")
    screen = json.loads(screen_file.read_text())
    roles, summary = select(screen, held)
    (OUT / "selection.json").write_text(json.dumps(
        {"summary": summary, "roles": {f"{e}:{s}": v for (e, s), v in roles.items()}}, indent=0) + "\n")
    print(json.dumps(summary), flush=True)
    jobs = []
    for (e, s), sel in sorted(roles.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        todo = {i: [r for r in rs if not (OUT / r / f"ep{e}_seed{s}_s{i}.npz").exists()] for i, rs in sel.items()}
        todo = {i: rs for i, rs in todo.items() if rs}
        if todo:
            jobs.append((e, s, "render", todo, str(OUT)))
    # D0 chains first so the gate can be read early
    jobs.sort(key=lambda j: not any("d0" in rs for rs in j[3].values()))
    run_pool(jobs, args.workers, "render")
    print(json.dumps({"D0": d0_gate()}), flush=True)


if __name__ == "__main__":
    main()
