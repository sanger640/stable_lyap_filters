"""Fully visual Bayesian v2 monitor on fresh panel 4 (shared with Step 3).

Frozen protocol: results/jenga/visual_v2_panel4_protocol.json. Same chain replay, perception and token
reading as panel 3 (eval/visual_pipeline_panel3.py); v2 hypotheses are now rendered and read from tokens
inside the workers. Also stores the close-camera frame of each true start state for Step 3.
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

import visual_pipeline_panel3 as p3  # noqa: E402

PROTOCOL = ROOT / "results/jenga/visual_v2_panel4_protocol.json"
OUT = ROOT / "results/program_b/panel4"
RESULT = ROOT / "results/jenga/visual_v2_panel4_result.json"
RESET_BASE = 8000
EPISODES = 80
SELECT_SEED = 20261043
SEEDS = {"hat": 20261050, "aware": 20261051, "v2_hyp": 20261052, "v2_probe": 20261053,
         "truth_A": 20261054, "truth_B": 20261055}


def _set_base():
    import fresh_fragility_panel as fp
    fp.RESET_BASE = RESET_BASE


def screen_panel(workers):
    import fresh_fragility_panel as fp
    from concurrent.futures import ProcessPoolExecutor
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    screen_file = OUT / "screen.json"
    if not screen_file.exists():
        replay = JengaReplay(DEFAULT_LMDB); ids = list(replay.episode_ids); replay.close()
        episodes = sorted(np.random.default_rng(SELECT_SEED).choice(ids, EPISODES, replace=False).tolist(), key=int)
        with ProcessPoolExecutor(workers, initializer=_set_base) as pool:
            screen = {e: s for e, s in pool.map(fp.chain_episode, [(e, set(), str(OUT)) for e in episodes])}
        screen_file.write_text(json.dumps(screen) + "\n")
    screen = json.loads(screen_file.read_text())
    states = [(e, int(i), v["topples"]) for e, rows in screen.items() for i, v in rows.items()]
    forks = [s for s in states if 2 <= s[2] <= 62]
    safe = [s for s in states if s[2] == 0]
    pick = np.random.default_rng(SELECT_SEED + 1)
    forks = [forks[i] for i in sorted(pick.choice(len(forks), min(60, len(forks)), replace=False))]
    safe = [safe[i] for i in sorted(pick.choice(len(safe), min(100, len(safe)), replace=False))]
    return {(e, i): ("topple_fork" if t >= 2 else "safe") for e, i, t in forks + safe}, screen, len(states)


def token_alarm(encoder, frames, device="cuda"):
    """Identical to Step 2 / panel 3 token reading: (64, 2, H, W, 3) uint8 -> split-rule alarm."""
    import torch
    from torchvision.transforms import functional as TF
    from counterfactual_monitor import persistent_mode_alarm
    flat = []
    for t in range(2):
        x = torch.as_tensor(frames[:, t], device=device).permute(0, 3, 1, 2).float() / 255
        x = (TF.resize(x, [196, 196], antialias=True) - 0.5) / 0.5
        with torch.inference_mode():
            flat.append(encoder.forward(x).float().reshape(len(x), -1).cpu().numpy())
    return bool(persistent_mode_alarm(flat[0], flat[1]).alarm)


def worker(job):
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
    encoder = perceive.encoder
    errors = p3.ErrorModel()
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    starts = set(chunk_starts(len(actions)))
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="panel4_"))))
    close = mujoco.Renderer(sim.model, height=384, width=384)
    last = max(selected)
    render = p3.rollout_render
    try:
        sim.reset(int(episode_id) + RESET_BASE)
        for index, action in enumerate(actions):
            if index in starts and index + 8 <= len(actions):
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                target = index in selected
                exact = [render(sim, snapshot, chunk, n, [], target) for n in snippets]
                warm_end = sim.data.qacc_warmstart.copy()
                if target:
                    key = (int(episode_id) + RESET_BASE, index)
                    sim.restore(snapshot)
                    close.update_scene(sim.data, camera=close_camera(mujoco))
                    close_frame = close.render().copy()
                    hat_poses = perceive(close_frame[None])[0]
                    true_pose = np.concatenate([sim.data.xpos[sim.tracked_block_ids],
                                                sim.data.xquat[sim.tracked_block_ids]], 1)
                    hat = p3.delta_to(sim, hat_poses)
                    traces = {"exact": np.stack([t for t, _ in exact]).astype(np.float32)}
                    tokens = {"exact": token_alarm(encoder, np.stack([f for _, f in exact]))}
                    traces["naive"] = np.stack([render(sim, snapshot, chunk, n, [hat], False)[0]
                                                for n in snippets]).astype(np.float32)
                    aware = [render(sim, snapshot, chunk, n, [hat, errors.sample(SEEDS["aware"], *key, j)], True)
                             for j, n in enumerate(snippets)]
                    traces["aware"] = np.stack([t for t, _ in aware]).astype(np.float32)
                    aware_frames = np.stack([f for _, f in aware])
                    tokens["aware"] = token_alarm(encoder, aware_frames)
                    for m in range(5):
                        d_m = errors.sample(SEEDS["v2_hyp"], *key, m)
                        hyp = [render(sim, snapshot, chunk, n,
                                      [hat, d_m, errors.sample(SEEDS["v2_probe"], *key, m, j)], True)
                               for j, n in enumerate(snippets)]
                        traces[f"v2h{m}"] = np.stack([t for t, _ in hyp]).astype(np.float32)
                        tokens[f"v2h{m}"] = token_alarm(encoder, np.stack([f for _, f in hyp]))
                    for name in ("truth_A", "truth_B"):
                        traces[name] = np.stack([render(sim, snapshot, chunk, n, [errors.sample(SEEDS[name], *key, j)],
                                                        False)[0] for j, n in enumerate(snippets)]).astype(np.float32)
                    path = Path(out) / f"ep{episode_id}_s{index}.npz"
                    np.savez_compressed(path.with_suffix(".tmp.npz"), hat_pose=hat_poses, true_pose=true_pose,
                                        close_frame=close_frame, chunk=chunk,
                                        **{f"tok_{k}": np.array(v) for k, v in tokens.items()}, **traces)
                    np.savez(Path(out) / f"ep{episode_id}_s{index}_frames.npz", aware=aware_frames)
                    path.with_suffix(".tmp.npz").replace(path)
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm_end
                if index == last:
                    break
            sim.execute(action)
    finally:
        close.close()
        sim.close()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    panel, screen, screened = screen_panel(args.workers)
    print(f"screened {screened} states; panel {len(panel)}", flush=True)
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

    import fragility_grading_dev as fg
    from fresh_fragility_panel import topple_count
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings, wilson
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    conds = ("exact", "naive", "aware", "truth_A", "truth_B")
    dec = {c: [] for c in conds}
    for extra in ("v2_state", "v2_token", "v1_token", "exact_token"):
        dec[extra] = []
    harm, wrong_side, hyp_agree = [], [], []
    for e, i in keys:
        with np.load(OUT / f"ep{e}_s{i}.npz") as f:
            for c in conds:
                dec[c].append(decide(*trace_endings(f[c], NON_ROBOT["jenga"]))[0])
            state_votes = [decide(*trace_endings(f[f"v2h{m}"], NON_ROBOT["jenga"]))[0] for m in range(5)]
            token_votes = [bool(f[f"tok_v2h{m}"]) for m in range(5)]
            hyp_agree.extend(a == b for a, b in zip(state_votes, token_votes))
            dec["v2_state"].append(sum(state_votes) >= 3)
            dec["v2_token"].append(sum(token_votes) >= 3)
            dec["v1_token"].append(bool(f["tok_aware"]))
            dec["exact_token"].append(bool(f["tok_exact"]))
            start_tilt = np.array(screen[e][str(i)]["start_tilt"])
            harm.append(2 <= topple_count(f["truth_A"], start_tilt) <= 62)
            wrong_side.append(all(topple_count(f[f"v2h{m}"], start_tilt) <= 3 for m in range(5)))
    harm, wrong_side = np.array(harm), np.array(wrong_side)
    truth_a = np.array(dec["truth_A"])
    ceiling = fg.grade(np.array(dec["truth_B"]), truth_a)
    graded = {c: fg.grade(dec[c], truth_a) for c in ("exact", "naive", "aware", "v2_state", "v2_token", "v1_token")}

    def harm_stats(name):
        d = np.array(dec[name])
        hits = int((d & harm).sum())
        return {"recall": hits / harm.sum() if harm.sum() else None, "recall_wilson": wilson(hits, int(harm.sum())),
                "alarm_on_non_harm": float((d & ~harm).sum() / max((~harm).sum(), 1)),
                "missed": int((~d & harm).sum()), "missed_wrong_side": int((~d & harm & wrong_side).sum())}

    harm_table = {c: harm_stats(c) for c in ("v2_token", "v2_state", "v1_token", "aware", "naive")}
    v2t = harm_table["v2_token"]
    gate = bool(v2t["recall"] is not None and v2t["recall"] >= 0.80 and v2t["alarm_on_non_harm"] <= 0.20
                and graded["v2_token"]["kappa"] >= 0.60)
    result = {
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "screened_states": screened, "panel_states": len(keys), "harm_fork_states": int(harm.sum()),
        "truth_fragile_A": int(truth_a.sum()), "ceiling_A_vs_B": ceiling, "graded_vs_truth_A": graded,
        "harm": harm_table, "v2_hypothesis_token_vs_state_agreement": float(np.mean(hyp_agree)),
        "gates": {"V2_fully_visual_v2": gate}, "confirmed": gate,
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("ceiling_A_vs_B", "harm", "gates")}, indent=1))


if __name__ == "__main__":
    main()
