"""W0: the frozen response-curve benchmark for world-model discontinuity.

Reality jumps when a chunk is pushed across an outcome boundary; the current models ramp. This
measures that directly and is the scoreboard for PLAN_WORLDMODEL W1-W3 -- never rollout MSE, which
fell 10x while the monitor stayed dead.

For each state, push every action of the chunk by a signed offset along a fixed direction, run the
chunk plus the 30-step hold, and record (a) the simulator's settled block positions and (b) each
model's predicted ending latent. Two scale-free metrics per curve:

  spread     - median pairwise distance between endings over the offset grid
  jump_ratio - largest slope between adjacent offsets / median slope (a smooth ramp gives ~1)

No labels enter either metric. Topple counts are recorded for reporting only.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay,  # noqa: E402
                           NUM_HIST, load_world_model)
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import HOLD  # noqa: E402
from jenga_stage3_predicted_forks import action_windows, predict_state  # noqa: E402

OFFSETS_MM = np.array([-50, -20, -10, -5, -2.5, -1.6, 0, 1.6, 2.5, 5, 10, 20, 50], float)
DIRECTIONS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def curve_metrics(endings, offsets_mm):
    """endings (n, d) along the offset grid -> spread and jump ratio."""
    endings = np.asarray(endings, float)
    pairs = [np.linalg.norm(endings[i] - endings[j])
             for i in range(len(endings)) for j in range(i + 1, len(endings))]
    steps = np.diff(np.asarray(offsets_mm, float))
    slopes = np.array([np.linalg.norm(endings[i + 1] - endings[i]) / steps[i]
                       for i in range(len(endings) - 1)])
    median = float(np.median(slopes))
    return {"spread": float(np.median(pairs)),
            "jump_ratio": float(np.max(slopes) / median) if median > 0 else None,
            "max_slope_at_mm": float(offsets_mm[int(np.argmax(slopes)) + 1])}


def offset_windows(actions, start, direction, offsets_mm):
    """Reuse the Stage 3 window builder: a constant offset is a constant 'noise' snippet."""
    snippets = np.stack([-np.tile(np.asarray(direction, np.float32) * (mm / 1000.0),
                                  (HORIZON, 1)) for mm in offsets_mm])
    return action_windows(actions, start, snippets, 1.0, own_hold=False)


def simulate_curve(sim, snapshot, chunk, direction, offsets_mm):
    positions, topples = [], []
    start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
    for mm in offsets_mm:
        sim.restore(snapshot)
        actions = np.asarray(chunk, np.float32).copy()
        actions[:, :3] += np.asarray(direction, np.float32) * (mm / 1000.0)
        for action in actions:
            sim.execute(action)
        for _ in range(HOLD):
            sim.execute(actions[-1])
        positions.append(1000 * sim.block_diagnostics()["position"].reshape(-1).copy())
        tilt = sim.block_diagnostics()["tilt"][1:]
        topples.append(bool(np.any((tilt >= TOPPLE_DEG) & (start_tilt < TOPPLE_DEG))))
    sim.restore(snapshot)
    return np.stack(positions), topples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--states", default=str(ROOT / "results/jenga/holdout2_stage2_shared.json"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/w0_response_curves.json"))
    ap.add_argument("--forks", type=int, default=20)
    ap.add_argument("--quiet", type=int, default=20)
    ap.add_argument("--model", action="append", default=[], help="name=path.pt (repeatable)")
    args = ap.parse_args()
    models = args.model or [f"shipped={DEFAULT_CHECKPOINT}",
                            f"fine-tuned={ROOT}/results/jenga/world_model_gtf.pt"]

    rows = json.loads(Path(args.states).read_text())["rows"]
    forks = [r for r in rows if r["by_scale"]["2.0"]["class"] == "topple_fork"][:args.forks]
    quiet = [r for r in rows if r["by_scale"]["2.0"]["class"] == "quiet"][:args.quiet]
    states = [dict(r, group=g) for g, part in (("fork", forks), ("quiet", quiet)) for r in part]
    by_episode = {}
    for r in states:
        by_episode.setdefault(r["episode_id"], []).append(r)

    replay = JengaReplay(args.lmdb)
    loaded = []
    for spec in models:
        name, path = spec.split("=", 1)
        loaded.append((name, load_world_model(path, "cuda")))
    records = []
    with tempfile.TemporaryDirectory(prefix="jenga_w0_") as temp:
        sim = DirectJengaSim(str(extract_sim(args.sim_archive, temp)))
        try:
            for number, episode_id in enumerate(sorted(by_episode, key=int)):
                episode = replay.episode(episode_id)
                starts = {r["chunk_start"]: r for r in by_episode[episode_id]}
                sim.reset(int(episode_id))
                frames, props = [sim.render()], [sim.proprio()]
                for step, action in enumerate(episode.actions):
                    if step in starts:
                        history = (np.stack(frames[-NUM_HIST:]), np.stack(props[-NUM_HIST:]))
                        chunk = episode.actions[step:step + HORIZON]
                        snapshot = sim.snapshot()
                        for axis, direction in DIRECTIONS.items():
                            positions, topples = simulate_curve(sim, snapshot, chunk, direction,
                                                                OFFSETS_MM)
                            record = {"episode_id": episode_id, "chunk_start": step,
                                      "group": starts[step]["group"], "direction": axis,
                                      "real": curve_metrics(positions, OFFSETS_MM),
                                      "real_topple_count": int(sum(topples)),
                                      "models": {}}
                            windows = offset_windows(episode.actions, step, direction, OFFSETS_MM)
                            for name, model in loaded:
                                tokens = predict_state(model, "cuda", history[0], history[1],
                                                       windows)[HOLD].cpu().numpy()
                                record["models"][name] = curve_metrics(tokens, OFFSETS_MM)
                            records.append(record)
                    sim.execute(action)
                    frames.append(sim.render()); props.append(sim.proprio())
                    frames, props = frames[-NUM_HIST:], props[-NUM_HIST:]
                print(f"  w0 {number + 1}/{len(by_episode)} ep{episode_id}", flush=True)
        finally:
            sim.close()
            replay.close()

    summary = {}
    for group in ("fork", "quiet"):
        part = [r for r in records if r["group"] == group]
        entry = {"curves": len(part),
                 "real": {"jump_ratio_median": float(np.median(
                     [r["real"]["jump_ratio"] for r in part])),
                     "spread_median": float(np.median([r["real"]["spread"] for r in part]))}}
        for name, _ in loaded:
            jumps = [r["models"][name]["jump_ratio"] for r in part]
            spreads = [r["models"][name]["spread"] for r in part]
            relative = [r["models"][name]["spread"] / r["real"]["spread"]
                        for r in part if r["real"]["spread"] > 0]
            entry[name] = {"jump_ratio_median": float(np.median(jumps)),
                           "spread_median": float(np.median(spreads)),
                           "spread_relative_to_real_median": float(np.median(relative))}
        summary[group] = entry

    result = {"protocol": {
        "offsets_mm": OFFSETS_MM.tolist(), "directions": list(DIRECTIONS),
        "horizon": HORIZON, "hold": HOLD,
        "metrics": "spread = median pairwise ending distance over the grid; jump_ratio = max "
                   "adjacent slope / median adjacent slope (smooth ramp ~1)",
        "labels_used": "none in the metrics; topple counts reported only",
        "grading_rule": "PLAN_WORLDMODEL gates are read off this file, never rollout MSE"},
        "summary": summary, "records": records}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
