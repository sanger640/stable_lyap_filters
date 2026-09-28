"""Generate a bounded TRAIN-only 64-action neighbourhood corpus for D4.

States are selected deterministically from the existing TRAIN trace corpus without consulting
topple/failure labels.  Each state is replayed in MuJoCo and all 64 execution-noise actions are
recorded for the eight-step intervention plus a shared 30-step hold.  The result is intentionally
small enough for a one-seed architecture gate and uses at most four simulator workers.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_state_data import step_state  # noqa: E402

HOLD = 30


def inventory(directory):
    rows = []
    for path in sorted(Path(directory).glob("ep*_seed*.npz")):
        data = np.load(path, allow_pickle=False)
        episode = str(data["episode_id"])
        seed = int(data["seed"])
        for state_index, start in enumerate(data["starts"]):
            rows.append((episode, seed, int(start), state_index))
    return rows


def select_states(rows, count, seed):
    """Episode-balanced deterministic selection, independent of outcomes."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    selected, episode_counts = [], {}
    cap = max(1, int(np.ceil(count / len({row[0] for row in rows}))))
    for index in order:
        row = rows[index]
        if episode_counts.get(row[0], 0) >= cap:
            continue
        selected.append(row); episode_counts[row[0]] = episode_counts.get(row[0], 0) + 1
        if len(selected) == count:
            break
    if len(selected) < count:
        used = set(selected)
        selected.extend(row for row in rows if row not in used)[:count - len(selected)]
    return sorted(selected, key=lambda row: (int(row[0]), row[1], row[2]))


def simulate(job):
    episode_id, seed, targets, lmdb, xml, snippets = job
    replay = JengaReplay(lmdb); episode = replay.episode(episode_id); replay.close()
    actions_all = np.asarray(episode.actions, np.float32)
    sim = DirectJengaSim(xml)
    output = []
    try:
        sim.reset(seed)
        for step, action in enumerate(actions_all):
            if step in targets:
                snapshot = sim.snapshot(); start = step_state(sim)
                nominal = actions_all[step:step + HORIZON]
                windows, traces = [], []
                for noise in snippets:
                    perturbed = nominal.copy(); perturbed[:, :3] -= noise
                    future = np.concatenate([perturbed, np.repeat(perturbed[-1:], HOLD, 0)])
                    sim.restore(snapshot); trace = []
                    for command in future:
                        sim.execute(command); trace.append(step_state(sim))
                    windows.append(future); traces.append(trace)
                output.append({"episode_id": episode_id, "seed": seed, "chunk_start": step,
                               "start": start, "actions": np.stack(windows).astype(np.float32),
                               "traces": np.stack(traces).astype(np.float32)})
                sim.restore(snapshot)
            sim.execute(action)
    finally:
        sim.close()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-data", default=str(ROOT / "results/jenga/trace_data"))
    parser.add_argument("--bench", default=str(ROOT / "results/jenga/bench/jenga_bench.npz"))
    parser.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    parser.add_argument("--output", default=str(ROOT / "results/jenga/d4_neighborhood_data.npz"))
    parser.add_argument("--manifest", default=str(ROOT / "results/jenga/d4_neighborhood_manifest.json"))
    parser.add_argument("--states", type=int, default=48)
    parser.add_argument("--selection-seed", type=int, default=404)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("D4 generation is capped at four workers to avoid exhausting the host")

    selected = select_states(inventory(args.trace_data), args.states, args.selection_seed)
    grouped = {}
    for episode, seed, step, _ in selected:
        grouped.setdefault((episode, seed), []).append(step)
    snippets = np.load(args.bench, allow_pickle=False)["snippets"]
    with tempfile.TemporaryDirectory(prefix="jenga_d4_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        jobs = [(ep, seed, set(steps), args.lmdb, xml, snippets)
                for (ep, seed), steps in grouped.items()]
        rows = []
        with ProcessPoolExecutor(args.workers) as pool:
            for number, result in enumerate(pool.map(simulate, jobs), 1):
                rows.extend(result)
                print(f"D4 replay {number}/{len(jobs)} ({len(rows)} states)", flush=True)
    rows.sort(key=lambda row: (int(row["episode_id"]), row["seed"], row["chunk_start"]))
    arrays = {name: np.stack([row[name] for row in rows])
              for name in ("start", "actions", "traces")}
    arrays.update({"episode_id": np.asarray([row["episode_id"] for row in rows]),
                   "seed": np.asarray([row["seed"] for row in rows]),
                   "chunk_start": np.asarray([row["chunk_start"] for row in rows])})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    output_path = Path(args.output).resolve()
    try:
        data_name = str(output_path.relative_to(ROOT))
    except ValueError:
        data_name = str(output_path)
    manifest = {"purpose": "D4 full-neighbourhood TRAIN-only architecture gate",
                "selection": "deterministic episode-balanced sample; no outcome labels",
                "states": len(rows), "probes": int(len(snippets)), "horizon": HORIZON,
                "hold": HOLD, "selection_seed": args.selection_seed,
                "workers_cap": 4, "source": "results/jenga/trace_data",
                "data": data_name}
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output}: {arrays['traces'].shape}")


if __name__ == "__main__":
    main()
