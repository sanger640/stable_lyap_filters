"""All-three-block physical tail 0/5/10 scan over 100 Jenga trajectories."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_failures import failure_masks  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import (CONTACT_LABELS, DirectJengaSim, HORIZON, TAILS,
                                    chunk_starts, extract_sim)  # noqa: E402


def branch(sim, snapshot, chunk, tail):
    sim.restore(snapshot); start = sim.block_diagnostics(); contact = sim.contact_signature()
    trace = {"peak_tilt": start["tilt"].copy(),
             "min_z": start["position"][:, 2].copy(),
             "max_z": start["position"][:, 2].copy(),
             "contact_seen": contact.copy(),
             "contact_transitions": np.zeros(len(CONTACT_LABELS), np.int32),
             "previous_contact": contact.copy()}
    for action in chunk:
        sim.execute(action, trace)
    for _ in range(int(tail)):
        sim.execute(chunk[-1], trace)
    end = sim.block_diagnostics()
    return {"peak_tilt": trace["peak_tilt"], "min_z": trace["min_z"],
            "max_z": trace["max_z"], "end_tilt": end["tilt"],
            "end_position": end["position"], "contact_end": sim.contact_signature(),
            "contact_seen": trace["contact_seen"]}


def collect(replay, episode_ids, xml):
    sim = DirectJengaSim(xml); episodes = []; starts = []; start_tilt = []; start_position = []
    values = {tail: {key: [] for key in (
        "peak_tilt", "min_z", "max_z", "end_tilt", "end_position",
        "contact_end", "contact_seen")} for tail in TAILS}
    try:
        for number, episode_id in enumerate(episode_ids):
            episode = replay.episode(episode_id); sim.reset(int(episode_id))
            wanted = set(chunk_starts(len(episode.actions)))
            for start, action in enumerate(episode.actions):
                if start in wanted:
                    snapshot = sim.snapshot(); beginning = sim.block_diagnostics()
                    chunk = episode.actions[start:start + HORIZON]
                    records = {tail: branch(sim, snapshot, chunk, tail) for tail in TAILS}
                    sim.restore(snapshot)
                    episodes.append(str(episode_id)); starts.append(start)
                    start_tilt.append(beginning["tilt"]); start_position.append(beginning["position"])
                    for tail in TAILS:
                        for key, value in records[tail].items():
                            values[tail][key].append(value)
                sim.execute(action)
            print(f"  {number + 1}/{len(episode_ids)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    arrays = {"episode_ids": np.asarray(episodes), "chunk_starts": np.asarray(starts),
              "start_tilt": np.asarray(start_tilt), "start_position": np.asarray(start_position)}
    for tail in TAILS:
        arrays.update({f"tail{tail}_{key}": np.asarray(value)
                       for key, value in values[tail].items()})
    return arrays


def analyse(arrays):
    start_safe = np.max(arrays["start_tilt"], axis=1) < 45
    result = {"chunks": len(start_safe), "safe_starts": int(start_safe.sum()), "tails": {}}
    masks = {}
    for tail in TAILS:
        mode = failure_masks(arrays[f"tail{tail}_peak_tilt"], arrays[f"tail{tail}_min_z"])
        masks[tail] = mode
        safe_extraction = (~mode["any_failure"]
                           & ((arrays[f"tail{tail}_end_position"][:, 0, 2]
                               - arrays["start_position"][:, 0, 2]) >= 0.02))
        result["tails"][str(tail)] = {
            "neighbor_failures": int(mode["neighbor_failure"].sum()),
            "middle_topples": int(mode["middle_topple"].sum()),
            "middle_falls": int(mode["middle_fall"].sum()),
            "middle_failures": int(mode["middle_failure"].sum()),
            "any_failures": int(mode["any_failure"].sum()),
            "new_any_failures_from_safe_start": int((start_safe & mode["any_failure"]).sum()),
            "safe_middle_lifts_20mm": int(safe_extraction.sum()),
        }
    result["outcome_changes"] = {
        "tail0_to_5": int((masks[0]["any_failure"] != masks[5]["any_failure"]).sum()),
        "tail5_to_10": int((masks[5]["any_failure"] != masks[10]["any_failure"]).sum()),
        "middle_tail0_to_5": int((masks[0]["middle_failure"]
                                  != masks[5]["middle_failure"]).sum()),
        "middle_tail5_to_10": int((masks[5]["middle_failure"]
                                   != masks[10]["middle_failure"]).sum()),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/all_block_tails_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/all_block_tails.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    if args.reuse_cache:
        arrays = dict(np.load(args.cache, allow_pickle=False))
    else:
        replay = JengaReplay(args.lmdb)
        try:
            ids = replay.episode_ids[:args.episodes]
            with tempfile.TemporaryDirectory(prefix="jenga_all_block_tails_") as temp:
                arrays = collect(replay, ids, extract_sim(args.sim_archive, temp))
        finally:
            replay.close()
        output = Path(args.cache); output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
    result = analyse(arrays)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2)); print(f"wrote {output}")


if __name__ == "__main__":
    main()
