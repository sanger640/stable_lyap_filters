"""Expand neighbor-only physical controls with silent episodes and moving near-misses."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
from local_expansion import detect_local_expansion  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import probe_scalars  # noqa: E402
from jenga_multipeak_oracle import original_physical_responses  # noqa: E402
from jenga_near_boundary_discovery import offset_probe_chunks  # noqa: E402
from jenga_ordered_holdout import binomial_interval  # noqa: E402
from jenga_short_held_tails import HORIZON, extract_sim  # noqa: E402
from jenga_trajectory_oracle import (collect, read_cache, score_rows,
                                     summarise, write_cache)  # noqa: E402


def choose_expansion(scan_cache, reference_cache, hard_count=100, silent_limit=None):
    data = dict(np.load(scan_cache, allow_pickle=False))
    ids = data["episode_ids"].astype(str); starts = data["chunk_starts"].astype(int)
    start_tilt = np.max(data["start_tilt"][:, 1:], axis=1)
    peak = np.max(data["tail5_peak_tilt"][:, 1:], axis=1)
    displacement = np.max(np.linalg.norm(
        data["tail5_end_position"][:, 1:] - data["start_position"][:, 1:], axis=2), axis=1)
    silent_episodes = [episode for episode in sorted(set(ids), key=int)
                       if np.all(peak[ids == episode] < 5.0)]
    silent = [i for i in range(len(ids)) if ids[i] in silent_episodes]
    if silent_limit is not None:
        silent = silent[:int(silent_limit)]
    old, *_ = read_cache(reference_cache)
    used = {(row["episode_id"], int(row["chunk_start"])) for row in old}
    candidates = [i for i in range(len(ids)) if start_tilt[i] < 45.0
                  and 5.0 <= peak[i] < 45.0 and displacement[i] >= .002
                  and (ids[i], int(starts[i])) not in used]
    # One state per episode first; then fill with strongest nominal movement.
    candidates.sort(key=lambda i: (-float(displacement[i]), int(ids[i]), int(starts[i])))
    first, rest, seen = [], [], set()
    for i in candidates:
        (first if ids[i] not in seen else rest).append(i); seen.add(ids[i])
    hard = (first + rest)[:int(hard_count)]
    if len(hard) < int(hard_count):
        raise ValueError(f"only {len(hard)} hard negative candidates")
    rows = [{"episode_id": ids[i], "chunk_start": int(starts[i]),
             "stratum": "silent_episode", "screen_peak_neighbor": float(peak[i]),
             "screen_displacement_m": float(displacement[i]), "center_offset": 0.0}
            for i in silent]
    rows += [{"episode_id": ids[i], "chunk_start": int(starts[i]),
              "stratum": "moving_nominal_non_topple", "screen_peak_neighbor": float(peak[i]),
              "screen_displacement_m": float(displacement[i]), "center_offset": 0.0}
             for i in hard]
    return rows, [str(x) for x in silent_episodes]


def report(rows, silent_episodes):
    groups = {}
    for stratum in ("silent_episode", "moving_nominal_non_topple"):
        subset = [row for row in rows if row["stratum"] == stratum]
        negatives = [row for row in subset if row["physical_counts"]["toppled"] == 0]
        certain_failures = [row for row in subset if row["physical_counts"]["intact"] == 0]
        alarms = sum(row["alarm"] for row in negatives)
        groups[stratum] = {
            "chunks": len(subset),
            "mixed_outcome_chunks": int(sum(row["physical_boundary"] for row in subset)),
            "certain_failures": len(certain_failures),
            "low_support_mixed_chunks": int(sum(
                0 < row["physical_counts"]["toppled"] < 8 for row in subset)),
            "verified_safe_nonboundaries": len(negatives), "false_alarms": int(alarms),
            "false_alarm_rate": alarms / max(len(negatives), 1),
            "false_alarm_rate_exact_95_percent": binomial_interval(alarms, len(negatives)),
        }
    by_episode = {}
    for row in rows:
        if row["stratum"] == "silent_episode":
            by_episode.setdefault(row["episode_id"], []).append(row)
    return {"groups": groups, "silent_episodes": silent_episodes,
            "silent_episode_alarm_counts": {episode: int(sum(row["alarm"] for row in values))
                                            for episode, values in by_episode.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--scan-cache", default=str(ROOT / "results/jenga/all_block_tails_cache.npz"))
    ap.add_argument("--reference-cache", default=str(ROOT / "results/jenga/trajectory_oracle_cache.npz"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/expanded_controls_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/expanded_controls.json"))
    ap.add_argument("--hard-count", type=int, default=100)
    ap.add_argument("--silent-limit", type=int)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args(); scalars = probe_scalars(50)
    selected, episodes = choose_expansion(args.scan_cache, args.reference_cache,
                                           args.hard_count, args.silent_limit)
    if args.reuse_cache:
        metadata, scalars, trajectories, outcomes = read_cache(args.cache)
    else:
        replay = JengaReplay(args.lmdb)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_expanded_controls_") as temp:
                metadata, trajectories, outcomes = collect(
                    selected, replay, extract_sim(args.sim_archive, temp), scalars, .10)
        finally:
            replay.close()
        write_cache(args.cache, metadata, scalars, trajectories, outcomes)
    scale = robust_component_scale(original_physical_responses(args.original_cache))
    rows = score_rows(scalars, trajectories, outcomes, metadata, scale)
    result = {"protocol": {"horizon": HORIZON, "tail": 5, "probes": len(scalars),
                           "eps": .10, "hard_nominal_displacement_min_m": .002,
                           "hard_nominal_tilt_deg": "[5,45)",
                           "silent_episode_nominal_max_tilt_deg": 5.0,
                           "physical_outcomes_verified_after_probing": True},
              **report(rows, episodes), "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
