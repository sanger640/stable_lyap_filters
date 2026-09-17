"""Frozen physical BIC sensitivity to narrower subsets of operational probes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_multipeak_oracle import original_physical_responses  # noqa: E402
from jenga_trajectory_oracle import read_cache, score_rows  # noqa: E402

SIZES = (50, 40, 30, 20, 16)


def center_slice(total, count):
    first = (int(total) - int(count)) // 2
    return slice(first, first + int(count))


def group_result(rows, name):
    if name == "boundary_benchmark":
        boundary = [r for r in rows if r["stratum"] == "recentered_neighbor_boundary"]
        quiet = [r for r in rows if r["stratum"] == "quiet_control"]
        return {"boundary_alarms": int(sum(r["alarm"] for r in boundary)),
                "boundary_cases": len(boundary),
                "boundary_mixed_in_subset": int(sum(
                    0 < r["physical_counts"]["toppled"] < r["physical_counts"]["intact"]
                    + r["physical_counts"]["toppled"] for r in boundary)),
                "quiet_alarms": int(sum(r["alarm"] for r in quiet)),
                "quiet_cases": len(quiet)}
    if name == "expanded_controls":
        silent = [r for r in rows if r["stratum"] == "silent_episode"
                  and r["physical_counts"]["toppled"] == 0]
        moving = [r for r in rows if r["stratum"] == "moving_nominal_non_topple"
                  and r["physical_counts"]["toppled"] == 0]
        return {"silent_safe_alarms": int(sum(r["alarm"] for r in silent)),
                "silent_safe_cases": len(silent),
                "moving_safe_alarms": int(sum(r["alarm"] for r in moving)),
                "moving_safe_cases": len(moving)}
    episodes = sorted(set(r["episode_id"] for r in rows), key=int)
    return {ep: {"alarms": int(sum(r["alarm"] for r in rows if r["episode_id"] == ep)),
                 "decisions": int(sum(r["episode_id"] == ep for r in rows))}
            for ep in episodes}


def commanded_displacements(replay, examples, scalars):
    out = {}
    for episode_id, start in examples:
        actions = replay.episode(episode_id).actions[start:start + 8]
        span_m = float(np.max(np.linalg.norm(actions[:, :3] - actions[0, :3], axis=1)))
        out[f"ep{episode_id}_chunk{start}"] = {
            "nominal_chunk_span_mm": span_m * 1000,
            "max_probe_change_from_nominal_mm":
                {str(n): float(.10 * np.max(np.abs(scalars[center_slice(len(scalars), n)]))
                               * span_m * 1000) for n in SIZES}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--benchmark-cache", default=str(ROOT / "results/jenga/trajectory_oracle_cache.npz"))
    ap.add_argument("--expanded-cache", default=str(ROOT / "results/jenga/expanded_controls_cache.npz"))
    ap.add_argument("--episode-cache", default=str(ROOT / "results/jenga/full_episode_alarm_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/action_strength_sensitivity.json"))
    args = ap.parse_args()
    scale = robust_component_scale(original_physical_responses(args.original_cache))
    datasets = {"boundary_benchmark": read_cache(args.benchmark_cache),
                "expanded_controls": read_cache(args.expanded_cache),
                "full_episodes": read_cache(args.episode_cache)}
    results = {}
    scalars = datasets["boundary_benchmark"][1]
    for size in SIZES:
        section = center_slice(len(scalars), size)
        scores = {}
        for name, (metadata, source_scalars, trajectories, outcomes) in datasets.items():
            if not np.allclose(source_scalars, scalars):
                raise ValueError(f"different scalar grid in {name}")
            rows = score_rows(scalars[section], trajectories[:, section],
                              outcomes[:, section], metadata, scale)
            scores[name] = group_result(rows, name)
        results[str(size)] = {"probes": size,
                              "max_abs_scalar": float(np.max(np.abs(scalars[section]))),
                              "max_perturbation_coefficient": float(
                                  .10 * np.max(np.abs(scalars[section]))),
                              **scores}
        print(f"scored central {size} probes", flush=True)
    replay = JengaReplay(args.lmdb)
    try:
        examples = [("24", 106), ("74", 106), ("91", 98)]
        displacements = commanded_displacements(replay, examples, scalars)
    finally:
        replay.close()
    result = {"protocol": {"horizon": 8, "held_tail": 5, "original_eps": .10,
                            "frozen_scale_and_BIC": True,
                            "subset_warning": "central subsets reduce probe count and search power "
                                              "as well as action range; re-simulate evenly spaced "
                                              "strengths before concluding causality"},
              "commanded_displacements": displacements, "results": results}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
