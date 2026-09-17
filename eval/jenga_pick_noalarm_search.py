"""Find physical no-alarm probes during middle-block lifts, not static controls."""
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
sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import probe_scalars  # noqa: E402
from jenga_multipeak_oracle import original_physical_responses  # noqa: E402
from jenga_short_held_tails import extract_sim  # noqa: E402
from jenga_trajectory_oracle import collect, read_cache, score_rows, write_cache  # noqa: E402


def select_picks(path, min_lift=.025, min_lateral=.02):
    data = dict(np.load(path, allow_pickle=False))
    lift = data["tail5_max_z"][:, 0] - data["start_position"][:, 0, 2]
    lateral = np.linalg.norm(data["tail5_end_position"][:, 0, :2]
                             - data["start_position"][:, 0, :2], axis=1)
    safe = np.max(data["tail5_peak_tilt"][:, 1:], axis=1) < 45
    chosen = np.flatnonzero((lift >= min_lift) & (lateral >= min_lateral) & safe)
    return [{"episode_id": str(data["episode_ids"][i]),
             "chunk_start": int(data["chunk_starts"][i]),
             "stratum": "middle_pick_nominal_safe",
             "center_offset": 0.0,
             "nominal_middle_lift_m": float(lift[i]),
             "nominal_middle_lateral_m": float(lateral[i])} for i in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--all-block-cache", default=str(ROOT / "results/jenga/all_block_tails_cache.npz"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/pick_noalarm_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/pick_noalarm_search.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    selected = select_picks(args.all_block_cache)
    scalars = probe_scalars(50)
    if args.reuse_cache:
        metadata, scalars, trajectories, outcomes = read_cache(args.cache)
    else:
        replay = JengaReplay(args.lmdb)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_pick_search_") as temp:
                metadata, trajectories, outcomes = collect(
                    selected, replay, extract_sim(args.sim_archive, temp), scalars, .10)
        finally:
            replay.close()
        write_cache(args.cache, metadata, scalars, trajectories, outcomes)
    scale = robust_component_scale(original_physical_responses(args.original_cache))
    rows = score_rows(scalars, trajectories, outcomes, metadata, scale)
    no_alarm_safe = [row for row in rows if not row["alarm"]
                     and row["physical_counts"]["toppled"] == 0]
    result = {"protocol": {"nominal_middle_lift_min_m": .025,
                            "nominal_middle_lateral_min_m": .02,
                            "nominal_neighbor_topple": False,
                            "probes": 50, "horizon": 8, "held_tail": 5},
              "candidate_chunks": len(rows),
              "safe_under_all_probes": sum(row["physical_counts"]["toppled"] == 0
                                           for row in rows),
              "safe_no_alarm": len(no_alarm_safe),
              "safe_no_alarm_keys": [[r["episode_id"], r["chunk_start"]]
                                     for r in no_alarm_safe],
              "rows": rows}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
