"""Zoomed no-alarm picks that start at the neighbor-block grasp interface."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_probe_comparison_videos import render_case  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_trajectory_oracle import read_cache  # noqa: E402

CASES = (("74", 106), ("91", 98))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/pick_noalarm_cache.npz"))
    ap.add_argument("--search-result", default=str(ROOT / "results/jenga/pick_noalarm_search.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "results/jenga/near_grasp_noalarm_videos"))
    args = ap.parse_args()
    rows = json.loads(Path(args.search_result).read_text())["rows"]
    metadata, scalars, _, outcomes = read_cache(args.cache)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    replay = JengaReplay(args.lmdb)
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_near_grasp_videos_") as temp:
            sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
            try:
                for episode_id, start in CASES:
                    i = next(i for i, r in enumerate(metadata)
                             if r["episode_id"] == episode_id and r["chunk_start"] == start)
                    row = rows[i]
                    if row["alarm"] or row["physical_counts"]["toppled"]:
                        raise ValueError("selected case is not safe/no-alarm")
                    episode = replay.episode(episode_id)
                    sim.reset(int(episode_id))
                    for action in episode.actions[:start]:
                        sim.execute(action)
                    snapshot = sim.snapshot()
                    positions = sim.block_diagnostics()["position"]
                    ee = sim.proprio()[:3]
                    neighbor_distance = float(np.min(np.linalg.norm(positions[1:] - ee, axis=1)))
                    start_red_z = float(positions[0, 2])
                    name = f"near_grasp_no_alarm_ep{episode_id}_chunk{start}.mp4"
                    shown = render_case(sim, snapshot, episode.actions[start:start + HORIZON],
                                        row, scalars, outcomes[i], "moving_safe", output_dir / name,
                                        indices=[23, 24, 25], zoom=True)
                    if not all(p["middle_lift_m"] >= .025 for p in shown):
                        raise ValueError("red block did not lift on every shown probe")
                    manifest.append({"video": name, "episode_id": episode_id,
                                     "chunk_start": start, "detector_alarm": False,
                                     "all_50_neighbor_safe": True,
                                     "red_start_z_m": start_red_z,
                                     "start_ee_to_nearest_neighbor_center_m": neighbor_distance,
                                     "nominal_neighbor_robot_contact": False,
                                     "shown_probes": shown})
                    print(f"wrote {name}", flush=True)
            finally:
                sim.close()
    finally:
        replay.close()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
