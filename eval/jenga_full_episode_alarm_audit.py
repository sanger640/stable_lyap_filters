"""Score every non-overlapping H=8 decision point in selected full episodes."""
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
from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim  # noqa: E402
from jenga_trajectory_oracle import collect, read_cache, score_rows, write_cache  # noqa: E402


def nominal_episode(sim, episode):
    sim.reset(int(episode.episode_id))
    positions, tilts, frames = [], [], []
    for action in episode.actions:
        sim.execute(action)
        block = sim.block_diagnostics()
        positions.append(block["position"])
        tilts.append(block["tilt"])
    positions = np.asarray(positions); tilts = np.asarray(tilts)
    baseline = positions[min(9, len(positions) - 1), 0]
    after = positions[10:, 0]
    lift = float(np.max(after[:, 2] - baseline[2]))
    lateral = float(np.max(np.linalg.norm(after[:, :2] - baseline[:2], axis=1)))
    neighbor_tilt = float(np.max(tilts[:, 1:]))
    return {"actions": len(episode.actions), "red_peak_lift_after_settle_m": lift,
            "red_peak_lateral_after_settle_m": lateral,
            "neighbor_peak_tilt_deg": neighbor_tilt,
            "successful_pick_proxy": bool(lift >= .025 and lateral >= .02 and neighbor_tilt < 45)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", default=["1", "24", "74", "91"])
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/full_episode_alarm_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/full_episode_alarm_audit.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    replay = JengaReplay(args.lmdb)
    try:
        episodes = [replay.episode(str(e)) for e in args.episodes]
        selected = [{"episode_id": str(ep.episode_id), "chunk_start": start,
                     "center_offset": 0.0, "stratum": "full_episode"}
                    for ep in episodes for start in chunk_starts(len(ep.actions))]
        scalars = probe_scalars(50)
        with tempfile.TemporaryDirectory(prefix="jenga_full_episode_audit_") as temp:
            xml = extract_sim(args.sim_archive, temp)
            if args.reuse_cache:
                metadata, scalars, trajectories, outcomes = read_cache(args.cache)
            else:
                metadata, trajectories, outcomes = collect(selected, replay, xml, scalars, .10)
                write_cache(args.cache, metadata, scalars, trajectories, outcomes)
            sim = DirectJengaSim(xml)
            try:
                nominal = {str(ep.episode_id): nominal_episode(sim, ep) for ep in episodes}
            finally:
                sim.close()
    finally:
        replay.close()
    scale = robust_component_scale(original_physical_responses(args.original_cache))
    rows = score_rows(scalars, trajectories, outcomes, metadata, scale)
    summaries = {}
    for episode_id in args.episodes:
        selected_rows = [r for r in rows if r["episode_id"] == str(episode_id)]
        summaries[str(episode_id)] = {**nominal[str(episode_id)],
                                      "decision_points": len(selected_rows),
                                      "alarm_starts": [r["chunk_start"] for r in selected_rows
                                                       if r["alarm"]],
                                      "alarm_count": sum(r["alarm"] for r in selected_rows),
                                      "all_quiet": not any(r["alarm"] for r in selected_rows)}
    result = {"protocol": {"horizon": 8, "held_tail": 5, "probes": 50,
                            "decision_points": "non-overlapping chunks starting at action 2"},
              "episodes": summaries, "rows": rows}
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
