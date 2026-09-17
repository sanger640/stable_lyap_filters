"""Render true red/middle-block pick chunks on which the physical monitor stays quiet."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_probe_comparison_videos import render_case  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_trajectory_oracle import read_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/pick_noalarm_cache.npz"))
    ap.add_argument("--search-result", default=str(ROOT / "results/jenga/pick_noalarm_search.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "results/jenga/pick_noalarm_videos"))
    ap.add_argument("--max-videos", type=int, default=3)
    args = ap.parse_args()
    rows = json.loads(Path(args.search_result).read_text())["rows"]
    metadata, scalars, _, outcomes = read_cache(args.cache)
    if len(metadata) != len(rows):
        raise ValueError("search cache and JSON have different rows")
    candidates = [i for i, r in enumerate(rows) if not r["alarm"]
                  and r["physical_counts"]["toppled"] == 0]
    candidates.sort(key=lambda i: -rows[i]["nominal_middle_lift_m"])
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    replay = JengaReplay(args.lmdb)
    manifest, used_episodes = [], set()
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_noalarm_video_") as temp:
            sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
            try:
                for i in candidates:
                    row = rows[i]
                    if row["episode_id"] in used_episodes:
                        continue
                    episode_id = row["episode_id"]; start = row["chunk_start"]
                    assert metadata[i]["episode_id"] == episode_id
                    assert metadata[i]["chunk_start"] == start
                    episode = replay.episode(episode_id)
                    sim.reset(int(episode_id))
                    for action in episode.actions[:start]:
                        sim.execute(action)
                    snapshot = sim.snapshot()
                    name = f"no_alarm_pick_ep{episode_id}_chunk{start}.mp4"
                    shown = render_case(sim, snapshot, episode.actions[start:start + HORIZON],
                                        row, scalars, outcomes[i], "moving_safe", output_dir / name,
                                        indices=[23, 24, 25])
                    if not all(x["middle_lift_m"] >= .025 and x["middle_lateral_m"] >= .02
                               for x in shown):
                        print(f"skip {name}: nominal pick not preserved on shown probes", flush=True)
                        continue
                    manifest.append({"video": name, "episode_id": episode_id,
                                     "chunk_start": start, "detector_alarm": row["alarm"],
                                     "trajectory_score": row["trajectory_score"],
                                     "all_50_safe": row["physical_counts"]["toppled"] == 0,
                                     "shown_probes": shown})
                    used_episodes.add(episode_id)
                    print(f"wrote {name}", flush=True)
                    if len(manifest) >= args.max_videos:
                        break
            finally:
                sim.close()
    finally:
        replay.close()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest:
        print("No no-alarm middle picks met the per-probe visual criterion")


if __name__ == "__main__":
    main()
