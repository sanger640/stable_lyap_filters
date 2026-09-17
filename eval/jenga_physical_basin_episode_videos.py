"""Full nominal episode videos with physical-regime dissent at tails 5, 10, and 20."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402

FPS = 10
TAILS = (5, 10, 20)


def annotate(frame, episode_id, step, total, active):
    image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (640, 480),
                       interpolation=cv2.INTER_CUBIC)
    canvas = cv2.copyMakeBorder(image, 170, 0, 0, 0, cv2.BORDER_CONSTANT,
                                value=(245, 245, 245))
    cv2.putText(canvas, f"Episode {episode_id}   action {step + 1}/{total}   physical-regime dissent",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .57, (30, 30, 30), 1)
    if active is None:
        cv2.putText(canvas, "No 8-action probe decision yet", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, .56, (80, 80, 80), 1)
    else:
        cv2.putText(canvas, f"Latest probe begins at action {active[5]['chunk_start']}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, .53, (40, 40, 40), 1)
        for line, tail in enumerate(TAILS):
            row = active[tail]
            if row["coverage"] == 0.0:
                status, color = "UNASSIGNED", (90, 90, 90)
            elif row["alarm"]:
                status, color = "DISSENT", (0, 0, 180)
            else:
                status, color = "NO DISSENT", (75, 75, 75)
            cv2.putText(canvas, f"tail {tail:2d}: {status:12s} k={row['dissent']:2d} "
                        f"coverage={row['coverage']:.0%}",
                        (10, 81 + 29 * line), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--result", default=str(ROOT / "results/jenga/physical_basin_tails.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "results/jenga/physical_basin_episode_videos"))
    args = ap.parse_args()
    result = json.loads(Path(args.result).read_text())
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    metadata = result["protocol"]
    episode_ids = metadata["reference_excludes_test_episodes"]
    by_tail = {tail: {(r["episode_id"], r["chunk_start"]): r
                      for r in result["rows"][str(tail)]} for tail in TAILS}
    manifest = []
    replay = JengaReplay(args.lmdb)
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_basin_episode_video_") as temp:
            sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
            try:
                for episode_id in episode_ids:
                    episode = replay.episode(episode_id)
                    sim.reset(int(episode_id))
                    name = f"physical_basin_full_ep{episode_id}.mp4"
                    writer = cv2.VideoWriter(str(output_dir / name),
                                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 650))
                    if not writer.isOpened():
                        raise RuntimeError("MP4 writer unavailable")
                    active = None
                    pending = {}
                    try:
                        for i, action in enumerate(episode.actions):
                            if (episode_id, i) in by_tail[5]:
                                pending[i + 7] = {tail: by_tail[tail][(episode_id, i)]
                                                  for tail in TAILS}
                            sim.execute(action)
                            if i in pending:
                                active = pending.pop(i)
                            frame = annotate(sim.render(), episode_id, i,
                                             len(episode.actions), active)
                            for _ in range(FPS if i in (0, len(episode.actions) - 1) else 1):
                                writer.write(frame)
                    finally:
                        writer.release()
                    manifest.append({"video": name, "episode_id": episode_id,
                                     "tail_alarm_counts": {str(t): result["summary"][str(t)]
                                                           ["episode_alarm_counts"][episode_id]
                                                           for t in TAILS}})
                    print(f"wrote {name}", flush=True)
            finally:
                sim.close()
    finally:
        replay.close()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
