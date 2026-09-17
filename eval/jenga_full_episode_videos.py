"""Render complete nominal Jenga episodes with the precomputed physical alarm timeline."""
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


def frame_panel(frame, episode_id, action_index, n_actions, decision, red_z, peak_tilt):
    image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (640, 480),
                       interpolation=cv2.INTER_CUBIC)
    canvas = cv2.copyMakeBorder(image, 96, 0, 0, 0, cv2.BORDER_CONSTANT,
                                value=(245, 245, 245))
    if decision is None:
        label = "NO DECISION YET"; color = (70, 70, 70); source = ""
    else:
        label = "ALARM" if decision["alarm"] else "NO ALARM"
        color = (0, 0, 190) if decision["alarm"] else (25, 110, 25)
        source = f"decision at action {decision['chunk_start']}   score {decision['trajectory_score']:.1f}"
    cv2.putText(canvas, f"Episode {episode_id}   action {action_index + 1}/{n_actions}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (25, 25, 25), 2)
    cv2.putText(canvas, label, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .75, color, 2)
    cv2.putText(canvas, source, (170, 51), cv2.FONT_HERSHEY_SIMPLEX, .48, (25, 25, 25), 1)
    cv2.putText(canvas, f"red z {red_z:.3f} m   neighbor peak tilt {peak_tilt:.1f} deg",
                (12, 78), cv2.FONT_HERSHEY_SIMPLEX, .52, (25, 25, 25), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--audit", default=str(ROOT / "results/jenga/full_episode_alarm_audit.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "results/jenga/full_episode_alarm_videos"))
    args = ap.parse_args()
    audit = json.loads(Path(args.audit).read_text())
    rows = audit["rows"]
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    replay = JengaReplay(args.lmdb)
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_full_episode_video_") as temp:
            sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
            try:
                for episode_id, summary in audit["episodes"].items():
                    episode = replay.episode(episode_id)
                    decisions = {r["chunk_start"]: r for r in rows
                                 if r["episode_id"] == episode_id}
                    sim.reset(int(episode_id))
                    name = f"full_episode_ep{episode_id}.mp4"
                    writer = cv2.VideoWriter(str(output_dir / name),
                                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 576))
                    if not writer.isOpened():
                        raise RuntimeError("MP4 writer unavailable")
                    active = None; peak_tilt = 0.0
                    try:
                        for i, action in enumerate(episode.actions):
                            if i in decisions:
                                active = decisions[i]
                            sim.execute(action)
                            block = sim.block_diagnostics()
                            peak_tilt = max(peak_tilt, float(max(block["tilt"][1:])))
                            panel = frame_panel(sim.render(), episode_id, i,
                                                len(episode.actions), active,
                                                float(block["position"][0, 2]), peak_tilt)
                            for _ in range(FPS if i in (0, len(episode.actions) - 1) else 1):
                                writer.write(panel)
                    finally:
                        writer.release()
                    manifest.append({"video": name, "episode_id": episode_id,
                                     "success_proxy": summary["successful_pick_proxy"],
                                     "alarm_starts": summary["alarm_starts"],
                                     "all_quiet": summary["all_quiet"],
                                     "neighbor_peak_tilt_deg": summary["neighbor_peak_tilt_deg"]})
                    print(f"wrote {name}", flush=True)
            finally:
                sim.close()
    finally:
        replay.close()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
