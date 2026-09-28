"""Render one complete nominal Jenga-pick episode that topples a neighbour.

This is a presentation artifact generator, not an evaluation.  Episode 2 and reset
seed 2002 are taken from the frozen 100-episode intervention-v0 result, where the
nominal arm completes the pick but topples a neighbouring block.
"""
import argparse
from pathlib import Path
import sys
import tempfile

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, extract_sim  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", default="2")
    parser.add_argument("--reset-seed", type=int, default=2002)
    parser.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "results/presentation/3d_jenga/pick_failure.mp4"),
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    replay = JengaReplay(args.lmdb)
    try:
        episode = replay.episode(args.episode)
    finally:
        replay.close()

    with tempfile.TemporaryDirectory(prefix="jenga_presentation_failure_") as temp:
        sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 576)
        )
        if not writer.isOpened():
            raise RuntimeError("MP4 writer unavailable")
        peak_tilt = 0.0
        try:
            sim.reset(args.reset_seed)
            for index, action in enumerate(episode.actions):
                sim.execute(action)
                state = sim.block_diagnostics()
                peak_tilt = max(peak_tilt, float(max(state["tilt"][1:])))
                frame = cv2.resize(
                    cv2.cvtColor(sim.render(), cv2.COLOR_RGB2BGR),
                    (640, 480), interpolation=cv2.INTER_CUBIC,
                )
                canvas = cv2.copyMakeBorder(
                    frame, 96, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)
                )
                failed = peak_tilt >= 45.0
                status = "FAILURE: NEIGHBOUR TOPPLED" if failed else "NOMINAL PICK IN PROGRESS"
                color = (0, 0, 190) if failed else (25, 110, 25)
                cv2.putText(canvas, f"3D Jenga pick   action {index + 1}/{len(episode.actions)}",
                            (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (25, 25, 25), 2)
                cv2.putText(canvas, status, (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                            .72, color, 2)
                cv2.putText(canvas, f"neighbour peak tilt: {peak_tilt:.1f} deg",
                            (12, 84), cv2.FONT_HERSHEY_SIMPLEX, .55, (25, 25, 25), 1)
                repeats = 10 if index in (0, len(episode.actions) - 1) else 1
                for _ in range(repeats):
                    writer.write(canvas)
        finally:
            writer.release()
            sim.close()
    print(output)


if __name__ == "__main__":
    main()
