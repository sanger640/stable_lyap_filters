"""Render inspectable nominal/boundary examples for pushing and insertion benchmarks."""
import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from systems.contact_benchmarks import InsertionBenchmark, PushingBenchmark  # noqa: E402

OUTPUT = ROOT / "results/contact_benchmarks/previews"
FPS = 10


def _label(frame, title, subtitle):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], 62), (18, 20, 26), -1)
    cv2.putText(image, title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, .66,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, subtitle, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, .50,
                (205, 215, 225), 1, cv2.LINE_AA)
    return image


def _rollout(env, controls, title, subtitle):
    frames = [_label(env.render(), title, subtitle)]
    for control in controls:
        env.step(control)
        frames.append(_label(env.render(), title, subtitle))
    return frames, env.outcome()


def _paired_video(path, left, right):
    count = max(len(left), len(right))
    left = left + [left[-1]] * (count - len(left))
    right = right + [right[-1]] * (count - len(right))
    height, width = left[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (2 * width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for a, b in zip(left, right):
            writer.write(np.concatenate([a, b], axis=1))
    finally:
        writer.release()


def pushing_preview(output):
    nominal = PushingBenchmark(); excessive = PushingBenchmark()
    try:
        nominal.reset(); excessive.reset()
        nominal_controls = [[x, 0] for x in np.linspace(0, .55, 80)] + [[.55, 0]] * 25
        excessive_controls = [[x, 0] for x in np.linspace(0, .78, 100)] + [[.78, 0]] * 40
        good, good_outcome = _rollout(
            nominal, nominal_controls, "PUSHING: task success",
            "red object reaches green goal and remains supported")
        bad, bad_outcome = _rollout(
            excessive, excessive_controls, "PUSHING: nearby over-push",
            "same contact mode crosses the support edge and falls")
    finally:
        nominal.close(); excessive.close()
    path = output / "pushing_goal_vs_edge_fall.mp4"
    _paired_video(path, good, bad)
    return path, good, bad, good_outcome, bad_outcome


def insertion_preview(output):
    centered = InsertionBenchmark(); offset = InsertionBenchmark()
    try:
        centered.reset(); offset.reset()
        centre_controls = [[0, 0, z, 0] for z in np.linspace(0, -.145, 80)] + [
            [0, 0, -.145, 0]] * 25
        jam_controls = [[.012, 0, z, 0] for z in np.linspace(0, -.145, 80)] + [
            [.012, 0, -.145, 0]] * 25
        good, good_outcome = _rollout(
            centered, centre_controls, "INSERTION: aligned success",
            "orange peg enters the tight square socket")
        bad, bad_outcome = _rollout(
            offset, jam_controls, "INSERTION: 12 mm execution offset",
            "peg contacts the rim and remains jammed above the socket")
    finally:
        centered.close(); offset.close()
    path = output / "insertion_success_vs_rim_jam.mp4"
    _paired_video(path, good, bad)
    return path, good, bad, good_outcome, bad_outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    push_path, push_good, push_bad, push_success, push_failure = pushing_preview(output)
    insert_path, insert_good, insert_bad, insert_success, insert_failure = insertion_preview(output)
    preview = np.vstack([
        np.concatenate([push_good[-1], push_bad[-1]], axis=1),
        np.concatenate([insert_good[-1], insert_bad[-1]], axis=1),
    ])
    preview_path = output / "preview.jpg"
    cv2.imwrite(str(preview_path), preview)
    manifest = {
        "purpose": "visual design validation before freezing cross-task monitor benchmarks",
        "pushing": {
            "task": "push a free object into a goal region while retaining table support",
            "mechanisms": ["stick/slide", "glancing deflection", "support/edge-fall"],
            "success_example": push_success.__dict__, "failure_example": push_failure.__dict__,
            "video": push_path.name,
        },
        "insertion": {
            "task": "insert a tight rectangular peg into a square socket",
            "mechanisms": ["free motion", "rim contact", "aligned insertion", "wedging/jam"],
            "success_example": insert_success.__dict__, "failure_example": insert_failure.__dict__,
            "video": insert_path.name,
        },
        "design_scope": "controlled tool pose rather than full arm; isolates contact regime "
                        "sensitivity from robot-controller and perception confounds",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
