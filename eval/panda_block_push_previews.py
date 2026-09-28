"""Render full-episode visual validation for the Panda single-Jenga-block push task."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from jenga_short_held_tails import extract_sim  # noqa: E402
from systems.panda_block_push import PandaBlockPush, write_push_xml  # noqa: E402

OUTPUT = ROOT / "results/panda_block_push/previews"
FPS = 10


def segment(start, stop, count, gripper=1.):
    start, stop = np.asarray(start, float), np.asarray(stop, float)
    xyz = np.linspace(start, stop, count)
    return np.column_stack([xyz, np.full(count, gripper)])


def controls(name, initial):
    behind = np.array([.455, 0., .435])
    contact = np.array([.500, 0., .435])
    if name == "success":
        path = [segment(initial, behind, 20), segment(behind, contact, 15),
                segment(contact, [.600, 0., .435], 45),
                segment([.600, 0., .435], [.600, 0., .510], 10)]
        retract_start = 80
    elif name == "overshoot":
        behind[2] = contact[2] = .425
        path = [segment(initial, behind, 20), segment(behind, contact, 15),
                segment(contact, [.650, 0., .425], 55),
                segment([.650, 0., .425], [.650, 0., .510], 10)]
        retract_start = 90
    elif name == "topple":
        behind[2] = contact[2] = .455
        path = [segment(initial, behind, 20), segment(behind, contact, 15),
                segment(contact, [.580, 0., .455], 35),
                segment([.580, 0., .455], [.580, 0., .510], 10)]
        retract_start = 70
    elif name == "contact_loss":
        path = [segment(initial, behind, 20), segment(behind, contact, 15),
                segment(contact, [.545, 0., .435], 20),
                segment([.545, 0., .435], [.600, .075, .435], 30),
                segment([.600, .075, .435], [.600, .075, .510], 10)]
        retract_start = 85
    else:
        raise ValueError(name)
    return np.concatenate(path), retract_start


def label(frame, title, status, contact):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (image.shape[1], 68), (18, 20, 26), -1)
    cv2.putText(image, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, .68,
                (255, 255, 255), 2, cv2.LINE_AA)
    colour = (80, 220, 100) if contact else (180, 190, 205)
    cv2.putText(image, f"{status}  |  robot contact: {'YES' if contact else 'no'}", (16, 55),
                cv2.FONT_HERSHEY_SIMPLEX, .47, colour, 1, cv2.LINE_AA)
    return image


def rollout(xml, name):
    env = PandaBlockPush(xml)
    titles = {"success": "SUCCESS: upright block reaches green goal",
              "overshoot": "INCOMPLETE: upright block overshoots goal",
              "topple": "FAILURE: block topples during push",
              "contact_loss": "INCOMPLETE: end effector loses contact"}
    try:
        env.reset(); actions, retract_start = controls(name, env.proprio()[:3])
        frames = [label(env.render(), titles[name], "approach", env.robot_contact())]
        contact_seen = False; contact_history = []
        for action in actions:
            env.step(action); contact = env.robot_contact()
            contact_seen |= contact
            contact_history.append(contact)
            frames.append(label(env.render(), titles[name], "executing", contact))
        for _ in range(15):
            env.step(actions[-1]); contact = env.robot_contact()
            frames.append(label(env.render(), titles[name], "settled", contact))
        outcome = env.outcome()
        outcome.update({"contact_seen": bool(contact_seen),
                        "contact_lost_before_retract": bool(
                            contact_seen and not any(contact_history[max(0, retract_start-3):retract_start])),
                        "control_steps": int(len(actions) + 15)})
        return frames, outcome
    finally:
        env.close()


def write_video(path, case_frames):
    order = ("success", "overshoot", "topple", "contact_loss")
    count = max(len(case_frames[name]) for name in order)
    padded = {name: case_frames[name] + [case_frames[name][-1]] *
              (count-len(case_frames[name])) for name in order}
    height, width = padded[order[0]][0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (2*width, 2*height))
    if not writer.isOpened(): raise RuntimeError(f"cannot create {path}")
    try:
        for i in range(count):
            top = np.concatenate([padded["success"][i], padded["overshoot"][i]], axis=1)
            bottom = np.concatenate([padded["topple"][i], padded["contact_loss"][i]], axis=1)
            writer.write(np.concatenate([top, bottom], axis=0))
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    args = parser.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="panda_block_push_") as temp:
        original = extract_sim(args.sim_archive, temp)
        xml = write_push_xml(original.parent)
        case_frames, outcomes = {}, {}
        for name in ("success", "overshoot", "topple", "contact_loss"):
            case_frames[name], outcomes[name] = rollout(xml, name)
    video = output / "panda_block_push_four_cases.mp4"; write_video(video, case_frames)
    final = np.concatenate([
        np.concatenate([case_frames["success"][-1], case_frames["overshoot"][-1]], axis=1),
        np.concatenate([case_frames["topple"][-1], case_frames["contact_loss"][-1]], axis=1),
    ], axis=0)
    preview = output / "panda_block_push_four_cases.jpg"; cv2.imwrite(str(preview), final)
    manifest = {"environment": "Panda operational-space controller pushing one upright Jenga block",
                "goal": "green non-colliding planar target",
                "grading": "success iff upright block reaches goal; failure iff peak tilt reaches 45 degrees at any time",
                "cases": outcomes, "video": video.name, "preview": preview.name}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
