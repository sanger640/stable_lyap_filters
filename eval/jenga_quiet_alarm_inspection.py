"""Targeted image/robot inspection for encoded-real quiet-control alarms."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import coherent_probe_chunks, probe_scalars  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402


def rotation_difference_deg(left, right):
    relative = np.asarray(left).reshape(3, 3) @ np.asarray(right).reshape(3, 3).T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def render_pair_sheet(path, pairs):
    width, height = 640, len(pairs) * 280
    sheet = Image.new("RGB", (width, height), "white"); draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(pairs):
        y = row * 280
        sheet.paste(Image.fromarray(pair["left_frame"]), (0, y + 40))
        sheet.paste(Image.fromarray(pair["right_frame"]), (320, y + 40))
        draw.text((5, y + 5), pair["title"], fill="black")
        draw.text((5, y + 22), "probe left", fill="black")
        draw.text((325, y + 22), "probe right", fill="black")
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--ordered-result", default=str(
        ROOT / "results/jenga/ordered_change_points.json"))
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/quiet_alarm_inspection.json"))
    ap.add_argument("--sheet", default=str(
        ROOT / "results/jenga/quiet_alarm_inspection.png"))
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--probes", type=int, default=50)
    args = ap.parse_args()

    ordered = json.loads(Path(args.ordered_result).read_text())
    targets = [row for row in ordered["actual_rows"]
               if row["alarm"] and row["stratum"] == "quiet_control"]
    scalars = probe_scalars(args.probes)
    replay = JengaReplay(args.lmdb); rows = []; pairs = []
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_quiet_inspect_") as temp:
            sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
            try:
                for target in targets:
                    episode = replay.episode(target["episode_id"])
                    sim.reset(int(target["episode_id"]))
                    for action in episode.actions[:target["chunk_start"]]:
                        sim.execute(action)
                    probes = coherent_probe_chunks(
                        episode.actions[target["chunk_start"]:target["chunk_start"] + HORIZON],
                        scalars, args.eps)
                    snapshot = sim.snapshot(); observations = []
                    for probe_index in (target["split_index"] - 1, target["split_index"]):
                        sim.restore(snapshot)
                        probe = probes[probe_index]
                        for action in probe:
                            sim.execute(action)
                        for _ in range(5):
                            sim.execute(probe[-1])
                        observations.append({
                            "frame": sim.render(), "qpos": sim.data.qpos.copy(),
                            "ee_position": sim.data.site_xpos[sim.site].copy(),
                            "ee_rotation": sim.data.site_xmat[sim.site].copy(),
                            "blocks": sim.block_diagnostics(),
                            "contact": sim.contact_signature(),
                            "scalar": float(scalars[probe_index]),
                        })
                    left, right = observations
                    pixel = np.abs(left["frame"].astype(np.int16) - right["frame"].astype(np.int16))
                    block_position = np.linalg.norm(
                        left["blocks"]["position"] - right["blocks"]["position"], axis=1) * 1000
                    row = {
                        "episode_id": target["episode_id"],
                        "chunk_start": target["chunk_start"],
                        "split_scalar": target["split_scalar"],
                        "probe_scalars": [left["scalar"], right["scalar"]],
                        "mean_absolute_pixel_difference": float(pixel.mean()),
                        "pixels_changed_fraction": float(np.any(pixel > 0, axis=2).mean()),
                        "pixels_changed_over_8_fraction": float(np.any(pixel > 8, axis=2).mean()),
                        "qpos_l2_difference": float(np.linalg.norm(left["qpos"] - right["qpos"])),
                        "qpos_max_difference": float(np.max(np.abs(left["qpos"] - right["qpos"]))),
                        "ee_position_difference_mm": float(np.linalg.norm(
                            left["ee_position"] - right["ee_position"]) * 1000),
                        "ee_rotation_difference_deg": rotation_difference_deg(
                            left["ee_rotation"], right["ee_rotation"]),
                        "block_position_difference_mm": block_position.tolist(),
                        "contact_changed": bool(np.any(left["contact"] != right["contact"])),
                    }
                    rows.append(row)
                    pairs.append({"title": f"ep{target['episode_id']} chunk {target['chunk_start']} "
                                           f"s={left['scalar']:.3f}/{right['scalar']:.3f}",
                                  "left_frame": left["frame"], "right_frame": right["frame"]})
            finally:
                sim.close()
    finally:
        replay.close()
    result = {"quiet_alarm_pairs": len(rows), "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    render_pair_sheet(args.sheet, pairs)
    print(json.dumps(result, indent=2))
    print(f"wrote {output} and {args.sheet}")


if __name__ == "__main__":
    main()
