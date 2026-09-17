"""Explain ordered real-latent alarms with continuous MuJoCo measurements."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ordered_change_point import robust_component_scale  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import read_probe_cache_with_diagnostics  # noqa: E402
from jenga_ordered_change_points import score_responses, summarise  # noqa: E402


def quaternion_angle_deg(left, right):
    dot = np.abs(np.sum(np.asarray(left) * np.asarray(right), axis=-1))
    return np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))


def adjacent_ratio(values, index):
    values = np.asarray(values, np.float64)
    differences = np.linalg.norm(np.diff(values.reshape(len(values), -1), axis=0), axis=1)
    selected = float(differences[index])
    baseline = float(np.median(np.delete(differences, index)))
    return selected, selected / max(baseline, 1e-12)


def explain_split(chunk, split_index, diagnostics):
    """Physical differences between the two probes adjacent to a fitted split."""
    i = int(split_index) - 1
    peak = diagnostics["peak_tilt"][chunk]
    end_tilt = diagnostics["end_tilt"][chunk]
    positions = diagnostics["block_end_position"][chunk]
    quaternions = diagnostics["block_end_quaternion"][chunk]
    contacts_end = diagnostics["contact_end"][chunk]
    contacts_seen = diagnostics["contact_seen"][chunk]
    labels = diagnostics["contact_labels"].astype(str)

    peak_jump = np.abs(peak[i + 1] - peak[i])
    end_tilt_jump = np.abs(end_tilt[i + 1] - end_tilt[i])
    position_jump_mm = np.linalg.norm(positions[i + 1] - positions[i], axis=1) * 1000
    rotation_jump = quaternion_angle_deg(quaternions[i + 1], quaternions[i])
    end_flips = labels[contacts_end[i + 1] != contacts_end[i]].tolist()
    seen_flips = labels[contacts_seen[i + 1] != contacts_seen[i]].tolist()
    _, peak_ratio = adjacent_ratio(peak, i)
    _, position_ratio = adjacent_ratio(positions * 1000, i)
    _, ee_ratio = adjacent_ratio(diagnostics["ee_end_position"][chunk] * 1000, i)
    ee_jump_mm = float(np.linalg.norm(
        diagnostics["ee_end_position"][chunk, i + 1]
        - diagnostics["ee_end_position"][chunk, i]) * 1000)
    side = slice(1, 3)
    material_block_change = bool(
        np.max(position_jump_mm[side]) >= 2.0
        or np.max(rotation_jump[side]) >= 2.0
        or np.max(peak_jump[side]) >= 2.0)
    return {
        "max_side_peak_tilt_jump_deg": float(np.max(peak_jump[side])),
        "max_side_endpoint_tilt_jump_deg": float(np.max(end_tilt_jump[side])),
        "max_side_position_jump_mm": float(np.max(position_jump_mm[side])),
        "max_side_rotation_jump_deg": float(np.max(rotation_jump[side])),
        "middle_position_jump_mm": float(position_jump_mm[0]),
        "middle_rotation_jump_deg": float(rotation_jump[0]),
        "ee_position_jump_mm": ee_jump_mm,
        "peak_tilt_adjacent_ratio": peak_ratio,
        "block_position_adjacent_ratio": position_ratio,
        "ee_position_adjacent_ratio": ee_ratio,
        "endpoint_contact_flips": end_flips,
        "trajectory_contact_flips": seen_flips,
        "material_side_block_change": material_block_change,
        "any_contact_change": bool(end_flips or seen_flips),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(
        ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/physical_jump_diagnostics.json"))
    ap.add_argument("--dimension", type=int, default=4)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--min-side", type=int, default=8)
    ap.add_argument("--min-jump-ratio", type=float, default=3.0)
    args = ap.parse_args()

    metadata, scalars, _, actual, physical, diagnostics = read_probe_cache_with_diagnostics(
        args.cache)
    required = {"peak_tilt", "end_tilt", "block_end_position",
                "block_end_quaternion", "contact_end", "contact_seen", "ee_end_position"}
    missing = sorted(required - diagnostics.keys())
    if missing:
        raise ValueError(f"probe cache lacks physical diagnostics: {missing}; regenerate it")
    scale = robust_component_scale(actual)
    rows = score_responses(
        scalars, actual, physical, metadata, dimension=args.dimension, degree=args.degree,
        min_side=args.min_side, min_jump_ratio=args.min_jump_ratio, scale=scale)
    alarm_rows = []
    for chunk, row in enumerate(rows):
        if not row["alarm"]:
            continue
        explained = explain_split(chunk, row["split_index"], diagnostics)
        item = dict(row); item["physical_jump"] = explained
        item["topple_switch_at_fitted_split"] = bool(
            physical[chunk, row["split_index"] - 1]
            != physical[chunk, row["split_index"]])
        alarm_rows.append(item)
    quiet = [row for row in alarm_rows if row["stratum"] == "quiet_control"]
    summary = {
        "ordered_detector": summarise(rows),
        "alarms": len(alarm_rows),
        "quiet_alarms": len(quiet),
        "quiet_with_material_side_block_change": int(sum(
            row["physical_jump"]["material_side_block_change"] for row in quiet)),
        "quiet_with_contact_change": int(sum(
            row["physical_jump"]["any_contact_change"] for row in quiet)),
        "quiet_with_either_block_or_contact_change": int(sum(
            row["physical_jump"]["material_side_block_change"]
            or row["physical_jump"]["any_contact_change"] for row in quiet)),
    }
    result = {
        "protocol": {"chunks": len(rows), "probes": len(scalars),
                     "dimension": args.dimension, "material_change_definition":
                     "adjacent side-block endpoint position >=2 mm, rotation >=2 deg, peak tilt >=2 deg, or a contact-signature change"},
        "summary": summary,
        "alarm_rows": alarm_rows,
        "block_names": diagnostics["block_names"].astype(str).tolist(),
        "contact_labels": diagnostics["contact_labels"].astype(str).tolist(),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("physical jump diagnostics")
    print(json.dumps(summary, indent=2))
    for row in quiet:
        p = row["physical_jump"]
        print(f"quiet ep{row['episode_id']} chunk{row['chunk_start']}: "
              f"side_pos={p['max_side_position_jump_mm']:.2f}mm "
              f"side_rot={p['max_side_rotation_jump_deg']:.2f}deg "
              f"side_peak={p['max_side_peak_tilt_jump_deg']:.2f}deg "
              f"ee={p['ee_position_jump_mm']:.2f}mm contacts="
              f"{p['endpoint_contact_flips'] + p['trajectory_contact_flips']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
