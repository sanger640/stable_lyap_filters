"""Render paired physical-probe comparisons from cached boundary and hard-negative sets."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_near_boundary_discovery import offset_probe_chunks  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim  # noqa: E402
from jenga_trajectory_oracle import read_cache  # noqa: E402

CASES = (("boundary", "48", 74), ("boundary", "33", 122),
         ("moving_safe", "3", 106), ("moving_safe", "79", 98))
TAIL = 5
FPS = 4


def chosen_indices(kind, scalars, outcomes):
    if kind == "boundary":
        cuts = np.flatnonzero(outcomes[1:] != outcomes[:-1])
        if not len(cuts):
            raise ValueError("boundary example has no outcome switch")
        cut = int(cuts[np.argmin(np.abs((scalars[cuts] + scalars[cuts + 1]) / 2))])
        nominal = next(i for i in np.argsort(np.abs(scalars)) if i not in (cut, cut + 1))
        return [cut, cut + 1, int(nominal)]
    if np.any(outcomes):
        raise ValueError("moving-safe example includes a toppled probe")
    return [0, int(np.argmin(np.abs(scalars))), len(scalars) - 1]


def panel(frame, label, step, tilt, expected_failure, zoom=False):
    if zoom:
        frame = cv2.resize(frame[:125, 95:225], (400, 385), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    canvas = np.full((image.shape[0] + 72, image.shape[1], 3), 245, np.uint8)
    canvas[72:] = image
    cv2.putText(canvas, label, (9, 20), cv2.FONT_HERSHEY_SIMPLEX, .49, (20, 20, 20), 1)
    cv2.putText(canvas, f"step {step:02d}/13  peak tilt {tilt:.1f} deg", (9, 42),
                cv2.FONT_HERSHEY_SIMPLEX, .46, (20, 20, 20), 1)
    cv2.putText(canvas, "TOPPLED" if expected_failure else "SAFE", (9, 62),
                cv2.FONT_HERSHEY_SIMPLEX, .49,
                (0, 0, 180) if expected_failure else (30, 110, 30), 1)
    return canvas


def render_case(sim, snapshot, actions, row, scalars, outcomes, kind, output,
                indices=None, zoom=False):
    indices = chosen_indices(kind, scalars, outcomes) if indices is None else list(indices)
    probes = offset_probe_chunks(actions, scalars[indices], .10, row.get("center_offset", 0.0))
    sequences, measured = [], []
    for probe, index in zip(probes, indices):
        sim.restore(snapshot)
        start_middle = sim.block_diagnostics()["position"][0].copy()
        middle_positions = [start_middle]
        trace = {"peak_tilt": sim.block_diagnostics()["tilt"].copy()}
        frames = [sim.render()]
        tilts = [float(np.max(trace["peak_tilt"][1:]))]
        for action in list(probe) + [probe[-1]] * TAIL:
            sim.execute(action, trace)
            frames.append(sim.render())
            tilts.append(float(np.max(trace["peak_tilt"][1:])))
            middle_positions.append(sim.block_diagnostics()["position"][0].copy())
        actual = bool(tilts[-1] >= TOPPLE_DEG)
        if actual != bool(outcomes[index]):
            raise AssertionError(f"cached outcome mismatch at probe {index}")
        sequences.append((frames, tilts))
        middle_positions = np.asarray(middle_positions)
        measured.append({"probe_index": int(index), "scalar": float(scalars[index]),
                         "toppled": actual, "peak_neighbor_tilt_deg": tilts[-1],
                         "middle_lift_m": float(np.max(middle_positions[:, 2] - start_middle[2])),
                         "middle_lateral_m": float(np.linalg.norm(
                             middle_positions[-1, :2] - start_middle[:2]))})
    height, width = (385, 400) if zoom else sequences[0][0][0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (width * len(indices), height + 72))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer unavailable")
    try:
        for step in range(HORIZON + TAIL + 1):
            panes = []
            for (frames, tilts), item in zip(sequences, measured):
                label = f"probe {item['probe_index']:02d}   s={item['scalar']:+.3f}"
                panes.append(panel(frames[step], label, step, tilts[step],
                                   item["toppled"], zoom=zoom))
            joined = np.concatenate(panes, axis=1)
            for _ in range(4 if step in (0, HORIZON + TAIL) else 1):
                writer.write(joined)
    finally:
        writer.release()
        sim.restore(snapshot)
    return measured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/trajectory_oracle_cache.npz"))
    ap.add_argument("--expanded-cache", default=str(ROOT / "results/jenga/expanded_controls_cache.npz"))
    ap.add_argument("--original-result", default=str(ROOT / "results/jenga/trajectory_oracle.json"))
    ap.add_argument("--expanded-result", default=str(ROOT / "results/jenga/expanded_controls.json"))
    ap.add_argument("--output-dir", default=str(ROOT / "results/jenga/probe_comparison_videos"))
    args = ap.parse_args()
    caches = {"boundary": read_cache(args.original_cache),
              "moving_safe": read_cache(args.expanded_cache)}
    results = {"boundary": json.loads(Path(args.original_result).read_text())["rows"],
               "moving_safe": json.loads(Path(args.expanded_result).read_text())["rows"]}
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    replay = JengaReplay(args.lmdb)
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_probe_videos_") as temp:
            sim = DirectJengaSim(extract_sim(args.sim_archive, temp))
            try:
                for kind, episode_id, start in CASES:
                    metadata, scalars, _, outcomes = caches[kind]
                    matches = [i for i, row in enumerate(metadata)
                               if row["episode_id"] == episode_id and row["chunk_start"] == start]
                    if len(matches) != 1:
                        raise ValueError(f"missing or duplicate case {kind} ep{episode_id} chunk{start}")
                    i = matches[0]; row = results[kind][i]
                    assert row["episode_id"] == episode_id and row["chunk_start"] == start
                    episode = replay.episode(episode_id)
                    sim.reset(int(episode_id))
                    for action in episode.actions[:start]:
                        sim.execute(action)
                    snapshot = sim.snapshot()
                    name = f"{kind}_ep{episode_id}_chunk{start}.mp4"
                    probes = render_case(sim, snapshot, episode.actions[start:start + HORIZON],
                                         row, scalars, outcomes[i], kind, output_dir / name)
                    manifest.append({"video": name, "kind": kind, "episode_id": episode_id,
                                     "chunk_start": start, "detector_alarm": row["alarm"],
                                     "trajectory_score": row["trajectory_score"],
                                     "cached_counts": row["physical_counts"], "shown_probes": probes})
                    print(f"wrote {name}", flush=True)
            finally:
                sim.close()
    finally:
        replay.close()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
