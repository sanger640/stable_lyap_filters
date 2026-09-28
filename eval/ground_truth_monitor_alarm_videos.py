"""Render ten exact alarm-pair videos each for Jenga TEST and fresh Panda pushing TEST."""
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

from jenga_bench import BENCH_FILE, SPLITS
from jenga_generic_regime_audit import reconstruct_endpoints
from jenga_regime_monitor_v0_freeze import sha256_file, verify as verify_v0
from jenga_runtime import DEFAULT_LMDB, JengaReplay
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim
from jenga_stage0_noise_oracle import HOLD
import panda_push_dev_panel as panda_dev
import panda_push_fresh_test as panda_test
from systems.panda_block_push import PandaBlockPush, write_push_xml

OUT = ROOT / "results/ground_truth_monitor_alarm_videos"
JENGA_RESULT = ROOT / "results/jenga/regime_monitor_v0/test_ground_truth.json"
FPS = 8


def consequential_pair(row):
    boundary = row["boundary_evidence"]
    for index, (early, full, commitment, persistence) in enumerate(zip(
            boundary["early_delta_bic"], boundary["full_delta_bic"],
            row["commitment_delta_bic"], row["persistence_delta_bic"])):
        if early > 0 and full > 0 and commitment > 0 and persistence > 0:
            return index
    raise ValueError("alarm row has no consequential pair")


def _phase(step):
    if step == 0: return "common start"
    if step <= HORIZON: return f"perturbed action {step}/{HORIZON}"
    return f"shared settle {step-HORIZON}/{HOLD}"


def _panel(frame, title, side, step, status, detail):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (448, 336), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.copyMakeBorder(image, 102, 0, 0, 0, cv2.BORDER_CONSTANT,
                                value=(242, 242, 242))
    cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .51, (25, 25, 25), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, f"ALARM | nearby side {side} | {_phase(step)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, .54, (0, 0, 185), 2, cv2.LINE_AA)
    colour = (0, 0, 185) if status != "upright" else (25, 115, 25)
    cv2.putText(canvas, f"physical status: {status}", (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, .48, colour, 1, cv2.LINE_AA)
    cv2.putText(canvas, detail, (10, 96), cv2.FONT_HERSHEY_SIMPLEX, .40, (35, 35, 35), 1,
                cv2.LINE_AA)
    return canvas


def _write_pair(path, left, right):
    height, width = left[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (2 * width, height))
    if not writer.isOpened(): raise RuntimeError(f"cannot create {path}")
    try:
        for step, panels in enumerate(zip(left, right)):
            joined = np.concatenate(panels, axis=1)
            repeats = FPS if step in (0, HORIZON, HORIZON + HOLD) else 1
            for _ in range(repeats): writer.write(joined)
    finally: writer.release()


def _jenga_sequence(sim, snapshot, chunk, noise, title, side):
    sim.restore(snapshot); actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise; frames = []; peak = 0.
    for step in range(HORIZON + HOLD + 1):
        diagnostic = sim.block_diagnostics()
        peak = max(peak, float(np.max(diagnostic["tilt"][1:])))
        fallen = peak >= 45.
        frames.append(_panel(sim.render(), title, side, step,
                             "neighbor toppled" if fallen else "upright",
                             f"neighbor peak tilt {peak:.1f} deg"))
        if step < HORIZON: sim.execute(actions[step])
        elif step < HORIZON + HOLD: sim.execute(chunk[-1])
    return frames, {"neighbor_peak_tilt_deg": peak, "neighbor_toppled": bool(peak >= 45.)}


def render_jenga(lmdb, archive, count):
    verify_v0(); result = json.loads(JENGA_RESULT.read_text())
    rows = [row for row in result["rows"] if row["alarm"] and row["class"] == "topple_fork"][:count]
    if len(rows) != count: raise RuntimeError(f"only {len(rows)} alarmed Jenga topple forks")
    snippets = np.load(BENCH_FILE, allow_pickle=False)["snippets"]
    directory = OUT / "jenga"; directory.mkdir(parents=True, exist_ok=True); manifest = []
    replay = JengaReplay(lmdb)
    try:
        with tempfile.TemporaryDirectory(prefix="jenga_alarm_videos_") as temp:
            sim = DirectJengaSim(extract_sim(archive, temp))
            try:
                for number, row in enumerate(rows, 1):
                    episode = replay.episode(row["episode_id"]); start = row["chunk_start"]
                    sim.reset(int(row["episode_id"]) + int(SPLITS["test"]["reset_base"]))
                    for action in episode.actions[:start]: sim.execute(action)
                    snapshot = sim.snapshot(); chunk = episode.actions[start:start + HORIZON]
                    pair = consequential_pair(row)
                    endpoints = reconstruct_endpoints(snippets, row["pair_details"])[pair]
                    title = f"Jenga TEST ep{row['episode_id']} action {start} pair {pair}"
                    sequences = [_jenga_sequence(sim, snapshot, chunk, noise, title, side)
                                 for noise, side in zip(endpoints, ("A", "B"))]
                    name = f"jenga_alarm_{number:02d}_ep{row['episode_id']}_a{start}.mp4"
                    _write_pair(directory / name, sequences[0][0], sequences[1][0])
                    manifest.append({"video": name, "episode_id": row["episode_id"],
                                     "chunk_start": start, "class": row["class"],
                                     "topple_count_64": row["topple_count"], "pair": pair,
                                     "sides": [sequences[0][1], sequences[1][1]]})
                    print(f"wrote {name}", flush=True)
            finally: sim.close()
    finally: replay.close()
    value = {"source": str(JENGA_RESULT.relative_to(ROOT)),
             "source_sha256": sha256_file(JENGA_RESULT), "count": count, "videos": manifest}
    (directory / "manifest.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


def _panda_sequence(env, snapshot, chunk, noise, title, side):
    env.restore(snapshot); actions = np.asarray(chunk, float).copy(); actions[:, :3] -= noise
    frames = []
    for step in range(HORIZON + HOLD + 1):
        tilt = env.block_tilt_deg(); toppled = env.peak_tilt_deg >= 45.
        frames.append(_panel(env.render(), title, side, step,
                             "block toppled" if toppled else "upright",
                             f"current tilt {tilt:.1f} deg | peak {env.peak_tilt_deg:.1f} deg"))
        if step < HORIZON: env.step(actions[step])
        elif step < HORIZON + HOLD: env.step(actions[-1])
    outcome = env.outcome()
    return frames, {"peak_tilt_deg": outcome["peak_tilt_deg"],
                    "toppled": outcome["failure"], "supported": outcome["supported"]}


def render_pushing(archive, count):
    panda_test.verify(); result = json.loads(panda_test.RESULT.read_text())
    rows = [row for row in result["rows"] if row["alarm"] and row["cohort"] == "topple_fork"][:count]
    if len(rows) != count: raise RuntimeError(f"only {len(rows)} alarmed pushing topple forks")
    cache = np.load(panda_test.CACHE, allow_pickle=False); snippets = cache["snippets"]
    directory = OUT / "pushing"; directory.mkdir(parents=True, exist_ok=True); manifest = []
    with tempfile.TemporaryDirectory(prefix="panda_alarm_videos_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        env = PandaBlockPush(xml, width=448, height=336, render=True)
        try:
            for number, row in enumerate(rows, 1):
                index = row["panel_index"]; snapshot = panda_dev._snapshot(cache, index)
                chunk = cache["chunks"][index]; pair = consequential_pair(row)
                endpoints = reconstruct_endpoints(snippets, row["pair_details"])[pair]
                title = f"Pushing TEST state {index} pair {pair} | {row['topple_count']}/64 topple"
                sequences = [_panda_sequence(env, snapshot, chunk, noise, title, side)
                             for noise, side in zip(endpoints, ("A", "B"))]
                name = f"pushing_alarm_{number:02d}_state{index:03d}.mp4"
                _write_pair(directory / name, sequences[0][0], sequences[1][0])
                manifest.append({"video": name, "panel_index": index, "cohort": row["cohort"],
                                 "topple_count_64": row["topple_count"], "pair": pair,
                                 "sides": [sequences[0][1], sequences[1][1]]})
                print(f"wrote {name}", flush=True)
        finally: env.close()
    value = {"source": str(panda_test.RESULT.relative_to(ROOT)),
             "source_sha256": sha256_file(panda_test.RESULT), "count": count, "videos": manifest}
    (directory / "manifest.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if (OUT / "jenga/manifest.json").exists() or (OUT / "pushing/manifest.json").exists():
        raise SystemExit(f"refusing to overwrite existing video manifests under {OUT}")
    summary = {"jenga": render_jenga(args.lmdb, args.sim_archive, args.count),
               "pushing": render_pushing(args.sim_archive, args.count)}
    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__": main()
