"""Generic high-resolution optical-flow geometry on frozen Panda upright-push DEV.

This is the prospectively specified follow-up to the negative DINO patch-correspondence arm.
Generic Shi--Tomasi points are detected once in each state's common pre-action image and tracked
through every control frame with pyramidal Lucas--Kanade flow.  The monitor sees only normalized
image position, one-step velocity and visibility.  There is no object mask, object identity,
simulator state, task label or outcome-calibrated threshold.

The experiment is deliberately resource-safe: one simulator process, two math-library threads,
and one trajectory rendered/tracked at a time.  Dense RGB frames are never retained in the cache.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

# Keep OpenCV and its numerical libraries from oversubscribing the desktop host.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (action_branch_partition, boundary_refinement_alarm,
                                   nearest_cross_branch_pairs, result_dict,
                                   smooth_vs_branch_alarm)
from consequence_monitor import result_dict as consequence_result_dict
from jenga_short_held_tails import extract_sim
from panda_push_dev_panel import (CACHE as PANEL_CACHE, HORIZON, PAIRS, PANEL,
                                  REFINEMENTS, _projector, _reconstruct)
import panda_push_visual_pca as pca_baseline
import panda_push_visual_v0 as visual_baseline
from systems.panda_block_push import PandaBlockPush, write_push_xml
from visual_consequence_monitor import visual_whole_trajectory_consequence_alarm

OUT = ROOT / "results/panda_block_push/optical_flow"
CACHE_DIR = OUT / "evaluated_states"
RESULT = OUT / "optical_flow_diagnostic.json"
START_DIR = ROOT / "results/panda_block_push/visual_regions/input_cache"
MAX_CORNERS = 128
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 5
BLOCK_SIZE = 7
WINDOW_SIZE = (21, 21)
PYRAMID_LEVELS = 3
TERMINATION = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
FB_MAX_ERROR_PX = 1.0
PCA_DIMS = 45

PROTOCOL = {
    "id": "panda-upright-push-generic-optical-flow-dev",
    "status": "DEV-only universal representation diagnostic; not a holdout claim",
    "source_panel_sha256": visual_baseline.PROTOCOL["source_panel_sha256"],
    "anchors": {
        "detector": "Shi-Tomasi corners in each state's common pre-action RGB image",
        "max_corners": MAX_CORNERS,
        "quality_level": QUALITY_LEVEL,
        "minimum_distance_px": MIN_DISTANCE,
        "block_size_px": BLOCK_SIZE,
    },
    "tracking": {
        "method": "frame-to-frame pyramidal Lucas-Kanade",
        "window_px": list(WINDOW_SIZE),
        "pyramid_levels": PYRAMID_LEVELS,
        "termination": ["epsilon_or_30_iterations", 0.01],
        "forward_backward_max_error_px": FB_MAX_ERROR_PX,
        "loss_policy": "once invalid, carry last position and keep visibility zero",
    },
    "geometry": "per-anchor normalized x/y position, one-control-step vx/vy, visibility",
    "compression": "per-state unlabeled PCA45 fitted to 64 probe trajectories",
    "monitor": "unchanged action branch/refinement plus proper visual-native consequence stage",
    "object_mask_or_identity": False,
    "simulator_state_in_representation": False,
    "reference_or_outcome_calibration": False,
    "task_labels_in_monitor": False,
    "resource_policy": "one simulator process; one trajectory at a time; OpenCV/BLAS <=2 threads",
}


def _gray(frame):
    return cv2.cvtColor(np.asarray(frame, np.uint8), cv2.COLOR_RGB2GRAY)


def detect_points(start_frame, max_corners=MAX_CORNERS):
    """Detect generic image corners without an object mask or semantic selection."""
    points = cv2.goodFeaturesToTrack(
        _gray(start_frame), maxCorners=int(max_corners), qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE, blockSize=BLOCK_SIZE, useHarrisDetector=False)
    if points is None or len(points) == 0:
        raise RuntimeError("no generic corners found")
    return np.asarray(points, np.float32).reshape(-1, 1, 2)


def track_geometry(start_frame, frames, points, fb_max_error_px=FB_MAX_ERROR_PX):
    """Track common anchors and return [x,y,vx,vy,visible] at every supplied frame."""
    images = np.asarray(frames, np.uint8)
    anchors = np.asarray(points, np.float32).reshape(-1, 1, 2)
    if images.ndim != 4 or images.shape[-1] != 3 or len(anchors) == 0:
        raise ValueError("expected RGB frames and at least one (x,y) point")
    height, width = images.shape[1:3]
    previous_gray = _gray(start_frame)
    previous = anchors.copy()
    active = np.ones(len(anchors), dtype=bool)
    rows = []
    lk = dict(winSize=WINDOW_SIZE, maxLevel=PYRAMID_LEVELS, criteria=TERMINATION)
    for frame in images:
        current_gray = _gray(frame)
        forward, status_f, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray,
                                                       previous, None, **lk)
        if forward is None:
            valid = np.zeros(len(previous), dtype=bool)
            forward = previous.copy()
        else:
            backward, status_b, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray,
                                                             forward, None, **lk)
            if backward is None:
                valid = np.zeros(len(previous), dtype=bool)
            else:
                fb_error = np.linalg.norm(backward[:, 0] - previous[:, 0], axis=1)
                x, y = forward[:, 0, 0], forward[:, 0, 1]
                valid = ((status_f[:, 0] > 0) & (status_b[:, 0] > 0) &
                         np.isfinite(forward[:, 0]).all(axis=1) &
                         (fb_error <= float(fb_max_error_px)) &
                         (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1))
        active &= valid
        current = previous.copy()
        current[active] = forward[active]
        delta = current[:, 0] - previous[:, 0]
        position = current[:, 0].copy()
        position[:, 0] = 2. * position[:, 0] / max(width - 1, 1) - 1.
        position[:, 1] = 2. * position[:, 1] / max(height - 1, 1) - 1.
        velocity = delta.copy()
        velocity[:, 0] *= 2. / max(width - 1, 1)
        velocity[:, 1] *= 2. / max(height - 1, 1)
        rows.append(np.column_stack([position, velocity, active.astype(np.float32)]).reshape(-1))
        previous = current
        previous_gray = current_gray
    return np.asarray(rows, np.float32)


def render_geometry(env, start_frame, points, snapshot, chunk, noises, dense=False):
    """Render and immediately reduce trajectories; RGB frames are not retained."""
    output = []
    outcomes = []
    sample_indices = np.asarray(visual_baseline._sample_indices(), int)
    for noise in noises:
        frames, outcome = visual_baseline._execute(env, snapshot, chunk, noise, dense=True)
        geometry = track_geometry(start_frame, frames, points)
        output.append(geometry if dense else geometry[sample_indices])
        outcomes.append(outcome)
    return np.stack(output), outcomes


def evaluate_state(index, row, physical, panel_cache, env):
    destination = CACHE_DIR / f"state_{index:03d}.npz"
    if destination.exists():
        return json.loads(str(np.load(destination, allow_pickle=False)["result_json"]))
    start_path = START_DIR / f"state_{index:03d}.npz"
    if not start_path.exists():
        raise FileNotFoundError(f"missing common start RGB cache: {start_path}")
    start_frame = np.load(start_path, allow_pickle=False)["start"]
    points = detect_points(start_frame)
    if len(points) * 5 < PCA_DIMS:
        raise RuntimeError(f"state {index}: only {len(points)} corners; PCA45 needs at least 9")
    snapshot = visual_baseline._restore_snapshot(panel_cache, index)
    chunk = panel_cache["chunks"][index]
    snippets = np.asarray(panel_cache["snippets"], float)
    raw, outcomes = render_geometry(env, start_frame, points, snapshot, chunk, snippets)
    expected = np.load(visual_baseline.RENDER_DIR / f"state_{index:03d}.npz",
                       allow_pickle=False)["failures"].astype(bool)
    observed = np.asarray([outcome["failure"] for outcome in outcomes], bool)
    if not np.array_equal(observed, expected):
        raise RuntimeError(f"state {index}: dense optical-flow replay changed physical outcomes")

    pca, visual = pca_baseline.fit_pca(raw)
    errors = -snippets - (-snippets).mean(0, keepdims=True)
    candidate = smooth_vs_branch_alarm(errors, visual[:, :2], visual)
    out = {**row, "initial_alarm": bool(candidate.alarm), "boundary_refined_alarm": False,
           "alarm": False, "initial_evidence": result_dict(candidate),
           "physical_initial_alarm": bool(physical["initial_alarm"]),
           "physical_alarm": bool(physical["alarm"]), "tracked_points": int(len(points)),
           "mean_final_visibility": float(raw[:, -1, 4::5].mean()),
           "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())}
    payload = {"points": points[:, 0].astype(np.float32),
               "initial_geometry": raw.astype(np.float32), "initial_features": visual,
               "pca_mean": pca.mean_.astype(np.float32),
               "pca_components": pca.components_.astype(np.float32)}

    if candidate.alarm:
        x, labels = action_branch_partition(errors, visual[:, :2], visual)
        pairs = nearest_cross_branch_pairs(x, labels, PAIRS)
        early_project, full_project = _projector(visual[:, :2]), _projector(visual)
        original_early, original_full = early_project(visual[:, :2]), full_project(visual)
        centroids = np.stack([original_full[labels == group].mean(0) for group in (0, 1)])
        early_gaps, full_gaps, details = [], [], []
        for left, right in pairs:
            endpoint_noise = [snippets[left].copy(), snippets[right].copy()]
            endpoint_visual = [visual[left].copy(), visual[right].copy()]
            early_gap = [float(np.linalg.norm(original_early[left] - original_early[right]))]
            full_gap = [float(np.linalg.norm(original_full[left] - original_full[right]))]
            sides = []
            for _ in range(REFINEMENTS):
                midpoint = .5 * (endpoint_noise[0] + endpoint_noise[1])
                geometry, _ = render_geometry(env, start_frame, points, snapshot, chunk, [midpoint])
                midpoint_visual = pca.transform(
                    geometry.reshape(-1, geometry.shape[-1])).reshape(
                        *geometry.shape[:-1], PCA_DIMS).astype(np.float32)[0]
                projected = full_project(midpoint_visual[None])[0]
                side = int(np.argmin(np.linalg.norm(centroids - projected[None], axis=1)))
                endpoint_noise[side] = midpoint
                endpoint_visual[side] = midpoint_visual
                sides.append(side)
                early_pair = early_project(np.stack([value[:2] for value in endpoint_visual]))
                full_pair = full_project(np.stack(endpoint_visual))
                early_gap.append(float(np.linalg.norm(early_pair[0] - early_pair[1])))
                full_gap.append(float(np.linalg.norm(full_pair[0] - full_pair[1])))
            early_gaps.append(early_gap); full_gaps.append(full_gap)
            details.append({"probe_indices": [int(left), int(right)], "midpoint_sides": sides})
        boundary = boundary_refinement_alarm(early_gaps, full_gaps)
        endpoints = np.stack([_reconstruct(snippets, pair, detail["midpoint_sides"])
                              for pair, detail in zip(pairs, details)])
        dense_features = []
        for endpoint_pair in endpoints:
            geometry, _ = render_geometry(env, start_frame, points, snapshot, chunk,
                                          endpoint_pair, dense=True)
            dense_features.append(pca.transform(
                geometry.reshape(-1, geometry.shape[-1])).reshape(
                    *geometry.shape[:-1], PCA_DIMS).astype(np.float32))
        dense_features = np.stack(dense_features)
        evidence = visual_whole_trajectory_consequence_alarm(
            dense_features, endpoints[:, 0] - endpoints[:, 1],
            boundary.early_delta_bic, boundary.full_delta_bic)
        out.update({"boundary_refined_alarm": bool(boundary.alarm),
                    "boundary_evidence": result_dict(boundary), "pair_details": details,
                    **consequence_result_dict(evidence)})
        payload.update({"dense_features": dense_features,
                        "endpoints": endpoints.astype(np.float32)})

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload["result_json"] = np.asarray(json.dumps(out, allow_nan=False))
    np.savez_compressed(destination, **payload)
    return out


def evaluate(archive):
    visual_baseline.verify_render()
    panel = json.loads(PANEL.read_text())
    panel_cache = np.load(PANEL_CACHE, allow_pickle=False)
    physical = json.loads((PANEL.parent / "v0_stage_diagnostic.json").read_text())["rows"]
    rows = []
    cv2.setNumThreads(2)
    with tempfile.TemporaryDirectory(prefix="panda_push_optical_flow_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        env = PandaBlockPush(xml, width=visual_baseline.WIDTH,
                             height=visual_baseline.HEIGHT, render=True)
        try:
            for index, (row, physical_row) in enumerate(zip(panel["rows"], physical)):
                rows.append(evaluate_state(index, row, physical_row, panel_cache, env))
                print(f"optical flow {index + 1}/{len(panel['rows'])} state={index}", flush=True)
        finally:
            env.close()
    value = {"protocol": PROTOCOL,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "summary": visual_baseline.summarise(rows, physical), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(value["summary"], indent=2))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    args = parser.parse_args()
    evaluate(args.sim_archive)


if __name__ == "__main__":
    main()
