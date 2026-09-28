"""Universal DINO patch-correspondence geometry on frozen Panda upright-push DEV.

DINO is used only to identify the same visual region across frames.  The monitor receives normalized
2-D patch positions and finite-difference velocities, not descriptor differences.  Every patch in
the common pre-action frame is an anchor; no object mask, identity, outcome, reference population
or fitted confidence threshold is used.  A fixed top-4 soft correspondence makes sub-patch motion
continuous enough to test whether generic geometry repairs DINO's response coordinates.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

# One process and bounded math-library pools protect the desktop host.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"eval"))

from action_branch_monitor import (action_branch_partition, boundary_refinement_alarm,
                                   nearest_cross_branch_pairs, result_dict,
                                   smooth_vs_branch_alarm)
from consequence_monitor import result_dict as consequence_result_dict
from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model, preprocess_frames
from jenga_short_held_tails import extract_sim
from panda_push_dev_panel import (CACHE as PANEL_CACHE, HORIZON, PAIRS, PANEL,
                                  RECORD_HOLDS, REFINEMENTS, _projector, _reconstruct)
import panda_push_visual_pca as pca_baseline
import panda_push_visual_v0 as visual_baseline
from systems.panda_block_push import PandaBlockPush, write_push_xml
from visual_consequence_monitor import visual_whole_trajectory_consequence_alarm

OUT = ROOT/"results/panda_block_push/patch_correspondence"
CACHE_DIR = OUT/"evaluated_states"
RESULT = OUT/"patch_correspondence_diagnostic.json"
TOP_K = 4
PCA_DIMS = 45
SAMPLE_TIMES = np.asarray([HORIZON + held for held in RECORD_HOLDS], float)

PROTOCOL = {
    "id": "panda-upright-push-dino-patch-correspondence-dev",
    "status": "DEV-only universal representation diagnostic; not a holdout claim",
    "source_panel_sha256": visual_baseline.PROTOCOL["source_panel_sha256"],
    "encoder_role": "frozen DINOv2-S/14 patch descriptors used only for correspondence",
    "anchors": "all 16x16 patches in each state's common pre-action image",
    "matching": "cosine top-4; per-anchor standardized softmax barycenter; direct start-to-frame matching",
    "geometry": "normalized x/y position plus finite-difference vx/vy for every anchor",
    "compression": "per-state unlabeled PCA45 fitted to 64 probe trajectories",
    "monitor": "unchanged action branch/refinement plus proper visual-native consequence stage",
    "object_mask_or_identity": False,
    "confidence_threshold": None,
    "reference_or_outcome_calibration": False,
    "task_labels_in_monitor": False,
    "resource_policy": "one simulator process; Torch <=8, BLAS <=2 CPU threads",
}


def encode_tokens(encoder, frames, device, batch_size):
    frames = np.asarray(frames); leading = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:]); rows = []
    for first in range(0, len(flat), batch_size):
        with torch.inference_mode():
            image = preprocess_frames(flat[first:first+batch_size, None], device)[:, 0]
            rows.append(encoder.forward(image).float().cpu().numpy())
    tokens = np.concatenate(rows)
    return tokens.reshape(*leading, *tokens.shape[-2:])


def patch_coordinates(patches):
    side = int(round(np.sqrt(patches)))
    if side*side != patches: raise ValueError("patch count must form a square grid")
    axis = np.linspace(-1., 1., side)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def correspondence_positions(start_tokens, frame_tokens, top_k=TOP_K):
    """Map each start anchor to a soft image coordinate in every subsequent frame."""
    start = np.asarray(start_tokens, np.float32)
    frames = np.asarray(frame_tokens, np.float32)
    if start.ndim != 2 or frames.shape[-2:] != start.shape:
        raise ValueError("expected start (patches, features) and frames (..., patches, features)")
    leading = frames.shape[:-2]; flat = frames.reshape(-1, *start.shape)
    start = start/np.maximum(np.linalg.norm(start, axis=1, keepdims=True), 1e-12)
    flat = flat/np.maximum(np.linalg.norm(flat, axis=2, keepdims=True), 1e-12)
    coordinates = patch_coordinates(len(start)); output = []
    k = min(int(top_k), len(start))
    for current in flat:
        similarity = start@current.T
        indices = np.argpartition(similarity, -k, axis=1)[:, -k:]
        values = np.take_along_axis(similarity, indices, axis=1)
        order = np.argsort(values, axis=1)[:, ::-1]
        indices = np.take_along_axis(indices, order, axis=1)
        values = np.take_along_axis(values, order, axis=1)
        centred = values-values.mean(axis=1, keepdims=True)
        scale = np.maximum(values.std(axis=1, keepdims=True), 1e-6)
        logits = centred/scale; logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits); weights /= weights.sum(axis=1, keepdims=True)
        output.append(np.sum(coordinates[indices]*weights[..., None], axis=1))
    return np.stack(output).reshape(*leading, len(start), 2).astype(np.float32)


def geometric_trajectories(start_tokens, frame_tokens, times):
    """Return aligned [x,y,vx,vy] for every start-frame patch anchor."""
    positions = correspondence_positions(start_tokens, frame_tokens)
    times = np.asarray(times, float)
    if positions.shape[-3] != len(times) or np.any(np.diff(times) <= 0) or times[0] <= 0:
        raise ValueError("times must be positive, increasing and match trajectory length")
    # Use the matcher's self-correspondence as the geometric origin. Soft top-k matching is
    # intentionally not a one-hot lookup, so raw grid centers would create artificial velocity in
    # a perfectly static image.
    anchors = correspondence_positions(start_tokens, np.asarray(start_tokens)[None])[0]
    previous = np.concatenate([
        np.broadcast_to(anchors, (*positions.shape[:-3], 1, *anchors.shape)), positions[..., :-1, :, :]
    ], axis=-3)
    dt = np.diff(np.r_[0., times]).reshape(*([1]*(positions.ndim-3)), len(times), 1, 1)
    velocity = (positions-previous)/dt
    return np.concatenate([positions, velocity], axis=-1).reshape(*positions.shape[:-2], -1)


def render_geometry(env, encoder, start_tokens, snapshot, chunk, noises, device, batch_size,
                    dense=False):
    frames = np.stack([visual_baseline._execute(env, snapshot, chunk, noise, dense=dense)[0]
                       for noise in noises])
    tokens = encode_tokens(encoder, frames, device, batch_size)
    times = np.arange(1, HORIZON+30+1, dtype=float) if dense else SAMPLE_TIMES
    return frames, geometric_trajectories(start_tokens, tokens, times)


def evaluate_state(index, row, physical, panel_cache, env, encoder, device, batch_size):
    destination = CACHE_DIR/f"state_{index:03d}.npz"
    if destination.exists():
        return json.loads(str(np.load(destination, allow_pickle=False)["result_json"]))
    rendered = np.load(visual_baseline.RENDER_DIR/f"state_{index:03d}.npz", allow_pickle=False)
    region = np.load(ROOT/f"results/panda_block_push/visual_regions/input_cache/state_{index:03d}.npz",
                     allow_pickle=False)
    start_tokens = encode_tokens(encoder, region["start"][None], device, batch_size)[0]
    initial_tokens = encode_tokens(encoder, rendered["frames"], device, batch_size)
    raw = geometric_trajectories(start_tokens, initial_tokens, SAMPLE_TIMES)
    pca, visual = pca_baseline.fit_pca(raw)
    snippets = np.asarray(panel_cache["snippets"], float)
    errors = -snippets-(-snippets).mean(0, keepdims=True)
    candidate = smooth_vs_branch_alarm(errors, visual[:, :2], visual)
    out = {**row, "initial_alarm": bool(candidate.alarm), "boundary_refined_alarm": False,
           "alarm": False, "initial_evidence": result_dict(candidate),
           "physical_initial_alarm": bool(physical["initial_alarm"]),
           "physical_alarm": bool(physical["alarm"]),
           "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())}
    payload = {"initial_geometry": raw.astype(np.float32), "initial_features": visual,
               "pca_mean": pca.mean_.astype(np.float32),
               "pca_components": pca.components_.astype(np.float32)}
    if candidate.alarm:
        x, labels = action_branch_partition(errors, visual[:, :2], visual)
        pairs = nearest_cross_branch_pairs(x, labels, PAIRS)
        early_project, full_project = _projector(visual[:, :2]), _projector(visual)
        original_early, original_full = early_project(visual[:, :2]), full_project(visual)
        centroids = np.stack([original_full[labels == group].mean(0) for group in (0, 1)])
        snapshot = visual_baseline._restore_snapshot(panel_cache, index)
        chunk = panel_cache["chunks"][index]
        early_gaps, full_gaps, details = [], [], []
        for left, right in pairs:
            endpoint_noise = [snippets[left].copy(), snippets[right].copy()]
            endpoint_visual = [visual[left].copy(), visual[right].copy()]
            eg = [float(np.linalg.norm(original_early[left]-original_early[right]))]
            fg = [float(np.linalg.norm(original_full[left]-original_full[right]))]; sides = []
            for _ in range(REFINEMENTS):
                midpoint = .5*(endpoint_noise[0]+endpoint_noise[1])
                _, geometry = render_geometry(env, encoder, start_tokens, snapshot, chunk,
                                              [midpoint], device, batch_size)
                midpoint_visual = pca.transform(
                    geometry.reshape(-1, geometry.shape[-1])).reshape(
                        *geometry.shape[:-1], PCA_DIMS).astype(np.float32)[0]
                projected = full_project(midpoint_visual[None])[0]
                side = int(np.argmin(np.linalg.norm(centroids-projected[None], axis=1)))
                endpoint_noise[side] = midpoint; endpoint_visual[side] = midpoint_visual
                sides.append(side)
                ee = early_project(np.stack([value[:2] for value in endpoint_visual]))
                ff = full_project(np.stack(endpoint_visual))
                eg.append(float(np.linalg.norm(ee[0]-ee[1])))
                fg.append(float(np.linalg.norm(ff[0]-ff[1])))
            early_gaps.append(eg); full_gaps.append(fg)
            details.append({"probe_indices": [int(left), int(right)], "midpoint_sides": sides})
        boundary = boundary_refinement_alarm(early_gaps, full_gaps)
        endpoints = np.stack([_reconstruct(snippets, pair, detail["midpoint_sides"])
                              for pair, detail in zip(pairs, details)])
        dense_features = []
        for endpoint_pair in endpoints:
            _, geometry = render_geometry(env, encoder, start_tokens, snapshot, chunk,
                                          endpoint_pair, device, batch_size, dense=True)
            dense_features.append(pca.transform(geometry.reshape(-1, geometry.shape[-1])).reshape(
                *geometry.shape[:-1], PCA_DIMS).astype(np.float32))
        dense_features = np.stack(dense_features)
        evidence = visual_whole_trajectory_consequence_alarm(
            dense_features, endpoints[:, 0]-endpoints[:, 1],
            boundary.early_delta_bic, boundary.full_delta_bic)
        out.update({"boundary_refined_alarm": bool(boundary.alarm),
                    "boundary_evidence": result_dict(boundary), "pair_details": details,
                    **consequence_result_dict(evidence)})
        payload.update({"dense_features": dense_features, "endpoints": endpoints.astype(np.float32)})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload["result_json"] = np.asarray(json.dumps(out, allow_nan=False))
    np.savez_compressed(destination, **payload)
    return out


def evaluate(archive, checkpoint, device, batch_size):
    visual_baseline.verify_render()
    panel = json.loads(PANEL.read_text()); panel_cache = np.load(PANEL_CACHE, allow_pickle=False)
    physical = json.loads((PANEL.parent/"v0_stage_diagnostic.json").read_text())["rows"]
    model = load_world_model(checkpoint, device); encoder = model.encoder; rows = []
    with tempfile.TemporaryDirectory(prefix="panda_push_patch_correspondence_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        env = PandaBlockPush(xml, width=visual_baseline.WIDTH,
                             height=visual_baseline.HEIGHT, render=True)
        try:
            for index, (row, physical_row) in enumerate(zip(panel["rows"], physical)):
                rows.append(evaluate_state(index, row, physical_row, panel_cache, env, encoder,
                                           device, batch_size))
                print(f"patch correspondence {index+1}/{len(panel['rows'])} state={index}", flush=True)
        finally:
            env.close()
    value = {"protocol": PROTOCOL,
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "summary": visual_baseline.summarise(rows, physical), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(json.dumps(value["summary"], indent=2)); return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-archive", default=str(ROOT/"vendor/panda_express_sim.tar"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, choices=(1, 2, 4, 6, 8), default=8)
    args = parser.parse_args(); torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    evaluate(args.sim_archive, args.checkpoint, args.device, args.batch_size)


if __name__ == "__main__": main()
