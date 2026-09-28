"""Label-free PCA visual-adapter comparison on the frozen Panda push DEV panel.

This changes exactly one representational choice from ``panda_push_visual_v0.py``: the fixed
Gaussian projection is replaced by PCA45 fitted independently to the current state's 64 unlabeled
probe trajectories.  No task outcome, physical-v0 decision, reference state, or fitted alarm
threshold enters PCA.  The frozen v0 monitor and physical panel remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import (action_branch_partition, boundary_refinement_alarm,
                                   nearest_cross_branch_pairs, result_dict,
                                   smooth_vs_branch_alarm)
from consequence_monitor import (result_dict as consequence_result_dict,
                                 whole_trajectory_consequence_alarm)
from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model, preprocess_frames
from jenga_short_held_tails import extract_sim
from panda_push_dev_panel import (CACHE as PANEL_CACHE, PAIRS, PANEL, REFINEMENTS,
                                  _projector, _reconstruct)
import panda_push_visual_v0 as baseline
from systems.panda_block_push import PandaBlockPush, write_push_xml

OUT = ROOT / "results/panda_block_push/visual_pca45"
CACHE_DIR = OUT / "evaluated_states"
RESULT = OUT / "visual_pca45_diagnostic.json"
PCA_DIMS = 45
PCA_SEED = 640

PROTOCOL = {
    "id": "panda-upright-push-rendered-v0-pca45-transfer-dev",
    "status": "DEV-only representation comparison; not a holdout claim",
    "source_panel_sha256": baseline.PROTOCOL["source_panel_sha256"],
    "initial_render_source": "verified panda_push_visual_v0 exact rendered cache",
    "encoder": baseline.ENCODER,
    "pre_pca_features": "full-frame DINO patch tokens adaptively pooled to 4x4 (6144-D)",
    "adapter": "per-state PCA45 fitted to all 64 unlabeled probe trajectories at hold 5/10/20/29/30",
    "pca_solver": "sklearn randomized SVD with fixed seed 640",
    "refinement": "PCA-selected 3 nearest cross-branch pairs, 5 exact bisections",
    "dense_final": "both refined endpoints at every H8+hold30 control step",
    "monitor": "frozen Regime Monitor v0 functions unchanged",
    "labels_or_reference_states_in_adapter": False,
    "task_labels_in_monitor": False,
    "adapter_limitation": "PCA coordinates satisfy v0's 45-field input shape but not its semantic physical channel grouping",
}


def pooled_dino_features(encoder, frames, device, batch_size):
    """Encode arbitrary leading frame dimensions to label-free 4x4 pooled DINO features."""
    frames = np.asarray(frames); leading = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:]); out = []
    for first in range(0, len(flat), batch_size):
        with torch.inference_mode():
            image = preprocess_frames(flat[first:first+batch_size, None], device)[:, 0]
            tokens = encoder.forward(image).float()
            side = int(round(np.sqrt(tokens.shape[1])))
            if side * side != tokens.shape[1] or tokens.shape[2] != 384:
                raise RuntimeError(f"unexpected DINO patch-token shape {tuple(tokens.shape)}")
            grid = tokens.reshape(len(tokens), side, side, 384).permute(0, 3, 1, 2)
            out.append(F.adaptive_avg_pool2d(grid, (baseline.POOL_GRID,
                                                    baseline.POOL_GRID)).flatten(1).cpu().numpy())
    return np.concatenate(out).reshape(*leading, -1)


def fit_pca(initial):
    values = np.asarray(initial, np.float32)
    flat = values.reshape(-1, values.shape[-1])
    model = PCA(n_components=PCA_DIMS, svd_solver="randomized", random_state=PCA_SEED)
    projected = model.fit_transform(flat).reshape(*values.shape[:-1], PCA_DIMS)
    return model, projected.astype(np.float32)


def _render_projected(env, encoder, pca, snapshot, chunk, noises, device, batch_size,
                      dense=False):
    frames = np.stack([baseline._execute(env, snapshot, chunk, noise, dense=dense)[0]
                       for noise in noises])
    raw = pooled_dino_features(encoder, frames, device, batch_size)
    shape = raw.shape[:-1]
    projected = pca.transform(raw.reshape(-1, raw.shape[-1])).reshape(*shape, PCA_DIMS)
    return frames, projected.astype(np.float32)


def evaluate_state(index, row, physical, panel_cache, env, encoder, device, batch_size):
    destination = CACHE_DIR/f"state_{index:03d}.npz"
    if destination.exists():
        return json.loads(str(np.load(destination, allow_pickle=False)["result_json"]))
    initial = np.load(baseline.RENDER_DIR/f"state_{index:03d}.npz", allow_pickle=False)
    raw = pooled_dino_features(encoder, initial["frames"], device, batch_size)
    pca, visual = fit_pca(raw)
    snippets = np.asarray(panel_cache["snippets"], float)
    errors = -snippets - (-snippets).mean(0, keepdims=True)
    candidate = smooth_vs_branch_alarm(errors, visual[:, :2], visual)
    out = {**row, "initial_alarm": bool(candidate.alarm), "boundary_refined_alarm": False,
           "alarm": False, "initial_evidence": result_dict(candidate),
           "physical_initial_alarm": bool(physical["initial_alarm"]),
           "physical_alarm": bool(physical["alarm"]),
           "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())}
    payload = {"initial_features": visual, "pca_mean": pca.mean_.astype(np.float32),
               "pca_components": pca.components_.astype(np.float32)}
    if candidate.alarm:
        x, labels = action_branch_partition(errors, visual[:, :2], visual)
        pairs = nearest_cross_branch_pairs(x, labels, PAIRS)
        early_project, full_project = _projector(visual[:, :2]), _projector(visual)
        original_early, original_full = early_project(visual[:, :2]), full_project(visual)
        centroids = np.stack([original_full[labels == group].mean(0) for group in (0, 1)])
        snapshot = baseline._restore_snapshot(panel_cache, index)
        chunk = panel_cache["chunks"][index]
        early_gaps, full_gaps, details, midpoint_frames = [], [], [], []
        for left, right in pairs:
            endpoint_noise = [snippets[left].copy(), snippets[right].copy()]
            endpoint_visual = [visual[left].copy(), visual[right].copy()]
            eg = [float(np.linalg.norm(original_early[left]-original_early[right]))]
            fg = [float(np.linalg.norm(original_full[left]-original_full[right]))]
            sides = []; pair_frames = []
            for _ in range(REFINEMENTS):
                midpoint = .5*(endpoint_noise[0]+endpoint_noise[1])
                frames, features = _render_projected(env, encoder, pca, snapshot, chunk,
                                                      [midpoint], device, batch_size)
                midpoint_visual = features[0]; pair_frames.append(frames[0])
                projected = full_project(midpoint_visual[None])[0]
                side = int(np.argmin(np.linalg.norm(centroids-projected[None], axis=1)))
                endpoint_noise[side] = midpoint; endpoint_visual[side] = midpoint_visual
                sides.append(side)
                ee = early_project(np.stack([value[:2] for value in endpoint_visual]))
                ff = full_project(np.stack(endpoint_visual))
                eg.append(float(np.linalg.norm(ee[0]-ee[1])))
                fg.append(float(np.linalg.norm(ff[0]-ff[1])))
            early_gaps.append(eg); full_gaps.append(fg); midpoint_frames.append(pair_frames)
            details.append({"probe_indices": [int(left), int(right)], "midpoint_sides": sides})
        boundary = boundary_refinement_alarm(early_gaps, full_gaps)
        endpoints = np.stack([_reconstruct(snippets, pair, detail["midpoint_sides"])
                              for pair, detail in zip(pairs, details)])
        dense_frames, dense_features = [], []
        for endpoint_pair in endpoints:
            frames, features = _render_projected(env, encoder, pca, snapshot, chunk,
                                                  endpoint_pair, device, batch_size, dense=True)
            dense_frames.append(frames); dense_features.append(features)
        dense_features = np.stack(dense_features)
        consequence = whole_trajectory_consequence_alarm(
            dense_features, endpoints[:, 0]-endpoints[:, 1],
            boundary.early_delta_bic, boundary.full_delta_bic)
        out.update({"boundary_refined_alarm": bool(boundary.alarm),
                    "boundary_evidence": result_dict(boundary), "pair_details": details,
                    **consequence_result_dict(consequence)})
        payload.update({"midpoint_frames": np.asarray(midpoint_frames, np.uint8),
                        "dense_frames": np.stack(dense_frames),
                        "dense_features": dense_features, "endpoints": endpoints.astype(np.float32)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload["result_json"] = np.asarray(json.dumps(out, allow_nan=False))
    np.savez_compressed(destination, **payload)
    return out


def evaluate(archive, checkpoint, device, batch_size):
    baseline.verify_render()
    panel = json.loads(PANEL.read_text()); panel_cache = np.load(PANEL_CACHE, allow_pickle=False)
    physical = json.loads((PANEL.parent/"v0_stage_diagnostic.json").read_text())["rows"]
    model = load_world_model(checkpoint, device); encoder = model.encoder; rows = []
    with tempfile.TemporaryDirectory(prefix="panda_push_visual_pca_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        env = PandaBlockPush(xml, width=baseline.WIDTH, height=baseline.HEIGHT, render=True)
        try:
            for index, (row, physical_row) in enumerate(zip(panel["rows"], physical)):
                rows.append(evaluate_state(index, row, physical_row, panel_cache, env, encoder,
                                           device, batch_size))
                print(f"PCA evaluate {index+1}/{len(panel['rows'])} state={index}", flush=True)
        finally:
            env.close()
    value = {"protocol": PROTOCOL, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "summary": baseline.summarise(rows, physical), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(json.dumps(value["summary"], indent=2)); return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-archive", default=str(ROOT/"vendor/panda_express_sim.tar"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args.sim_archive, args.checkpoint, args.device, args.batch_size)


if __name__ == "__main__": main()
