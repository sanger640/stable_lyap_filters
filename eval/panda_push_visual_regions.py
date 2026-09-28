"""Background-suppressed and block-region visual diagnostics on frozen Panda push DEV.

Two predeclared readouts reuse the exact rendered initial futures and frozen Regime Monitor v0:

``motion_pca45``
    Weight each DINO patch token by that patch's mean absolute RGB change from the state's common
    pre-action image, then spatially pool to 4x4 and fit per-state unlabeled PCA45.  This removes
    unchanged background but retains every moving object, including the robot.

``block_pca45``
    Weight DINO tokens by simulator segmentation occupancy of the pushed block, spatially pool to
    4x4 and fit the same PCA45.  Segmentation is grading/diagnostic information and this arm is an
    oracle representation upper bound, not a deployable universal monitor.

Each arm selects, bisects and densely replays its own candidate boundaries.  No outcome label,
physical-v0 decision, reference population or fitted alarm threshold enters either readout.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

# Keep MuJoCo rendering, PCA/BLAS and DINO from multiplying thread pools.  The previous eight-way
# EGL attempt oversubscribed the desktop host; this experiment intentionally favors responsiveness.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
import numpy as np
import torch
from torch.nn import functional as F

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(ROOT/"eval"))

from action_branch_monitor import (action_branch_partition, boundary_refinement_alarm,
                                   nearest_cross_branch_pairs, result_dict,
                                   smooth_vs_branch_alarm)
from consequence_monitor import (result_dict as consequence_result_dict,
                                 whole_trajectory_consequence_alarm)
from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model, preprocess_frames
from jenga_short_held_tails import extract_sim
from panda_push_dev_panel import (CACHE as PANEL_CACHE, HORIZON, HOLD, PAIRS, PANEL,
                                  RECORD_HOLDS, REFINEMENTS, _projector, _reconstruct)
import panda_push_visual_pca as pca_baseline
import panda_push_visual_v0 as visual_baseline
from systems.panda_block_push import PandaBlockPush, write_push_xml

OUT = ROOT/"results/panda_block_push/visual_regions"
INPUT_CACHE = OUT/"input_cache"
PCA_DIMS = pca_baseline.PCA_DIMS
MODES = ("motion_pca45", "block_pca45")


def protocol(mode):
    common = {
        "id": f"panda-upright-push-rendered-v0-{mode}-dev",
        "status": "DEV-only representation diagnostic; not a holdout claim",
        "source_panel_sha256": visual_baseline.PROTOCOL["source_panel_sha256"],
        "encoder": visual_baseline.ENCODER,
        "spatial_pool": "weighted native DINO patch tokens pooled to 4x4",
        "adapter": "per-state PCA45 fitted to 64 unlabeled probe trajectories",
        "monitor": "frozen Regime Monitor v0 unchanged",
        "refinement": "representation-selected 3 pairs, 5 exact bisections, dense H8+hold30 endpoints",
        "labels_or_reference_states_in_adapter": False,
        "task_labels_in_monitor": False,
        "physical_channel_contract_caveat": "PCA coordinates are exchangeable, not semantic pose/velocity channel families",
        "resource_policy": "one simulator worker; BLAS capped at two and Torch at eight CPU threads",
    }
    if mode == "motion_pca45":
        common.update({
            "readout": "each patch token weighted by mean absolute RGB change from common pre-action frame",
            "object_identity_or_segmentation": False,
            "role": "deployable label-free background-suppression diagnostic",
        })
    elif mode == "block_pca45":
        common.update({
            "readout": "each patch token weighted by pushed-block occupancy from MuJoCo segmentation",
            "object_identity_or_segmentation": True,
            "role": "oracle object-centric representation upper bound; not deployable",
        })
    else:
        raise ValueError(mode)
    return common


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Segmenter:
    def __init__(self, env):
        self.env = env
        self.renderer = env.mj.Renderer(env.model, height=visual_baseline.HEIGHT,
                                       width=visual_baseline.WIDTH)
        self.renderer.enable_segmentation_rendering()

    def mask(self):
        self.renderer.update_scene(self.env.data, camera="cam_fixed")
        segmentation = self.renderer.render()
        return ((segmentation[..., 0] == self.env.block_geom_id)
                & (segmentation[..., 1] == int(self.env.mj.mjtObj.mjOBJ_GEOM)))

    def close(self):
        self.renderer.close()


def _execute(env, snapshot, chunk, noise, segmenter=None, dense=False):
    env.restore(snapshot)
    actions = np.asarray(chunk, float).copy(); actions[:, :3] -= np.asarray(noise)[:, :3]
    frames, masks = [], []
    wanted = {HORIZON + held - 1 for held in RECORD_HOLDS}; step = 0
    for action in actions:
        env.step(action)
        if dense or step in wanted:
            frames.append(env.render())
            if segmenter is not None: masks.append(segmenter.mask())
        step += 1
    for _ in range(HOLD):
        env.step(actions[-1])
        if dense or step in wanted:
            frames.append(env.render())
            if segmenter is not None: masks.append(segmenter.mask())
        step += 1
    expected = HORIZON + HOLD if dense else len(RECORD_HOLDS)
    if len(frames) != expected:
        raise RuntimeError("incorrect visual trajectory length")
    return np.asarray(frames, np.uint8), (None if segmenter is None
                                         else np.asarray(masks, bool)), env.outcome()


def _prepare_input_state(job):
    xml, index, snapshot, chunk, snippets, expected_topples, destination = job
    if Path(destination).exists(): return index, "cached"
    env = PandaBlockPush(xml, width=visual_baseline.WIDTH,
                         height=visual_baseline.HEIGHT, render=True)
    segmenter = Segmenter(env)
    try:
        env.restore(snapshot); start = env.render(); masks = []; failures = []
        for noise in snippets:
            _, mask, outcome = _execute(env, snapshot, chunk, noise,
                                        segmenter=segmenter, dense=False)
            masks.append(mask); failures.append(outcome["failure"])
        if int(np.sum(failures)) != int(expected_topples):
            raise RuntimeError(f"state {index}: segmented replay changed physical grade")
        np.savez_compressed(destination, start=start, block_masks=np.stack(masks))
        return index, "rendered"
    finally:
        segmenter.close(); env.close()


def prepare_inputs(xml, panel, panel_cache, workers):
    """Cache one common start image and exact block masks for all initial rendered futures."""
    INPUT_CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(xml, index, visual_baseline._restore_snapshot(panel_cache, index),
             panel_cache["chunks"][index], panel_cache["snippets"], row["topple_count"],
             INPUT_CACHE/f"state_{index:03d}.npz")
            for index, row in enumerate(panel["rows"])]
    if workers == 1:
        for done, job in enumerate(jobs, 1):
            index, status = _prepare_input_state(job)
            print(f"region input {done}/{len(jobs)} state={index} {status}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_prepare_input_state, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                index, status = future.result()
                print(f"region input {done}/{len(jobs)} state={index} {status}", flush=True)


def weighted_dino_features(encoder, frames, mode, device, batch_size,
                           start=None, block_masks=None):
    frames = np.asarray(frames); leading = frames.shape[:-3]
    flat = frames.reshape(-1, *frames.shape[-3:]); features = []
    if mode == "motion_pca45":
        if start is None: raise ValueError("motion readout requires the common start frame")
        reference = np.asarray(start)
        if reference.shape != frames.shape[-3:]: raise ValueError("start frame shape mismatch")
        flat_masks = None
    elif mode == "block_pca45":
        flat_masks = np.asarray(block_masks).reshape(-1, *frames.shape[-3:-1])
        if len(flat_masks) != len(flat): raise ValueError("block masks and frames do not align")
        reference = None
    else:
        raise ValueError(mode)
    for first in range(0, len(flat), batch_size):
        last = first + batch_size
        with torch.inference_mode():
            image = preprocess_frames(flat[first:last, None], device)[:, 0]
            tokens = encoder.forward(image).float()
            side = int(round(np.sqrt(tokens.shape[1])))
            if side*side != tokens.shape[1] or tokens.shape[2] != 384:
                raise RuntimeError(f"unexpected DINO token shape {tuple(tokens.shape)}")
            if mode == "motion_pca45":
                rgb = torch.as_tensor(flat[first:last], dtype=torch.float32, device=device)
                ref = torch.as_tensor(reference, dtype=torch.float32, device=device)[None]
                weights = torch.mean(torch.abs(rgb-ref), dim=-1, keepdim=False)/255.
            else:
                weights = torch.as_tensor(flat_masks[first:last], dtype=torch.float32, device=device)
            weights = F.adaptive_avg_pool2d(weights[:, None], (side, side)).flatten(1)
            weighted = tokens * weights[..., None]
            grid = weighted.reshape(len(tokens), side, side, 384).permute(0, 3, 1, 2)
            features.append(F.adaptive_avg_pool2d(
                grid, (visual_baseline.POOL_GRID, visual_baseline.POOL_GRID)).flatten(1).cpu().numpy())
    return np.concatenate(features).reshape(*leading, -1)


def _render_projected(env, segmenter, encoder, pca, snapshot, chunk, noises, mode,
                      start, device, batch_size, dense=False):
    frame_rows, mask_rows = [], []
    for noise in noises:
        frames, masks, _ = _execute(env, snapshot, chunk, noise,
                                    segmenter=segmenter if mode == "block_pca45" else None,
                                    dense=dense)
        frame_rows.append(frames)
        if masks is not None: mask_rows.append(masks)
    frames = np.stack(frame_rows); masks = np.stack(mask_rows) if mask_rows else None
    raw = weighted_dino_features(encoder, frames, mode, device, batch_size,
                                 start=start, block_masks=masks)
    projected = pca.transform(raw.reshape(-1, raw.shape[-1])).reshape(*raw.shape[:-1], PCA_DIMS)
    return frames, masks, projected.astype(np.float32)


def evaluate_state(index, row, physical, panel_cache, env, segmenter, encoder, mode,
                   device, batch_size):
    cache_dir = OUT/mode/"evaluated_states"; destination = cache_dir/f"state_{index:03d}.npz"
    if destination.exists():
        return json.loads(str(np.load(destination, allow_pickle=False)["result_json"]))
    rendered = np.load(visual_baseline.RENDER_DIR/f"state_{index:03d}.npz", allow_pickle=False)
    region_input = np.load(INPUT_CACHE/f"state_{index:03d}.npz", allow_pickle=False)
    start = region_input["start"]
    raw = weighted_dino_features(encoder, rendered["frames"], mode, device, batch_size,
                                 start=start, block_masks=region_input["block_masks"])
    pca, visual = pca_baseline.fit_pca(raw)
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
                _, _, features = _render_projected(env, segmenter, encoder, pca, snapshot, chunk,
                                                    [midpoint], mode, start, device, batch_size)
                midpoint_visual = features[0]
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
            _, _, features = _render_projected(env, segmenter, encoder, pca, snapshot, chunk,
                                                endpoint_pair, mode, start, device, batch_size,
                                                dense=True)
            dense_features.append(features)
        dense_features = np.stack(dense_features)
        consequence = whole_trajectory_consequence_alarm(
            dense_features, endpoints[:, 0]-endpoints[:, 1],
            boundary.early_delta_bic, boundary.full_delta_bic)
        out.update({"boundary_refined_alarm": bool(boundary.alarm),
                    "boundary_evidence": result_dict(boundary), "pair_details": details,
                    **consequence_result_dict(consequence)})
        payload.update({"dense_features": dense_features, "endpoints": endpoints.astype(np.float32)})
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload["result_json"] = np.asarray(json.dumps(out, allow_nan=False))
    np.savez_compressed(destination, **payload)
    return out


def evaluate_mode(mode, panel, physical, panel_cache, env, segmenter, encoder, device, batch_size):
    rows = []
    for index, (row, physical_row) in enumerate(zip(panel["rows"], physical)):
        rows.append(evaluate_state(index, row, physical_row, panel_cache, env, segmenter,
                                   encoder, mode, device, batch_size))
        print(f"{mode} {index+1}/{len(panel['rows'])} state={index}", flush=True)
    value = {"protocol": protocol(mode),
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "summary": visual_baseline.summarise(rows, physical), "rows": rows}
    output = OUT/mode/f"{mode}_diagnostic.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(json.dumps({mode: value["summary"]}, indent=2)); return value


def run(archive, checkpoint, device, batch_size, modes, workers):
    visual_baseline.verify_render()
    panel = json.loads(PANEL.read_text()); panel_cache = np.load(PANEL_CACHE, allow_pickle=False)
    physical = json.loads((PANEL.parent/"v0_stage_diagnostic.json").read_text())["rows"]
    with tempfile.TemporaryDirectory(prefix="panda_push_visual_regions_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        # Complete input rendering before constructing Torch or a parent-process EGL context.
        # Forking either one caused severe contention in the discarded parallel attempt.
        prepare_inputs(xml, panel, panel_cache, workers)
        model = load_world_model(checkpoint, device); encoder = model.encoder
        env = PandaBlockPush(xml, width=visual_baseline.WIDTH,
                             height=visual_baseline.HEIGHT, render=True)
        segmenter = Segmenter(env)
        try:
            return {mode: evaluate_mode(mode, panel, physical, panel_cache, env, segmenter,
                                        encoder, device, batch_size) for mode in modes}
        finally:
            segmenter.close(); env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-archive", default=str(ROOT/"vendor/panda_express_sim.tar"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--torch-threads", type=int, choices=(1, 2, 3, 4, 6, 8), default=4)
    parser.add_argument("--mode", choices=("all",)+MODES, default="all")
    args = parser.parse_args(); torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    modes = MODES if args.mode == "all" else (args.mode,)
    run(args.sim_archive, args.checkpoint, args.device, args.batch_size, modes, args.workers)


if __name__ == "__main__": main()
