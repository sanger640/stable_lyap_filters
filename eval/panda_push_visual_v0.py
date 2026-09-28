"""Rendered-future transfer diagnostic for the frozen Panda upright-push DEV panel.

This file is deliberately additive: it verifies, but never edits, the frozen physical panel and
Regime Monitor v0.  ``render`` materialises the exact 64 H8+hold30 counterfactuals at the five
observation times used by the physical diagnostic.  ``evaluate`` applies the same v0 stages to a
fixed, label-free visual representation and renders refinement/dense endpoint futures on demand.

The visual adapter is frozen before scoring: DINOv2-S/14 patch tokens are spatially pooled from
their native square grid to 4x4 and mapped to 45 exchangeable coordinates by a seeded Gaussian projection.  This avoids
fitting a representation on outcomes or reference/calibration states.  The 45-D adapter is needed
because v0's unchanged consequence stage has a 45-field input contract; unlike the physical
input, its four channel slices do not have semantic pose/velocity meanings, so this is a transfer
diagnostic rather than a final visual-monitor claim.
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

import numpy as np
import torch
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
from panda_push_dev_panel import (CACHE as PANEL_CACHE, HORIZON, HOLD, PAIRS, PANEL,
                                  RECORD_HOLDS, REFINEMENTS, _projector, _reconstruct,
                                  _snapshot, verify as verify_panel)
from systems.panda_block_push import PandaBlockPush, write_push_xml

OUT = ROOT / "results/panda_block_push/visual_v0"
RENDER_DIR = OUT / "rendered_initial"
EVAL_CACHE_DIR = OUT / "evaluated_states"
RENDER_MANIFEST = OUT / "render_manifest.json"
RESULT = OUT / "visual_v0_diagnostic.json"
HEIGHT = WIDTH = 224
POOL_GRID = 4
VISUAL_DIMS = 45
PROJECTION_SEED = 640
ENCODER = "dinov2_vits14/x_norm_patchtokens"

PROTOCOL = {
    "id": "panda-upright-push-rendered-v0-transfer-dev",
    "status": "DEV-only representation-transfer diagnostic; not a holdout claim",
    "source_panel_sha256": "20dc10ea8eb455fceb958e0c409f94fdeafacddfb577ef479db6b443742d1aed",
    "initial_futures": "all 56 states x 64 exact frozen H8+hold30 probes",
    "observation_holds": list(RECORD_HOLDS),
    "refinement": "visual-selected 3 nearest cross-branch pairs, 5 exact bisections",
    "dense_final": "both refined endpoints at every H8+hold30 control step",
    "encoder": ENCODER,
    "visual_adapter": {
        "spatial_pool": "adaptive average pooling of the full native patch-token grid to 4x4",
        "projection": "fixed seeded Gaussian random projection; no fit or calibration data",
        "dimensions": VISUAL_DIMS,
        "seed": PROJECTION_SEED,
    },
    "monitor": "frozen Regime Monitor v0 functions unchanged",
    "adapter_limitation": "v0 consequence channel slices are exchangeable projected visual coordinates, not semantic physical families",
    "task_labels_in_monitor": False,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_indices():
    return tuple(HORIZON + held - 1 for held in RECORD_HOLDS)


def _restore_snapshot(cache, index):
    return (cache["snapshots"][index], cache["snapshot_ctrl"][index],
            cache["snapshot_target"][index], float(cache["snapshot_gripper"][index]),
            float(cache["snapshot_peak_tilt"][index]))


def _execute(env, snapshot, chunk, noise, dense=False):
    env.restore(snapshot)
    actions = np.asarray(chunk, float).copy(); actions[:, :3] -= np.asarray(noise)[:, :3]
    frames = []
    wanted = set(_sample_indices())
    step = 0
    for action in actions:
        env.step(action)
        if dense or step in wanted: frames.append(env.render())
        step += 1
    for _ in range(HOLD):
        env.step(actions[-1])
        if dense or step in wanted: frames.append(env.render())
        step += 1
    expected = HORIZON + HOLD if dense else len(RECORD_HOLDS)
    if len(frames) != expected:
        raise RuntimeError(f"recorded {len(frames)} frames, expected {expected}")
    return np.asarray(frames, np.uint8), env.outcome()


def _render_initial_state(job):
    xml, index, snapshot, chunk, snippets, expected_topples, destination = job
    if Path(destination).exists():
        return index, "cached"
    env = PandaBlockPush(xml, width=WIDTH, height=HEIGHT, render=True)
    try:
        frames, failures = [], []
        for noise in snippets:
            rendered, outcome = _execute(env, snapshot, chunk, noise, dense=False)
            frames.append(rendered); failures.append(outcome["failure"])
        if int(np.sum(failures)) != int(expected_topples):
            raise RuntimeError(f"state {index}: rendered replay changed physical grade")
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, frames=np.stack(frames), failures=failures)
        return index, "rendered"
    finally:
        env.close()


def render_initial(archive, workers):
    manifest = verify_panel()
    panel = json.loads(PANEL.read_text()); cache = np.load(PANEL_CACHE, allow_pickle=False)
    OUT.mkdir(parents=True, exist_ok=True); RENDER_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="panda_push_visual_render_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        jobs = [(xml, i, _restore_snapshot(cache, i), cache["chunks"][i], cache["snippets"],
                 row["topple_count"], RENDER_DIR/f"state_{i:03d}.npz")
                for i, row in enumerate(panel["rows"])]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_render_initial_state, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                index, status = future.result()
                print(f"render {done}/{len(jobs)} state={index} {status}", flush=True)
    files = {path.name: _sha256(path) for path in sorted(RENDER_DIR.glob("state_*.npz"))}
    if len(files) != len(panel["rows"]):
        raise RuntimeError("render cache is incomplete")
    value = {"protocol": PROTOCOL, "source_panel_sha256": manifest["protocol_sha256"],
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "files_sha256": files}
    RENDER_MANIFEST.write_text(json.dumps(value, indent=2)+"\n")
    return value


def verify_render():
    verify_panel(); value = json.loads(RENDER_MANIFEST.read_text())
    if value.get("protocol") != PROTOCOL:
        raise SystemExit("render protocol mismatch")
    for name, expected in value.get("files_sha256", {}).items():
        if _sha256(RENDER_DIR/name) != expected:
            raise SystemExit(f"render cache changed: {name}")
    if len(value.get("files_sha256", {})) != 56:
        raise SystemExit("render manifest is incomplete")
    return value


def fixed_projection(device):
    generator = torch.Generator(device="cpu").manual_seed(PROJECTION_SEED)
    matrix = torch.randn(POOL_GRID * POOL_GRID * 384, VISUAL_DIMS,
                         generator=generator, dtype=torch.float32)
    matrix /= np.sqrt(VISUAL_DIMS)
    return matrix.to(device)


def visual_features(encoder, projection, frames, device, batch_size):
    """Encode arbitrary leading frame dimensions to the fixed 45-D visual representation."""
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
            pooled = F.adaptive_avg_pool2d(grid, (POOL_GRID, POOL_GRID))
            out.append((pooled.flatten(1) @ projection).cpu().numpy())
    return np.concatenate(out).reshape(*leading, VISUAL_DIMS)


def _render_features(env, encoder, projection, snapshot, chunk, noises, device,
                     batch_size, dense=False):
    rendered = [_execute(env, snapshot, chunk, noise, dense=dense)[0] for noise in noises]
    frames = np.stack(rendered)
    return frames, visual_features(encoder, projection, frames, device, batch_size)


def _evaluate_state(index, row, cache, physical, env, encoder, projection, device, batch_size):
    state_cache = EVAL_CACHE_DIR/f"state_{index:03d}.npz"
    if state_cache.exists():
        saved = np.load(state_cache, allow_pickle=False)
        return json.loads(str(saved["result_json"]))
    initial = np.load(RENDER_DIR/f"state_{index:03d}.npz", allow_pickle=False)
    visual = visual_features(encoder, projection, initial["frames"], device, batch_size)
    snippets = np.asarray(cache["snippets"], float)
    errors = -snippets - (-snippets).mean(0, keepdims=True)
    candidate = smooth_vs_branch_alarm(errors, visual[:, :2], visual)
    out = {**row, "initial_alarm": bool(candidate.alarm), "boundary_refined_alarm": False,
           "alarm": False, "initial_evidence": result_dict(candidate),
           "physical_initial_alarm": bool(physical["initial_alarm"]),
           "physical_alarm": bool(physical["alarm"])}
    cache_payload = {"initial_features": visual.astype(np.float32)}
    if candidate.alarm:
        x, labels = action_branch_partition(errors, visual[:, :2], visual)
        pairs = nearest_cross_branch_pairs(x, labels, PAIRS)
        early_project, full_project = _projector(visual[:, :2]), _projector(visual)
        original_early, original_full = early_project(visual[:, :2]), full_project(visual)
        centroids = np.stack([original_full[labels == group].mean(0) for group in (0, 1)])
        snapshot = _restore_snapshot(cache, index); chunk = cache["chunks"][index]
        early_gaps, full_gaps, details, midpoint_frames = [], [], [], []
        for left, right in pairs:
            endpoint_noise = [snippets[left].copy(), snippets[right].copy()]
            endpoint_visual = [visual[left].copy(), visual[right].copy()]
            eg = [float(np.linalg.norm(original_early[left]-original_early[right]))]
            fg = [float(np.linalg.norm(original_full[left]-original_full[right]))]
            sides = []; pair_frames = []
            for _ in range(REFINEMENTS):
                midpoint = .5*(endpoint_noise[0]+endpoint_noise[1])
                frames, features = _render_features(env, encoder, projection, snapshot, chunk,
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
            frames, features = _render_features(env, encoder, projection, snapshot, chunk,
                                                endpoint_pair, device, batch_size, dense=True)
            dense_frames.append(frames); dense_features.append(features)
        dense_features = np.stack(dense_features)
        consequence = whole_trajectory_consequence_alarm(
            dense_features, endpoints[:, 0]-endpoints[:, 1],
            boundary.early_delta_bic, boundary.full_delta_bic)
        out.update({"boundary_refined_alarm": bool(boundary.alarm),
                    "boundary_evidence": result_dict(boundary), "pair_details": details,
                    **consequence_result_dict(consequence)})
        cache_payload.update({"midpoint_frames": np.asarray(midpoint_frames, np.uint8),
                              "dense_frames": np.stack(dense_frames),
                              "dense_features": dense_features.astype(np.float32),
                              "endpoints": endpoints.astype(np.float32)})
    state_cache.parent.mkdir(parents=True, exist_ok=True)
    cache_payload["result_json"] = np.asarray(json.dumps(out, allow_nan=False))
    np.savez_compressed(state_cache, **cache_payload)
    return out


def summarise(rows, physical_rows):
    cohorts = tuple(dict.fromkeys(row["cohort"] for row in rows))
    by_cohort = {}
    for cohort in cohorts:
        selected = [row for row in rows if row["cohort"] == cohort]
        by_cohort[cohort] = {
            "states": len(selected),
            "initial_candidates": sum(row["initial_alarm"] for row in selected),
            "refined_boundaries": sum(row["boundary_refined_alarm"] for row in selected),
            "commitment_majority": sum(row.get("commitment_pairs", 0) >= 2 for row in selected),
            "persistence_majority": sum(row.get("persistence_pairs", 0) >= 2 for row in selected),
            "final_alarms": sum(row["alarm"] for row in selected),
            "physical_final_alarms": sum(row["physical_alarm"] for row in selected),
        }
    visual = np.asarray([row["alarm"] for row in rows], bool)
    physical = np.asarray([row["alarm"] for row in physical_rows], bool)
    topple_fork = np.asarray([row["cohort"] == "topple_fork" for row in rows], bool)
    return {"by_cohort": by_cohort,
            "agreement": {
                "states": len(rows), "exact_final_decision_agreement": int(np.sum(visual == physical)),
                "agreement_rate": float(np.mean(visual == physical)),
                "physical_alarm_count": int(np.sum(physical)),
                "visual_alarm_count": int(np.sum(visual)),
                "physical_alarm_recall": float(np.sum(visual & physical)/max(np.sum(physical), 1)),
                "visual_only_alarms": int(np.sum(visual & ~physical)),
                "missed_physical_alarms": int(np.sum(~visual & physical)),
                "topple_fork_physical_alarm_recall": float(
                    np.sum(visual & physical & topple_fork)/max(np.sum(physical & topple_fork), 1)),
            }}


def evaluate(archive, checkpoint, device, batch_size):
    render_manifest = verify_render()
    panel = json.loads(PANEL.read_text()); cache = np.load(PANEL_CACHE, allow_pickle=False)
    physical_result = json.loads((PANEL.parent/"v0_stage_diagnostic.json").read_text())
    physical_rows = physical_result["rows"]
    model = load_world_model(checkpoint, device); encoder = model.encoder
    projection = fixed_projection(device); rows = []
    with tempfile.TemporaryDirectory(prefix="panda_push_visual_eval_") as temp:
        xml = str(write_push_xml(extract_sim(archive, temp).parent))
        env = PandaBlockPush(xml, width=WIDTH, height=HEIGHT, render=True)
        try:
            for index, (row, physical) in enumerate(zip(panel["rows"], physical_rows)):
                rows.append(_evaluate_state(index, row, cache, physical, env, encoder,
                                            projection, device, batch_size))
                print(f"evaluate {index+1}/{len(panel['rows'])} state={index}", flush=True)
        finally:
            env.close()
    value = {"protocol": PROTOCOL,
             "render_manifest_sha256": _sha256(RENDER_MANIFEST),
             "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "summary": summarise(rows, physical_rows), "rows": rows}
    RESULT.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(json.dumps(value["summary"], indent=2)); return value


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("render", "run"):
        child = sub.add_parser(command); child.add_argument("--workers", type=int, default=8)
        child.add_argument("--sim-archive", default=str(ROOT/"vendor/panda_express_sim.tar"))
        if command == "run":
            child.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
            child.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            child.add_argument("--batch-size", type=int, default=16)
    sub.add_parser("verify-render")
    args = parser.parse_args()
    if args.command == "render": render_initial(args.sim_archive, args.workers)
    elif args.command == "verify-render": print(verify_render()["source_panel_sha256"])
    else:
        if not RENDER_MANIFEST.exists(): render_initial(args.sim_archive, args.workers)
        evaluate(args.sim_archive, args.checkpoint, args.device, args.batch_size)


if __name__ == "__main__": main()
