"""Matched real-frame trajectory expansion and oracle neighbor-mask nuisance audit."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay,
                           load_world_model, normalise_proprio,
                           preprocess_frames)  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_multipeak_oracle import original_physical_responses  # noqa: E402
from jenga_near_boundary_discovery import offset_probe_chunks  # noqa: E402
from jenga_ordered_holdout import binomial_interval  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim, projection_basis  # noqa: E402
from jenga_trajectory_oracle import score_rows, temporal_log_mean_evidence  # noqa: E402
from local_expansion import detect_local_expansion  # noqa: E402

TAIL = 5


def neighbor_patch_weights(sim):
    sim.segment_renderer.update_scene(sim.data, camera="cam_fixed")
    seg = sim.segment_renderer.render()
    geom = seg[..., 1] == int(sim.mj.mjtObj.mjOBJ_GEOM)
    neighbor_bodies = sim.tracked_block_ids[1:]
    neighbor_geoms = np.flatnonzero(np.isin(sim.model.geom_bodyid, neighbor_bodies))
    pixels = geom & np.isin(seg[..., 0], neighbor_geoms)
    weights = np.zeros((14, 14), np.float32)
    ys, xs = np.nonzero(pixels)
    np.add.at(weights, (np.minimum(ys * 14 // pixels.shape[0], 13),
                        np.minimum(xs * 14 // pixels.shape[1], 13)), 1)
    weights /= max(float(weights.sum()), 1.0)
    return weights.reshape(-1)


def select_rows(oracle_cache, control_count, boundary_count):
    data = dict(np.load(oracle_cache, allow_pickle=False))
    metadata = json.loads(str(data["metadata_json"]))
    by_kind = {}
    for row in metadata:
        by_kind.setdefault(row["stratum"], []).append(row)
    controls = by_kind.get("quiet_control", [])[:int(control_count)]
    boundaries = by_kind.get("recentered_neighbor_boundary", [])[:int(boundary_count)]
    if len(controls) < control_count or len(boundaries) < boundary_count:
        raise ValueError("requested more rows than in oracle cache")
    return controls + boundaries, data["scalars"]


def simulate_frames(sim, snapshot, probes):
    frames, props, masks, outcomes = [], [], [], []
    for probe in probes:
        sim.restore(snapshot); f, p, w = [], [], []
        trace = {"peak_tilt": sim.block_diagnostics()["tilt"].copy()}
        for action in list(probe) + [probe[-1]] * TAIL:
            sim.execute(action, trace)
            f.append(sim.render()); p.append(sim.proprio()); w.append(neighbor_patch_weights(sim))
        frames.append(f); props.append(p); masks.append(w)
        outcomes.append(bool(np.max(trace["peak_tilt"][1:]) >= TOPPLE_DEG))
    sim.restore(snapshot)
    return np.asarray(frames), np.asarray(props), np.asarray(masks), np.asarray(outcomes, bool)


def encode(model, device, frames, props, masks, projection, batch_size):
    n, t = frames.shape[:2]; flat_frames = frames.reshape(n * t, *frames.shape[2:])
    flat_props = props.reshape(n * t, props.shape[-1]); flat_masks = masks.reshape(n * t, 196)
    mean, components = projection
    mean_t = torch.as_tensor(mean, device=device); comp_t = torch.as_tensor(components[:4], device=device)
    full, pooled = [], []
    for first in range(0, len(flat_frames), batch_size):
        last = first + batch_size
        with torch.inference_mode():
            tokens = model.encode_obs({
                "visual": preprocess_frames(flat_frames[first:last, None], device),
                "proprio": normalise_proprio(flat_props[first:last, None], device),
            })["visual"][:, 0].float()
        full.append(((tokens.flatten(1) - mean_t) @ comp_t.T).cpu().numpy())
        weights = torch.as_tensor(flat_masks[first:last], device=device)
        pooled.append(torch.einsum("bp,bpd->bd", weights, tokens).cpu().numpy())
    return (np.concatenate(full).reshape(n, t, 4),
            np.concatenate(pooled).reshape(n, t, -1))


def collect(selected, replay, model, projection, xml, scalars, eps, device, batch_size, cache_dir):
    wanted = {}
    for row in selected:
        wanted.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    sim = DirectJengaSim(xml); records = []
    try:
        for number, episode_id in enumerate(sorted(wanted, key=int)):
            episode = replay.episode(episode_id); sim.reset(int(episode_id))
            for start, action in enumerate(episode.actions):
                if start in wanted[episode_id]:
                    path = cache_dir / f"ep{episode_id}_chunk{start}.npz"
                    row = wanted[episode_id][start]
                    if not path.exists():
                        snapshot = sim.snapshot()
                        probes = offset_probe_chunks(
                            episode.actions[start:start + HORIZON], scalars, eps,
                            row.get("center_offset", 0.0))
                        frames, props, masks, outcomes = simulate_frames(sim, snapshot, probes)
                        full, pooled = encode(model, device, frames, props, masks,
                                              projection, batch_size)
                        np.savez_compressed(path, metadata_json=np.asarray(json.dumps(row)),
                                            full=full, neighbor_pool=pooled,
                                            mask_nonzero=(masks.sum(-1) > 0).astype(np.uint8),
                                            physical_outcomes=outcomes)
                    records.append(path)
                sim.execute(action)
            print(f"  visual {number + 1}/{len(wanted)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return records


def load_records(selected, cache_dir):
    records = []
    for row in selected:
        path = Path(cache_dir) / f"ep{row['episode_id']}_chunk{row['chunk_start']}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        records.append(dict(np.load(path, allow_pickle=False)))
    return records


def score_visual(selected, scalars, records, original_cache):
    full = np.stack([record["full"] for record in records])
    pool = np.stack([record["neighbor_pool"] for record in records])
    outcomes = np.stack([record["physical_outcomes"] for record in records])
    mask_coverage = np.stack([record["mask_nonzero"] for record in records]).mean(axis=(1, 2))
    full -= full[:, :1, :1]
    # A single fixed PCA basis for the oracle-mask readout, fitted without labels
    # to the selected start-state block tokens. Transductive upper bound only.
    start_pool = pool[:, :, 0].mean(axis=1)
    pca = PCA(n_components=4, svd_solver="full").fit(start_pool)
    pooled = np.einsum("nptd,dk->nptk", pool - pool[:, :1, :1], pca.components_.T)
    old = dict(np.load(original_cache, allow_pickle=False))
    full_scale = robust_component_scale(old["actual_delta"])[:4]
    # Do not estimate a response scale from evaluation outcomes. A common scalar
    # scale leaves BIC and zero-evidence alarms invariant; use unit scale here.
    reports = {}
    for name, values, scale in (("full_frame_pca4", full, full_scale),
                                ("oracle_neighbor_patch_pool_pca4", pooled, np.ones(4))):
        rows = score_rows(scalars, values, outcomes, selected, scale)
        for row, coverage in zip(rows, mask_coverage):
            row["visible_neighbor_mask_fraction"] = float(coverage)
        truth = np.asarray([row["physical_boundary"] for row in rows], bool)
        alarms = np.asarray([row["alarm"] for row in rows], bool)
        controls = np.asarray([row["stratum"] == "quiet_control" for row in rows], bool)
        boundary = truth
        fp = int((alarms & controls).sum()); tp = int((alarms & boundary).sum())
        from jenga_ordered_change_points import confusion
        summary = {"controls": int(controls.sum()), "false_alarms": fp,
                   "control_fpr": fp / max(int(controls.sum()), 1),
                   "control_fpr_exact_95_percent": binomial_interval(fp, int(controls.sum())),
                   "boundaries": int(boundary.sum()), "detected_boundaries": tp,
                   "boundary_recall": tp / max(int(boundary.sum()), 1),
                   "boundary_recall_exact_95_percent": binomial_interval(tp, int(boundary.sum())),
                   "confusion": confusion(truth, alarms),
                   "score_roc_auc": float(roc_auc_score(truth, [r["trajectory_score"] for r in rows]))}
        reports[name] = {"summary": summary, "rows": rows}
    return reports


def temporal_audit(reports, physical_result):
    physical = json.loads(Path(physical_result).read_text())
    matched = {(row["episode_id"], row["chunk_start"]): row
               for row in physical["rows"]}
    audit = {}
    for name, report in reports.items():
        rows = report["rows"]
        controls = [row for row in rows if row["stratum"] == "quiet_control"]
        boundaries = [row for row in rows if row["physical_boundary"]]
        physical_controls = [matched[(row["episode_id"], row["chunk_start"])]
                             for row in controls]
        physical_boundaries = [matched[(row["episode_id"], row["chunk_start"])]
                               for row in boundaries]
        audit[name] = {
            "median_control_positive_times_of_13": float(np.median(
                [len(row["positive_times"]) for row in controls])),
            "controls_first_positive_by_time": {
                str(time): int(sum(row["positive_times"] and row["positive_times"][0] == time
                                   for row in controls)) for time in range(1, 14)},
            "matched_physical_control_alarms": int(sum(row["alarm"] for row in physical_controls)),
            "median_visual_control_score": float(np.median(
                [row["trajectory_score"] for row in controls])),
            "median_physical_control_score": float(np.median(
                [row["trajectory_score"] for row in physical_controls])),
            "boundary_visual_physical_score_correlation": float(np.corrcoef(
                [row["trajectory_score"] for row in boundaries],
                [row["trajectory_score"] for row in physical_boundaries])[0, 1]),
            "minimum_neighbor_mask_visible_fraction": float(min(
                row["visible_neighbor_mask_fraction"] for row in rows)),
        }
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--projection-cache", default=str(ROOT / "results/jenga/j4_truth_and_short_predictions.npz"))
    ap.add_argument("--oracle-cache", default=str(ROOT / "results/jenga/trajectory_oracle_cache.npz"))
    ap.add_argument("--original-cache", default=str(ROOT / "results/jenga/local_delta_probe_arrays.npz"))
    ap.add_argument("--physical-result", default=str(ROOT / "results/jenga/trajectory_oracle.json"))
    ap.add_argument("--cache-dir", default=str(ROOT / "results/jenga/visual_trajectories_cache"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/visual_trajectories.json"))
    ap.add_argument("--controls", type=int, default=100)
    ap.add_argument("--boundaries", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    selected, scalars = select_rows(args.oracle_cache, args.controls, args.boundaries)
    if not args.reuse_cache:
        replay = JengaReplay(args.lmdb); model = load_world_model(args.checkpoint, args.device)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_visual_trajectory_") as temp:
                collect(selected, replay, model, projection_basis(args.projection_cache),
                        extract_sim(args.sim_archive, temp), scalars, .10,
                        args.device, args.batch_size, args.cache_dir)
        finally:
            replay.close()
    reports = score_visual(selected, scalars, load_records(selected, args.cache_dir),
                           args.original_cache)
    result = {"protocol": {"states": len(selected), "probes": len(scalars),
                           "times": HORIZON + TAIL, "eps": .10,
                           "full_frame_projection": "frozen terminal PCA4",
                           "neighbor_pool": "simulator segmentation weighted DINO patch pool; transductive unlabeled PCA4; diagnostic upper bound"},
              "representations": reports,
              "temporal_audit": temporal_audit(reports, args.physical_result)}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value["summary"] for key, value in reports.items()}, indent=2))
    print(json.dumps(result["temporal_audit"], indent=2))


if __name__ == "__main__":
    main()
