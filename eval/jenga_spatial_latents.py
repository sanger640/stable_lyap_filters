"""Balanced all-block probe experiment comparing full-frame and block-ROI latents."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_failures import boundary_mask, failure_masks  # noqa: E402
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay, load_world_model,
                           normalise_proprio, preprocess_frames)  # noqa: E402
from ordered_change_point import robust_component_scale  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from jenga_local_delta_probes import coherent_probe_chunks, probe_scalars  # noqa: E402
from jenga_ordered_change_points import confusion, score_responses, summarise  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402

TAIL = 10
ROI_ROWS = range(6, 10)
ROI_COLS = range(5, 10)
ROI_INDICES = np.asarray([row * 14 + col for row in ROI_ROWS for col in ROI_COLS])
MODES = ("full", "block_roi_flat", "block_roi_pool", "oracle_block_pool")


def representation(tokens, mode):
    x = np.asarray(tokens)
    if mode == "full":
        return x.reshape(len(x), -1)
    roi = x[:, ROI_INDICES]
    if mode == "block_roi_flat":
        return roi.reshape(len(x), -1)
    if mode == "block_roi_pool":
        return roi.mean(axis=1)
    raise ValueError(mode)


def projection_bases(path, dimension=16):
    data = dict(np.load(path, allow_pickle=False))
    terminal = np.asarray(data["tail_truth"][:, -1], np.float32)
    bases = {}
    for mode in MODES:
        if mode == "oracle_block_pool":
            continue
        features = representation(terminal, mode)
        mean = features.mean(0)
        components = np.linalg.svd(features - mean, full_matrices=False)[2][:dimension]
        bases[mode] = (mean.astype(np.float32), components.astype(np.float32))
    return bases


def project(tokens, bases):
    output = {}
    for mode, (mean, components) in bases.items():
        features = representation(tokens, mode)
        output[mode] = (features - mean) @ components.T
    return output


def spread_take(indices, count, episode_ids, score=None):
    indices = list(indices)
    if score is not None:
        indices.sort(key=lambda i: (float(score[i]), int(episode_ids[i])))
    else:
        indices.sort(key=lambda i: (int(episode_ids[i]), i))
    first, remainder, seen = [], [], set()
    for i in indices:
        if episode_ids[i] not in seen:
            first.append(i); seen.add(episode_ids[i])
        else:
            remainder.append(i)
    ordered = first + remainder
    if len(ordered) < count:
        raise ValueError(f"requested {count} rows from only {len(ordered)} candidates")
    if score is None and len(ordered) > count:
        positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
        return [ordered[i] for i in positions]
    return ordered[:count]


def choose_balanced(scan_cache, per_stratum=20):
    data = dict(np.load(scan_cache, allow_pickle=False))
    ids = data["episode_ids"].astype(str); starts = data["chunk_starts"].astype(int)
    start_safe = np.max(data["start_tilt"], axis=1) < 45
    modes = failure_masks(data["tail10_peak_tilt"], data["tail10_min_z"])
    middle_score = np.abs(data["tail10_peak_tilt"][:, 0] - 45)
    neighbor_score = np.abs(np.max(data["tail10_peak_tilt"][:, 1:], axis=1) - 45)
    lift = (data["tail10_end_position"][:, 0, 2]
            - data["start_position"][:, 0, 2] >= 0.02)
    quiet = (start_safe & (np.max(data["tail10_peak_tilt"], axis=1) < 5)
             & (np.max(np.linalg.norm(data["tail10_end_position"]
                                      - data["start_position"], axis=2), axis=1) < 0.002))
    candidates = {
        "middle_failure": (np.flatnonzero(start_safe & modes["middle_failure"]), middle_score),
        "neighbor_failure": (np.flatnonzero(start_safe & modes["neighbor_failure"]
                                              & ~modes["middle_failure"]), neighbor_score),
        "safe_extraction": (np.flatnonzero(lift & ~modes["any_failure"]), None),
        "quiet_control": (np.flatnonzero(quiet), None),
    }
    rows = []
    for stratum, (indices, score) in candidates.items():
        for i in spread_take(indices, per_stratum, ids, score):
            rows.append({"episode_id": ids[i], "chunk_start": int(starts[i]),
                         "stratum": stratum, "screen_index": int(i)})
    return rows


def simulate_probes(sim, snapshot, probes):
    sim.restore(snapshot); start_mask = sim.block_patch_mask()
    frames, props, masks, peak_tilt, min_z = [], [], [], [], []
    for probe in probes:
        sim.restore(snapshot); blocks = sim.block_diagnostics()
        trace = {"peak_tilt": blocks["tilt"].copy(),
                 "min_z": blocks["position"][:, 2].copy()}
        for action in probe:
            sim.execute(action, trace)
        for _ in range(TAIL):
            sim.execute(probe[-1], trace)
        frames.append(sim.render()); props.append(sim.proprio())
        masks.append(sim.block_patch_mask())
        peak_tilt.append(trace["peak_tilt"]); min_z.append(trace["min_z"])
    sim.restore(snapshot)
    return (np.stack(frames), np.stack(props), start_mask, np.stack(masks),
            np.stack(peak_tilt), np.stack(min_z))


def encode_deltas(model, device, start_frame, start_prop, endpoint_frames, endpoint_props,
                  start_mask, endpoint_masks, bases, batch_size):
    with torch.inference_mode():
        start_tokens = model.encode_obs({
            "visual": preprocess_frames(start_frame[None, None], device),
            "proprio": normalise_proprio(start_prop[None, None], device)})["visual"][:, 0]
    endpoints = []
    for first in range(0, len(endpoint_frames), batch_size):
        with torch.inference_mode():
            tokens = model.encode_obs({
                "visual": preprocess_frames(endpoint_frames[first:first + batch_size, None], device),
                "proprio": normalise_proprio(endpoint_props[first:first + batch_size, None], device)
            })["visual"][:, 0]
        endpoints.append(tokens.float().cpu().numpy())
    end_tokens = np.concatenate(endpoints)
    start_projection = project(start_tokens.float().cpu().numpy(), bases)
    end_projection = project(end_tokens, bases)
    output = {mode: end_projection[mode] - start_projection[mode][0] for mode in bases}
    union = start_mask | np.any(endpoint_masks, axis=0)
    indices = np.flatnonzero(union.reshape(-1))
    if not len(indices):
        raise ValueError("segmentation found no block patches")
    start_pool = start_tokens.float().cpu().numpy()[0, indices].mean(axis=0)
    output["oracle_block_pool"] = end_tokens[:, indices].mean(axis=1) - start_pool
    return output


def collect(args, selected, replay, model, bases, xml):
    by_episode = {}
    for row in selected:
        by_episode.setdefault(row["episode_id"], {})[row["chunk_start"]] = row
    scalars = probe_scalars(args.probes); metadata = []; deltas = {mode: [] for mode in MODES}
    peak_tilt, min_z = [], []
    sim = DirectJengaSim(xml)
    try:
        for number, episode_id in enumerate(sorted(by_episode, key=int)):
            episode = replay.episode(episode_id); wanted = by_episode[episode_id]
            sim.reset(int(episode_id))
            for start, action in enumerate(episode.actions):
                if start in wanted:
                    snapshot = sim.snapshot(); start_frame = sim.render(); start_prop = sim.proprio()
                    probes = coherent_probe_chunks(
                        episode.actions[start:start + HORIZON], scalars, args.eps)
                    frames, props, start_mask, endpoint_masks, peak, low = simulate_probes(
                        sim, snapshot, probes)
                    encoded = encode_deltas(model, args.device, start_frame, start_prop,
                                            frames, props, start_mask, endpoint_masks,
                                            bases, args.batch_size)
                    metadata.append(wanted[start]); peak_tilt.append(peak); min_z.append(low)
                    for mode in MODES:
                        deltas[mode].append(encoded[mode])
                sim.execute(action)
            print(f"  {number + 1}/{len(by_episode)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return (metadata, scalars, {mode: np.stack(value) for mode, value in deltas.items()},
            np.stack(peak_tilt), np.stack(min_z))


def score(metadata, scalars, deltas, peak_tilt, min_z):
    failures = failure_masks(peak_tilt, min_z)
    output = {}
    for mode, response in deltas.items():
        if response.shape[-1] > 16:
            flat = response.reshape(-1, response.shape[-1])
            components = np.linalg.svd(flat - flat.mean(0), full_matrices=False)[2][:16]
            response = ((flat - flat.mean(0)) @ components.T).reshape(
                *response.shape[:-1], 16)
        scale = robust_component_scale(response)
        rows = score_responses(scalars, response, failures["any_failure"], metadata,
                               dimension=4, degree=3, min_side=8,
                               min_jump_ratio=4.0, scale=scale)
        summary = summarise(rows)
        for failure_mode, outcomes in failures.items():
            truth = boundary_mask(outcomes)
            supported_truth = boundary_mask(outcomes, minimum_each=8)
            summary[f"{failure_mode}_confusion"] = confusion(
                truth, [row["alarm"] for row in rows])
            summary[f"{failure_mode}_boundaries"] = int(truth.sum())
            summary[f"{failure_mode}_supported_confusion"] = confusion(
                supported_truth, [row["alarm"] for row in rows])
            summary[f"{failure_mode}_supported_boundaries"] = int(supported_truth.sum())
        summary["alarms_by_stratum"] = {
            stratum: int(sum(row["alarm"] for row in rows if row["stratum"] == stratum))
            for stratum in sorted(set(row["stratum"] for row in rows))}
        output[mode] = {"summary": summary, "rows": rows}
    return output


def roi_overlay(path, frame, block_mask=None):
    image = Image.fromarray(frame); draw = ImageDraw.Draw(image)
    x0, x1 = min(ROI_COLS) * 320 / 14, (max(ROI_COLS) + 1) * 320 / 14
    y0, y1 = min(ROI_ROWS) * 240 / 14, (max(ROI_ROWS) + 1) * 240 / 14
    draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
    if block_mask is not None:
        for row, col in zip(*np.nonzero(block_mask)):
            draw.rectangle((col * 320 / 14, row * 240 / 14,
                            (col + 1) * 320 / 14, (row + 1) * 240 / 14),
                           outline="lime", width=2)
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True); image.save(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB)); ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--scan-cache", default=str(ROOT / "results/jenga/all_block_tails_cache.npz"))
    ap.add_argument("--projection-cache", default=str(ROOT / "results/jenga/j4_truth_and_short_predictions.npz"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/spatial_latents_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/spatial_latents.json"))
    ap.add_argument("--overlay", default=str(ROOT / "results/jenga/spatial_roi.png"))
    ap.add_argument("--per-stratum", type=int, default=20); ap.add_argument("--probes", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.10); ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args(); bases = projection_bases(args.projection_cache)
    if args.reuse_cache:
        data = dict(np.load(args.cache, allow_pickle=False)); metadata = json.loads(str(data["metadata_json"]))
        scalars, peak_tilt, min_z = data["scalars"], data["peak_tilt"], data["min_z"]
        deltas = {mode: data[f"delta_{mode}"] for mode in MODES
                  if f"delta_{mode}" in data}
    else:
        selected = choose_balanced(args.scan_cache, args.per_stratum)
        replay = JengaReplay(args.lmdb); model = load_world_model(args.checkpoint, args.device)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_spatial_") as temp:
                xml = extract_sim(args.sim_archive, temp)
                metadata, scalars, deltas, peak_tilt, min_z = collect(
                    args, selected, replay, model, bases, xml)
                first = selected[0]; episode = replay.episode(first["episode_id"])
                sim = DirectJengaSim(xml); sim.reset(int(first["episode_id"]))
                for action in episode.actions[:first["chunk_start"]]:
                    sim.execute(action)
                roi_overlay(args.overlay, sim.render(), sim.block_patch_mask()); sim.close()
        finally:
            replay.close()
        arrays = {"metadata_json": np.asarray(json.dumps(metadata)), "scalars": scalars,
                  "peak_tilt": peak_tilt, "min_z": min_z}
        arrays.update({f"delta_{mode}": value for mode, value in deltas.items()})
        output = Path(args.cache); output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
    scored = score(metadata, scalars, deltas, peak_tilt, min_z)
    result = {"protocol": {"chunks": len(metadata), "probes": len(scalars), "eps": args.eps,
                           "tail": TAIL,
                           "strata": {name: sum(row["stratum"] == name for row in metadata)
                                      for name in sorted(set(row["stratum"] for row in metadata))},
                           "roi_patch_rows": list(ROI_ROWS), "roi_patch_columns": list(ROI_COLS),
                           "detector": "frozen PCA4/cubic/min-side8/jump-ratio4"},
              "representations": scored}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for mode in scored:
        print(mode, scored[mode]["summary"])
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
