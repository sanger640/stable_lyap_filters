"""Stage 2: the universal version of the Stage 1 fork test, on image latents.

The same 64 noisy executions per panel state (Stage 0 snippets, all scales) are re-simulated.
The settled scene at hold steps 10 and 30 is rendered from the world model's fixed camera and
encoded with the world model's DINOv2 encoder (full frame, all 196 patch tokens). No object
identity, segmentation, geometry, contact, or millimetre scale enters the detector.

Rule, frozen before results (src/outcome_modes.two_mode_test with its floor disabled):
PC1 best split; BIC prefers two groups; Ashman D > 2; minority >= 2; the same split (<=1 probe)
at hold steps 10 and 30.

Grading classes use physics only after scoring:
  topple_fork  - Stage 0 mixed (both topple and no-topple >= 2 runs)
  nudge_fork   - unanimous no-topple, Stage 1 physical fork with floor (the 8 rendered at 1x)
  small_split  - unanimous no-topple, physical split only without the floor
  quiet        - unanimous no-topple, no physical split
  weak         - one dissenting topple run; reported, not graded
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import (DEFAULT_CHECKPOINT, DEFAULT_LMDB, JengaReplay,  # noqa: E402
                           load_world_model, normalise_proprio, preprocess_frames)
from outcome_modes import same_partition, two_mode_test  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import HOLD, SCALES, grade  # noqa: E402
from jenga_stage1_outcome_modes import auc  # noqa: E402

FRAME_AT = (10, 30)


def render_episode(job):
    episode_id, starts, lmdb, xml, snippets, own_hold, reset_base = job
    replay = JengaReplay(lmdb)
    episode = replay.episode(episode_id)
    replay.close()
    sim = DirectJengaSim(xml)
    out = {}
    try:
        sim.reset(int(episode_id) + reset_base if reset_base is not None
                  else int(episode_id))
        for index, action in enumerate(episode.actions):
            if index in starts:
                chunk = episode.actions[index:index + HORIZON]
                snapshot = sim.snapshot()
                frames = np.empty((len(SCALES), len(snippets), len(FRAME_AT), 240, 320, 3),
                                  np.uint8)
                proprio = np.empty((len(SCALES), len(snippets), len(FRAME_AT), 4), np.float32)
                topple = np.zeros((len(SCALES), len(snippets)), bool)
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                for s, scale in enumerate(SCALES):
                    for p, noise in enumerate(snippets):
                        sim.restore(snapshot)
                        actions = np.asarray(chunk, np.float32).copy()
                        actions[:, :3] -= noise * scale
                        for a in actions:
                            sim.execute(a)
                        # Arm check: hold each probe's own perturbed final target.
                        hold = actions[-1] if own_hold else chunk[-1]
                        for held in range(1, HOLD + 1):
                            sim.execute(hold)
                            if held in FRAME_AT:
                                t = FRAME_AT.index(held)
                                frames[s, p, t] = sim.render()
                                proprio[s, p, t] = sim.proprio()
                        tilt = sim.block_diagnostics()["tilt"][1:]
                        topple[s, p] = bool(np.any((tilt >= TOPPLE_DEG)
                                                   & (start_tilt < TOPPLE_DEG)))
                sim.restore(snapshot)
                out[index] = (frames, proprio, topple)
            sim.execute(action)
    finally:
        sim.close()
    return episode_id, out


def encode_coordinates(model, device, frames, proprio, batch_size=128):
    """(S,P,T,H,W,3) frames -> per (S,T) exact PCA coordinates (P,P) of flattened tokens."""
    s_n, p_n, t_n = frames.shape[:3]
    flat = frames.reshape(-1, *frames.shape[3:])
    flat_p = proprio.reshape(-1, proprio.shape[-1])
    tokens = []
    for first in range(0, len(flat), batch_size):
        with torch.inference_mode():
            enc = model.encode_obs({
                "visual": preprocess_frames(flat[first:first + batch_size, None], device),
                "proprio": normalise_proprio(flat_p[first:first + batch_size, None], device),
            })["visual"][:, 0]
        tokens.append(enc.flatten(1).float())
    tokens = torch.cat(tokens).reshape(s_n, p_n, t_n, -1)
    coords = np.empty((s_n, t_n, p_n, p_n), np.float32)
    for s in range(s_n):
        for t in range(t_n):
            x = tokens[s, :, t].double()
            x = x - x.mean(0)
            u, sv, _ = torch.linalg.svd(x, full_matrices=False)
            coords[s, t] = (u * sv).cpu().numpy()  # preserves all pairwise distances
    return coords


def remove_arm(coords, ee):
    """Residualize each time's latent coordinates on [1, end-effector xyz] (robot's own state)."""
    out = np.empty_like(coords)
    for t in range(coords.shape[0]):
        design = np.column_stack([np.ones(len(ee[t])), ee[t] - ee[t].mean(0)])
        beta = np.linalg.lstsq(design, coords[t], rcond=None)[0]
        out[t] = coords[t] - design @ beta
    return out


def fork_test(coords):
    """coords (T=2, P, P). Returns alarm plus diagnostics at the final time."""
    early = two_mode_test(coords[0], 0.0)
    late = two_mode_test(coords[1], 0.0)
    persistent = same_partition(early.labels, late.labels)
    ok = lambda r: r.bic_prefers_two and r.ashman_d > 2 and r.minority >= 2  # noqa: E731
    x = coords[1]
    separation = float(np.linalg.norm(x[late.labels == 0].mean(0) - x[late.labels == 1].mean(0)))
    spread = float(np.sqrt(np.mean(np.sum((x - x.mean(0)) ** 2, 1))))
    return {"alarm": bool(ok(early) and ok(late) and persistent),
            "final_alone": bool(ok(late)), "persistent_partition": bool(persistent),
            "ashman_d": None if not np.isfinite(late.ashman_d) else float(late.ashman_d),
            "bic_prefers_two": late.bic_prefers_two, "minority": late.minority,
            "latent_separation": separation, "latent_spread_rms": spread,
            "labels": late.labels.tolist()}


def physical_class(row1, scale):
    grade = row1["grade"][scale]
    if grade == "mixed":
        return "topple_fork"
    if grade != "unanimous_safe":
        return "weak"
    score = row1["scores"][scale]
    if score["full"]:
        return "nudge_fork"
    return "small_split" if score["no_floor"] else "quiet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--stage0-cache", default=str(ROOT / "results/jenga/stage0_noise_oracle_cache.npz"))
    ap.add_argument("--stage1", default=str(ROOT / "results/jenga/stage1_outcome_modes.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/stage2_visual_forks_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/stage2_visual_forks.json"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--reset-seed-base", type=int, default=None,
                    help="reset with seed BASE + episode_id (new configurations); default: episode_id")
    ap.add_argument("--own-hold", action="store_true",
                    help="arm check: each probe holds its own perturbed final target")
    args = ap.parse_args()
    stage1 = json.loads(Path(args.stage1).read_text())
    stage0 = np.load(args.stage0_cache, allow_pickle=False)
    keys = [(r["episode_id"], r["chunk_start"]) for r in stage1["rows"]]

    if args.reuse_cache:
        data = np.load(args.cache, allow_pickle=False)
        coords, arm_spread = data["coords"], data["arm_spread_mm"]
        ee, topples = data["ee"], data["topple"]
    else:
        device = "cuda"
        model = load_world_model(args.checkpoint, device)
        by_episode = {}
        for ep, start in keys:
            by_episode.setdefault(ep, set()).add(start)
        coords = np.empty((len(keys), len(SCALES), len(FRAME_AT), 64, 64), np.float32)
        arm_spread = np.empty((len(keys), len(SCALES)), np.float32)
        ee = np.empty((len(keys), len(SCALES), len(FRAME_AT), 64, 3), np.float32)
        topples = np.empty((len(keys), len(SCALES), 64), bool)
        index = {k: i for i, k in enumerate(keys)}
        with tempfile.TemporaryDirectory(prefix="jenga_stage2_") as temp:
            xml = str(extract_sim(args.sim_archive, temp))
            jobs = [(ep, starts, args.lmdb, xml, stage0["snippets"], args.own_hold, args.reset_seed_base)
                    for ep, starts in sorted(by_episode.items(), key=lambda x: int(x[0]))]
            with ProcessPoolExecutor(args.workers) as pool:
                for done, (ep, out) in enumerate(pool.map(render_episode, jobs), 1):
                    for start, (frames, proprio, topple) in out.items():
                        i = index[(ep, start)]
                        ee[i] = proprio[..., :3].transpose(0, 2, 1, 3)
                        topples[i] = topple
                        coords[i] = encode_coordinates(model, device, frames, proprio)
                        final_ee = proprio[:, :, -1, :3]
                        arm_spread[i] = 1000 * np.sqrt(np.mean(np.sum(
                            (final_ee - final_ee.mean(1, keepdims=True)) ** 2, -1), 1))
                    print(f"  stage2 {done}/{len(jobs)} ep{ep}", flush=True)
        target = Path(args.cache); target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, coords=coords, arm_spread_mm=arm_spread, ee=ee,
                            topple=topples)

    rows, classes = [], ("topple_fork", "nudge_fork", "small_split", "quiet", "weak")
    for i, row1 in enumerate(stage1["rows"]):
        row = {"episode_id": row1["episode_id"], "chunk_start": row1["chunk_start"],
               "stratum": row1["stratum"], "by_scale": {}}
        for s, scale in enumerate(map(str, SCALES)):
            test = fork_test(coords[i, s])
            test["arm_removed"] = fork_test(remove_arm(coords[i, s], ee[i, s]))["alarm"]
            count = int(topples[i, s].sum())
            cls = physical_class(row1, scale)
            if args.own_hold:
                g = grade(topples[i, s])
                cls = ("topple_fork" if g == "mixed" else "weak" if g != "unanimous_safe"
                       else ("quiet" if cls in ("topple_fork", "weak") else cls))
            row["by_scale"][scale] = {"class": cls, "topple_count": count,
                                      "arm_end_spread_mm": float(arm_spread[i, s]),
                                      "visual": test}
        rows.append(row)

    summary = {}
    for scale in map(str, SCALES):
        entry = {}
        for c in classes:
            part = [r["by_scale"][scale] for r in rows if r["by_scale"][scale]["class"] == c]
            entry[c] = {"states": len(part),
                        "alarms": sum(p["visual"]["alarm"] for p in part),
                        "final_time_alone_alarms": sum(p["visual"]["final_alone"] for p in part),
                        "arm_removed_alarms": sum(p["visual"]["arm_removed"] for p in part),
                        "median_arm_end_spread_mm": (float(np.median(
                            [p["arm_end_spread_mm"] for p in part])) if part else None)}
        graded = [r["by_scale"][scale] for r in rows if r["by_scale"][scale]["class"] != "weak"]
        for positive in ("topple_fork", "nudge_fork"):
            subset = [g for g in graded if g["class"] in (positive, "quiet")]
            labels = [g["class"] == positive for g in subset]
            entry[f"auc_{positive}_vs_quiet"] = {
                "latent_separation": auc([g["visual"]["latent_separation"] for g in subset], labels),
                "latent_spread": auc([g["visual"]["latent_spread_rms"] for g in subset], labels)}
        summary[scale] = entry

    result = {"protocol": {
                  "input": "cam_fixed render at hold steps 10 and 30, world-model DINOv2 "
                           "patch tokens (196x384), full frame, per-state exact PCA",
                  "executions": "Stage 0 snippets, 64 per state, scales 0.5/1/2",
                  "rule": "BIC 2>1, Ashman D>2, minority>=2 at both times; same partition "
                          "(<=1 probe) at hold 10 and 30; no floor",
                  "no_object_or_task_features": True,
                  "own_hold": bool(args.own_hold),
                  "arm_removed_variant": "latent coordinates residualized on end-effector "
                                         "xyz per time before the same rule (declared before "
                                         "arm-check results)",
                  "own_hold_class_note": "topple class re-graded from own-hold runs; nudge/"
                                         "small-split classes inherited from shared-hold Stage 1", "constants_fixed_before_results": True,
                  "classes": list(classes)},
              "summary": summary, "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
