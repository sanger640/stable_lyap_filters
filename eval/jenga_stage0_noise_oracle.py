"""Stage 0: an answer key under realistic execution noise, before any detector.

Each of the frozen 55 panel states replays its nominal 8-action chunk N=64 times, once per
contiguous 8-step tracking-residual snippet, then holds the chunk's final target for 30 steps.
Snippets come only from episodes outside the panel and are shared across states (common random
numbers). They are applied at 0.5x, 1x, and 2x; every scale is reported and none is selected.

Evaluation-only oracle: a probe topples if any neighbor that started below 45 degrees ends the
30-step hold at or above 45 degrees. A state is MIXED when both outcomes have >=2 of 64 probes,
UNANIMOUS when one outcome has all 64, and WEAK otherwise (a single dissenting probe).

The residual injection is a proxy for execution variation (see src/action_uncertainty.py); the
simulated controller also lags, so part of the lag may be counted twice. No detector is scored.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from action_uncertainty import tracking_arrays  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_predictive_regime_probe import all_block_pose  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim  # noqa: E402

N_PROBES = 64
SCALES = (0.5, 1.0, 2.0)
HOLD = 30
RECORD_AT = (5, 10, 20, 29, 30)
MIN_SIDE = 2


def sample_snippets(pools, count=N_PROBES, horizon=HORIZON, seed=0):
    """Contiguous (horizon,3) residual windows, episode then start chosen uniformly."""
    rng = np.random.default_rng(seed)
    usable = [p for p in pools if len(p) >= horizon]
    if not usable:
        raise ValueError("no residual sequence is long enough")
    out = []
    for _ in range(count):
        seq = usable[int(rng.integers(len(usable)))]
        start = int(rng.integers(len(seq) - horizon + 1))
        out.append(seq[start:start + horizon])
    return np.asarray(out, np.float32)


def grade(topples, min_side=MIN_SIDE):
    """topples: bool (n,). Returns mixed / unanimous_safe / unanimous_topple / weak_*."""
    n, k = len(topples), int(np.sum(topples))
    if k >= min_side and n - k >= min_side:
        return "mixed"
    if k == 0:
        return "unanimous_safe"
    if k == n:
        return "unanimous_topple"
    return "weak_topple_minority" if k < min_side else "weak_safe_minority"


def detection_probability(p, n=N_PROBES, min_side=MIN_SIDE):
    """P(at least min_side of n draws land in a minority outcome of probability p)."""
    from math import comb
    return 1.0 - sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(min_side))


def rollout(sim, snapshot, chunk, noise, start_pose, start_tilt):
    sim.restore(snapshot)
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise
    for action in actions:
        sim.execute(action)
    poses, tilts = [], []
    for held in range(1, HOLD + 1):
        # Same nominal target for every probe: the chunk being judged ends, the scene settles.
        sim.execute(chunk[-1])
        if held in RECORD_AT:
            poses.append(all_block_pose(sim) - start_pose)
            tilts.append(sim.block_diagnostics()["tilt"][1:].copy())
    tilts = np.asarray(tilts, np.float32)
    eligible = start_tilt < TOPPLE_DEG
    topple = np.any((tilts >= TOPPLE_DEG) & eligible[None, :], axis=1)
    return np.stack(poses), tilts, topple


def run_episode(job):
    episode_id, starts, lmdb, xml, noise_by_scale, reset_base = job
    replay = JengaReplay(lmdb)
    episode = replay.episode(episode_id)
    replay.close()
    sim = DirectJengaSim(xml)
    out = {}
    try:
        sim.reset(int(episode_id) + (reset_base or 0) if reset_base is not None
                  else int(episode_id))
        for index, action in enumerate(episode.actions):
            if index in starts:
                chunk = episode.actions[index:index + HORIZON]
                snapshot = sim.snapshot()
                start_pose = all_block_pose(sim)
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                nominal = rollout(sim, snapshot, chunk, np.zeros((HORIZON, 3)),
                                  start_pose, start_tilt)
                probes = [[rollout(sim, snapshot, chunk, noise, start_pose, start_tilt)
                           for noise in noises] for noises in noise_by_scale]
                sim.restore(snapshot)
                out[index] = {
                    "nominal_pose": nominal[0], "nominal_tilt": nominal[1],
                    "nominal_topple": nominal[2],
                    "pose": np.asarray([[p[0] for p in s] for s in probes], np.float32),
                    "tilt": np.asarray([[p[1] for p in s] for s in probes], np.float32),
                    "topple": np.asarray([[p[2] for p in s] for s in probes], bool),
                    "start_tilt": start_tilt.astype(np.float32),
                    "start_pose": start_pose.astype(np.float32),
                    "span_mm": float(1000 * np.max(np.linalg.norm(
                        chunk[:, :3] - chunk[0, :3], axis=1)))}
            sim.execute(action)
    finally:
        sim.close()
    return episode_id, out


def settling(pose, tilt, topple):
    """pose (..., times, D) at RECORD_AT; per-step motion over the last held step."""
    i29, i30 = RECORD_AT.index(29), RECORD_AT.index(30)
    position = pose[..., :9].reshape(*pose.shape[:-1], 3, 3)
    last_mm = 1000 * np.linalg.norm(position[..., i30, :, :] - position[..., i29, :, :], axis=-1)
    last_deg = np.abs(tilt[..., i30, :] - tilt[..., i29, :])
    final = topple[..., i30]
    agree = {f"tail{t}": float(np.mean(topple[..., RECORD_AT.index(t)] == final))
             for t in (5, 10, 20)}
    return {"last_step_block_motion_mm_p99": float(np.quantile(last_mm, .99)),
            "last_step_block_motion_mm_max": float(np.max(last_mm)),
            "last_step_neighbor_tilt_change_deg_p99": float(np.quantile(last_deg, .99)),
            "last_step_neighbor_tilt_change_deg_max": float(np.max(last_deg)),
            "topple_label_agreement_with_tail30": agree}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--panel", default=str(ROOT / "results/jenga/persistent_branch_margin.json"))
    ap.add_argument("--snippet-panel",
                    default=str(ROOT / "results/jenga/persistent_branch_margin.json"),
                    help="snippets come from episodes NOT in this panel (keeps the same 64)")
    ap.add_argument("--uncertainty", default=str(ROOT / "results/jenga/tracking_uncertainty.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/stage0_noise_oracle_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/stage0_noise_oracle.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reset-seed-base", type=int, default=None,
                    help="reset with seed BASE + episode_id (new configurations); default: episode_id")
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    panel = json.loads(Path(args.panel).read_text())["rows"]
    coefficients = np.asarray(json.loads(Path(args.uncertainty).read_text())["lag_coefficients"])
    panel_episodes = {r["episode_id"] for r in
                      json.loads(Path(args.snippet_panel).read_text())["rows"]}
    replay = JengaReplay(args.lmdb)
    try:
        pools, pool_episodes = [], []
        for ep in replay.episode_ids:
            if ep in panel_episodes:
                continue
            design, errors = tracking_arrays([replay.episode(ep)])
            pools.append(errors - design @ coefficients)
            pool_episodes.append(ep)
    finally:
        replay.close()
    snippets = sample_snippets(pools)
    snippet_mm = 1000 * np.linalg.norm(snippets, axis=2)

    keys = [(r["episode_id"], r["chunk_start"]) for r in panel]
    if args.reuse_cache:
        cache = np.load(args.cache, allow_pickle=False)
        arrays = {k: cache[k] for k in cache.files if k != "snippets"}
    else:
        by_episode = {}
        for ep, start in keys:
            by_episode.setdefault(ep, set()).add(start)
        noise_by_scale = [snippets * s for s in SCALES]
        with tempfile.TemporaryDirectory(prefix="jenga_stage0_") as temp:
            xml = str(extract_sim(args.sim_archive, temp))
            jobs = [(ep, starts, args.lmdb, xml, noise_by_scale, args.reset_seed_base)
                    for ep, starts in sorted(by_episode.items(), key=lambda x: int(x[0]))]
            results = {}
            with ProcessPoolExecutor(args.workers) as pool:
                for done, (ep, out) in enumerate(pool.map(run_episode, jobs), 1):
                    results[ep] = out
                    print(f"  stage0 {done}/{len(jobs)} ep{ep}", flush=True)
        per_key = [results[ep][start] for ep, start in keys]
        arrays = {name: np.stack([np.asarray(r[name]) for r in per_key])
                  for name in per_key[0]}
        target = Path(args.cache); target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, snippets=snippets, **arrays)

    rows = []
    for i, meta in enumerate(panel):
        by_scale = {}
        for s, scale in enumerate(SCALES):
            final = arrays["topple"][i, s, :, RECORD_AT.index(30)]
            by_scale[str(scale)] = {"topple_count": int(final.sum()),
                                    "grade": grade(final)}
        rows.append({"episode_id": meta["episode_id"], "chunk_start": meta["chunk_start"],
                     "stratum": meta["stratum"],
                     "nominal_topple_tail30": bool(arrays["nominal_topple"][i, -1]),
                     "start_neighbor_tilt_deg": arrays["start_tilt"][i].tolist(),
                     "prior_oracle_margin": meta.get("oracle_margin"),
                     "by_scale": by_scale})

    grades = ("mixed", "unanimous_safe", "unanimous_topple",
              "weak_topple_minority", "weak_safe_minority")
    summary = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        part = [r for r in rows if r["stratum"] == stratum]
        summary[stratum] = {"states": len(part), "nominal_topples": sum(
            r["nominal_topple_tail30"] for r in part)}
        for scale in map(str, SCALES):
            summary[stratum][scale] = {g: sum(r["by_scale"][scale]["grade"] == g for r in part)
                                       for g in grades}
            counts = [r["by_scale"][scale]["topple_count"] for r in part]
            summary[stratum][scale]["topple_counts"] = counts

    result = {
        "protocol": {
            "panel": "frozen 55 states from persistent_branch_margin.json",
            "probes_per_state": N_PROBES, "scales": list(SCALES), "horizon": HORIZON,
            "hold_steps": HOLD, "recorded_hold_steps": list(RECORD_AT),
            "noise": "contiguous 8-step lag-model tracking residuals from non-panel episodes, "
                     "subtracted from commanded xyz targets; same snippets at every state",
            "snippet_source_episodes": len(pool_episodes),
            "hold": "nominal chunk final target (gripper included) for every probe",
            "oracle": "any neighbor with start tilt <45 deg ends hold step 30 at >=45 deg",
            "grades": f"mixed if both outcomes >= {MIN_SIDE}/{N_PROBES}; unanimous if all "
                      "agree; weak otherwise",
            "evaluation_only": True, "detector_scored": False,
            "proxy_warning": "tracking residuals are an execution-variation proxy; the "
                             "simulated controller also lags, so lag may be double counted"},
        "snippet_norm_mm": {"median_step": float(np.median(snippet_mm)),
                            "p90_step": float(np.quantile(snippet_mm, .9)),
                            "median_chunk_max": float(np.median(snippet_mm.max(1)))},
        "sizing": {f"minority_p_{p}": detection_probability(p)
                   for p in (.02, .05, .10, .20)},
        "settling_by_scale": {str(scale): settling(arrays["pose"][:, s], arrays["tilt"][:, s],
                                                   arrays["topple"][:, s])
                              for s, scale in enumerate(SCALES)},
        "summary": summary, "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("snippet_norm_mm", "settling_by_scale")}, indent=2))
    for stratum, part in summary.items():
        print(stratum, part["states"], "nominal topples", part["nominal_topples"])
        for scale in map(str, SCALES):
            print("   ", scale, {g: part[scale][g] for g in grades if part[scale][g]},
                  part[scale]["topple_counts"])


if __name__ == "__main__":
    main()
