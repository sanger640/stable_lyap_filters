"""Short-horizon, multi-direction predictive-regime probe in physical state.

This is an experimental physical-state monitor, not a certified safety filter.
All probes share the same nominal held-action continuation after their 8-action
chunk. The detector sees continuous scene pose only; topple outcomes are used
after scoring as an evaluation oracle.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from ordered_change_point import detect_ordered_jump, robust_component_scale  # noqa: E402
sys.path.insert(0, str(ROOT / "eval"))
from jenga_short_held_tails import DirectJengaSim, HORIZON, TOPPLE_DEG, extract_sim  # noqa: E402

TAILS = (0, 5, 10)
STRENGTHS = np.linspace(-1.0, 1.0, 17, dtype=np.float32)


def all_block_pose(sim):
    position = sim.block_diagnostics()["position"].reshape(-1)
    rotation = sim.data.xmat[sim.tracked_block_ids].reshape(3, 3, 3)[:, :, :2]
    return np.concatenate([position, rotation.reshape(-1)]).astype(np.float32)


def probe_chunk(chunk, displacement, strength):
    actions = np.asarray(chunk, np.float32).copy()
    if actions.shape != (HORIZON, 4):
        raise ValueError(f"expected ({HORIZON},4) action chunk")
    ramp = np.arange(1, HORIZON + 1, dtype=np.float32) / HORIZON
    actions[:, :3] += ramp[:, None] * (np.asarray(displacement, np.float32) * float(strength))
    return actions


def simulate_probe(sim, snapshot, chunk, displacement, strength, start):
    sim.restore(snapshot)
    probe = probe_chunk(chunk, displacement, strength)
    trace = {"peak_tilt": sim.block_diagnostics()["tilt"].copy()}
    for action in probe:
        sim.execute(action, trace)
    pose = [all_block_pose(sim) - start]
    endpoint_tilt = [float(np.max(sim.block_diagnostics()["tilt"][1:]))]
    # Exactly the SAME nominal target after H=8 for every counterfactual.
    for held in range(1, max(TAILS) + 1):
        sim.execute(chunk[-1], trace)
        if held in TAILS:
            pose.append(all_block_pose(sim) - start)
            endpoint_tilt.append(float(np.max(sim.block_diagnostics()["tilt"][1:])))
    sim.restore(snapshot)
    return np.stack(pose), np.asarray(endpoint_tilt, np.float32)


def collect(rows, replay, xml, axes):
    wanted = {(r["episode_id"], r["chunk_start"]): r for r in rows}
    episodes = sorted({r["episode_id"] for r in rows}, key=int)
    sim = DirectJengaSim(xml)
    metadata, responses, tilts, repeat_errors = [], [], [], []
    try:
        for number, episode_id in enumerate(episodes):
            episode = replay.episode(episode_id)
            sim.reset(int(episode_id))
            for start_index, action in enumerate(episode.actions):
                key = (episode_id, start_index)
                if key in wanted:
                    chunk = episode.actions[start_index:start_index + HORIZON]
                    snapshot = sim.snapshot()
                    start = all_block_pose(sim)
                    state_response, state_tilts = [], []
                    for axis in axes.T:
                        direction_response, direction_tilts = [], []
                        for strength in STRENGTHS:
                            outcome, angle = simulate_probe(sim, snapshot, chunk,
                                                            axis, strength, start)
                            direction_response.append(outcome)
                            direction_tilts.append(angle)
                        state_response.append(direction_response)
                        state_tilts.append(direction_tilts)
                    # Identical intended action replay checks simulator repeatability.
                    repeats = [simulate_probe(sim, snapshot, chunk, np.zeros(3), 0, start)[0]
                               for _ in range(3)]
                    repeat_errors.append(float(max(np.linalg.norm(x - repeats[0])
                                                   for x in repeats[1:])))
                    metadata.append(wanted[key])
                    responses.append(state_response)
                    tilts.append(state_tilts)
                sim.execute(action)
            print(f"  predictive probe {number + 1}/{len(episodes)} ep{episode_id}", flush=True)
    finally:
        sim.close()
    return (metadata, np.asarray(responses, np.float32), np.asarray(tilts, np.float32),
            np.asarray(repeat_errors, np.float32))


def score_state(response, strength_grid, axes):
    """Unlabeled within-state PCA and smooth-versus-jump BIC at short future times."""
    scale = robust_component_scale(response)
    flat = (response / scale).reshape(-1, response.shape[-1])
    mean = flat.mean(0)
    components = np.linalg.svd(flat - mean, full_matrices=False)[2][:6]
    projected = ((response / scale - mean) @ components.T)
    directions = []
    for index, axis_response in enumerate(projected):
        time_results = [detect_ordered_jump(strength_grid, axis_response[:, time],
                                            degree=2, min_side=4, min_jump_ratio=3.0)
                        for time in (1, 2)]
        consistent = bool(all(r.alarm for r in time_results)
                          and abs(time_results[0].split_index
                                  - time_results[1].split_index) <= 1)
        if consistent:
            split = time_results[1].split_index
            lower = min(abs(strength_grid[split - 1]), abs(strength_grid[split]))
            upper = max(abs(strength_grid[split - 1]), abs(strength_grid[split]))
            interval = [float(lower), float(upper)]
        else:
            interval = None
        directions.append({"axis": index, "alarm": consistent,
                           "margin_radial_interval": interval,
                           "tail5": {"bic_evidence": time_results[0].jump_evidence_bic,
                                     "adjacent_jump_ratio": time_results[0].adjacent_jump_ratio,
                                     "split_index": time_results[0].split_index},
                           "tail10": {"bic_evidence": time_results[1].jump_evidence_bic,
                                      "adjacent_jump_ratio": time_results[1].adjacent_jump_ratio,
                                      "split_index": time_results[1].split_index}})
    positives = [r for r in directions if r["alarm"]]
    nearest = min(positives, key=lambda r: r["margin_radial_interval"][1]) if positives else None
    margin = None
    if nearest is not None:
        axis_norm = float(np.linalg.norm(axes[:, nearest["axis"]]))
        margin = {"axis": nearest["axis"],
                  "radial_interval": nearest["margin_radial_interval"],
                  "commanded_mm_interval": [1000 * axis_norm * x
                                            for x in nearest["margin_radial_interval"]]}
    return {"alarm": bool(positives), "margin": margin, "directions": directions}


def physical_oracle(tilt):
    outcome = np.all(tilt[:, :, 1:] >= TOPPLE_DEG, axis=2)
    mixed = np.any(outcome, axis=1) & np.any(~outcome, axis=1)
    return {"mixed_directions": int(mixed.sum()),
            "persistent_topple_probes": int(outcome.sum()),
            "nominal_persistent_topple": bool(np.any(outcome[:, len(STRENGTHS) // 2]))}


def summarise(rows):
    result = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        part = [r for r in rows if r["stratum"] == stratum]
        result[stratum] = {"states": len(part),
                           "alarms": int(sum(r["detector"]["alarm"] for r in part)),
                           "oracle_mixed": int(sum(r["oracle"]["mixed_directions"] > 0
                                                   for r in part)),
                           "alarms_with_oracle_mixed": int(sum(
                               r["detector"]["alarm"]
                               and r["oracle"]["mixed_directions"] > 0 for r in part)),
                           "median_repeat_error": float(np.median([
                               r["identical_probe_repeat_error"] for r in part]))}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--panel", default=str(ROOT / "results/jenga/persistent_branch_margin.json"))
    ap.add_argument("--uncertainty", default=str(ROOT / "results/jenga/tracking_uncertainty.json"))
    ap.add_argument("--cache", default=str(ROOT / "results/jenga/predictive_regime_probe_cache.npz"))
    ap.add_argument("--output", default=str(ROOT / "results/jenga/predictive_regime_probe.json"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    panel = json.loads(Path(args.panel).read_text())
    uncertainty = json.loads(Path(args.uncertainty).read_text())
    axes = (np.asarray(uncertainty["one_sigma_axes_m"], np.float32)
            * uncertainty["training_radial_quantile_90"])
    if args.reuse_cache:
        cache = np.load(args.cache, allow_pickle=False)
        metadata = json.loads(str(cache["metadata_json"]))
        responses, tilts, repeats = (cache[k] for k in
                                     ("responses", "tilts", "repeat_errors"))
    else:
        replay = JengaReplay(args.lmdb)
        try:
            with tempfile.TemporaryDirectory(prefix="jenga_predictive_regime_") as temp:
                metadata, responses, tilts, repeats = collect(
                    panel["rows"], replay, extract_sim(args.sim_archive, temp), axes)
        finally:
            replay.close()
        target = Path(args.cache); target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, metadata_json=np.asarray(json.dumps(metadata)),
                            responses=responses, tilts=tilts, repeat_errors=repeats)
    rows = []
    for meta, response, tilt, repeat_error in zip(metadata, responses, tilts, repeats):
        rows.append({"episode_id": meta["episode_id"],
                     "chunk_start": meta["chunk_start"], "stratum": meta["stratum"],
                     "detector": score_state(response, STRENGTHS, axes),
                     "oracle": physical_oracle(tilt),
                     "identical_probe_repeat_error": float(repeat_error)})
    result = {"protocol": {"horizon": HORIZON, "common_continuation": "nominal last target",
                            "held_tails": list(TAILS), "directions": 3,
                            "strengths_per_direction": len(STRENGTHS),
                            "strengths": STRENGTHS.tolist(),
                            "physical_features": "all three block positions and rotation6; "
                                                 "no contact bits or topple labels",
                            "action_envelope": "principal axes of held-out-fitted tracking "
                                               "residual covariance times empirical 90th radial quantile",
                            "detector": "within-state PCA6; BIC smooth quadratic vs jump; "
                                        "same/adjacent split at held tails 5 and 10; "
                                        "preexisting fixed adjacent jump ratio >=3",
                            "oracle_used_only_after_scoring": True,
                            "uncertainty_proxy_not_certified_bound": True,
                            "no_alarm_interpretation": "unresolved on tested directions/grid, not safe"},
              "summary": summarise(rows), "rows": rows}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
