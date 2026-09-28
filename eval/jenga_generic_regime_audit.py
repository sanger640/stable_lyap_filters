"""Generic physical-regime audit for refined ground-truth action boundaries.

This is diagnostic, not an alarm. It records anonymous pose, velocity and contact-edge traces for
the final 32x-narrow action brackets in forks, retained/rejected quiet alarms and matched quiet
controls. Jenga outcome classes define report cohorts only and never enter refinement or metrics.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from action_branch_monitor import action_branch_partition, nearest_cross_branch_pairs  # noqa: E402
from jenga_action_boundary_refine import PAIRS, _refine_state  # noqa: E402
from jenga_action_branch import action_errors, pose_features  # noqa: E402
from jenga_bench import BENCH_FILE, EVAL_CONFIG, SPLITS, verify  # noqa: E402
from jenga_predictive_regime_probe import all_block_pose  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_stage0_noise_oracle import HOLD, RECORD_AT  # noqa: E402
from jenga_state_data import step_state  # noqa: E402


def reconstruct_endpoints(snippets, pair_details):
    """Recover the final bisection bracket from its original probes and midpoint decisions."""
    endpoints = []
    for pair in pair_details:
        left, right = pair["probe_indices"]
        noise = [np.asarray(snippets[left]).copy(), np.asarray(snippets[right]).copy()]
        for side in pair["midpoint_sides"]:
            noise[int(side)] = .5 * (noise[0] + noise[1])
        endpoints.append(np.stack(noise))
    return np.stack(endpoints)


def rollout_trace(sim, snapshot, chunk, noise):
    """H=8 perturbed actions plus the shared hold, recording generic state every control step."""
    sim.restore(snapshot)
    actions = np.asarray(chunk, np.float32).copy()
    actions[:, :3] -= noise
    states = []
    for action in actions:
        sim.execute(action); states.append(step_state(sim))
    for _ in range(HOLD):
        sim.execute(chunk[-1]); states.append(step_state(sim))
    return np.stack(states)


def _max_true_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def trace_pair_metrics(left, right):
    """Task-independent differences between two physical state trajectories."""
    left, right = np.asarray(left), np.asarray(right)
    contact_difference = (left[:, 45:57] > .5) != (right[:, 45:57] > .5)
    any_contact_difference = np.any(contact_difference, axis=1)
    pose_delta = np.concatenate([(left[:, :9] - right[:, :9]) / .05,
                                 left[:, 9:27] - right[:, 9:27]], axis=1)
    velocity_delta = left[:, 27:45] - right[:, 27:45]
    pose_gap = np.linalg.norm(pose_delta, axis=1)
    velocity_gap = np.linalg.norm(velocity_delta, axis=1)
    first = np.flatnonzero(any_contact_difference)
    return {
        "contact_difference_fraction": float(np.mean(contact_difference)),
        "contact_difference_step_fraction": float(np.mean(any_contact_difference)),
        "contact_difference_max_run": _max_true_run(any_contact_difference),
        "persistent_contact_difference": bool(_max_true_run(any_contact_difference) >= 3),
        "late_contact_difference": bool(np.any(np.sum(contact_difference[-3:], axis=0) >= 2)),
        "first_contact_difference_step": int(first[0]) if len(first) else None,
        "contact_transition_count_left": int(np.sum(
            (left[1:, 45:57] > .5) != (left[:-1, 45:57] > .5))),
        "contact_transition_count_right": int(np.sum(
            (right[1:, 45:57] > .5) != (right[:-1, 45:57] > .5))),
        "pose_gap_action_end": float(pose_gap[HORIZON - 1]),
        "pose_gap_hold10": float(pose_gap[HORIZON + 10 - 1]),
        "pose_gap_final": float(pose_gap[-1]),
        "pose_gap_max": float(np.max(pose_gap)),
        "pose_gap_final_over_max": float(pose_gap[-1] / max(np.max(pose_gap), 1e-12)),
        "velocity_gap_action_end": float(velocity_gap[HORIZON - 1]),
        "velocity_gap_hold10": float(velocity_gap[HORIZON + 10 - 1]),
        "velocity_gap_final": float(velocity_gap[-1]),
        "velocity_gap_max": float(np.max(velocity_gap)),
    }


def _run_episode(job):
    episode_id, targets, lmdb, xml, snippets = job
    replay = JengaReplay(lmdb)
    episode = replay.episode(episode_id)
    replay.close()
    sim = DirectJengaSim(xml)
    out = {}
    try:
        sim.reset(int(episode_id))
        for step, action in enumerate(episode.actions):
            if step in targets:
                target = targets[step]
                snapshot = sim.snapshot()
                start_pose = all_block_pose(sim)
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                _, pair_details = _refine_state(
                    sim, snapshot, episode.actions[step:step + HORIZON], snippets,
                    target["pose"], target["labels"], target["pairs"], start_pose, start_tilt)
                endpoint_noises = reconstruct_endpoints(snippets, pair_details)
                metrics = []
                for pair_noises in endpoint_noises:
                    traces = [rollout_trace(sim, snapshot,
                                            episode.actions[step:step + HORIZON], noise)
                              for noise in pair_noises]
                    metrics.append(trace_pair_metrics(traces[0], traces[1]))
                out[step] = {"pair_metrics": metrics, "pair_details": pair_details}
                sim.restore(snapshot)
            sim.execute(action)
    finally:
        sim.close()
    return episode_id, out


def _matching_features(bench):
    state = np.asarray(bench["dev_start"][:, :27], float)
    actions = np.asarray(bench["dev_windows"][:, 1, :, 2:10, :3].mean(1), float)
    relative_actions = actions - actions[:, :1]
    return np.concatenate([state, relative_actions.reshape(len(state), -1)], axis=1)


def matched_controls(bench, audit_rows, count):
    """Match never-alarmed quiet controls to retained quiet states by state/action geometry."""
    retained = [i for i, row in enumerate(audit_rows)
                if row["class"] == "quiet" and row["alarm"]]
    available = {i for i, row in enumerate(audit_rows)
                 if row["class"] == "quiet" and not row["initial_alarm"]}
    features = _matching_features(bench)
    scale = features.std(0)
    keep = scale > 1e-10
    z = (features[:, keep] - features[:, keep].mean(0)) / scale[keep]
    chosen = []
    for target in retained[:count]:
        candidate = min(available, key=lambda i: float(np.linalg.norm(z[i] - z[target])))
        chosen.append(candidate); available.remove(candidate)
    return chosen


def prepare(refinement_path):
    verify()
    bench = np.load(BENCH_FILE, allow_pickle=False)
    refinement = json.loads(Path(refinement_path).read_text())["rows"]
    row_index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(refinement)}
    controls = set(matched_controls(bench, refinement, 21))
    spec = SPLITS["dev"]
    stage0 = json.loads((ROOT / spec["stage0"]).read_text())["rows"]
    cache_index = {(r["episode_id"], r["chunk_start"]): i for i, r in enumerate(stage0)}
    cache = np.load(ROOT / spec["cache"], allow_pickle=False)
    scale_index = EVAL_CONFIG["scales"].index(1.0)
    rows, wanted = [], {}
    for i, episode_id in enumerate(bench["dev_episode"]):
        chunk = int(bench["dev_chunk"][i])
        prior = refinement[row_index[(str(episode_id), chunk)]]
        if prior["class"] == "topple_fork" and prior["alarm"]:
            cohort = "fork"
        elif prior["class"] == "quiet" and prior["alarm"]:
            cohort = "retained_quiet"
        elif prior["class"] == "quiet" and prior["initial_alarm"]:
            cohort = "rejected_quiet"
        elif i in controls:
            cohort = "matched_quiet_control"
        else:
            continue
        cache_row = cache_index[(str(episode_id), chunk)]
        pose = cache["pose"][cache_row, scale_index]
        windows = bench["dev_windows"][i, scale_index]
        features = pose_features(pose)
        early_stop = RECORD_AT.index(10) + 1
        x, labels = action_branch_partition(
            action_errors(windows), features[:, :early_stop], features)
        pairs = nearest_cross_branch_pairs(x, labels, PAIRS)
        row = {"episode_id": str(episode_id), "chunk_start": chunk, "cohort": cohort,
               "class": prior["class"], "initial_alarm": prior["initial_alarm"],
               "refined_alarm": prior["alarm"]}
        rows.append(row)
        wanted.setdefault(str(episode_id), {})[chunk] = {
            "pose": pose, "labels": labels, "pairs": pairs, "row": len(rows) - 1}
    return rows, wanted, cache["snippets"]


def _state_summary(pair_metrics):
    return {
        "pairs_with_persistent_contact_difference": sum(
            p["persistent_contact_difference"] for p in pair_metrics),
        "pairs_with_late_contact_difference": sum(p["late_contact_difference"]
                                                   for p in pair_metrics),
        "contact_difference_fraction_median": float(np.median(
            [p["contact_difference_fraction"] for p in pair_metrics])),
        "contact_difference_step_fraction_median": float(np.median(
            [p["contact_difference_step_fraction"] for p in pair_metrics])),
        "pose_gap_final_median": float(np.median([p["pose_gap_final"] for p in pair_metrics])),
        "pose_gap_final_over_max_median": float(np.median(
            [p["pose_gap_final_over_max"] for p in pair_metrics])),
        "velocity_gap_max_median": float(np.median(
            [p["velocity_gap_max"] for p in pair_metrics])),
    }


def summarise(rows):
    fields = ("contact_difference_fraction_median", "contact_difference_step_fraction_median",
              "pose_gap_final_median", "pose_gap_final_over_max_median",
              "velocity_gap_max_median")
    out = {}
    for cohort in ("fork", "retained_quiet", "rejected_quiet", "matched_quiet_control"):
        selected = [r for r in rows if r["cohort"] == cohort]
        out[cohort] = {
            "states": len(selected),
            "persistent_contact_majority_rate": float(np.mean([
                r["pairs_with_persistent_contact_difference"] >= 2 for r in selected])),
            "late_contact_majority_rate": float(np.mean([
                r["pairs_with_late_contact_difference"] >= 2 for r in selected])),
            **{field: {"median": float(np.median([r[field] for r in selected])),
                       "q25": float(np.quantile([r[field] for r in selected], .25)),
                       "q75": float(np.quantile([r[field] for r in selected], .75))}
               for field in fields},
        }
    return out


def compare_cohorts(rows):
    """Predeclared descriptive comparisons; these do not define or tune an alarm."""
    from scipy.stats import fisher_exact, mannwhitneyu

    groups = {name: [r for r in rows if r["cohort"] == name]
              for name in ("fork", "retained_quiet", "rejected_quiet",
                           "matched_quiet_control")}
    fields = ("contact_difference_fraction_median", "pose_gap_final_median",
              "velocity_gap_max_median")
    out = {}
    for left, right in (("fork", "retained_quiet"),
                        ("retained_quiet", "matched_quiet_control"),
                        ("retained_quiet", "rejected_quiet")):
        a, b = groups[left], groups[right]
        key = f"{left}_vs_{right}"
        a_contact = sum(r["pairs_with_persistent_contact_difference"] >= 2 for r in a)
        b_contact = sum(r["pairs_with_persistent_contact_difference"] >= 2 for r in b)
        out[key] = {
            "persistent_contact_majority": {
                left: [a_contact, len(a) - a_contact],
                right: [b_contact, len(b) - b_contact],
                "fisher_two_sided_p": float(fisher_exact(
                    [[a_contact, len(a) - a_contact],
                     [b_contact, len(b) - b_contact]])[1]),
            },
            **{field: {
                "median_difference": float(np.median([r[field] for r in a])
                                           - np.median([r[field] for r in b])),
                "mann_whitney_two_sided_p": float(mannwhitneyu(
                    [r[field] for r in a], [r[field] for r in b],
                    alternative="two-sided").pvalue),
            } for field in fields},
        }
    return out


def evaluate(lmdb, xml, workers, refinement_path):
    rows, wanted, snippets = prepare(refinement_path)
    jobs = [(episode_id, targets, lmdb, xml, snippets)
            for episode_id, targets in wanted.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_episode, job): job[0] for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            episode_id, results = future.result()
            for step, result in results.items():
                row = rows[wanted[episode_id][step]["row"]]
                row.update(result)
                row.update(_state_summary(result["pair_metrics"]))
            print(f"audited episode {number}/{len(jobs)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--refinement", default=str(
        ROOT / "results/jenga/action_boundary_refine_dev.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse", default=None,
                    help="reuse completed row traces and regenerate summaries without simulation")
    ap.add_argument("--output", default=str(ROOT / "results/jenga/generic_regime_audit_dev.json"))
    args = ap.parse_args()
    if args.reuse:
        rows = json.loads(Path(args.reuse).read_text())["rows"]
    else:
        with tempfile.TemporaryDirectory(prefix="jenga_regime_audit_") as temp:
            xml = str(extract_sim(args.sim_archive, temp))
            rows = evaluate(args.lmdb, xml, args.workers, args.refinement)
    result = {
        "protocol": {
            "diagnostic_only": True,
            "split": "dev only",
            "horizon": "H=8 plus existing shared hold 30; no extended settling",
            "cohorts": "fork, retained/rejected quiet, state/action-matched never-alarmed quiet",
            "boundary": "three pairs after five midpoint bisections (32x narrower)",
            "signals": "anonymous object pose, velocity and binary physical contact edges",
            "task_information": "Jenga classes organize the report only; no task feature or "
                                "failure rule enters refinement or metrics",
        },
        "summary": summarise(rows),
        "comparisons": compare_cohorts(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
