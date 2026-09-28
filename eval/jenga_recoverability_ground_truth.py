"""DEV-only local corrective-reachability experiment on refined physical branches.

From each final 32x bracket endpoint, apply the same seven bounded correction options (neutral and
plus/minus three execution-uncertainty axes) for five steps, then return to the shared nominal target
for ten steps. An alarm requires the two correction-reachable sets to remain separated at their own
sampling resolution on a majority of boundary pairs. TEST and D2 are not read.
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

from jenga_action_branch import summarise  # noqa: E402
from jenga_bench import SPLITS, verify  # noqa: E402
from jenga_generic_regime_audit import reconstruct_endpoints  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402
from jenga_state_data import step_state  # noqa: E402
from recoverability_monitor import recoverability_alarm, result_dict  # noqa: E402

CORRECTION_STEPS = 5
SETTLE_STEPS = 10


def correction_offsets(snippets):
    """Neutral and +/- three one-sigma axes from the existing execution-error model."""
    values = np.asarray(snippets, float).reshape(-1, 3)
    centred = values - values.mean(0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    coordinates = centred @ axes.T
    scales = coordinates.std(0)
    offsets = [np.zeros(3)]
    for axis, scale in zip(axes[:3], scales[:3]):
        offsets.extend([axis * scale, -axis * scale])
    return np.asarray(offsets, np.float32)


def correction_outcomes(sim, endpoint_snapshot, nominal, offsets):
    outcomes = []
    for offset in offsets:
        sim.restore(endpoint_snapshot)
        corrected = np.asarray(nominal, np.float32).copy()
        corrected[:3] -= offset
        for _ in range(CORRECTION_STEPS):
            sim.execute(corrected)
        for _ in range(SETTLE_STEPS):
            sim.execute(nominal)
        outcomes.append(step_state(sim)[:45])
    return np.stack(outcomes)


def _run_episode(job):
    episode_id, targets, lmdb, xml, snippets, offsets = job
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
                start = sim.snapshot()
                chunk = np.asarray(episode.actions[step:step + HORIZON], np.float32)
                endpoints = reconstruct_endpoints(snippets, target["pair_details"])
                pair_outcomes = []
                for pair in endpoints:
                    branches = []
                    for noise in pair:
                        sim.restore(start)
                        perturbed = chunk.copy(); perturbed[:, :3] -= noise
                        for command in perturbed:
                            sim.execute(command)
                        branches.append(correction_outcomes(
                            sim, sim.snapshot(), chunk[-1], offsets))
                    pair_outcomes.append(np.stack(branches))
                evidence = recoverability_alarm(
                    np.stack(pair_outcomes), target["early_delta_bic"],
                    target["full_delta_bic"])
                out[step] = result_dict(evidence)
                sim.restore(start)
            sim.execute(action)
    finally:
        sim.close()
    return episode_id, out


def prepare(refinement_path):
    verify()
    refinement = json.loads(Path(refinement_path).read_text())
    if refinement["protocol"].get("split") != "dev only; test remains untouched":
        raise ValueError("refinement input is not the frozen DEV-only result")
    rows, wanted = [], {}
    for prior in refinement["rows"]:
        row = {key: prior[key] for key in (
            "split", "episode_id", "chunk_start", "scale", "class", "topple_count")}
        row.update({"initial_alarm": bool(prior["initial_alarm"]),
                    "boundary_refined_alarm": bool(prior["alarm"]), "alarm": False})
        rows.append(row)
        if "pair_details" in prior:
            wanted.setdefault(prior["episode_id"], {})[prior["chunk_start"]] = {
                "pair_details": prior["pair_details"],
                "early_delta_bic": prior["early_delta_bic"],
                "full_delta_bic": prior["full_delta_bic"],
                "row": len(rows) - 1}
    cache = np.load(ROOT / SPLITS["dev"]["cache"], allow_pickle=False)
    snippets = cache["snippets"]
    return rows, wanted, snippets, correction_offsets(snippets)


def evaluate(lmdb, xml, workers, refinement_path, limit=0):
    rows, wanted, snippets, offsets = prepare(refinement_path)
    flat = [(episode_id, step, target) for episode_id, targets in wanted.items()
            for step, target in targets.items()]
    if limit:
        flat = flat[:limit]
    selected = {}
    for episode_id, step, target in flat:
        selected.setdefault(episode_id, {})[step] = target
    jobs = [(episode_id, targets, lmdb, xml, snippets, offsets)
            for episode_id, targets in selected.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_episode, job): job[0] for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            episode_id, results = future.result()
            for step, result in results.items():
                rows[selected[episode_id][step]["row"]].update(result)
            print(f"recoverability episode {number}/{len(jobs)}", flush=True)
    return rows, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--refinement", default=str(
        ROOT / "results/jenga/action_boundary_refine_dev.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/recoverability_ground_truth_dev.json"))
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="jenga_recoverability_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        rows, offsets = evaluate(args.lmdb, xml, args.workers, args.refinement, args.limit)
    summary = summarise(rows, [1.0])
    scored = summary["dev"]["1.0"]
    gate_passed = (scored["topple_fork"]["alarms"] >= 21
                   and scored["quiet"]["alarms"] <= 4)
    result = {
        "protocol": {
            "calibration_free": True,
            "split": "DEV only; TEST untouched",
            "trajectory_source": "MuJoCo ground-truth state",
            "corrections": "neutral plus/minus three one-sigma PCA axes of the frozen execution "
                           "error snippets; five correction steps then ten nominal settle steps",
            "reachable_set": "seven anonymous pose/velocity endpoints from each boundary side",
            "overlap": "minimum cross-set distance <= geometric mean within-set nearest-neighbour "
                       "resolution",
            "decision": "the same >=2/3 refined pairs pass local boundary scaling and remain "
                        "non-overlapping after correction",
            "gate": ">=21/23 topple forks and <=4/89 quiet alarms",
            "task_information": "none in alarm; Jenga classes are grading only",
            "correction_offsets_xyz": offsets.tolist(),
        },
        "gate_passed": bool(gate_passed),
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"gate_passed": gate_passed, **scored}, indent=2))


if __name__ == "__main__":
    main()
