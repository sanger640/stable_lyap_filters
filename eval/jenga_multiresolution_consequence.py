"""DEV-only dense multi-resolution commitment-convergence experiment.

Every saved action-boundary bracket is replayed at its original width and all five midpoint
bisections. The monitor requires commitment evidence, onset time and normalized trajectory shape to
converge at narrow widths. Jenga classes are grading only; TEST and D2 are never read.
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

from consequence_monitor import multiresolution_consequence_alarm, result_dict  # noqa: E402
from jenga_action_branch import summarise  # noqa: E402
from jenga_bench import SPLITS, verify  # noqa: E402
from jenga_generic_regime_audit import rollout_trace  # noqa: E402
from jenga_runtime import DEFAULT_LMDB, JengaReplay  # noqa: E402
from jenga_short_held_tails import DirectJengaSim, HORIZON, extract_sim  # noqa: E402


def reconstruct_levels(snippets, pair_details):
    """Recover both bracket endpoints before and after every recorded midpoint decision."""
    pairs = []
    for pair in pair_details:
        left, right = pair["probe_indices"]
        endpoints = [np.asarray(snippets[left]).copy(), np.asarray(snippets[right]).copy()]
        levels = [np.stack(endpoints)]
        for side in pair["midpoint_sides"]:
            endpoints[int(side)] = .5 * (endpoints[0] + endpoints[1])
            levels.append(np.stack(endpoints))
        pairs.append(np.stack(levels))
    return np.stack(pairs)


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
                endpoints = reconstruct_levels(snippets, target["pair_details"])
                traces = np.stack([[[rollout_trace(
                    sim, snapshot, episode.actions[step:step + HORIZON], noise)
                    for noise in level] for level in pair] for pair in endpoints])
                action_differences = endpoints[:, :, 0] - endpoints[:, :, 1]
                evidence = multiresolution_consequence_alarm(
                    traces[:, :, :, :, :45], action_differences,
                    target["early_delta_bic"], target["full_delta_bic"])
                out[step] = result_dict(evidence)
                sim.restore(snapshot)
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
    return rows, wanted, cache["snippets"]


def evaluate(lmdb, xml, workers, refinement_path, limit=0):
    rows, wanted, snippets = prepare(refinement_path)
    flat = [(episode_id, step, target) for episode_id, targets in wanted.items()
            for step, target in targets.items()]
    if limit:
        flat = flat[:limit]
    selected = {}
    for episode_id, step, target in flat:
        selected.setdefault(episode_id, {})[step] = target
    jobs = [(episode_id, targets, lmdb, xml, snippets)
            for episode_id, targets in selected.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_episode, job): job[0] for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            episode_id, results = future.result()
            for step, result in results.items():
                rows[selected[episode_id][step]["row"]].update(result)
            print(f"multiresolution episode {number}/{len(jobs)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default=str(DEFAULT_LMDB))
    ap.add_argument("--sim-archive", default=str(ROOT / "vendor/panda_express_sim.tar"))
    ap.add_argument("--refinement", default=str(
        ROOT / "results/jenga/action_boundary_refine_dev.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=str(
        ROOT / "results/jenga/multiresolution_consequence_dev.json"))
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="jenga_multires_consequence_") as temp:
        xml = str(extract_sim(args.sim_archive, temp))
        rows = evaluate(args.lmdb, xml, args.workers, args.refinement, args.limit)
    summary = summarise(rows, [1.0])
    scored = summary["dev"]["1.0"]
    gate_passed = (scored["topple_fork"]["alarms"] >= 21
                   and scored["quiet"]["alarms"] <= 4)
    result = {
        "protocol": {
            "calibration_free": True,
            "split": "DEV only; TEST untouched",
            "trajectory_source": "dense MuJoCo ground-truth state",
            "resolutions": "original bracket plus all five midpoint bisections (1x through 32x)",
            "boundary": "saved per-pair early+late BIC plateau evidence",
            "stable_commitment": "commitment evidence positive at the final three widths and "
                                 "fitted onset span <= one control step",
            "shape_convergence": "BIC favours successive normalized-curve changes vanishing with "
                                 "bracket width rather than retaining a nonzero plateau",
            "persistence": "final-width late decay-to-zero versus nonzero-asymptote BIC",
            "decision": "the same >=2/3 pairs pass boundary, stable commitment, shape convergence "
                        "and persistence",
            "gate": ">=21/23 topple forks and <=4/89 quiet alarms",
            "task_information": "none in alarm; Jenga classes are grading only",
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
