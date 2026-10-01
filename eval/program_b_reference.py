"""Program B reference arm R: the true Jenga simulator from a misperceived starting state.

Frozen in results/jenga/program_b_stage1_protocol.json (reference_arm_R) and descriptive only.
For every held-out TRAIN neighbourhood (64 probes) and nested group (6 widths x 2 paths) on the
Program A episode and configuration axes, the state is reached exactly as its generator reached
it (D4: uninterrupted replay with snapshot/restore per probe; D3: clean snapshots with warm start,
then the window prefix). The block poses are then perturbed ONCE per state, as one perception
estimate: per-axis Gaussian position noise sigma in {0, 1.0, 2.65, 4.44} mm and rotation-vector
noise sigma / 0.0375 rad (the block half-length), with the gripper exact (proprioception). At
2.65 mm, four further friction "cousins" (block sliding friction x U(0.9, 1.1)) share that
perceived state. Every probe is then simulated for 38 control steps.

``--simulate`` writes traces under results/program_b/reference/ (gitignored); ``--score`` compares
the frozen v0 monitor decisions with those on the recorded physical traces.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

PROTOCOL = ROOT / "results/jenga/program_b_stage1_protocol.json"
NEIGHBOURHOODS = ROOT / "results/jenga/d4_neighborhood_data_large.npz"
NESTED = ROOT / "results/jenga/d3_data"
TRACES = ROOT / "results/jenga/trace_data"
OUTPUT_DIR = ROOT / "results/program_b/reference"
RESULT = ROOT / "results/jenga/program_b_reference_result.json"
SIM_ARCHIVE = ROOT / "vendor/panda_express_sim.tar"
STATE_SCALE_MODEL = ROOT / "results/jenga/w6_cw_d2_s1.pt"

HALF_LENGTH = 0.0375
SIGMAS_MM = (0.0, 1.0, 2.65, 4.44)
COUSIN_SIGMA_MM = 2.65
EXTRA_COUSINS = 4
FRICTION_RANGE = (0.9, 1.1)
EPISODE_AXIS = ("0", "1", "5", "7", "12", "13")
CONFIG_SEEDS = (108, 109)
NOISE_SEED = 20261001


def conditions():
    """[(name, sigma_m, friction_factor_index)]; index 0 means friction unchanged."""
    out = [(f"sigma_{s:g}", s / 1000.0, 0) for s in SIGMAS_MM]
    out += [(f"cousin_{k}", COUSIN_SIGMA_MM / 1000.0, k) for k in range(1, EXTRA_COUSINS + 1)]
    return out


def friction_factors():
    rng = np.random.default_rng([NOISE_SEED, 7])
    return np.r_[1.0, rng.uniform(*FRICTION_RANGE, EXTRA_COUSINS)]


def in_axes(episode, seed):
    return str(episode) in EPISODE_AXIS or int(seed) in CONFIG_SEEDS


def pose_noise(key, sigma, blocks=3):
    """One perception draw per (state, sigma): positions (blocks, 3) m and rotation vectors."""
    rng = np.random.default_rng([NOISE_SEED, *key, int(round(sigma * 1e6))])
    position = rng.normal(0.0, sigma, (blocks, 3))
    rotation = rng.normal(0.0, sigma / HALF_LENGTH, (blocks, 3))
    return position, rotation


def perturb_blocks(sim, position, rotation):
    """Shift and rotate the three blocks' free joints in place; velocities are untouched."""
    from scipy.spatial.transform import Rotation
    from jenga_short_held_tails import BLOCK_NAMES
    m, d = sim.model, sim.data
    for k, name in enumerate(BLOCK_NAMES):
        adr = m.jnt_qposadr[m.body_jntadr[m.body(name).id]]
        d.qpos[adr:adr + 3] += position[k]
        w, x, y, z = d.qpos[adr + 3:adr + 7]
        quat = (Rotation.from_rotvec(rotation[k]) * Rotation.from_quat([x, y, z, w])).as_quat()
        d.qpos[adr + 3:adr + 7] = [quat[3], quat[0], quat[1], quat[2]]
    sim.mj.mj_forward(m, d)


class Friction:
    """Scale the blocks' sliding friction; restore on exit."""

    def __init__(self, sim, factor):
        from jenga_short_held_tails import BLOCK_NAMES
        bodies = [sim.model.body(n).id for n in BLOCK_NAMES]
        self.geoms = np.flatnonzero(np.isin(sim.model.geom_bodyid, bodies))
        self.sim, self.factor = sim, factor

    def __enter__(self):
        self.base = self.sim.model.geom_friction[self.geoms, 0].copy()
        self.sim.model.geom_friction[self.geoms, 0] = self.base * self.factor
        return self

    def __exit__(self, *_):
        self.sim.model.geom_friction[self.geoms, 0] = self.base


def run(sim, actions):
    from jenga_state_data import step_state
    states = []
    for action in actions:
        sim.execute(action)
        states.append(step_state(sim))
    return np.stack(states)


def counterfactuals(sim, restore, actions_set, key, sigma, factor):
    position, rotation = pose_noise(key, sigma)
    out = []
    with Friction(sim, factor):
        for actions in actions_set:
            restore()
            if sigma > 0:
                perturb_blocks(sim, position, rotation)
            out.append(run(sim, actions))
    return np.stack(out)


def simulate_source(job):
    """All in-axis neighbourhoods and nested groups of one (episode, reset seed) source."""
    episode_id, seed, rows, groups, output = job
    path = Path(output) / f"ep{episode_id}_seed{seed}.npz"
    if path.exists():
        return str(path), 0
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    xml = extract_sim(SIM_ARCHIVE, tempfile.mkdtemp(prefix="program_b_ref_"))
    replay = JengaReplay(DEFAULT_LMDB)
    episode_actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    sim = DirectJengaSim(str(xml))
    neighbourhoods = np.load(NEIGHBOURHOODS, allow_pickle=False)
    factors = friction_factors()
    result = {}
    try:
        # D4 neighbourhoods: uninterrupted replay; snapshot at the chunk start; restore per probe;
        # continue the replay from the restored snapshot (exactly as the D4 generator did).
        targets = {int(neighbourhoods["chunk_start"][r]): r for r in rows}
        sim.reset(seed)
        for step, action in enumerate(episode_actions):
            if step in targets:
                row = targets[step]
                snapshot = sim.snapshot()
                # The D4 generator ran its 64 probes back to back (each starting from the previous
                # probe's solver warm start) and then continued the replay from the last probe's
                # warm start. Reproduce that exactly: sigma_0 first, remember its final warm
                # start, run the perturbed conditions, then restore both before continuing.
                warm_end = None
                for name, sigma, cousin in conditions():
                    result[f"n{row}_{name}"] = counterfactuals(
                        sim, lambda: sim.restore(snapshot), neighbourhoods["actions"][row],
                        (0, row), sigma, factors[cousin]).astype(np.float32)
                    if name == "sigma_0":
                        warm_end = sim.data.qacc_warmstart.copy()
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm_end
                if len(result) == len(rows) * len(conditions()):
                    break
            sim.execute(action)
        # D3 nested groups: clean snapshots with warm start, then the probe window prefix.
        if groups:
            source = np.load(TRACES / f"ep{episode_id}_seed{seed}.npz", allow_pickle=False)
            nested = np.load(NESTED / f"ep{episode_id}_seed{seed}.npz", allow_pickle=False)
            starts, windows = source["starts"], source["actions"]
            needed = {int(starts[nested["state_index"][g]]) for g in groups}
            clean = {}
            sim.reset(seed)
            for step, action in enumerate(episode_actions):
                if step in needed:
                    clean[step] = (sim.snapshot(), sim.data.qacc_warmstart.copy())
                    if len(clean) == len(needed):
                        break
                sim.execute(action)
            for g in groups:
                state, probe = int(nested["state_index"][g]), int(nested["probe_index"][g])
                contact = int(nested["contact_step"][g])
                snapshot, warm = clean[int(starts[state])]
                sim.restore(snapshot); sim.data.qacc_warmstart[:] = warm
                for action in windows[state, probe, 2:2 + contact]:
                    sim.execute(action)
                branch, branch_warm = sim.snapshot(), sim.data.qacc_warmstart.copy()

                def restore():
                    sim.restore(branch); sim.data.qacc_warmstart[:] = branch_warm
                paths = nested["actions"][g].reshape(-1, 38, 4)
                for name, sigma, cousin in conditions():
                    traces = counterfactuals(sim, restore, paths, (1, int(episode_id), seed, g),
                                             sigma, factors[cousin])
                    result[f"g{g}_{name}"] = traces.reshape(6, 2, 38, 61).astype(np.float32)
    finally:
        sim.close()
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **result)
    tmp.replace(path)
    return str(path), len(result)


def jobs(output):
    data = np.load(NEIGHBOURHOODS, allow_pickle=False)
    sources = {}
    for row, (episode, seed) in enumerate(zip(data["episode_id"], data["seed"])):
        if in_axes(episode, seed):
            sources.setdefault((str(episode), int(seed)), [[], []])[0].append(row)
    for path in sorted(NESTED.glob("ep*_seed*.npz")):
        episode, seed = path.stem.split("_")[0][2:], int(path.stem.split("seed")[1])
        if in_axes(episode, seed):
            count = len(np.load(path, allow_pickle=False)["start"])
            sources.setdefault((episode, seed), [[], []])[1].extend(range(count))
    return [(e, s, rows, groups, str(output)) for (e, s), (rows, groups) in sorted(sources.items())]


# ---------------------------------------------------------------------------------------------
# Scoring


def score(output, workers):
    import torch
    from jenga_reobservation_interface import binary_metrics, compare_nested, nested_stage_rows
    from program_a_feasibility import _init_worker, candidate_metrics, evaluate_candidates
    state_scale = torch.load(STATE_SCALE_MODEL, map_location="cpu",
                             weights_only=False)["state_scale"].numpy()
    data = np.load(NEIGHBOURHOODS, allow_pickle=False)
    names = [c[0] for c in conditions()]
    axes = {"episode": lambda e, s: str(e) in EPISODE_AXIS,
            "configuration": lambda e, s: int(s) in CONFIG_SEEDS}
    stored = {}
    for path in Path(output).glob("ep*_seed*.npz"):
        stem = path.stem
        with np.load(path, allow_pickle=False) as f:
            stored[stem] = {k: f[k] for k in f.files}
    pool = ProcessPoolExecutor(workers, initializer=_init_worker)
    result = {"conditions": names, "friction_factors": friction_factors().tolist(), "axes": {}}
    floor = {"neighbourhood_max_abs": 0.0, "nested_max_abs": 0.0}
    for axis, keep in axes.items():
        rows = [r for r, (e, s) in enumerate(zip(data["episode_id"], data["seed"])) if keep(e, s)]
        nested = []
        for path in sorted(NESTED.glob("ep*_seed*.npz")):
            episode, seed = path.stem.split("_")[0][2:], int(path.stem.split("seed")[1])
            if keep(episode, seed):
                f = np.load(path, allow_pickle=False)
                nested += [(path.stem, g, f["actions"][g], f["traces"][g]) for g in range(len(f["start"]))]
        source = lambda r: f"ep{data['episode_id'][r]}_seed{int(data['seed'][r])}"
        actions = data["actions"][rows]
        physical = evaluate_candidates(pool, actions, data["traces"][rows])
        nested_actions = np.stack([n[2] for n in nested])
        physical_nested = nested_stage_rows(nested_actions, np.stack([n[3] for n in nested]), state_scale)
        per_condition, decisions = {}, {}
        for name in names:
            traces = np.stack([stored[source(r)][f"n{r}_{name}"] for r in rows])
            nested_traces = np.stack([stored[n[0]][f"g{n[1]}_{name}"] for n in nested])
            if name == "sigma_0":
                floor["neighbourhood_max_abs"] = max(floor["neighbourhood_max_abs"], float(
                    np.abs(traces[..., :45] - data["traces"][rows][..., :45]).max()))
                floor["nested_max_abs"] = max(floor["nested_max_abs"], float(np.abs(
                    nested_traces[..., :45] - np.stack([n[3] for n in nested])[..., :45]).max()))
            candidates = evaluate_candidates(pool, actions, traces)
            stages = nested_stage_rows(nested_actions, nested_traces, state_scale)
            decisions[name] = ([c[0] if c is not None else None for c in candidates],
                               [s["alarm"] for s in stages])
            per_condition[name] = {"candidate": candidate_metrics(physical, candidates),
                                   "nested": compare_nested(physical_nested, stages)}
        cousin_names = ["sigma_2.65"] + [f"cousin_{k}" for k in range(1, EXTRA_COUSINS + 1)]
        final = np.array([decisions[n][1] for n in cousin_names])
        candidate = np.array([[x if x is not None else False for x in decisions[n][0]]
                              for n in cousin_names])
        majority = final.mean(0) > 0.5
        result["axes"][axis] = {
            "neighbourhood_states": len(rows), "nested_groups": len(nested),
            "physical_candidate_positives": int(sum(p is not None and p[0] for p in physical)),
            "physical_final_positives": int(sum(r["alarm"] for r in physical_nested)),
            "conditions": per_condition,
            "cousins": {
                "final_disagreement_rate": float(np.mean(final.min(0) != final.max(0))),
                "candidate_disagreement_rate": float(np.mean(candidate.min(0) != candidate.max(0))),
                "majority_final": binary_metrics([r["alarm"] for r in physical_nested], majority.tolist()),
                "disagreement_on_physical_positives": float(np.mean(
                    (final.min(0) != final.max(0))[[r["alarm"] for r in physical_nested]]))
                if any(r["alarm"] for r in physical_nested) else None,
            },
        }
    pool.shutdown()
    result["replay_floor"] = floor
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-workers", type=int, default=16)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=0, help="first N sources only (smoke test)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("simulator work is capped at four workers")
    protocol = json.loads(PROTOCOL.read_text())
    if not protocol["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.simulate:
        todo = jobs(args.output)[:args.limit or None]
        with ProcessPoolExecutor(args.workers) as pool:
            for number, (path, count) in enumerate(pool.map(simulate_source, todo), 1):
                print(f"{number}/{len(todo)} {Path(path).name} ({count} condition sets)", flush=True)
    if args.score:
        result = score(args.output, args.score_workers)
        result.update({"protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                       "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        RESULT.write_text(json.dumps(result, indent=2) + "\n")
        for axis, row in result["axes"].items():
            for name, c in row["conditions"].items():
                print(axis, name, "candidate recall %.3f added %.3f | final recall %.3f added %.3f" % (
                    c["candidate"]["recall"], c["candidate"]["added_positive_rate"],
                    c["nested"]["alarm"]["recall"], c["nested"]["alarm"]["added_positive_rate"]))
            print(axis, "cousins", json.dumps(row["cousins"]))
        print("replay floor", result["replay_floor"])


if __name__ == "__main__":
    main()
