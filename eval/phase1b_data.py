"""Branching-WM Phase 1b data: more fork-rich TRAIN chains (reset seeds 150-189) and the start-state
recovery replay (true 61-D start state of every training/H state, verified against stored traces).

Frozen protocol: results/jenga/phase1b_branching_protocol.json. Reuses eval/phase1_data.py with Phase-1b
seeds; recovery replays each data run's chain exactly (same visited states and futures, no rendering).
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import phase1_data as p1  # noqa: E402
import step3_data as sd  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase1b_branching_protocol.json"
OUT = ROOT / "results/program_b/phase1b"
SEEDS_1B = {"screen": 20261100, "select": 20261101, "futures": 20261102, "h_select": 20261083, "h_futures": 20261084}


def _patch():
    p1.SEEDS = dict(SEEDS_1B); p1.OUT = OUT; p1.NEW_SEEDS = range(150, 190)
    sd.SEEDS = dict(sd.SEEDS, screen=SEEDS_1B["screen"])


def screen_job(job):
    _patch(); return p1.screen_job(job)


def render_job(job):
    _patch(); return p1.render_job(job)


# ------------------------------------------------------------------------------ recovery
RUNS = {   # data run -> (selection file, output folders, per-role (futures, seed, snippet rule, frame index))
    "step3b": (ROOT / "results/program_b/step3/selection.json", {"train": ROOT / "results/program_b/step3b/train"},
               {"train": (8, 20261041, "random"), "val": (8, 20261041, "random"), "d0": (64, 20261046, "index")}),
    "phase1": (ROOT / "results/program_b/phase1/selection.json",
               {"t16": ROOT / "results/program_b/phase1/train", "t8": ROOT / "results/program_b/phase1/train",
                "h": ROOT / "results/program_b/phase1/h"},
               {"t16": (16, 20261082, "random"), "t8": (8, 20261082, "random"), "h": (64, 20261084, "index")}),
    "phase1b": (OUT / "selection.json", {"t16": OUT / "train", "t8": OUT / "train"},
                {"t16": (16, 20261102, "random"), "t8": (8, 20261102, "random")}),
}


def recover_job(job):
    import traceback
    try:
        return job[1], job[2], _recover(job), None
    except Exception:
        return job[1], job[2], None, traceback.format_exc()


def _recover(job):
    run, episode_id, seed, roles = job
    import program_b_reference as pr
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    from jenga_state_data import step_state
    _, folders, spec = RUNS[run]
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="recover_"))))
    report = {}
    try:
        sim.reset(int(seed))
        for index, action in enumerate(actions):
            if index in roles:
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                warm0 = sim.data.qacc_warmstart.copy()
                sim.block_diagnostics()
                key = (int(episode_id), int(seed), int(index))
                for role in (roles[index] if isinstance(roles[index], list) else [roles[index]]):
                    sim.restore(snapshot)
                    start = step_state(sim).copy()
                    n, fseed, rule = spec[role]
                    traces = []
                    for k in range(n):
                        noise = snippets[k] if rule == "index" else snippets[
                            np.random.default_rng([fseed, *key, k]).integers(len(snippets))]
                        d = errors.sample(fseed, *key, k)
                        traces.append(sd.future(sim, snapshot, chunk, noise, d)[0])
                    if role in folders:
                        target = folders[role] / f"ep{episode_id}_seed{seed}_s{index}.npz"
                        if target.exists():
                            with np.load(target) as f:
                                diff = float(np.abs(np.stack(traces) - f["traces"]).max())
                            out = target.with_name(target.stem + ".start.npy")
                            np.save(out, start)
                            report[f"{role}:{index}"] = diff
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm0
                if index == max(roles):
                    break
            sim.execute(action)
    finally:
        sim.close()
    return report


def recovery_jobs():
    jobs = []
    for run, (sel_file, _, _) in RUNS.items():
        if not Path(sel_file).exists():
            continue
        roles = json.loads(Path(sel_file).read_text())["roles"]
        for chain, sel in roles.items():
            e, s = chain.split(":")
            jobs.append((run, e, int(s), {int(i): r for i, r in sel.items()}))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    _patch()
    (OUT / "train").mkdir(parents=True, exist_ok=True)
    chains_all, held = sd.train_chains()
    episodes = sorted({e for e, _ in chains_all if e not in held}, key=int)
    screen_file = OUT / "screen.json"
    if not screen_file.exists():
        screen = p1.pool_run(screen_job, [(e, s) for e in episodes for s in range(150, 190)], args.workers, "screen")
        screen_file.write_text(json.dumps(screen) + "\n")
    screen = json.loads(screen_file.read_text())
    rows = [(k.split(":")[0], int(k.split(":")[1]), int(i), v["topples"]) for k, r in screen.items() for i, v in r.items()]
    rng = np.random.default_rng(SEEDS_1B["select"])
    dis = [r for r in rows if 0 < r[3] < sd.K_SCREEN]
    oth = [r for r in rows if not 0 < r[3] < sd.K_SCREEN]
    oth = [oth[i] for i in rng.permutation(len(oth))[:len(dis)]]
    roles = {}
    for group, role in ((dis, "t16"), (oth, "t8")):
        for e, s, i, _ in group:
            roles.setdefault((e, s), {})[i] = role
    (OUT / "selection.json").write_text(json.dumps({"summary": {"screened": len(rows), "disagree": len(dis)},
                                                    "roles": {f"{e}:{s}": v for (e, s), v in roles.items()}}) + "\n")
    print(json.dumps({"screened": len(rows), "disagree": len(dis)}), flush=True)
    jobs = [(e, s, {i: r for i, r in sel.items() if not (OUT / "train" / f"ep{e}_seed{s}_s{i}.npz").exists()})
            for (e, s), sel in sorted(roles.items(), key=lambda x: (x[0][1], int(x[0][0])))]
    jobs = [j for j in jobs if j[2]]
    p1.pool_run(render_job, jobs, args.workers, "render")
    report = p1.pool_run(recover_job, recovery_jobs(), args.workers, "recover")
    diffs = [d for r in report.values() for d in r.values()]
    summary = {"states": len(diffs), "exact_lt_1e-6": int(sum(d < 1e-6 for d in diffs)), "max_diff": float(max(diffs))}
    (OUT / "recovery.json").write_text(json.dumps({"summary": summary, "per_chain": report}) + "\n")
    print(json.dumps(summary), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
