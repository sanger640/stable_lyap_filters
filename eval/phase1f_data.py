"""Branching-WM Phase 1f data: label-free curation of training states.

Frozen protocol: results/jenga/phase1f_protocol.json. Screen = new TRAIN chains (reset seeds 190-229), 8 futures
per chunk start rendered only at hold 10 / hold 30 and encoded to the channel-PCA-8 latent; a state is selected
if the unchanged split test (persistent_mode_alarm) alarms on those latents. Topple counts are computed from the
screen traces for REPORTING ONLY. Rendering of the selected states reuses eval/phase1_data.py (as Phase 1b does).
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

PROTOCOL = ROOT / "results/jenga/phase1f_protocol.json"
OUT = ROOT / "results/program_b/phase1f"
NEW_SEEDS = range(190, 230)
SEEDS_1F = {"screen": 20261120, "select": 20261121, "futures": 20261122, "h_select": 20261083, "h_futures": 20261084}
TOTAL, ALARM_CAP = 4_766, 2_383


def _patch():
    p1.SEEDS = dict(SEEDS_1F); p1.OUT = OUT; p1.NEW_SEEDS = NEW_SEEDS


def render_job(job):
    _patch(); return p1.render_job(job)


def screen_job(job):
    import traceback
    try:
        return job[0], job[1], _screen(job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _screen(job):
    episode_id, seed = job
    import mujoco
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    targets = [s for s in chunk_starts(len(actions)) if s + 8 <= len(actions)]
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="phase1f_"))))
    renderer, camera, latent = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent()
    result = {}
    try:
        sim.reset(int(seed))
        for index, action in enumerate(actions):
            if index in targets:
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                warm0 = sim.data.qacc_warmstart.copy()
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                key = (int(episode_id), int(seed), int(index))
                traces, frames = [], []
                for k in range(sd.K_SCREEN):
                    rng = np.random.default_rng([SEEDS_1F["screen"], *key, k])
                    t, _, f = sd.future(sim, snapshot, chunk, snippets[rng.integers(len(snippets))],
                                        errors.sample(SEEDS_1F["screen"], *key, k), sd.D0_INDEX, renderer, camera)
                    traces.append(t); frames.extend(f)
                z = latent(np.stack(frames)).astype(np.float64).reshape(sd.K_SCREEN, 2, -1)
                alarm = bool(persistent_mode_alarm(z[:, 0], z[:, 1]).alarm)
                result[index] = {"alarm": alarm, "topples_reporting_only": int(topple_count(traces, start_tilt))}
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm0
                if index == max(targets):
                    break
            sim.execute(action)
    finally:
        renderer.close()
        sim.close()
    return result


def select(screen, held):
    rows = [(k.split(":")[0], int(k.split(":")[1]), int(i), v["alarm"]) for k, r in screen.items() for i, v in r.items()]
    train = [r for r in rows if r[0] not in held]
    rng = np.random.default_rng(SEEDS_1F["select"])
    alarm = [r for r in train if r[3]]
    other = [r for r in train if not r[3]]
    alarm = [alarm[i] for i in sorted(rng.permutation(len(alarm))[:ALARM_CAP])]
    other = [other[i] for i in sorted(rng.permutation(len(other))[:TOTAL - len(alarm)])]
    roles = {}
    for group, role in ((alarm, "t16"), (other, "t8")):
        for e, s, i, _ in group:
            roles.setdefault((e, s), {})[i] = role
    return roles, {"screened_train_states": len(train), "alarming": sum(r[3] for r in train),
                   "selected_alarming": len(alarm), "selected_other": len(other)}


def describe(screen, held):
    """Reporting only: how the image-split screen relates to topple disagreement."""
    v = [x for k, r in screen.items() if k.split(":")[0] not in held for x in r.values()]
    a = np.array([x["alarm"] for x in v]); t = np.array([x["topples_reporting_only"] for x in v])
    dis = (t > 0) & (t < sd.K_SCREEN)
    return {"states": len(v), "alarm_rate": float(a.mean()), "topple_disagree_rate": float(dis.mean()),
            "alarming_that_topple_disagree": float(dis[a].mean()) if a.any() else None,
            "topple_disagree_that_alarm": float(a[dis].mean()) if dis.any() else None,
            "alarming_with_no_topple_in_any_future": int((a & (t == 0)).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    (OUT / "train").mkdir(parents=True, exist_ok=True)
    chains_all, held = sd.train_chains()
    episodes = sorted({e for e, _ in chains_all}, key=int)
    screen_file = OUT / "screen.json"
    if not screen_file.exists():
        jobs = [(e, s) for e in episodes if e not in held for s in NEW_SEEDS]
        screen = p1.pool_run(screen_job, jobs, args.workers, "screen")
        screen_file.write_text(json.dumps(screen) + "\n")
    screen = json.loads(screen_file.read_text())
    roles, summary = select(screen, held)
    summary["reporting_only"] = describe(screen, held)
    (OUT / "selection.json").write_text(json.dumps({"summary": summary,
                                                    "roles": {f"{e}:{s}": v for (e, s), v in roles.items()}}) + "\n")
    print(json.dumps(summary), flush=True)
    jobs = []
    for (e, s), sel in sorted(roles.items(), key=lambda x: (x[0][1], int(x[0][0]))):
        todo = {i: r for i, r in sel.items() if not (OUT / "train" / f"ep{e}_seed{s}_s{i}.npz").exists()}
        if todo:
            jobs.append((e, s, todo))
    p1.pool_run(render_job, jobs, args.workers, "render")
    print("done", flush=True)


if __name__ == "__main__":
    main()
