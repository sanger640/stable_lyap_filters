"""Step 3 iteration 2 on fresh panel 5: generate the panel (screen, truths, start frames, oracle start
latents) and grade the uncertain-start world-model monitor with its controls.

Frozen protocol: results/jenga/step3b_uncertain_start_protocol.json. The panel is graded only if P1 passed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import visual_pipeline_panel3 as p3  # noqa: E402

PROTOCOL = ROOT / "results/jenga/step3b_uncertain_start_protocol.json"
OUT = ROOT / "results/program_b/panel5"
RESULT = ROOT / "results/jenga/step3b_panel5_result.json"
P1_RESULT = ROOT / "results/jenga/step3b_p1_result.json"
RESET_BASE, EPISODES, SELECT_SEED = 9000, 80, 20261070
SEEDS = {"monitor_d": 20261072, "truth_A": 20261074, "truth_B": 20261075, "hat": 20261076}


def _set_base():
    import fresh_fragility_panel as fp
    fp.RESET_BASE = RESET_BASE


def screen_panel(workers):
    import fresh_fragility_panel as fp
    from concurrent.futures import ProcessPoolExecutor
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    screen_file = OUT / "screen.json"
    if not screen_file.exists():
        replay = JengaReplay(DEFAULT_LMDB); ids = list(replay.episode_ids); replay.close()
        episodes = sorted(np.random.default_rng(SELECT_SEED).choice(ids, EPISODES, replace=False).tolist(), key=int)
        with ProcessPoolExecutor(workers, initializer=_set_base) as pool:
            screen = {e: s for e, s in pool.map(fp.chain_episode, [(e, set(), str(OUT)) for e in episodes])}
        screen_file.write_text(json.dumps(screen) + "\n")
    screen = json.loads(screen_file.read_text())
    states = [(e, int(i), v["topples"]) for e, rows in screen.items() for i, v in rows.items()]
    forks = [s for s in states if 2 <= s[2] <= 62]
    safe = [s for s in states if s[2] == 0]
    pick = np.random.default_rng(SELECT_SEED + 1)
    forks = [forks[i] for i in sorted(pick.choice(len(forks), min(60, len(forks)), replace=False))]
    safe = [safe[i] for i in sorted(pick.choice(len(safe), min(100, len(safe)), replace=False))]
    return {(e, i): ("topple_fork" if t >= 2 else "safe") for e, i, t in forks + safe}, screen, len(states)


def worker(job):
    import traceback
    try:
        return job[0], _worker(job)
    except Exception:
        return job[0], traceback.format_exc()


def _worker(job):
    episode_id, selected = job
    import mujoco
    import program_b_reference as pr
    import uncertainty_monitor_dev as um
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim
    from step3_data import Latent
    from visual_perception_model import Perceiver
    perceive, latent, errors = Perceiver(), Latent(), p3.ErrorModel()
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    starts = set(chunk_starts(len(actions)))
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="panel5_"))))
    close, camera = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco)
    render = p3.rollout_render
    try:
        sim.reset(int(episode_id) + RESET_BASE)
        for index, action in enumerate(actions):
            if index in starts and index + 8 <= len(actions):
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                exact = [render(sim, snapshot, chunk, n, [], False)[0] for n in snippets]
                warm_end = sim.data.qacc_warmstart.copy()
                if index in selected:
                    key = (int(episode_id) + RESET_BASE, index)
                    sim.restore(snapshot)
                    close.update_scene(sim.data, camera=camera)
                    close_frame = close.render().copy()
                    hat = p3.delta_to(sim, perceive(close_frame[None])[0])
                    traces = {"exact": np.stack(exact).astype(np.float32),
                              "naive": np.stack([render(sim, snapshot, chunk, n, [hat], False)[0]
                                                 for n in snippets]).astype(np.float32)}
                    oracle = []
                    for name in ("truth_A", "truth_B"):
                        out = []
                        for j, n in enumerate(snippets):
                            d = errors.sample(SEEDS[name], *key, j)
                            if name == "truth_A":                     # oracle: render the perturbed start
                                sim.restore(snapshot)
                                pr.perturb_blocks(sim, *d)
                                close.update_scene(sim.data, camera=camera)
                                oracle.append(close.render().copy())
                            out.append(render(sim, snapshot, chunk, n, [d], False)[0])
                        traces[name] = np.stack(out).astype(np.float32)
                    path = OUT / f"ep{episode_id}_s{index}.npz"
                    np.savez_compressed(path.with_suffix(".tmp.npz"), close_frame=close_frame, chunk=chunk,
                                        oracle_z0=latent(np.stack(oracle)), **traces)
                    path.with_suffix(".tmp.npz").replace(path)
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm_end
                if index == max(selected):
                    break
            sim.execute(action)
    finally:
        close.close()
        sim.close()
    return None


def generate(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    panel, screen, screened = screen_panel(workers)
    print(f"screened {screened} states; panel {len(panel)}", flush=True)
    by_episode = {}
    for e, i in panel:
        by_episode.setdefault(e, set()).add(i)
    todo = [(e, idx) for e, idx in sorted(by_episode.items(), key=lambda x: int(x[0]))
            if not all((OUT / f"ep{e}_s{i}.npz").exists() for i in idx)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        failures = []
        for number, (e, error) in enumerate(pool.imap_unordered(worker, todo), 1):
            print(f"{number}/{len(todo)} ep{e}" + (" FAILED" if error else ""), flush=True)
            if error:
                failures.append(e); print(error, flush=True)
    if failures:
        raise SystemExit(f"failed episodes: {failures}")
    return panel, screen, screened


def grade(panel, screen, screened):
    import torch
    import fragility_grading_dev as fg
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings, wilson
    from step3_data import Latent
    p1 = json.loads(P1_RESULT.read_text())
    if not p1["P1_pass"]:
        raise SystemExit("P1 failed: panel 5 is not graded")
    keys = sorted(panel, key=lambda k: (int(k[0]), k[1]))
    snippets = np.load(um.BENCH)["snippets"]
    latent, errors = Latent(), p3.ErrorModel()
    g = sm.G.load(sm.DATA / "G.pt")
    gauss = torch.load(sm.DATA / "G_gauss.pt")
    full = wm.WorldModel.load(sm.DATA / "dyn_full.pt")
    stochastic1 = wm.WorldModel.load(ROOT / "results/program_b/step3/wm_stochastic.pt")
    gen = torch.Generator(device=sm.DEVICE).manual_seed(20261077)
    names = ("wm_G", "wm_noG", "wm_gauss", "wm_oracle", "iter1_stochastic", "exact", "naive", "truth_A", "truth_B")
    dec = {n: [] for n in names}
    harm = []
    for n, (e, i) in enumerate(keys):
        key = (int(e) + RESET_BASE, i)
        with np.load(OUT / f"ep{e}_s{i}.npz") as f:
            z0 = torch.as_tensor(latent(f["close_frame"][None])[0], device=sm.DEVICE).float()
            acts = wm.panel_actions(f["chunk"], snippets)
            d = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in
                                       (errors.sample(SEEDS["monitor_d"], *key, j) for j in range(64))]),
                             dtype=torch.float32)
            starts = {"wm_G": g(z0[None].expand(64, -1, -1, -1), d),
                      "wm_noG": z0[None].expand(64, -1, -1, -1),
                      "wm_gauss": sm.gauss_starts(gauss, z0, 64, gen),
                      "wm_oracle": torch.as_tensor(f["oracle_z0"], device=sm.DEVICE).float()}
            for name, st in starts.items():
                dec[name].append(bool(persistent_mode_alarm(*sm.roll(full, st, acts)).alarm))
            dec["iter1_stochastic"].append(bool(persistent_mode_alarm(
                *wm.wm_rollouts(stochastic1, z0.cpu().numpy(), acts, seed=20261078 + n)).alarm))
            for name in ("exact", "naive", "truth_A", "truth_B"):
                dec[name].append(decide(*trace_endings(f[name], NON_ROBOT["jenga"]))[0])
            harm.append(2 <= topple_count(f["truth_A"], np.array(screen[e][str(i)]["start_tilt"])) <= 62)
    truth_a, harm = np.array(dec["truth_A"]), np.array(harm)
    graded = {n: fg.grade(dec[n], truth_a) for n in names if n != "truth_A"}

    def harm_stats(name):
        dd = np.array(dec[name]); hits = int((dd & harm).sum())
        return {"recall": hits / harm.sum() if harm.sum() else None, "recall_wilson": wilson(hits, int(harm.sum())),
                "alarm_on_non_harm": float((dd & ~harm).sum() / max((~harm).sum(), 1))}

    harms = {n: harm_stats(n) for n in names if not n.startswith("truth")}
    w1 = bool(harms["wm_G"]["recall"] is not None and harms["wm_G"]["recall"] >= 0.80
              and harms["wm_G"]["alarm_on_non_harm"] <= 0.20)
    w2 = bool(graded["wm_G"]["kappa"] >= 0.60)
    w3 = bool(graded["wm_G"]["kappa"] - graded["wm_noG"]["kappa"] >= 0.15)
    result = {
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "screened_states": screened, "panel_states": len(keys), "harm_fork_states": int(harm.sum()),
        "truth_fragile_A": int(truth_a.sum()), "P1": p1["summary"]["full_G"], "graded_vs_truth_A": graded,
        "harm": harms, "gates": {"P1": True, "W1_harm": w1, "W2_fragility": w2, "W3_uncertain_start_matters": w3},
        "confirmed": bool(w1 and w2 and w3),
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("harm", "gates", "confirmed")}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("generate", "grade"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    panel, screen, screened = generate(args.workers) if args.stage == "generate" else screen_panel(args.workers)
    if args.stage == "grade":
        grade(panel, screen, screened)


if __name__ == "__main__":
    main()
