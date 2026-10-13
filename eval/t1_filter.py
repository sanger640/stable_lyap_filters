"""T1 (three-block pick) safety filter: monitor-in-the-loop search over small offsets of the intended chunk.

Frozen protocol: results/jenga/t1_filter_protocol.json.
  screen   chains on a fresh panel (reset +12000), exact-probe topple counts at every chunk start (CPU)
  states   replay the selected chains and save the full simulator state + close-camera frame (CPU, no encoding)
  truth    for every state x candidate offset: 64 truth futures (perception draw + bench snippet j) -> topple count (CPU)
  monitor  US + G starts + H30_floor readout on every candidate; engine reference = truth_B split (GPU)
  analyse  policies: no filter / ours / engine; topple probability, deviation, abstentions
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

PROTOCOL = ROOT / "results/jenga/t1_filter_protocol.json"
OUT = ROOT / "results/program_b/t1_filter"
RESULT = ROOT / "results/jenga/t1_filter_result.json"
CONST = {"RESET_BASE": 12000, "EPISODES": 60, "SELECT_SEED": 20261300,
         "SEEDS": {"monitor_d": 20261301, "truth_A": 20261302, "truth_B": 20261303}}
OFFSETS_MM = [(dx, dy) for dx in (-6, -3, 0, 3, 6) for dy in (-6, -3, 0, 3, 6)]


def _set_base():
    import fresh_fragility_panel as fp
    fp.RESET_BASE = CONST["RESET_BASE"]


def screen_job(job):
    _set_base()
    import fresh_fragility_panel as fp
    return fp.chain_episode(job)


def states_job(job):
    """Mirror of fresh_fragility_panel.chain_episode's chain (exact probes at every start, warm start restored
    to the probes' end), saving the state at the selected starts."""
    import traceback
    try:
        return job[0], _states(job), None
    except Exception:
        return job[0], None, traceback.format_exc()


def _states(job):
    episode_id, selected = job
    import mujoco
    import uncertainty_monitor_dev as um
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, chunk_starts, extract_sim
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    starts = set(chunk_starts(len(actions)))
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="t1f_"))))
    close, camera = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco)
    last = max(selected)
    try:
        sim.reset(int(episode_id) + CONST["RESET_BASE"])
        for index, action in enumerate(actions):
            if index in starts and index + 8 <= len(actions):
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
                for noise in snippets:
                    um.rollout(sim, snapshot, chunk, noise)
                warm_end = sim.data.qacc_warmstart.copy()
                if index in selected:
                    sim.restore(snapshot)
                    close.update_scene(sim.data, camera=camera)
                    frame = close.render().copy()
                    state, ctrl, target, gripper = snapshot
                    np.savez_compressed(OUT / "states" / f"ep{episode_id}_s{index}.npz", state=state, ctrl=ctrl,
                                        target=target, gripper=gripper, warm=warm_end, close_frame=frame,
                                        chunk=chunk, start_tilt=start_tilt)
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm_end
                if index == last:
                    break
            sim.execute(action)
    finally:
        close.close(); sim.close()
    return sorted(selected)


def candidate(chunk, off_mm):
    c = np.asarray(chunk, np.float32).copy()
    c[:, 0] += off_mm[0] / 1000.0; c[:, 1] += off_mm[1] / 1000.0
    return c


def truth_job(path):
    """Topple counts of every candidate under truth_A and truth_B (64 futures each)."""
    import traceback
    try:
        return str(path), _truth(Path(path)), None
    except Exception:
        return str(path), None, traceback.format_exc()


def _truth(path):
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import topple_count
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="t1t_"))))
    e, s = path.stem[2:].split("_s")
    key = (int(e) + CONST["RESET_BASE"], int(s))
    with np.load(path) as f:
        snap = (f["state"], f["ctrl"], f["target"], float(f["gripper"]))
        chunk, tilt = f["chunk"], f["start_tilt"]
    out = {}
    try:
        for off in OFFSETS_MM:
            cand = candidate(chunk, off)
            for name in ("truth_A", "truth_B"):
                traces = [um.rollout(sim, snap, cand, snippets[j], [errors.sample(CONST["SEEDS"][name], *key, j)])
                          for j in range(64)]
                out[f"{off[0]},{off[1]}:{name}"] = int(topple_count(traces, tilt))
    finally:
        sim.close()
    return out


def truth_v2_job(path):
    """v2: topple counts AND the target block's end position for every candidate and future (truth_A, truth_B)."""
    import traceback
    try:
        return str(path), _truth_v2(Path(path)), None
    except Exception:
        return str(path), None, traceback.format_exc()


def _truth_v2(path):
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from fresh_fragility_panel import topple_count
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="t1t_"))))
    e, s = path.stem[2:].split("_s")
    key = (int(e) + CONST["RESET_BASE"], int(s))
    with np.load(path) as f:
        snap = (f["state"], f["ctrl"], f["target"], float(f["gripper"]))
        chunk, tilt = f["chunk"], f["start_tilt"]
    out = {}
    try:
        for off in OFFSETS_MM:
            cand = candidate(chunk, off)
            for name in ("truth_A", "truth_B"):
                traces = [um.rollout(sim, snap, cand, snippets[j], [errors.sample(CONST["SEEDS"][name], *key, j)])
                          for j in range(64)]
                out[f"{off[0]},{off[1]}:{name}"] = int(topple_count(traces, tilt))
                if name == "truth_A":
                    out[f"{off[0]},{off[1]}:target_end"] = [t[-1][:3].round(5).tolist() for t in traces]
    finally:
        sim.close()
    return out


def truth_v3_job(path):
    """v3 (data correction 2026-10-13): as v2 with physically valid perturbations (program_b_reference default) and
    an at-rest flag for the start state."""
    import traceback
    try:
        import program_b_reference as pr
        assert pr.VALID_PERTURBATION
        return str(path), _truth_v3(Path(path)), None
    except Exception:
        return str(path), None, traceback.format_exc()


def _truth_v3(path):
    import program_b_reference as pr
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    import uncertainty_monitor_dev as um
    out = _truth_v2(path)
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="t1r_"))))
    try:
        with np.load(path) as f:
            sim.restore((f["state"], f["ctrl"], f["target"], float(f["gripper"])))
        out["at_rest"] = pr.blocks_at_rest(sim)
    finally:
        sim.close()
    return out


def stage_truth_v3(workers):
    f = OUT / "truth_v3.json"
    truth = json.loads(f.read_text()) if f.exists() else {}
    paths = [str(p) for p in sorted((OUT / "states").glob("*.npz")) if str(p) not in truth]
    if paths:
        truth.update(pool(truth_v3_job, paths, workers, "truth_v3"))
        f.write_text(json.dumps(truth) + "\n")
    print("truth_v3 done", flush=True)


def pool(fn, jobs, workers, label):
    res, fail = {}, []
    with mp.get_context("spawn").Pool(workers, initializer=_set_base) as p:
        for n, (k, v, err) in enumerate(p.imap_unordered(fn, jobs), 1):
            if err:
                fail.append(k); print(err, flush=True)
            else:
                res[k] = v
            if n % 10 == 0 or n == len(jobs):
                print(f"{label} {n}/{len(jobs)}", flush=True)
    if fail:
        raise SystemExit(f"{label} failures: {fail[:5]}")
    return res


def stage_cpu(workers):
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    (OUT / "states").mkdir(parents=True, exist_ok=True)
    screen_file = OUT / "screen.json"
    if not screen_file.exists():
        replay = JengaReplay(DEFAULT_LMDB); ids = list(replay.episode_ids); replay.close()
        eps = sorted(np.random.default_rng(CONST["SELECT_SEED"]).choice(ids, CONST["EPISODES"], replace=False).tolist(), key=int)
        with mp.get_context("spawn").Pool(workers, initializer=_set_base) as p:
            screen = dict(p.map(screen_job, [(e, set(), str(OUT)) for e in eps]))
        screen_file.write_text(json.dumps(screen) + "\n")
    screen = json.loads(screen_file.read_text())
    rows = [(e, int(i), v["topples"]) for e, r in screen.items() for i, v in r.items()]
    rng = np.random.default_rng(CONST["SELECT_SEED"] + 1)
    forks = [r for r in rows if 2 <= r[2] <= 62]; safe = [r for r in rows if r[2] == 0]
    forks = [forks[i] for i in sorted(rng.permutation(len(forks))[:60])]
    safe = [safe[i] for i in sorted(rng.permutation(len(safe))[:40])]
    sel = {}
    for e, i, _ in forks + safe:
        sel.setdefault(e, set()).add(i)
    (OUT / "selection.json").write_text(json.dumps({"forks": len(forks), "safe": len(safe), "screened": len(rows),
                                                    "states": [f"{e}:{i}" for e, i, _ in forks + safe]}) + "\n")
    todo = [(e, idx) for e, idx in sorted(sel.items(), key=lambda x: int(x[0]))
            if not all((OUT / "states" / f"ep{e}_s{i}.npz").exists() for i in idx)]
    if todo:
        pool(states_job, todo, workers, "states")
    truth_file = OUT / "truth.json"
    truth = json.loads(truth_file.read_text()) if truth_file.exists() else {}
    paths = [str(p) for p in sorted((OUT / "states").glob("*.npz")) if str(p) not in truth]
    if paths:
        truth.update(pool(truth_job, paths, workers, "truth"))
        truth_file.write_text(json.dumps(truth) + "\n")
    print("cpu stages done", flush=True)


def stage_truth_v2(workers):
    f = OUT / "truth_v2.json"
    truth = json.loads(f.read_text()) if f.exists() else {}
    paths = [str(p) for p in sorted((OUT / "states").glob("*.npz")) if str(p) not in truth]
    if paths:
        truth.update(pool(truth_v2_job, paths, workers, "truth_v2"))
        f.write_text(json.dumps(truth) + "\n")
    print("truth_v2 done", flush=True)


def stage_monitor_v2():
    """v2: for every candidate, alarm (as v1) and the distance of its mean predicted hold-30 future to the centroid
    of the intended action's majority group."""
    import torch
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from outcome_modes import two_mode_test
    from phase2_amend4 import alarm
    from step3_data import Latent
    res1e = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    model, tau = wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"), res1e["US"]["tau"]
    g, latent, errors = sm.G.load(sm.DATA / "G.pt"), Latent(), p3.ErrorModel()
    snippets = np.load(um.BENCH)["snippets"]
    out = {}
    with torch.no_grad():
        for path in sorted((OUT / "states").glob("*.npz")):
            e, s = path.stem[2:].split("_s")
            key = (int(e) + CONST["RESET_BASE"], int(s))
            with np.load(path) as f:
                z0 = torch.as_tensor(latent(f["close_frame"][None])[0], device="cuda").float()
                chunk = f["chunk"]
            d = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in
                                       (errors.sample(CONST["SEEDS"]["monitor_d"], *key, j) for j in range(64))]), dtype=torch.float32)
            starts = g(z0[None].expand(64, -1, -1, -1), d)
            ends = {}
            for off in OFFSETS_MM:
                ends[off] = sm.roll(model, starts, wm.panel_actions(candidate(chunk, off), snippets))
            e10, e30 = ends[(0, 0)]
            lab = two_mode_test(e30, 0.0).labels
            major = 1 if lab.sum() > len(lab) / 2 else 0
            centroid = e30[lab == major].mean(0)
            out[str(path)] = {f"{o[0]},{o[1]}": {"alarm": bool(alarm(*ends[o], "H30_floor", tau)),
                                                 "dist": float(np.linalg.norm(ends[o][1].mean(0) - centroid))} for o in OFFSETS_MM}
            torch.cuda.empty_cache()
    (OUT / "monitor_v2.json").write_text(json.dumps(out) + "\n")
    print("monitor_v2 done", flush=True)


def analyse_v3():
    """analyse_v2 on corrected truth (truth_v3), starts not at rest excluded; monitor unchanged (monitor_v2)."""
    global RESULT_V2
    truth = json.loads((OUT / "truth_v3.json").read_text())
    excluded = [Path(p).stem for p, t in truth.items() if not t["at_rest"]]
    RESULT_V2 = ROOT / "results/jenga/t1_filter_v3_result.json"
    analyse_v2({p: t for p, t in truth.items() if t["at_rest"]}, extra={"not_at_rest_excluded": excluded})


RESULT_V2 = ROOT / "results/jenga/t1_filter_v2_result.json"


def analyse_v2(truth=None, extra=None):
    truth = truth if truth is not None else json.loads((OUT / "truth_v2.json").read_text()); mon = json.loads((OUT / "monitor_v2.json").read_text())
    order = sorted(OFFSETS_MM, key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    k = lambda o: f"{o[0]},{o[1]}"
    rows = []
    for path, t in truth.items():
        m = mon[path]
        tgt0 = np.array(t["0,0:target_end"])
        dev = lambda o: float(np.median(np.linalg.norm(np.array(t[f"{k(o)}:target_end"]) - tgt0, axis=1)) * 1000)
        topp = lambda o: None if o is None else t[f"{k(o)}:truth_A"] / 64.0
        fork = 2 <= t["0,0:truth_A"] <= 62
        # v1 rules (same data, re-derived)
        ours1 = next((o for o in order if not m[k(o)]["alarm"]), None)
        eng1 = next((o for o in order if not (2 <= t[f"{k(o)}:truth_B"] <= 62)), None)
        # v2 rules
        if not m["0,0"]["alarm"]:
            ours2 = (0, 0)
        else:
            ok = [o for o in order if not m[k(o)]["alarm"]]
            ours2 = min(ok, key=lambda o: (m[k(o)]["dist"], o[0] ** 2 + o[1] ** 2)) if ok else None
        maj0 = t["0,0:truth_B"] >= 32
        ok_e = [o for o in order if not (2 <= t[f"{k(o)}:truth_B"] <= 62) and ((t[f"{k(o)}:truth_B"] >= 32) == maj0)]
        eng2 = ok_e[0] if ok_e else None
        r = {"state": Path(path).stem, "fork": fork, "none": topp((0, 0))}
        for name, o in (("ours_v1", ours1), ("engine_v1", eng1), ("ours_v2", ours2), ("engine_v2", eng2)):
            r[name] = topp(o); r[name + "_off"] = o; r[name + "_dev_mm"] = None if o is None else dev(o)
        rows.append(r)
    out = {}
    for grp, sel in (("forks", [r for r in rows if r["fork"]]), ("safe", [r for r in rows if not r["fork"]])):
        g = {"states": len(sel), "topple_no_filter": float(np.mean([r["none"] for r in sel]))}
        for pol in ("ours_v1", "engine_v1", "ours_v2", "engine_v2"):
            acted = [r for r in sel if r[pol] is not None]
            g[pol] = {"abstain": len(sel) - len(acted),
                      "topple_when_acting": float(np.mean([r[pol] for r in acted])) if acted else None,
                      "topple_abstain_as_safe": float(np.mean([r[pol] if r[pol] is not None else 0.0 for r in sel])),
                      "changed": sum(1 for r in acted if tuple(r[pol + "_off"]) != (0, 0)),
                      "median_task_deviation_mm": float(np.median([r[pol + "_dev_mm"] for r in acted])) if acted else None,
                      "certain_topple_choices": sum(1 for r in acted if r[pol] >= 0.97)}
        out[grp] = g
    out.update(extra or {})
    out["rows"] = rows
    RESULT_V2.write_text(json.dumps(out, indent=1, default=list) + "\n")
    print(json.dumps({k2: v for k2, v in out.items() if k2 != "rows"}, indent=1), flush=True)


def stage_monitor():
    """GPU: US + G starts + H30_floor alarm for every candidate (the panel-confirmed monitor)."""
    import torch
    import step3_world_model as wm
    import step3b_models as sm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from phase2_amend4 import alarm
    from step3_data import Latent
    res1e = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    model, tau = wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"), res1e["US"]["tau"]
    g, latent, errors = sm.G.load(sm.DATA / "G.pt"), Latent(), p3.ErrorModel()
    snippets = np.load(um.BENCH)["snippets"]
    out = {}
    with torch.no_grad():
        for path in sorted((OUT / "states").glob("*.npz")):
            e, s = path.stem[2:].split("_s")
            key = (int(e) + CONST["RESET_BASE"], int(s))
            with np.load(path) as f:
                z0 = torch.as_tensor(latent(f["close_frame"][None])[0], device="cuda").float()
                chunk = f["chunk"]
            d = torch.tensor(np.stack([np.concatenate([p.reshape(9), r.reshape(9)]) for p, r in
                                       (errors.sample(CONST["SEEDS"]["monitor_d"], *key, j) for j in range(64))]), dtype=torch.float32)
            starts = g(z0[None].expand(64, -1, -1, -1), d)
            out[str(path)] = {}
            for off in OFFSETS_MM:
                e10, e30 = sm.roll(model, starts, wm.panel_actions(candidate(chunk, off), snippets))
                out[str(path)][f"{off[0]},{off[1]}"] = bool(alarm(e10, e30, "H30_floor", tau))
            torch.cuda.empty_cache()
    (OUT / "monitor.json").write_text(json.dumps(out) + "\n")
    print("monitor done", flush=True)


def analyse():
    truth = json.loads((OUT / "truth.json").read_text()); mon = json.loads((OUT / "monitor.json").read_text())
    order = sorted(OFFSETS_MM, key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    rows = []
    for path, t in truth.items():
        k0 = t["0,0:truth_A"]
        fork = 2 <= k0 <= 62
        def pick(alarm_of):
            for o in order:
                if not alarm_of(o):
                    return o
            return None
        ours = pick(lambda o: mon[path][f"{o[0]},{o[1]}"])
        engine = pick(lambda o: 2 <= t[f"{o[0]},{o[1]}:truth_B"] <= 62)
        def outcome(o):
            return None if o is None else t[f"{o[0]},{o[1]}:truth_A"] / 64.0
        rows.append({"state": Path(path).stem, "fork": fork, "none": k0 / 64.0, "ours": outcome(ours), "ours_off": ours,
                     "engine": outcome(engine), "engine_off": engine, "ours_alarm_at_0": mon[path]["0,0"]})
    out = {}
    for grp, sel in (("forks", [r for r in rows if r["fork"]]), ("safe", [r for r in rows if not r["fork"]])):
        g = {"states": len(sel), "topple_prob_no_filter": float(np.mean([r["none"] for r in sel]))}
        for pol in ("ours", "engine"):
            acted = [r for r in sel if r[pol] is not None]
            g[f"{pol}_abstain"] = len(sel) - len(acted)
            g[f"{pol}_topple_prob_when_acting"] = float(np.mean([r[pol] for r in acted])) if acted else None
            g[f"{pol}_topple_prob_abstain_as_safe"] = float(np.mean([r[pol] if r[pol] is not None else 0.0 for r in sel]))
            g[f"{pol}_changed"] = sum(r[f"{pol}_off"] not in (None, (0, 0)) and tuple(r[f"{pol}_off"]) != (0, 0) for r in sel)
            g[f"{pol}_mean_offset_mm"] = float(np.mean([np.hypot(*r[f"{pol}_off"]) for r in acted])) if acted else None
        out[grp] = g
    out["rows"] = rows
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cpu", "monitor", "analyse", "truth_v2", "monitor_v2", "analyse_v2", "truth_v3", "analyse_v3"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    {"cpu": lambda: stage_cpu(args.workers), "monitor": stage_monitor, "analyse": analyse,
     "truth_v2": lambda: stage_truth_v2(args.workers), "monitor_v2": stage_monitor_v2, "analyse_v2": analyse_v2,
     "truth_v3": lambda: stage_truth_v3(args.workers), "analyse_v3": analyse_v3}[args.stage]()


if __name__ == "__main__":
    main()
