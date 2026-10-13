"""Three-block retraining on corrected data (data_correction_2026_10_13 in results/jenga/phase1e_protocol.json).

Same training/validation states (episode, reset seed, chunk start), same future seeds and snippet rules as the
original step3b (train 1500, val 150) and phase1 (t16/t8, 3266) data; only the perception-error injection changes
(program_b_reference.VALID_PERTURBATION, the applied error is what is stored as d) and starts not at rest are dropped.
Everything goes to results/program_b/threeblock_v2/ (old data and models kept).

Stages:
  data     render + encode (one pool, GPU workers)       -> V2/train, V2/val, V2/p1train
  g        start sampler G on V2/train                   -> V2/G.pt
  modes    rare-mode detection (label-free)              -> V2/modes.json
  us       plain model US (fast recipe)                  -> V2/model_US.pt, floor on V2/val
  hybrid   flow head on the new US                       -> V2/model_HYB_T1.pt
  grade    corrected H (h_v2), gripper-relative label    -> results/jenga/threeblock_retrain_result.json
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

V2 = ROOT / "results/program_b/threeblock_v2"
RESULT = ROOT / "results/jenga/threeblock_retrain_result.json"


# ------------------------------------------------------------------------------------------------ data
def jobs():
    import step3_data as sd
    spec = {}   # (episode, seed) -> {index: [(folder, n, seed_value, frame_index)]}
    s3 = json.loads((sd.OUT / "selection.json").read_text())["roles"]
    for chain, sel in s3.items():
        e, s = chain.split(":")
        for i, roles in sel.items():
            for r in roles:
                if r in ("train", "val"):
                    spec.setdefault((e, int(s)), {}).setdefault(int(i), []).append(
                        ("train" if r == "train" else "val", sd.K_FUTURES, sd.SEEDS["futures"], sd.FRAME_INDEX))
    p1 = json.loads((ROOT / "results/program_b/phase1/selection.json").read_text())["roles"]
    for chain, sel in p1.items():
        e, s = chain.split(":")
        for i, r in sel.items():
            if r in ("t16", "t8"):
                spec.setdefault((e, int(s)), {}).setdefault(int(i), []).append(
                    ("p1train", {"t16": 16, "t8": 8}[r], 20261082, sd.FRAME_INDEX))
    return sorted(spec.items(), key=lambda x: (int(x[0][0]), x[0][1]))


def data_job(job):
    import traceback
    try:
        return job[0], _data(job), None
    except Exception:
        return job[0], None, traceback.format_exc()


def _data(job):
    (episode_id, seed), selected = job
    import mujoco
    import program_b_reference as pr
    import step3_data as sd
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from step3b_data import future_with_start
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    from jenga_state_data import step_state
    assert pr.VALID_PERTURBATION
    todo = {i: [r for r in rs if not (V2 / r[0] / f"ep{episode_id}_seed{seed}_s{i}.npz").exists()
                and not (V2 / r[0] / f"ep{episode_id}_seed{seed}_s{i}.notrest").exists()] for i, rs in selected.items()}
    todo = {i: rs for i, rs in todo.items() if rs}
    if not todo:
        return {}
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB); actions = np.asarray(replay.episode(episode_id).actions, np.float32); replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="tb2_"))))
    renderer, camera, latent = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent()
    done = {}
    try:
        sim.reset(int(seed))
        for index, action in enumerate(actions):
            if index in todo:
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot(); warm0 = sim.data.qacc_warmstart.copy()
                rest = pr.blocks_at_rest(sim)
                start_tilt = sim.block_diagnostics()["tilt"][1:].copy(); start = step_state(sim).copy()
                key = (int(episode_id), int(seed), int(index))
                for folder, n, fseed, at in todo[index]:
                    path = V2 / folder / f"ep{episode_id}_seed{seed}_s{index}.npz"
                    if not rest:
                        path.with_suffix(".notrest").write_text("")
                        done[f"{folder}:{index}"] = "not_at_rest"; continue
                    sim.restore(snapshot)
                    renderer.update_scene(sim.data, camera=camera); true_start = renderer.render().copy()
                    traces, executed, starts, frames, pos, rot, scales = [], [], [], [], [], [], []
                    for k in range(n):
                        noise = snippets[np.random.default_rng([fseed, *key, k]).integers(len(snippets))]
                        t, a, s0, f = future_with_start(sim, snapshot, chunk, noise, errors.sample(fseed, *key, k), at, renderer, camera)
                        traces.append(t); executed.append(a); starts.append(s0); frames.extend(f)
                        pos.append(sim.last_perturbation[0]); rot.append(sim.last_perturbation[1]); scales.append(sim.last_perturb_scale)
                    z = latent(np.concatenate([true_start[None], np.stack(starts), np.stack(frames)]))
                    np.save(path.with_name(path.stem + ".start.npy"), start)
                    np.savez(path.with_suffix(".tmp.npz"), z0=z[0], z0k=z[1:1 + n], z=z[1 + n:].reshape(n, len(at), 28, 28, 8),
                             d_pos=np.stack(pos), d_rot=np.stack(rot), scales=np.array(scales),
                             traces=np.stack(traces).astype(np.float32), actions=np.stack(executed),
                             start_frame=true_start, start_tilt=start_tilt, frame_index=np.array(at))
                    path.with_suffix(".tmp.npz").replace(path)
                    done[f"{folder}:{index}"] = "ok"
                sim.restore(snapshot); sim.data.qacc_warmstart[:] = warm0
                if index == max(todo):
                    break
            sim.execute(action)
    finally:
        renderer.close(); sim.close()
    return done


def stage_data(workers=6):
    for d in ("train", "val", "p1train"):
        (V2 / d).mkdir(parents=True, exist_ok=True)
    js = jobs(); fails = []; stats = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (key, res, err) in enumerate(pool.imap_unordered(data_job, js), 1):
            if err:
                fails.append(key); print(err, flush=True)
            else:
                for v in res.values():
                    stats[v] = stats.get(v, 0) + 1
            if n % 20 == 0 or n == len(js):
                print(f"data {n}/{len(js)} {stats}", flush=True)
    if fails:
        raise SystemExit(f"failures ({len(fails)}): {fails[:10]}")
    (V2 / "val_recovery.json").write_text(json.dumps({"inexact": []}) + "\n")
    summary = {d: len(list((V2 / d).glob("ep*.npz"))) for d in ("train", "val", "p1train")}
    summary["not_at_rest"] = {d: len(list((V2 / d).glob("*.notrest"))) for d in ("train", "val", "p1train")}
    (V2 / "data_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("data done", json.dumps(summary), flush=True)


# ------------------------------------------------------------------------------------------------ models
def _patch():
    import phase1b_models as p1b
    import phase1e_models as p1e
    import step3b_models as sm
    sm.DATA = V2
    p1b.FOLDERS_ENRICHED = [V2 / "train", V2 / "p1train"]
    p1e.OUT = V2; p1e.MODES = V2 / "modes.json"


def _names_states():
    import phase1b_models as p1b
    import phase1d_models as p1d
    names = [str(p) for f in p1b.FOLDERS_ENRICHED for p in sorted(Path(f).glob("ep*.npz")) if not p.name.endswith(".tmp.npz")]
    states = p1d.load(p1b.FOLDERS_ENRICHED, proprio=False)
    assert len(names) == len(states)
    return names, states


def stage_g():
    _patch()
    import step3b_models as sm
    sm.train_g(sm.load("train"))
    print("G done", flush=True)


def stage_modes():
    _patch()
    import phase1e_models as p1e
    names, states = _names_states()
    modes = p1e.detect_modes(states, names)
    (V2 / "modes.json").write_text(json.dumps(modes) + "\n")
    summary = p1e.describe(states, names, modes)
    (V2 / "modes_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


def stage_us():
    _patch()
    import fast_train as ft
    import phase1d_models as p1d
    names, states = _names_states()
    modes = json.loads((V2 / "modes.json").read_text())
    rare = np.array([k in set(modes[n]) for n, s in zip(names, states) for k in range(len(s["z"]))])
    has_rare = np.array([bool(modes[n]) for n, s in zip(names, states) for _ in range(len(s["z"]))])
    pools = (np.flatnonzero(rare), np.flatnonzero(~rare & has_rare))
    sampler = lambda rng, b: np.concatenate([rng.choice(pools[0], b // 2), rng.choice(pools[1], b - b // 2)])
    model = ft.train("US", V2, states, sampler)
    tau, n_val = p1d.floor(model, False)
    (V2 / "floor_US.json").write_text(json.dumps({"tau": tau, "validation_futures": n_val,
                                                   "possibly_undertrained": ft.undertrained(json.loads((V2 / "model_US_log.json").read_text()))}) + "\n")
    print("US done, floor", tau, flush=True)


def stage_hybrid():
    _patch()
    import phase2_hybrid as hy
    import phase2_models as pm
    names, states = _names_states()
    pm.load_train = lambda: (states, names)
    hy.OUT = V2
    hy.train(name="HYB_T1", det_name="US")
    print("hybrid done", flush=True)


def _hyb_floor(head, det, rs):
    import torch
    import phase1d_models as p1d
    import phase2_hybrid as hy
    import step3_world_model as wm
    errs = []
    for n, path in enumerate(sorted((V2 / "val").glob("ep*.npz"))):
        with np.load(path) as f:
            k = len(f["z"])
            z0 = torch.as_tensor(f["z0"], device="cuda").float()[None].expand(k, -1, -1, -1).contiguous()
            act = det._act(torch.as_tensor(p1d.augment(f["actions"], None, False), device="cuda"))
            real = f["z"][:, wm.HOLD30].reshape(k, -1).astype(np.float64)
        g = torch.Generator(device="cuda").manual_seed(hy.MONITOR_SEED + 10_000 + n)
        e30 = hy.sample(head, det, z0, act, rs, g)[:, 1].reshape(k, -1).double().cpu().numpy()
        errs.extend(np.linalg.norm(e30 - real, axis=1) / np.sqrt(real.shape[1] // 3))
        torch.cuda.empty_cache()
    return float(np.mean(errs))


def stage_grade():
    """Old models (trained on old data) vs retrained, on the corrected H with the gripper-relative label."""
    _patch()
    import torch
    import phase1d_models as p1d
    import phase2_hybrid as hy
    import step3_world_model as wm
    import step3b_models as sm
    from phase2_amend4 import alarm, score
    from regrade_any_outcome import events_three_block_v2, labels
    from fresh_fragility_panel import topple_count
    old_res = json.loads((ROOT / "results/jenga/threeblock_v2_result.json").read_text())
    us = wm.WorldModel.load(V2 / "model_US.pt"); us.net.eval()
    tau = json.loads((V2 / "floor_US.json").read_text())["tau"]
    gop = sm.G.load(V2 / "G.pt")
    hyb = None
    if (V2 / "model_HYB_T1.pt").exists():
        s = torch.load(V2 / "model_HYB_T1.pt", map_location="cuda")
        head = hy.build_head(s["action_dim"]).cuda(); head.load_state_dict(s["state_dict"]); head.eval()
        hyb = (head, s["residual_scale"], _hyb_floor(head, us, s["residual_scale"]))
    rows = []
    with torch.no_grad():
        for n, r in enumerate(old_res["rows"]):
            if not r["at_rest"]:
                continue
            with np.load(ROOT / "results/program_b/phase1/h_v2" / f"{r['state']}.npz") as f:
                f = {k: f[k] for k in f.files}
            af, asafe = labels(events_three_block_v2(f["traces"])); k = topple_count(f["traces"], f["start_tilt"])
            d = np.concatenate([f["d_pos"].reshape(64, 9), f["d_rot"].reshape(64, 9)], 1)
            z0 = torch.as_tensor(f["z0"], device="cuda").float()
            acts = p1d.augment(f["actions"], None, False)
            e10, e30 = sm.roll(us, gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d).float()), acts)
            al = {"old_" + m: r["alarms"][m] for m in ("US_H30_floor", "US_BOTH", "HYB_T1_H30_floor", "HYB_T1_BOTH")}
            al["new_US_H30_floor"] = bool(alarm(e10, e30, "H30_floor", tau)); al["new_US_BOTH"] = bool(alarm(e10, e30, "BOTH"))
            if hyb:
                g = torch.Generator(device="cuda").manual_seed(hy.MONITOR_SEED + n)
                fut = hy.sample(hyb[0], us, z0[None].expand(64, -1, -1, -1).contiguous(), us._act(torch.as_tensor(acts, device="cuda")), hyb[1], g)
                h10 = fut[:, 0].reshape(64, -1).double().cpu().numpy(); h30 = fut[:, 1].reshape(64, -1).double().cpu().numpy()
                al["new_HYB_T1_H30_floor"] = bool(alarm(h10, h30, "H30_floor", hyb[2])); al["new_HYB_T1_BOTH"] = bool(alarm(h10, h30, "BOTH"))
            rows.append({"state": r["state"], "fork": af, "safe": asafe, "harm": 2 <= k <= 62, "harm_safe": k == 0, "alarms": al})
            torch.cuda.empty_cache()
    out = {"floors": {"US_new": tau, "HYB_new": hyb[2] if hyb else None},
           "data": json.loads((V2 / "data_summary.json").read_text())}
    for lab, pk, nk in (("gripper_relative_any_outcome", "fork", "safe"), ("harm", "harm", "harm_safe")):
        sub = [r for r in rows if r[pk] or r[nk]]
        y = np.array([r[pk] for r in sub])
        out[lab] = {"forks": int(y.sum()), "safe": int((~y).sum()),
                    **{m: score(np.array([r["alarms"][m] for r in sub]), y) for m in sub[0]["alarms"]}}
    out["rows"] = rows
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)


if __name__ == "__main__":
    {"data": stage_data, "g": stage_g, "modes": stage_modes, "us": stage_us, "hybrid": stage_hybrid,
     "grade": stage_grade}[sys.argv[1]]()
