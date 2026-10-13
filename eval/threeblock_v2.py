"""Three-block selection set H with the 2026-10-13 data correction (physically valid perturbations, at-rest starts).

Same 160 states (episode, seed, chunk start), futures seed and snippet rule as results/program_b/phase1/h; only the
perception-error injection changes (program_b_reference.VALID_PERTURBATION). Stages:
  sim     CPU pool: 64 futures per state + 16 robot-still futures (unit test); frames saved for encoding
  encode  GPU, one process: DINOv2 latents -> results/program_b/phase1/h_v2/<state>.npz (same keys as h/ + extras)
  grade   re-grade the existing three-block monitors on h_v2 (any-outcome primary, harm beside)
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

OLD = ROOT / "results/program_b/phase1/h"
NEW = ROOT / "results/program_b/phase1/h_v2"
FRAMES = ROOT / "results/program_b/phase1/h_v2_frames"
H_FUTURES_SEED = 20261084
STILL_FUTURES = 16
RESULT = ROOT / "results/jenga/threeblock_v2_result.json"


def states():
    return sorted(p.stem for p in OLD.glob("ep*.npz"))


def sim_job(name):
    import traceback
    try:
        return name, _sim(name), None
    except Exception:
        return name, None, traceback.format_exc()


def _sim(name):
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
    out = FRAMES / f"{name}.npz"
    if out.exists():
        with np.load(out) as f:
            return {"at_rest": bool(f["at_rest"])}
    e, rest = name[2:].split("_seed"); seed, index = rest.split("_s"); e, seed, index = int(e), int(seed), int(index)
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB); actions = np.asarray(replay.episode(e).actions, np.float32); replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="h_v2_"))))
    renderer, camera = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco)
    try:
        sim.reset(seed)
        for i, a in enumerate(actions):
            if i == index:
                break
            sim.execute(a)
        at_rest = pr.blocks_at_rest(sim)
        snap = sim.snapshot(); start = step_state(sim).copy()
        start_tilt = sim.block_diagnostics()["tilt"][1:].copy()
        renderer.update_scene(sim.data, camera=camera); true_start = renderer.render().copy()
        chunk = actions[index:index + 8]; key = (e, seed, index)
        traces, executed, starts, frames, pos, rot, scales, still = [], [], [], [], [], [], [], []
        for k in range(64):
            t, a, s0, f = future_with_start(sim, snap, chunk, snippets[k], errors.sample(H_FUTURES_SEED, *key, k),
                                            sd.D0_INDEX, renderer, camera)
            traces.append(t); executed.append(a); starts.append(s0); frames.extend(f)
            pos.append(sim.last_perturbation[0]); rot.append(sim.last_perturbation[1]); scales.append(sim.last_perturb_scale)
        for k in range(STILL_FUTURES):                       # unit test: same draws, robot holds still
            sim.restore(snap); pr.perturb_blocks(sim, *errors.sample(H_FUTURES_SEED, *key, k))
            hold = np.r_[sim.target, 0.0].astype(np.float32); tr = []
            for _ in range(38):
                sim.execute(hold); tr.append(step_state(sim))
            still.append(np.stack(tr))
    finally:
        renderer.close(); sim.close()
    np.savez_compressed(out.with_suffix(".tmp.npz"), true_start=true_start, starts=np.stack(starts), frames=np.stack(frames),
                        traces=np.stack(traces).astype(np.float32), actions=np.stack(executed), d_pos=np.stack(pos),
                        d_rot=np.stack(rot), scales=np.array(scales), still=np.stack(still).astype(np.float32),
                        start=start, start_tilt=start_tilt, at_rest=at_rest, frame_index=np.array(sd.D0_INDEX))
    out.with_suffix(".tmp.npz").replace(out)
    return {"at_rest": at_rest}


def stage_sim(workers=24):
    FRAMES.mkdir(parents=True, exist_ok=True)
    names, fails, rest = states(), [], {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for n, (name, res, err) in enumerate(pool.imap_unordered(sim_job, names), 1):
            if err:
                fails.append(name); print(err, flush=True)
            else:
                rest[name] = res["at_rest"]
            if n % 20 == 0 or n == len(names):
                print(f"sim {n}/{len(names)}", flush=True)
    if fails:
        raise SystemExit(f"sim failures: {fails}")
    print("not at rest:", sum(not v for v in rest.values()), "of", len(rest), flush=True)


def stage_encode():
    import torch
    import step3_data as sd
    NEW.mkdir(parents=True, exist_ok=True)
    latent = sd.Latent()
    for name in states():
        path = NEW / f"{name}.npz"
        if path.exists():
            continue
        with np.load(FRAMES / f"{name}.npz") as f:
            d = {k: f[k] for k in f.files}
        z = latent(np.concatenate([d["true_start"][None], d["starts"], d["frames"]]))
        torch.cuda.empty_cache()
        np.savez(path.with_suffix(".tmp.npz"), z0=z[0], z0k=z[1:65], z=z[65:].reshape(64, 2, 28, 28, 8),
                 d_pos=d["d_pos"], d_rot=d["d_rot"], traces=d["traces"], actions=d["actions"], start_tilt=d["start_tilt"],
                 frame_index=d["frame_index"], scales=d["scales"], still=d["still"], start=d["start"], at_rest=d["at_rest"])
        path.with_suffix(".tmp.npz").replace(path)
    print("encoded", len(list(NEW.glob("ep*.npz"))), flush=True)


def stage_grade():
    import torch
    import phase1d_models as p1d
    import step3_world_model as wm
    import step3b_models as sm
    from fresh_fragility_panel import topple_count
    from phase2_amend4 import alarm, score
    from regrade_any_outcome import events_three_block, labels
    res1e = json.loads((ROOT / "results/jenga/phase1e_result.json").read_text())
    us, tau = wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt"), res1e["US"]["tau"]
    gop = sm.G.load(sm.DATA / "G.pt")
    rows, unit = [], {"states": 0, "still_futures": 0, "still_with_event": 0, "states_with_still_event": 0}
    with torch.no_grad():
        for name in states():
            with np.load(NEW / f"{name}.npz") as f:
                f = {k: f[k] for k in f.files}
            old_k = None
            with np.load(OLD / f"{name}.npz") as g:
                old_k = topple_count(g["traces"], g["start_tilt"]); old_ev = events_three_block(g["traces"])
            still_ev = events_three_block(f["still"]).any(axis=(1, 2))
            if f["at_rest"]:
                unit["states"] += 1; unit["still_futures"] += len(still_ev); unit["still_with_event"] += int(still_ev.sum())
                unit["states_with_still_event"] += int(still_ev.any())
            ev = events_three_block(f["traces"]); k = topple_count(f["traces"], f["start_tilt"])
            af, asafe = labels(ev)
            d = np.concatenate([f["d_pos"].reshape(64, 9), f["d_rot"].reshape(64, 9)], 1)
            z0 = torch.as_tensor(f["z0"], device="cuda").float()
            e10, e30 = sm.roll(us, gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d).float()), p1d.augment(f["actions"], None, False))
            real = f["z"].astype(np.float64).reshape(64, 2, -1)
            rows.append({"state": name, "at_rest": bool(f["at_rest"]), "harm": 2 <= k <= 62, "harm_safe": k == 0,
                         "any_fork": af, "any_safe": asafe, "old_harm_count": old_k, "new_harm_count": k,
                         "old_any_fork": labels(old_ev)[0], "scale_mean": float(f["scales"].mean()),
                         "alarms": {"real_futures": bool(alarm(real[:, 0], real[:, 1], "BOTH")),
                                    "US_BOTH": bool(alarm(e10, e30, "BOTH")),
                                    "US_H30_floor": bool(alarm(e10, e30, "H30_floor", tau))}})
            torch.cuda.empty_cache()
    keep = [r for r in rows if r["at_rest"]]
    out = {"unit_test_robot_still": unit, "states": len(rows), "not_at_rest_excluded": len(rows) - len(keep),
           "label_changes": {"old_any_fork_now_not": sum(r["old_any_fork"] and not r["any_fork"] for r in keep),
                             "old_not_any_fork_now_fork": sum(not r["old_any_fork"] and r["any_fork"] for r in keep)},
           "perturb_scale_mean": float(np.mean([r["scale_mean"] for r in keep]))}
    for lab, pos_key, neg_key in (("any_outcome", "any_fork", "any_safe"), ("harm", "harm", "harm_safe")):
        sub = [r for r in keep if r[pos_key] or r[neg_key]]
        y = np.array([r[pos_key] for r in sub])
        out[lab] = {"forks": int(y.sum()), "safe": int((~y).sum()),
                    **{m: score(np.array([r["alarms"][m] for r in sub]), y) for m in sub[0]["alarms"]}}
    out["rows"] = rows
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)


if __name__ == "__main__":
    {"sim": stage_sim, "encode": stage_encode, "grade": stage_grade}[sys.argv[1]]()
