"""Step 3 diagnostics (descriptive, held-out TRAIN states only; no panel data).

Why did the stochastic token world model fail on panel 4? Uses the 60 D0 states (held-out TRAIN episodes,
64 real physics futures each, encoded in the same latent) and the 150 validation states (8 futures x 19
steps). Compares real latent futures with world-model rollouts under several sampling variants:
  S_step   stochastic, fresh noise every step (as evaluated on panel 4)
  S_traj   stochastic, one noise draw per rollout reused at every step
  S_half   stochastic, fresh noise every step at temperature 0.5
  D        deterministic, action noise only (the control)
  D_pert   deterministic from perturbed start latents: rollout k starts from the rendered frame of the exact
           perturbed scene real future k started from (perception uncertainty moved to the input)
Nothing here selects or tunes a model for any panel; results only inform the next protocol.
"""
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import step3_data as sd  # noqa: E402
import step3_world_model as wm  # noqa: E402

OUT = sd.OUT / "diagnostics"
RESULT = ROOT / "results/jenga/step3_diagnostics.json"


# ------------------------------------------------------------------ perturbed start frames
def perturbed_starts_job(job):
    import traceback
    try:
        return job[0], job[1], _perturbed_starts(job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _perturbed_starts(job):
    episode_id, seed, indices, visited = job        # visited: {start: roles} the data chain ran
    import mujoco
    import program_b_reference as pr
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    from jenga_render_frames import close_camera
    from jenga_runtime import DEFAULT_LMDB, JengaReplay
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    errors = p3.ErrorModel()
    snippets = np.load(um.BENCH)["snippets"]
    replay = JengaReplay(DEFAULT_LMDB)
    actions = np.asarray(replay.episode(episode_id).actions, np.float32)
    replay.close()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="step3diag_"))))
    renderer, camera, latent = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent()
    done = {}
    try:
        sim.reset(int(seed))
        for index, action in enumerate(actions):
            if index in visited:
                # replay exactly what step3_data did at this start (same futures, same order, no rendering)
                chunk = actions[index:index + 8]
                snapshot = sim.snapshot()
                warm0 = sim.data.qacc_warmstart.copy()
                sim.block_diagnostics()
                key = (int(episode_id), int(seed), int(index))
                for role in visited[index]:
                    n, seed_name = (sd.D0_FUTURES, "d0_futures") if role == "d0" else (sd.K_FUTURES, "futures")
                    sim.restore(snapshot)
                    for k in range(n):
                        noise = snippets[k] if role == "d0" else snippets[
                            np.random.default_rng([sd.SEEDS[seed_name], *key, k]).integers(len(snippets))]
                        sd.future(sim, snapshot, chunk, noise, errors.sample(sd.SEEDS[seed_name], *key, k))
                if index in indices:
                    _render_perturbed(sim, snapshot, renderer, camera, latent, errors, episode_id, seed, index, key, done)
                sim.restore(snapshot)
                sim.data.qacc_warmstart[:] = warm0
                if index == max(indices):
                    break
            sim.execute(action)
    finally:
        renderer.close()
        sim.close()
    return done


def _render_perturbed(sim, snapshot, renderer, camera, latent, errors, episode_id, seed, index, key, done):
    import program_b_reference as pr
    sim.restore(snapshot)
    renderer.update_scene(sim.data, camera=camera)
    unperturbed = renderer.render().copy()
    frames = []
    for k in range(sd.D0_FUTURES):
        sim.restore(snapshot)
        pr.perturb_blocks(sim, *errors.sample(sd.SEEDS["d0_futures"], *key, k))
        renderer.update_scene(sim.data, camera=camera)
        frames.append(renderer.render().copy())
    z = latent(np.stack(frames))
    with np.load(sd.OUT / "d0" / f"ep{episode_id}_seed{seed}_s{index}.npz") as f:
        diff = np.abs(f["start_frame"].astype(int) - unperturbed.astype(int)).max(-1)
        # tolerance: renderer noise alone gives 1 grey level on ~1.5% of pixels at the first start (no futures run)
        same = bool(diff.max() <= 3 and (diff > 0).mean() < 0.05)
    np.savez(OUT / f"pert_ep{episode_id}_seed{seed}_s{index}.npz", z0=z, chain_matches=same, max_pixel_diff=diff.max(), frac_pixels_diff=(diff > 0).mean())
    done[index] = same


def render_perturbed(workers=6):
    states = {}
    for path in sorted((sd.OUT / "d0").glob("ep*.npz")):
        e, rest = path.stem[2:].split("_seed")
        s, i = rest.split("_s")
        if not (OUT / f"pert_{path.stem}.npz").exists():
            states.setdefault((e, int(s)), set()).add(int(i))
    if not states:
        return
    roles = json.loads((sd.OUT / "selection.json").read_text())["roles"]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for e, s, res, err in pool.imap_unordered(perturbed_starts_job, [
                (e, s, i, {int(k): v for k, v in roles[f"{e}:{s}"].items()}) for (e, s), i in states.items()]):
            if err:
                raise SystemExit(err)
            if not all(res.values()):
                raise SystemExit(f"chain mismatch at ep{e} seed{s}: {res}")


# ---------------------------------------------------------------------------- rollouts
@torch.no_grad()
def rollout(model, z0, actions, mode, seed):
    """z0 (64 or 1, 28, 28, 8); actions (64, 38, A) -> endings at hold 10 / 30, each (64, 6272)."""
    g = torch.Generator(device=wm.DEVICE).manual_seed(int(seed))
    n = len(actions)
    z = torch.as_tensor(z0, device=wm.DEVICE).float()
    z = z.expand(n, -1, -1, -1).contiguous() if len(z) == 1 else z
    pairs = torch.as_tensor(actions, device=wm.DEVICE).reshape(n, wm.STEPS, -1)
    fixed = torch.randn(z.shape, device=wm.DEVICE, generator=g)
    ends = {}
    for t in range(wm.STEPS):
        a = model._act(pairs[:, t])
        if model.kind == "deterministic":
            r = model.net(z, torch.zeros_like(z), a, torch.zeros(n, device=wm.DEVICE))
        else:
            x = {"S_step": torch.randn(z.shape, device=wm.DEVICE, generator=g),
                 "S_half": 0.5 * torch.randn(z.shape, device=wm.DEVICE, generator=g),
                 "S_traj": fixed.clone()}[mode]
            for k in range(wm.EULER):
                x = x + model.net(z, x, a, torch.full((n,), k / wm.EULER, device=wm.DEVICE)) / wm.EULER
            r = x
        z = z + r * model.stats.res_std
        if t in (wm.HOLD10, wm.HOLD30):
            ends[t] = z.reshape(n, -1).double().cpu().numpy()
    return ends[wm.HOLD10], ends[wm.HOLD30]


def describe(early, late):
    from counterfactual_monitor import persistent_mode_alarm
    from outcome_modes import two_mode_test
    t = two_mode_test(late, perturbation_mm=0.0)
    spread = float(np.mean(np.sum((late - late.mean(0)) ** 2, 1)))
    centred = late - late.mean(0)
    sv = np.linalg.svd(centred, compute_uv=False) ** 2
    return {"alarm": bool(persistent_mode_alarm(early, late).alarm), "ashman_d": float(min(t.ashman_d, 99)),
            "bic_two": bool(t.bic_prefers_two), "minority": int(t.minority), "spread": spread,
            "pc1_share": float(sv[0] / sv.sum()) if sv.sum() > 0 else 0.0}


def d0_comparison():
    import fragility_grading_dev as fg
    from monitor_v1_eval import NON_ROBOT, decide, trace_endings
    models = {"stochastic": wm.WorldModel.load(sd.OUT / "wm_stochastic.pt"),
              "deterministic": wm.WorldModel.load(sd.OUT / "wm_deterministic.pt")}
    variants = {"S_step": ("stochastic", "true"), "S_traj": ("stochastic", "true"), "S_half": ("stochastic", "true"),
                "D": ("deterministic", "true"), "D_pert": ("deterministic", "perturbed")}
    rows = []
    for n, path in enumerate(sorted((sd.OUT / "d0").glob("ep*.npz"))):
        with np.load(path) as f, np.load(OUT / f"pert_{path.stem}.npz") as p:
            real = f["z"].astype(np.float64)
            row = {"state": path.stem, "state_alarm": decide(*trace_endings(f["traces"], NON_ROBOT["jenga"]))[0],
                   "real": describe(real[:, 0].reshape(64, -1), real[:, 1].reshape(64, -1))}
            for name, (kind, start) in variants.items():
                z0 = f["z0"][None].astype(np.float32) if start == "true" else p["z0"].astype(np.float32)
                row[name] = describe(*rollout(models[kind], z0, f["actions"], name, seed=20261060 + n))
        rows.append(row)
    names = ["real"] + list(variants)
    real_alarm = [r["real"]["alarm"] for r in rows]
    state_alarm = [r["state_alarm"] for r in rows]
    summary = {}
    for name in names:
        alarms = [r[name]["alarm"] for r in rows]
        summary[name] = {
            "alarms": int(sum(alarms)),
            "kappa_vs_real_latent": fg.kappa(alarms, real_alarm),
            "kappa_vs_state": fg.kappa(alarms, state_alarm),
            "median_spread_ratio_to_real": float(np.median([r[name]["spread"] / max(r["real"]["spread"], 1e-9)
                                                            for r in rows])),
            "median_ashman_d_on_real_forks": float(np.median([r[name]["ashman_d"] for r in rows if r["real"]["alarm"]])),
            "median_pc1_share_on_real_forks": float(np.median([r[name]["pc1_share"] for r in rows if r["real"]["alarm"]])),
            "bic_two_rate": float(np.mean([r[name]["bic_two"] for r in rows])),
        }
    return summary, rows


@torch.no_grad()
def spread_over_time():
    """Validation states: between-future spread per model step, real vs model, on disagreeing vs other states."""
    sel = json.loads((sd.OUT / "selection.json").read_text())
    screen = json.loads((sd.OUT / "screen.json").read_text())
    models = {"S_step": wm.WorldModel.load(sd.OUT / "wm_stochastic.pt"),
              "D": wm.WorldModel.load(sd.OUT / "wm_deterministic.pt")}
    curves = {g: {k: [] for k in ("real", "S_step", "D")} for g in ("disagree", "other")}
    for path in sorted((sd.OUT / "val").glob("ep*.npz")):
        e, rest = path.stem[2:].split("_seed")
        s, i = rest.split("_s")
        group = "disagree" if 0 < screen[f"{e}:{s}"][i]["topples"] < sd.K_SCREEN else "other"
        with np.load(path) as f:
            real = f["z"].astype(np.float32)                                # (8, 19, 28, 28, 8)
            curves[group]["real"].append([float(np.mean(np.sum((real[:, t] - real[:, t].mean(0)) ** 2, (1, 2, 3))))
                                          for t in range(wm.STEPS)])
            for name, model in models.items():
                n = len(f["actions"])
                z = torch.as_tensor(f["z0"], device=wm.DEVICE).float()[None].expand(n, -1, -1, -1).contiguous()
                pairs = torch.as_tensor(f["actions"], device=wm.DEVICE).reshape(n, wm.STEPS, -1)
                per = []
                for t in range(wm.STEPS):
                    z = model.step(z, pairs[:, t])
                    per.append(float(((z - z.mean(0)) ** 2).sum((1, 2, 3)).mean()))
                curves[group][name].append(per)
    return {g: {k: np.median(np.array(v), 0).round(2).tolist() for k, v in c.items()} | {"states": len(c["real"])}
            for g, c in curves.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    render_perturbed()
    summary, rows = d0_comparison()
    curves = spread_over_time()
    result = {"note": "descriptive diagnostics on held-out TRAIN states only (D0 and validation); no panel data",
              "d0_states": len(rows), "d0_summary": summary, "val_spread_over_steps": curves}
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    (OUT / "d0_rows.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
