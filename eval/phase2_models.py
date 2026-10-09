"""Phase 2: selection set H2, the US recipe on F1+F2 with object-free Gauss-TE starts, grading on H2.

Frozen protocol: results/jenga/phase2_generality_protocol.json (amendments 1-3). Stages:
  h2      screen F1+F2 seeds 500000-500599 with 64 oracle futures per chunk start (harm), select 80 forks + 80 safe,
          render the selected states (true start latent, 64 futures' hold-10/30 latents, executed actions)
  train   fit Gauss-TE on the Phase-2 training deltas (z0k - z0), detect rare modes, train US2 (fast recipe, 50/50
          Gauss-TE or rendered starts, label-free rare-mode rebalancing)
  grade   US2 and the 3-block US model (zero-shot) on H2 with Gauss-TE starts; floor = each model's mean hold-30 error on
          H2's real futures (label-free); R0, R1, per family
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_data as pdata  # noqa: E402
import phase2_scenes as ps  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase2_generality_protocol.json"
OUT = ROOT / "results/program_b/phase2"
RESULT = ROOT / "results/jenga/phase2_h2_result.json"
H2_SEEDS = range(500_000, 500_600)
SEEDS = {"h2_futures": 20261212, "h2_select": 20261213, "monitor": 20261220}
D_INDEX = (17, 37)


# ------------------------------------------------------------------------------------------- H2 data
def h2_screen_job(job):
    import traceback
    family, seed = job
    done = OUT / "h2_screen" / f"{family}_{seed}.json"
    try:
        if done.exists():
            return family, seed, json.loads(done.read_text()), None
        res = _h2_screen(family, seed)
        tmp = done.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); tmp.replace(done)
        return family, seed, res, None
    except Exception:
        return family, seed, None, traceback.format_exc()


def _futures(sim, snap, chunk, n_obj, key, snippets, pos_cov, rot_cov, render_at=(), render=None):
    """64 oracle futures: snippet j, pose error draw j. -> traces, executed actions, frames, errors."""
    traces, executed, frames = [], [], []
    for j in range(64):
        err = ps.draw_error(n_obj, pos_cov, rot_cov, SEEDS["h2_futures"], *key, j)
        sim.restore(snap); sim.perturb(*err)
        a = np.asarray(chunk, np.float32).copy(); a[:, :3] -= snippets[j]
        ex = list(a) + [chunk[-1]] * 30
        tr = []
        for t, c in enumerate(ex):
            sim.execute(c); tr.append(sim.object_state())
            if t in render_at:
                frames.append(render())
        traces.append(np.stack(tr)); executed.append(np.asarray(ex, np.float32))
    return traces, executed, frames


def _h2_screen(family, seed):
    import uncertainty_monitor_dev as um
    prep = pdata.prepare(family, seed)
    if prep is None:
        return {"skipped": True}
    scene, sim, acts, starts = prep
    snippets = np.load(um.BENCH)["snippets"]
    pos_cov, rot_cov = ps.pooled_error_model()
    out = {}
    for index, act in enumerate(acts):
        if index in starts:
            snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy()
            start = sim.object_state()
            key = ({"F1": 1, "F2": 2}[family], seed, index)
            traces, _, _ = _futures(sim, snap, acts[index:index + 8], len(scene["objects"]), key, snippets, pos_cov, rot_cov)
            out[index] = int(sum(ps.harm(t, start) for t in traces))
            sim.restore(snap); sim.data.qacc_warmstart[:] = warm
        sim.execute(act)
    return {"skipped": False, "harm_counts": out}


def h2_render_job(job):
    import traceback
    try:
        return job[0], job[1], _h2_render(job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _h2_render(job):
    import mujoco
    import torch
    import step3_data as sd
    import uncertainty_monitor_dev as um
    from jenga_render_frames import close_camera
    family, seed, indices = job
    scene, sim, acts, starts = pdata.prepare(family, seed)
    snippets = np.load(um.BENCH)["snippets"]
    pos_cov, rot_cov = ps.pooled_error_model()
    renderer, camera, latent = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent()

    def render():
        renderer.update_scene(sim.data, camera=camera); return renderer.render().copy()
    try:
        for index, act in enumerate(acts):
            if index in indices:
                path = OUT / "h2" / f"{family}_{seed}_s{index}.npz"
                if not path.exists():
                    snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy()
                    start = sim.object_state()
                    true_start = render()
                    key = ({"F1": 1, "F2": 2}[family], seed, index)
                    traces, executed, frames = _futures(sim, snap, acts[index:index + 8], len(scene["objects"]), key,
                                                        snippets, pos_cov, rot_cov, D_INDEX, render)
                    z = latent(np.concatenate([true_start[None], np.stack(frames)]))
                    torch.cuda.empty_cache()
                    np.savez(path.with_suffix(".tmp.npz"), z0=z[0], z=z[1:].reshape(64, 2, 28, 28, 8),
                             actions=np.stack(executed), harm=np.array([ps.harm(t, start) for t in traces]),
                             traces=np.stack(traces).astype(np.float32), start_state=start, close_frame=true_start)
                    path.with_suffix(".tmp.npz").replace(path)
                    sim.restore(snap); sim.data.qacc_warmstart[:] = warm
                if index == max(indices):
                    break
            sim.execute(act)
    finally:
        renderer.close()
    return sorted(indices)


def h2(workers_screen, workers_render):
    for d in ("h2_screen", "h2"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    screens = pdata.pool_run(h2_screen_job, [(f, s) for f in ("F1", "F2") for s in H2_SEEDS], workers_screen, "h2 screen")
    rows = [(f, s, int(i), c) for (f, s), r in sorted(screens.items()) if not r["skipped"] for i, c in r["harm_counts"].items()]
    rng = np.random.default_rng(SEEDS["h2_select"])
    forks = [r for r in rows if 2 <= r[3] <= 62]
    safe = [r for r in rows if r[3] == 0]
    forks = [forks[i] for i in sorted(rng.permutation(len(forks))[:80])]
    safe = [safe[i] for i in sorted(rng.permutation(len(safe))[:80])]
    by = {}
    for f, s, i, _ in forks + safe:
        by.setdefault((f, s), set()).add(i)
    summary = {"scenes": len(screens), "skipped": sum(r["skipped"] for r in screens.values()), "chunk_starts": len(rows),
               "forks_available": len([r for r in rows if 2 <= r[3] <= 62]), "safe_available": len([r for r in rows if r[3] == 0]),
               "h2_forks": len(forks), "h2_safe": len(safe),
               "per_family": {f: {"forks": sum(r[0] == f for r in forks), "safe": sum(r[0] == f for r in safe)} for f in ("F1", "F2")}}
    (OUT / "h2_selection.json").write_text(json.dumps({"summary": summary, "states": [f"{f}:{s}:{i}" for f, s, i, _ in forks + safe]}) + "\n")
    print(json.dumps(summary), flush=True)
    pdata.pool_run(h2_render_job, [(f, s, idx) for (f, s), idx in sorted(by.items())], workers_render, "h2 render")
    print("h2 done", flush=True)


# ------------------------------------------------------------------------------------------- training
def load_train():
    import torch
    import phase1d_models as p1d
    states, names = [], []
    for path in sorted((OUT / "train").glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            n = len(f["z"])
            states.append({"z0": torch.from_numpy(f["z0"]), "z": torch.from_numpy(f["z"]), "z0k": torch.from_numpy(f["z0k"]),
                           "actions": torch.from_numpy(p1d.augment(f["actions"], None, False)),
                           "d": torch.zeros(n, 18), "top": f["harm"].astype(bool)})
        names.append(str(path))
    return states, names


class GaussTE:
    """Object-free, location-independent start sampler (P2-0b), refit on the Phase-2 training deltas."""

    def __init__(self, fitted):
        import torch
        self.root = fitted["root"].to("cuda")

    def __call__(self, z0, generator=None):
        import torch
        w = torch.randn(*z0.shape, device=z0.device, generator=generator).double()
        f = torch.fft.fft2(w.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = torch.einsum("abij,nabj->nabi", self.root, f)
        return z0.float() + torch.fft.ifft2(x.permute(0, 3, 1, 2)).real.permute(0, 2, 3, 1).float()


def fit_sampler(states):
    import torch
    import phase2_p0 as p0
    path = OUT / "Gauss_TE_phase2.pt"
    if not path.exists():
        p0.OUT = OUT
        fitted = p0.fit_te(states)                          # writes OUT/Gauss_TE.pt
        (OUT / "Gauss_TE.pt").replace(path)
    return GaussTE(torch.load(path))


def train(seed=0):
    import fast_train as ft
    import phase1e_models as p1e
    states, names = load_train()
    sampler_te = fit_sampler(states)
    modes_file = OUT / "modes.json"
    if not modes_file.exists():
        modes_file.write_text(json.dumps(p1e.detect_modes(states, names)) + "\n")
    modes = json.loads(modes_file.read_text())
    rare = np.array([k in set(modes[n]) for n, s in zip(names, states) for k in range(len(s["z"]))])
    has_rare = np.array([bool(modes[n]) for n, s in zip(names, states) for _ in range(len(s["z"]))])
    pools = (np.flatnonzero(rare), np.flatnonzero(~rare & has_rare))
    sampler = lambda rng, b: np.concatenate([rng.choice(pools[0], b // 2), rng.choice(pools[1], b - b // 2)])
    summary = p1e.describe(states, names, modes)            # reporting only ("top" = harm here)
    (OUT / "modes_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    ft.train("US2", OUT, states, sampler, seed=seed, start_fn=sampler_te)
    print("trained", json.dumps(summary), flush=True)


# ------------------------------------------------------------------------------------------- grading
def grade():
    import torch
    import fast_train as ft
    import step3_world_model as wm
    import step3b_models as sm
    from counterfactual_monitor import persistent_mode_alarm
    from outcome_modes import two_mode_test
    sampler_te = GaussTE(torch.load(OUT / "Gauss_TE_phase2.pt"))
    models = {"US2": wm.WorldModel.load(OUT / "model_US2.pt"),
              "US_3block_zero_shot": wm.WorldModel.load(ROOT / "results/program_b/phase1e/model_US.pt")}
    files = sorted((OUT / "h2").glob("*.npz"))
    rows, errs = [], {m: [] for m in models}
    for n_state, path in enumerate(files):
        with np.load(path) as f:
            k = int(f["harm"].sum())
            z0 = torch.as_tensor(f["z0"], device=sm.DEVICE).float()
            g = torch.Generator(device=sm.DEVICE).manual_seed(SEEDS["monitor"] + n_state)
            starts = sampler_te(z0[None].expand(64, -1, -1, -1).contiguous(), g)
            acts = np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, 8)
            real30 = f["z"][:, 1].reshape(64, -1).astype(np.float64)
            row = {"state": path.stem, "family": path.stem.split("_")[0], "fork": bool(2 <= k <= 62), "harm_count": k}
            for name, model in models.items():
                e10, e30 = sm.roll(model, starts, acts)
                row[f"{name}_alarm"] = bool(persistent_mode_alarm(e10, e30).alarm)
                row[f"{name}_sep"] = float(two_mode_test(e30, 0.0).separation_rms_mm)
                errs[name].extend(np.linalg.norm(e30 - real30, axis=1) / np.sqrt(real30.shape[1] // 3))
            rows.append(row)
        torch.cuda.empty_cache()
    fork = np.array([r["fork"] for r in rows]); fam = np.array([r["family"] for r in rows])
    out = {"states": len(rows), "forks": int(fork.sum()), "per_family_forks": {f: int((fork & (fam == f)).sum()) for f in ("F1", "F2")}}
    for name in models:
        tau = float(np.mean(errs[name]))
        r0 = np.array([r[f"{name}_alarm"] for r in rows])
        r1 = r0 & np.array([r[f"{name}_sep"] > tau for r in rows])

        def g_(a, mask=None):
            m = np.ones(len(a), bool) if mask is None else mask
            return {"caught": round(float((a & fork & m).sum() / max((fork & m).sum(), 1)), 4),
                    "false_alarm": round(float((a & ~fork & m).sum() / max((~fork & m).sum(), 1)), 4),
                    "forks_caught": int((a & fork & m).sum()), "false_alarms": int((a & ~fork & m).sum())}
        out[name] = {"tau": tau, "R0": g_(r0), "R1": g_(r1), "R0_F1": g_(r0, fam == "F1"), "R0_F2": g_(r0, fam == "F2")}
        out[name]["score_R0"] = round(out[name]["R0"]["caught"] - out[name]["R0"]["false_alarm"], 4)
    log = OUT / "model_US2_log.json"
    if log.exists():
        out["US2_possibly_undertrained"] = ft.undertrained(json.loads(log.read_text()))
    out["rows"] = rows
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("h2screen", "h2", "train", "grade"))
    parser.add_argument("--screen-workers", type=int, default=14)
    parser.add_argument("--render-workers", type=int, default=5)
    args = parser.parse_args()
    if "amendment_3" not in json.loads(PROTOCOL.read_text()):
        raise SystemExit("amendment 3 (training/grading details) not recorded")
    if args.stage == "h2screen":                            # CPU only; may overlap with GPU encoding jobs
        (OUT / "h2_screen").mkdir(parents=True, exist_ok=True)
        pdata.pool_run(h2_screen_job, [(f, s) for f in ("F1", "F2") for s in H2_SEEDS], args.screen_workers, "h2 screen")
        return
    {"h2": lambda: h2(args.screen_workers, args.render_workers), "train": train, "grade": grade}[args.stage]()


if __name__ == "__main__":
    main()
