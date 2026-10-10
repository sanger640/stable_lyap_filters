"""Phase 2 fresh panels (amendment 2 seeds; run only if the amendment-5 rule passes on H2).

Frozen protocol: results/jenga/phase2_generality_protocol.json. From the oracle screens (eval/phase2_models.py
panelscreen): select up to 60 forks (harm 2-62/64) + 100 safe (0/64) per panel; render and encode the true start;
references from the simulator: second oracle run (truth_B: independent pose-error draws) and the exact-start probe (no
pose error, the 64 bench snippets). Monitor: the selected seed-0 model (best on H2) and the 3-block US model zero-shot,
64 Gauss-TE starts (monitor seed per panel), executed chunk with snippet j, BOTH readout; floor = each model's H2 floor.
Gates: in-family >= 80% caught / <= 20% false; held-out F3 >= 70% / <= 20%.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_data as pdata  # noqa: E402
import phase2_models as pm  # noqa: E402
import phase2_scenes as ps  # noqa: E402

RESULT = ROOT / "results/jenga/phase2_panels_result.json"
SEEDS = {"in": {"truth_B": 20261215, "monitor": 20261216, "select": 20261221},
         "F3": {"truth_B": 20261218, "monitor": 20261219, "select": 20261222}}
GATES = {"in": (0.80, 0.20), "F3": (0.70, 0.20)}


def select(panel):
    rows = []
    for path in sorted((pm.OUT / f"panel_{panel}_screen").glob("*.json")):
        r = json.loads(path.read_text())
        if r["skipped"]:
            continue
        fam, seed = path.stem.split("_")
        rows += [(fam, int(seed), int(i), c) for i, c in r["harm_counts"].items()]
    rng = np.random.default_rng(SEEDS[panel]["select"])
    forks = [r for r in rows if 2 <= r[3] <= 62]
    safe = [r for r in rows if r[3] == 0]
    forks = [forks[i] for i in sorted(rng.permutation(len(forks))[:60])]
    safe = [safe[i] for i in sorted(rng.permutation(len(safe))[:100])]
    return forks + safe, {"chunk_starts": len(rows), "forks_available": sum(2 <= r[3] <= 62 for r in rows),
                          "forks": len(forks), "safe": len(safe)}


def render_job(job):
    import traceback
    try:
        return job[0], job[1], _render(*job), None
    except Exception:
        return job[0], job[1], None, traceback.format_exc()


def _render(panel, family, seed, indices):
    import mujoco
    import torch
    import step3_data as sd
    import uncertainty_monitor_dev as um
    from jenga_render_frames import close_camera
    scene, sim, acts, starts = pdata.prepare(family, seed)
    snippets = np.load(um.BENCH)["snippets"]
    pos_cov, rot_cov = ps.pooled_error_model()
    n_obj = len(scene["objects"])
    renderer, camera, latent = mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent()
    out_dir = pm.OUT / f"panel_{panel}"
    try:
        for index, act in enumerate(acts):
            if index in indices:
                path = out_dir / f"{family}_{seed}_s{index}.npz"
                if not path.exists():
                    snap = sim.snapshot(); warm = sim.data.qacc_warmstart.copy()
                    start = sim.object_state()
                    renderer.update_scene(sim.data, camera=camera); frame = renderer.render().copy()
                    key = ({"F1": 1, "F2": 2, "F3": 3}[family], seed, index)
                    chunk = acts[index:index + 8]
                    tA, exA, _ = pm._futures(sim, snap, chunk, n_obj, key, snippets, pos_cov, rot_cov,
                                             seed=pm.PANELS[panel][2])
                    tB, _, _ = pm._futures(sim, snap, chunk, n_obj, key, snippets, pos_cov, rot_cov, seed=SEEDS[panel]["truth_B"])
                    exact = []
                    for j in range(64):                     # exact-start probe: no pose error
                        sim.restore(snap)
                        a = np.asarray(chunk, np.float32).copy(); a[:, :3] -= snippets[j]
                        tr = []
                        for c in list(a) + [chunk[-1]] * 30:
                            sim.execute(c); tr.append(sim.object_state())
                        exact.append(np.stack(tr))
                    z0 = latent(frame[None])[0]
                    torch.cuda.empty_cache()
                    np.savez(path.with_suffix(".tmp.npz"), z0=z0, close_frame=frame, actions=np.stack(exA),
                             harm_A=np.array([ps.harm(t, start) for t in tA]), harm_B=np.array([ps.harm(t, start) for t in tB]),
                             harm_exact=np.array([ps.harm(t, start) for t in exact]), start_state=start)
                    path.with_suffix(".tmp.npz").replace(path)
                    sim.restore(snap); sim.data.qacc_warmstart[:] = warm
                if index == max(indices):
                    break
            sim.execute(act)
    finally:
        renderer.close()
    return sorted(indices)


def grade(panel, model_names, taus):
    import torch
    import step3_world_model as wm
    import step3b_models as sm
    from phase2_amend4 import alarm, score
    from outcome_modes import two_mode_test
    te = pm.GaussTE(torch.load(pm.OUT / "Gauss_TE_phase2.pt"))
    paths = {"US_3block_zero_shot": ROOT / "results/program_b/phase1e/model_US.pt"}
    models = {n: wm.WorldModel.load(paths.get(n, pm.OUT / f"model_{n}.pt")) for n in model_names}
    fork, refB, refE, al, alf = [], [], [], {n: [] for n in models}, {n: [] for n in models}
    with torch.no_grad():
        for n_state, path in enumerate(sorted((pm.OUT / f"panel_{panel}").glob("*.npz"))):
            with np.load(path) as f:
                k = int(f["harm_A"].sum())
                fork.append(2 <= k <= 62)
                refB.append(2 <= int(f["harm_B"].sum()) <= 62); refE.append(2 <= int(f["harm_exact"].sum()) <= 62)
                z0 = torch.as_tensor(f["z0"], device=sm.DEVICE).float()
                g = torch.Generator(device=sm.DEVICE).manual_seed(SEEDS[panel]["monitor"] + n_state)
                starts = te(z0[None].expand(64, -1, -1, -1).contiguous(), g)
                acts = np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, 8)
            for n, m in models.items():
                e10, e30 = sm.roll(m, starts, acts)
                a = alarm(e10, e30, "BOTH")
                al[n].append(a)
                alf[n].append(a and float(two_mode_test(e30, 0.0).separation_rms_mm) > taus[n])
            torch.cuda.empty_cache()
    fork = np.array(fork)
    out = {"states": len(fork), "forks": int(fork.sum()),
           "truth_B_split_label": score(np.array(refB), fork), "exact_start_probe_label": score(np.array(refE), fork)}
    for n in models:
        out[n] = {"BOTH": score(np.array(al[n]), fork), "BOTH_floor": score(np.array(alf[n]), fork), "tau": taus[n]}
    c, fa = GATES[panel]
    best = out[model_names[0]]["BOTH"]
    out["primary_passes_gate"] = bool(best["caught"] >= c and best["false_alarm"] <= fa)
    return out


def main():
    a4 = json.loads((ROOT / "results/jenga/phase2_amend4_result.json").read_text())
    g = a4["amendment5_grade_H2"]
    if not g["best_passes_in_family_bar_on_H2"]:
        raise SystemExit("amendment-5 rule not met on H2: panels are not generated")
    primary = g["ranking"][0]
    res = {"primary_model": primary, "selection": {}}
    for panel in ("in", "F3"):
        (pm.OUT / f"panel_{panel}").mkdir(parents=True, exist_ok=True)
        rows, summary = select(panel)
        res["selection"][panel] = summary
        by = {}
        for fam, seed, i, _ in rows:
            by.setdefault((fam, seed), set()).add(i)
        pdata.pool_run(render_job, [(panel, f, s, idx) for (f, s), idx in sorted(by.items())], 5, f"panel {panel} render")
    taus = {primary: g[primary]["tau"], "US_3block_zero_shot": g["US_3block_zero_shot"]["tau"]}
    for panel in ("in", "F3"):
        res[panel] = grade(panel, [primary, "US_3block_zero_shot"], taus)
        RESULT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
