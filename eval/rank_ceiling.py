"""Within-decision ranking ceiling: can the latent fork/topple readout rank a decision's candidate plans by real risk
when it is given the REAL futures (exact simulator rollouts, rendered + encoded) instead of imagined ones?

75 risky dev decisions x 13 plans (policy plan + 12 policy samples, results/program_b/cbf_qp/checks/plans.json) x 32
futures: perturbed start (perception error) + recorded execution noise x2 (the run's level), H15 horizon, latent at 1 s
and 3 s. Statistics on the real latents: fork (BOTH split test), q (imagined-topple-fraction readout), obar.
Real risk per plan: the cached 32-future truth (truth_32.json).
If the readout ranks real futures well within a decision -> the world model is the bottleneck; if not -> the readout.

usage: python3 eval/rank_ceiling.py run | report       (env RC_WORKERS 10)
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("EXEC_NOISE_SCALE", "2"); os.environ.setdefault("HARD_MASS", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import cbf_qp as q  # noqa: E402

OUT = q.OUT / "rank_ceiling"; N_FUT, N_PLANS, SEED = 32, 13, 20261800


def job(args):
    import traceback
    try:
        return args[0], _job(args), None
    except Exception:
        return args[0], None, traceback.format_exc()


def _job(args):
    i, state, ctrl, target, gripper, plans = args
    import mujoco
    import program_b_reference as pr
    import step3_data as sd
    import step3_world_model as wm
    import uncertainty_monitor_dev as um
    import visual_pipeline_panel3 as p3
    import lookahead_check as la
    from jenga_render_frames import close_camera
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    path = OUT / f"d{i}.npz"
    if path.exists():
        return "exists"
    g = globals()
    if "_W" not in g:
        sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="rc_"))))
        g["_W"] = (sim, mujoco.Renderer(sim.model, height=384, width=384), close_camera(mujoco), sd.Latent())
    sim, r, cam, latent = g["_W"]
    errors, snippets = p3.ErrorModel(), np.load(um.BENCH)["snippets"]
    snap = (state, ctrl, target, float(gripper))
    i10, i30 = sd.FRAME_INDEX[wm.HOLD10], sd.FRAME_INDEX[wm.HOLD30]
    frames = []
    for c, plan in enumerate(plans):
        for j in range(N_FUT):
            sim.restore(snap); pr.perturb_blocks(sim, *errors.sample(SEED, i, 0, j))      # same draws for every plan
            cmds = la.horizon_actions(plan, "H15", la.noise_for(snippets, j) * la.EXEC_SCALE)
            for t, a in enumerate(cmds):
                sim.execute(a)
                if t in (i10, i30):
                    r.update_scene(sim.data, camera=cam); frames.append(r.render().copy())
    sim.restore(snap); r.update_scene(sim.data, camera=cam); z0 = latent(r.render().copy()[None])[0]
    import torch
    fr = np.stack(frames)
    z = np.concatenate([latent(fr[k:k + 64]) for k in range(0, len(fr), 64)]).reshape(len(plans), N_FUT, 2, 28, 28, 8)
    torch.cuda.empty_cache()
    np.savez_compressed(path, z=z.astype(np.float16), z0=z0.astype(np.float16))
    return "ok"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    C, K = q.load_cache()
    rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
    jobs = [(r["i"], C["state"][r["i"]], C["ctrl"][r["i"]], C["target"][r["i"]], C["gripper"][r["i"]],
             np.asarray(r["plans"], np.float32)[:N_PLANS]) for r in rows]
    with mp.get_context("spawn").Pool(int(os.environ.get("RC_WORKERS", 10))) as pool:
        for n, (key, res, err) in enumerate(pool.imap_unordered(job, jobs), 1):
            if err:
                print(err, flush=True)
            if n % 10 == 0 or n == len(jobs):
                print(f"rc {n}/{len(jobs)}", flush=True)


def report():
    import torch
    import cbf_barrier as cb
    import cem_filter as cf
    from scipy.stats import spearmanr
    C, K = q.load_cache()
    T = json.loads((q.OUT / "truth_32.json").read_text())
    rows = json.loads((q.OUT / "checks" / "plans.json").read_text())
    sp_q, sp_o, sp_f, pick_q, pick_o, pol, rnd, best = [], [], [], [], [], [], [], []
    for r in rows:
        i = r["i"]; f = np.load(OUT / f"d{i}.npz")
        z = torch.as_tensor(f["z"].astype(np.float32), device="cuda"); z0 = torch.as_tensor(f["z0"].astype(np.float32), device="cuda")
        P = np.asarray(r["plans"], np.float32)[:N_PLANS]
        risk = np.array([T[q._hash(p[1:9])] for p in P])
        qs, obs, fk = [], [], []
        ok10, _, _ = cf.split_test_gpu(z[:, :, 0].reshape(len(P), N_FUT, -1)); ok30, lab, _ = cf.split_test_gpu(z[:, :, 1].reshape(len(P), N_FUT, -1))
        for c in range(len(P)):
            pts = np.vstack([C["site"][i][None], P[c][1:16, :3]])
            m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
            o_j = (((z[c, :, 1] - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum())
            fork = bool(ok10[c] & ok30[c]); L = lab[c].bool()
            qv = 0.0
            if fork and not (L.all() or (~L).all()):
                bad = L if o_j[L].mean() > o_j[~L].mean() else ~L
                qv = float(bad.float().mean())
            qs.append(qv); obs.append(float(o_j.mean())); fk.append(fork)
        qs, obs, fk = np.array(qs), np.array(obs), np.array(fk, float)
        if np.ptp(risk) > 0:
            if np.ptp(qs) > 0: sp_q.append(spearmanr(qs, risk).correlation)
            if np.ptp(obs) > 0: sp_o.append(spearmanr(obs, risk).correlation)
            if np.ptp(fk) > 0: sp_f.append(spearmanr(fk.astype(float), risk).correlation)
        pick_q.append(risk[np.argmin(qs + 1e-6 * np.arange(len(P)))]); pick_o.append(risk[np.argmin(obs)])
        pol.append(risk[0]); rnd.append(risk[1:].mean()); best.append(risk.min())
    res = {"decisions": len(rows), "plans_per_decision": N_PLANS, "real_futures": N_FUT,
           "within_decision_spearman_with_real_risk": {"q_real_latent": round(float(np.nanmean(sp_q)), 3), "obar_real_latent": round(float(np.nanmean(sp_o)), 3),
                                                        "fork_real_latent": round(float(np.nanmean(sp_f)), 3)},
           "real_risk_of_pick": {"policy_plan": round(float(np.mean(pol)), 4), "random_sample": round(float(np.mean(rnd)), 4),
                                 "min_q_real_latent": round(float(np.mean(pick_q)), 4), "min_obar_real_latent": round(float(np.mean(pick_o)), 4),
                                 "oracle": round(float(np.mean(best)), 4)}}
    (q.OUT / "rank_ceiling.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
