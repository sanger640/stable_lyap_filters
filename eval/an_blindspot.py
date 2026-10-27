"""Exploratory (user 2026-10-27: analyse the red-block blind spot). Policy-alone runs paired_record_h2h2 / _m3 (2x noise,
latents in results/jenga/hj2/data). 'pre' = decisions 1-3 chunks before the red block falls (first replayed step with
red tilt > 45 deg, episodes the audit marks red_fell_before_extraction); 'safe' = gripper-closed decisions of v4-success
episodes without a red fall. Per decision, CF3's undo test (15-step plan, 64 draws at 2x) residual |f_rev - f_hold|
read three ways: outside the hand mask (as deployed), all patches, the red block's region only (patches within 40 px of
its projected centre and top); and the PLAN branch (do the plan, then hold at its end) vs hold on the red region: does
the model predict the red block moving at all? Grading labels only.
usage: python3 eval/an_blindspot.py -> results/jenga/deep_dive/blindspot_rows.npz, blindspot.json"""
import json, os, sys
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ["EXEC_NOISE_SCALE"] = "2"
ROOT = "/home/corey/wksp/stable_lyap_filters"; sys.path[:0] = [ROOT + "/eval", ROOT + "/src"]
import torch
import cbf_barrier as cb, cbf_filter as cbff, cem_filter as cf, cf_fork as C, cf_rev as V
import inspector_data as insp, step3b_models as sm, step3_world_model as wm, visual_pipeline_panel3 as p3, uncertainty_monitor_dev as um
from sklearn.metrics import roc_auc_score
HERE = ROOT + "/results/jenga/deep_dive"; S_ = 2.0
model = wm.WorldModel.load(f"{ROOT}/results/program_b/oil_v1/model/model_OIL.pt"); model.net.eval()
G = sm.G.load(cf.V3 / "G.pt"); E = p3.ErrorModel(); snip = np.load(um.BENCH)["snippets"]
S0, NL = cbff.CFG["PLAN_START"], cbff.CFG["N_LOOK"]; total = 2 * wm.STEPS
sim = insp._sim(); RED = sim.tracked_block_ids[0]


def fall_chunk(run, ep):
    S = dict(np.load(f"{ROOT}/results/program_b/policy_rollouts/{run}/ep{ep}.npz"))
    nrng = np.random.default_rng([20261502, ep]); noises = [snip[nrng.integers(len(snip))] * S_ for _ in range(int(S["chunk_index"][-1]) + 1)]
    for d in range(len(S["chunk_index"])):
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d]))); k = int(S["chunk_index"][d])
        for j, a in enumerate(S["executed"][d]):
            a = np.array(a, np.float32); a[:3] -= noises[k][j]; sim.execute(a)
            if sim.block_diagnostics()["tilt"][0] > 45:
                return k
    return None


def red_mask(state_row, ctrl, target, grip):
    sim.restore((state_row, ctrl, target, float(grip))); c = sim.data.xpos[RED].copy()
    uv = cb.project(np.array([c, c + [0, 0, 0.0375]]))
    cc = (np.arange(cb.GRID) + 0.5) * cb.IMG / cb.GRID; U, Vv = np.meshgrid(cc, cc); m = np.zeros((cb.GRID, cb.GRID), bool)
    for u, v in uv:
        m |= (U - u) ** 2 + (Vv - v) ** 2 <= 40 ** 2
    return m


def residuals(z0, plan, cur, hand_xyz, rmask, seed):
    d_ = np.stack([np.concatenate([(p * [1, 1, 0] * S_).reshape(9), (r * [0, 0, 1] * S_).reshape(9)]) for p, r in (E.sample(seed, 0, 0, j) for j in range(C.N_FUT))])
    st = G(torch.as_tensor(z0, device="cuda")[None].expand(C.N_FUT, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
    rev = V.reverse_plan(plan, cur, NL, S0); rev = np.concatenate([rev[S0:], np.repeat(rev[-1:], total, 0)])[:total]
    do = np.concatenate([plan[S0:S0 + NL], np.repeat(plan[S0 + NL - 1:S0 + NL], total, 0)])[:total]
    hold = np.repeat(cur[None], total, 0)
    acts = []
    for base in (rev, do, hold):
        for j in range(C.N_FUT):
            a = base.copy(); a[:NL, :3] -= np.concatenate([snip[j % 64], snip[(j + 1) % 64]])[:NL] * S_; acts.append(a)
    pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(3 * C.N_FUT, wm.STEPS, -1); z = st.repeat(3, 1, 1, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for t in range(wm.STEPS):
            z = model.step(z, pairs[:, t]).float()
    z = z.float().reshape(3, C.N_FUT, 28, 28, -1)
    dr = (z[0] - z[2]).abs().mean(-1); dd = (z[1] - z[2]).abs().mean(-1)
    hm = torch.as_tensor(~cb.hand_mask(np.vstack([hand_xyz[None], plan[S0:S0 + NL, :3]])), device="cuda"); rm = torch.as_tensor(rmask, device="cuda")
    q = lambda D, m: float(torch.quantile(D[:, m], 0.98, dim=1).max())
    return q(dr, hm), q(dr, torch.ones_like(hm)), q(dr, rm), q(dd, rm), float(rm.float().mean()), float((rm & ~hm).float().sum() / max(rm.float().sum(), 1))


rows = []
rng = np.random.default_rng(20261027)
for run in ("paired_record_h2h2", "paired_record_m3"):
    aud = {r["ep"]: r for r in json.load(open(f"{ROOT}/results/jenga/outcome_audit_{run.split('_')[-1]}.json"))[run]["rows"]}
    red = [e for e, r in aud.items() if r["red_fell_before_extraction"]]
    ok = [e for e, r in aud.items() if r["v4_success"] and not r["red_fell_before_extraction"] and not r["red_fell_after_extraction"]]
    jobs = []
    for e in red:
        kf = fall_chunk(run, e)
        if kf is not None:
            jobs.append((e, 1, kf))
    for e in rng.permutation(ok)[:60]:
        jobs.append((int(e), 0, None))
    for e, lab, kf in jobs:
        H = dict(np.load(f"{ROOT}/results/jenga/hj2/data/{run}/ep{e}.npz")); R = dict(np.load(f"{ROOT}/results/program_b/policy_rollouts/{run}/ep{e}.npz"))
        ks = list(H["chunk_index"]); rk = list(R["chunk_index"])
        for i, k in enumerate(ks):
            if lab == 1 and not (1 <= kf - k <= 3):
                continue
            if lab == 0 and not H["proprio"][i, 3] > 0:
                continue
            if int(k) not in rk:
                continue
            d = rk.index(int(k)); plan = np.asarray(R["plan16"][d], np.float32); cur = np.r_[R["target"][d], plan[0, 3]].astype(np.float32)
            rm = red_mask(R["state"][d], R["ctrl"][d], R["target"][d], R["gripper"][d])
            vals = residuals(H["z"][i].astype(np.float32), plan, cur, H["proprio"][i, :3], rm, C.SEED + 1000 * e + i)
            rows.append((lab, e, int(k), kf if kf is not None else -1, float(H["proprio"][i, 3] > 0), float(plan[S0:S0 + NL, 3].min() < 0 and plan[0, 3] > 0)) + vals)
        print(run, e, lab, len(rows), flush=True)
R = np.array(rows, float)
cols = ["pre", "ep", "k", "kf", "closed", "plan_opens", "r_hand_masked", "r_all", "r_red", "do_red", "red_area", "red_outside_mask"]
np.savez(HERE + "/blindspot_rows.npz", rows=R, cols=np.array(cols))
pre = R[:, 0] == 1; out = {"pre": int(pre.sum()), "safe_holding": int((~pre).sum()),
                            "share_red_region_outside_hand_mask": round(float(R[:, 11].mean()), 3)}
for j, c in enumerate(cols[6:10], 6):
    out[c] = {"auc_pre_vs_safe": round(float(roc_auc_score(pre, R[:, j])), 3), "median_pre": round(float(np.median(R[pre, j])), 3),
              "median_safe": round(float(np.median(R[~pre, j])), 3), "flag_rate_at_1.3356": [round(float((R[pre, j] > 1.3356).mean()), 3), round(float((R[~pre, j] > 1.3356).mean()), 3)]}
json.dump(out, open(HERE + "/blindspot.json", "w"), indent=1); print(json.dumps(out, indent=1))
