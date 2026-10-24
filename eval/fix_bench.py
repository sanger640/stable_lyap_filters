"""Offline fix bench (results/jenga/fixbench_protocol.json): which KIND of fix delays the policy least, at equal safety?
Same 150 approach decisions and harness as eval/fix_counterfactual.py (exact saved state d of paired_cbfg3m_stl, exact
2-frame history, run's execution noise, same continuation seed in every branch, policy alone for up to M chunks).
First chunk per branch:
  ARMS=sel   nominal   the policy's own chunk (reference)
             gcem      the deployed 3x filter's fix (logged)
             sel       SELECTION from the policy's own distribution: 16 fresh policy samples from the same history,
                       certified with the 3x filter's certificate (OIL+ENS, 64 fresh futures per model); the certified
                       sample closest to the policy's plan (opt2 cost); none -> hold if certified, else best (h, -o)
             resample  control: the first of those 16 samples, uncertified (how much does ANY other policy sample cost?)
  ARMS=steer nominal, gcem, steer (G-steer recomputed at d: cbfgst3m's search and certificate, run regardless of the trigger)
  ARMS=proj  nominal, gcem, proj (amendment 1: the deployed fix PROJECTED onto the policy's distribution, SDEdit-style:
             noise the fix's 16-step plan to 1/4 of the policy's DDIM schedule (strength 0.25) and denoise it with the policy
             conditioned on the same observations, 8 samples; certified as sel; none certified -> the deployed fix)
usage: ARMS=sel|steer SHARD=i/n python3 eval/fix_bench.py run   -> results/jenga/deep_dive/fixbench/<arms>_<i>.jsonl
       python3 eval/fix_bench.py analyse                          -> results/jenga/fixbench_result.json
"""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import fix_counterfactual as fc  # noqa: E402

OUT = ROOT / "results/jenga/deep_dive/fixbench"
ARMS = os.environ.get("ARMS", "sel")
N_SEL = 16


def setup():
    os.environ["CBF_WM"] = str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"); os.environ["OPT2_CERT"] = "OIL+ENS"
    os.environ["OPT3_GRIP"] = "fixed"
    if ARMS == "steer":
        os.environ["OPT3_STEER"] = "1"
    import cbf_barrier
    import cbf_filter as cbff
    import opt3
    assert opt3.CFG["STEER"] == (ARMS == "steer") and ARMS in ("sel", "steer", "proj")
    cbf_barrier.SCALES = (0.5, 1.0, 2.0, 3.0, 4.0)
    cbff.CALIB = ROOT / "results/jenga/cbf_calibration_OIL.json"
    cbff.CFG["H_SAFE"] = float(np.log2(4.0)); cbff.CFG["FIX"] = "g3"
    import cem_filter as cf
    import step3_world_model as wm
    mon = cf.Monitor(); cbf = cbff.CBF(mon)
    certs = [None] + [wm.WorldModel.load(ROOT / f"results/program_b/ens_v1/model_ENS{e}.pt") for e in range(4)]
    return mon, cbf, certs


def first_chunks(sim, S, d, hist, pol, mon, cbf, certs, e):
    """-> {branch: first 8-step chunk}, info. sim is at the exact saved state d."""
    import torch
    import opt2
    import opt3
    import policy_rollout as pr
    PS = pr.PLAN_START
    mon.rng = np.random.default_rng([fc.SEED, e, d])
    z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
    p16 = np.asarray(S["plan16"][d], np.float32).copy(); p16[0, :3] = sim.target
    hold16 = np.repeat(np.r_[sim.target, 0.0][None].astype(np.float32), 16, 0)
    h, o = cbf.score(sim, z0, np.stack([p16, hold16]), 64); h_now = float(h[1])
    info = {"h_nom_re": float(h[0]), "o_nom_re": round(float(o[0]), 4), "h_now_re": h_now}
    import cbf_filter as cbff
    P = opt3.Problem(mon, cbf, sim, z0, p16, h_now, certs); P.cbf_O_MAX = cbff.CFG["O_MAX"]
    out = {"nominal": S["nominal"][d]}
    if ARMS == "sel":
        out["gcem"] = S["executed"][d]
        torch.manual_seed(int(fc.SEED + 7 + 1000 * e + d))
        samples = np.asarray(pr.policy_chunk(pol, hist, n=N_SEL), np.float32)            # (16, 16, 4)
        out["resample"] = samples[0][PS:PS + 8].copy()
        hold = np.repeat(np.r_[sim.target, p16[0, 3]][None], 16, 0).astype(np.float32)   # opt2.finish's hold
        ok, hc, oc = P.certify(list(samples) + [hold])
        costs = np.array([opt2.Problem.cost(P, s[1:9, :3] - p16[1:9, :3]) for s in samples])
        okc = ok[:-1]
        if okc.any():
            j = int(np.flatnonzero(okc)[np.argmin(costs[okc])]); best, kind = samples[j], "certified"
        elif ok[-1]:
            j, best, kind = -1, hold, "fallback_hold"
        else:
            j = int(np.lexsort((oc[:-1], -hc[:-1]))[0]); best, kind = samples[j], "fallback_best"
        out["sel"] = best[PS:PS + 8].copy()
        info.update({"sel_kind": kind, "sel_n_certified": int(okc.sum()), "sel_choice": j,
                     "sel_change_mm": round(float(np.abs(best[1:9, :3] - p16[1:9, :3]).max() * 1000), 1),
                     "sel_progress": round(opt3.progress(p16, best), 3),
                     "resample_change_mm": round(float(np.abs(samples[0][1:9, :3] - p16[1:9, :3]).max() * 1000), 1)})
    elif ARMS == "proj":
        out["gcem"] = S["executed"][d]
        fix16 = p16.copy(); D = np.asarray(S["executed"][d], np.float32)[:, :3] - p16[PS:PS + 8, :3]
        fix16[PS:PS + 8] = S["executed"][d]; fix16[PS + 8:, :3] += D[-1]          # the fix's 16-step plan (opt2 convention)
        torch.manual_seed(int(fc.SEED + 9 + 1000 * e + d))
        cand = project(pol, hist, fix16)
        ok, hc, oc = P.certify(list(cand))
        costs = np.array([opt2.Problem.cost(P, c[1:9, :3] - p16[1:9, :3]) for c in cand])
        if ok.any():
            j = int(np.flatnonzero(ok)[np.argmin(costs[ok])]); out["proj"] = cand[j][PS:PS + 8].copy(); kind = "certified"
        else:
            j, out["proj"], kind = -1, S["executed"][d], "fallback_fix"
        info.update({"proj_kind": kind, "proj_n_certified": int(ok.sum()), "proj_choice": j,
                     "proj_dist_to_fix_mm": round(float(np.abs(out["proj"][:, :3] - D[:, :3] - p16[PS:PS + 8, :3]).max() * 1000), 1),
                     "proj_change_mm": round(float(np.abs(out["proj"][:, :3] - p16[PS:PS + 8, :3]).max() * 1000), 1)})
    else:
        out["gcem"] = S["executed"][d]                                   # amendment 2: within-job reference (nodes differ)
        best, oi = opt3.solve_cem(P, int(mon.rng.integers(2 ** 31)))
        out["steer"] = np.asarray(best, np.float32)[PS:PS + 8].copy()
        info.update({"steer_kind": oi["kind"], "steer_theta": oi.get("theta"), "steer_change_mm": oi.get("change_mm"),
                     "steer_progress": round(opt3.progress(p16, best), 3)})
    return out, info


PROJ_N, PROJ_STRENGTH = 8, 0.25


def project(pol, hist, plan16, n=PROJ_N, strength=PROJ_STRENGTH):
    """SDEdit projection of a 16-step plan onto the policy's distribution: x_t = noised plan at the schedule step
    round(len * (1 - strength)), then the policy's own remaining DDIM steps (same observation conditioning). -> (n, 16, 4)"""
    import torch
    w = np.stack([h[0] for h in hist[-2:]]); f = np.stack([h[1] for h in hist[-2:]]); s = np.stack([h[2] for h in hist[-2:]])
    t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2)[None].float().cuda().expand(n, -1, -1, -1, -1) / 255.0
    obs = {"camera_1": t(w), "camera_2": t(f), "agent_pos": torch.from_numpy(s)[None].cuda().expand(n, -1, -1)}
    with torch.no_grad():
        nobs = pol.normalizer.normalize(obs)
        this = {k: v[:, :pol.n_obs_steps].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        gc = pol.obs_encoder(this).reshape(n, -1)
        x0 = pol.normalizer["action"].normalize(torch.as_tensor(plan16, dtype=torch.float32, device="cuda"))[None].expand(n, -1, -1)
        sch = pol.noise_scheduler; sch.set_timesteps(pol.num_inference_steps); ts = sch.timesteps
        i0 = int(round(len(ts) * (1 - strength)))
        x = sch.add_noise(x0, torch.randn_like(x0), ts[i0:i0 + 1].expand(n))
        for tt in ts[i0:]:
            x = sch.step(pol.model(x, tt, local_cond=None, global_cond=gc), tt, x).prev_sample
        return pol.normalizer["action"].unnormalize(x).cpu().numpy().astype(np.float32)


def branch(sim, cams, pol, S, d, e, hist0, first, noises, xy0):
    import torch
    import policy_rollout as pr
    sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
    hist = list(hist0); z0 = float(sim.data.xpos[sim.tracked_block_ids[0]][2])
    res = {"grasp": None, "extract": None, "topple": None}
    torch.manual_seed(int(fc.SEED + 1000 * e + d))
    for c in range(fc.M + 1):
        k = int(S["chunk_index"][d]) + c
        chunk = first if c == 0 else pr.policy_chunk(pol, hist, full=True)[pr.PLAN_START:pr.PLAN_START + 8]
        for j, a in enumerate(chunk):
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]; sim.execute(a); hist.append(cams.obs())
            if res["grasp"] is None and sim.gripper < 55:
                res["grasp"] = c
            p = sim.data.xpos[sim.tracked_block_ids[0]]
            if res["extract"] is None and (p[2] - z0 >= 0.02 or p[0] - xy0[0] <= -0.02):
                res["extract"] = c
            if res["topple"] is None and float(np.max(sim.block_diagnostics()["tilt"][1:])) >= 45:
                res["topple"] = c
        hist = hist[-2:]
        if res["extract"] is not None or res["topple"] is not None:
            break
    return res


def run():
    import plausibility as pl
    import policy_rollout as pr
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    shard, nshard = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
    OUT.mkdir(parents=True, exist_ok=True); fout = OUT / f"{ARMS}_{shard}.jsonl"
    done = {(r["ep"], r["d"]) for r in map(json.loads, fout.read_text().splitlines())} if fout.exists() else set()
    mon, cbf, certs = setup()
    pol = pr.load_policy()
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="fb_")))); cams = pr.Cameras(sim)
    left = int(os.environ.get("LIMIT", 10 ** 9))                     # smoke: stop after LIMIT new cases (resumable)
    for ci, (e, d) in enumerate(fc.cases()):
        if ci % nshard != shard or (e, d) in done:
            continue
        if left <= 0:
            break
        left -= 1
        with np.load(fc.D / f"ep{e}.npz") as f:
            S = {k: f[k] for k in f.files}
        k0 = int(S["chunk_index"][d])
        noises = pl.episode_noises(e, k0 + fc.M + 2, scale=fc.NOISE)
        sim.restore((S["state"][0], S["ctrl"][0], S["target"][0], float(S["gripper"][0])))
        xy0 = sim.data.xpos[sim.tracked_block_ids[0]][:2].copy()
        sim.restore((S["state"][d - 1], S["ctrl"][d - 1], S["target"][d - 1], float(S["gripper"][d - 1])))
        hist = []
        for j, a in enumerate(S["executed"][d - 1]):                     # exact history: replay chunk d-1
            a = np.asarray(a, np.float32).copy(); a[:3] -= noises[int(S["chunk_index"][d - 1])][j]; sim.execute(a)
            if j >= 6:
                hist.append(cams.obs())
        sim.restore((S["state"][d], S["ctrl"][d], S["target"][d], float(S["gripper"][d])))
        firsts, info = first_chunks(sim, S, d, hist, pol, mon, cbf, certs, e)
        row = {"ep": e, "d": d, "chunk": k0, **info}
        for name, first in firsts.items():
            row[name] = branch(sim, cams, pol, S, d, e, hist, first, noises, xy0)
        with open(fout, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(ci, row, flush=True)
    cams.close(); sim.close()


def analyse():
    P = json.loads((ROOT / "results/jenga/fixbench_protocol.json").read_text())
    R = {a: {(r["ep"], r["d"]): r for f in sorted(OUT.glob(f"{a}_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
         for a in ("sel", "steer")}
    g = lambda b: b["grasp"] if b["grasp"] is not None else fc.M + 1
    rng = np.random.default_rng(P["bootstrap_seed"])

    def boot(x):
        bs = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)])
        return round(float(x.mean()), 3), round(float(np.percentile(bs, 5)), 3), round(float(np.percentile(bs, 95)), 3)

    def delay(rows, arm):                    # grasp delay vs the SAME job's nominal branch, cases where neither topples
        ok = [r for r in rows if r[arm]["topple"] is None and r["nominal"]["topple"] is None]
        return {(r["ep"], r["d"]): g(r[arm]) - g(r["nominal"]) for r in ok}
    out = {"cases": {a: len(v) for a, v in R.items()}}
    S = list(R["sel"].values())
    for arm in ("nominal", "gcem", "sel", "resample"):
        out[f"topples_{arm}"] = sum(r[arm]["topple"] is not None for r in S)
        out[f"censored_{arm}"] = sum(r[arm]["grasp"] is None for r in S)
    dg, ds, dr = delay(S, "gcem"), delay(S, "sel"), delay(S, "resample")
    out["mean_delay"] = {"gcem": boot(np.array(list(dg.values()), float)), "sel": boot(np.array(list(ds.values()), float)),
                         "resample": boot(np.array(list(dr.values()), float))}
    K = sorted(set(dg) & set(ds)); x = np.array([ds[k] - dg[k] for k in K], float)
    out["PRIMARY_sel_minus_gcem_delay"] = {"n": len(K), "mean_lower_upper90": boot(x)}
    tg = np.array([r["gcem"]["topple"] is not None for r in S]); ts = np.array([r["sel"]["topple"] is not None for r in S])
    out["topple_discordant_sel_only_gcem_only"] = [int((ts & ~tg).sum()), int((tg & ~ts).sum())]
    tdiff = (ts.astype(float) - tg.astype(float)); out["topple_diff_pp_upper95"] = round(100 * boot(tdiff)[2], 1)
    F = {(r["ep"], r["d"]): r for f in sorted(fc.OUT.glob("rows_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    out["reproduces_fixcf"] = {b: sum(F[k][a] == R["sel"][k][b] for k in R["sel"] if k in F) for a, b in (("nominal", "nominal"), ("fix", "gcem"))}
    out["sel_kinds"] = {k: sum(r["sel_kind"] == k for r in S) for k in ("certified", "fallback_hold", "fallback_best")}
    out["PRIMARY_pass"] = bool(out["PRIMARY_sel_minus_gcem_delay"]["mean_lower_upper90"][2] < 0 and out["topple_diff_pp_upper95"] <= P["topple_margin_pp"])
    Rp = {(r["ep"], r["d"]): r for f in sorted(OUT.glob("proj_*.jsonl")) for r in map(json.loads, f.read_text().splitlines())}
    if Rp:
        T = list(Rp.values()); dp_, dg2 = delay(T, "proj"), delay(T, "gcem")
        K3 = sorted(set(dp_) & set(dg2)); z = np.array([dp_[k] - dg2[k] for k in K3], float)
        tp = np.array([r["proj"]["topple"] is not None for r in T]); tg2 = np.array([r["gcem"]["topple"] is not None for r in T])
        out["proj"] = {"cases": len(T), "topples": int(tp.sum()), "topples_gcem_same_job": int(tg2.sum()),
                       "kinds": {k: sum(r["proj_kind"] == k for r in T) for k in ("certified", "fallback_fix")},
                       "mean_delay": boot(np.array(list(dp_.values()), float)),
                       "proj_minus_gcem_delay": {"n": len(K3), "mean_lower_upper90": boot(z)},
                       "topple_diff_pp_upper95": round(100 * boot(tp.astype(float) - tg2.astype(float))[2], 1)}
        out["proj"]["pass"] = bool(out["proj"]["proj_minus_gcem_delay"]["mean_lower_upper90"][2] < 0 and out["proj"]["topple_diff_pp_upper95"] <= P["topple_margin_pp"])
    if R["steer"]:
        T = list(R["steer"].values()); dt = delay(T, "steer")
        out["steer"] = {"cases": len(T), "topples": sum(r["steer"]["topple"] is not None for r in T),
                        "topples_nominal_same_job": sum(r["nominal"]["topple"] is not None for r in T),
                        "mean_delay": boot(np.array(list(dt.values()), float))}
        dg3 = delay(T, "gcem")                                            # same job's gcem branch (amendment 2)
        K2 = sorted(set(dt) & set(dg3)); y = np.array([dt[k] - dg3[k] for k in K2], float)
        ts_, tg3 = (np.array([r[b]["topple"] is not None for r in T]) for b in ("steer", "gcem"))
        out["steer"]["topples_gcem_same_job"] = int(tg3.sum())
        out["steer"]["topple_diff_pp_upper95"] = round(100 * boot(ts_.astype(float) - tg3.astype(float))[2], 1)
        out["steer_minus_gcem_delay"] = {"n": len(K2), "mean_lower_upper90": boot(y)}
        out["nominal_reproduced_across_jobs"] = sum(R["steer"][k]["nominal"] == R["sel"][k]["nominal"] for k in set(R["steer"]) & set(R["sel"]))
    (ROOT / "results/jenga/fixbench_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "analyse": analyse}[sys.argv[1]]()
