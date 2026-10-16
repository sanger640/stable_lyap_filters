"""Single CBF optimisation filter (no hand-made primitives), validated offline on the 75 really risky dev decisions.

Decision variables per start: smooth xyz offsets at 4 waypoints (mm) + a speed factor sigma in [0.3, 1] (time-scaling
along the plan; sigma < 1 = slower).  Problem:
    min  ||p(theta)_{1:8} - p^pi_{1:8}||^2              (change the policy's intention as little as possible)
    s.t. obar(p(theta)) <= (1 - alpha) * obar(p^pi)       (relative barrier: predicted disturbance away from the hand)
obar = mean over imagined futures of the latent change outside the hand mask (tracks real nudge effects r = 0.76).
Solved by a penalty method with Adam, batched over starts = the policy's plan + 3 of the policy's own samples
(initialisations only). Answer: the cheapest start that satisfies the constraint (else the lowest obar).
Real risk of the answer: exact simulator (32 futures, same draws as every other plan of the decision -> paired).

usage: python3 eval/cbf_opt.py run [tag] | truth [tag] | report [tag]
env: OPT_ALPHA (0.3), OPT_STARTS (4), OPT_STEPS (15), OPT_N (16 futures), OPT_LAMBDA (400: cost of sigma = 0.5 ~ 10 mm)
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("EXEC_NOISE_SCALE", "2"); os.environ.setdefault("HARD_MASS", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import cbf_qp as q  # noqa: E402

E = lambda k, d: type(d)(os.environ.get(k, d))
CFG = {"ALPHA": E("OPT_ALPHA", 0.3), "STARTS": E("OPT_STARTS", 4), "STEPS": E("OPT_STEPS", 15), "N": E("OPT_N", 16),
       "LAMBDA": E("OPT_LAMBDA", 400.0), "MU": E("OPT_MU", 1e4), "LR_MM": E("OPT_LR_MM", 1.5), "MAX_MM": E("OPT_MAX_MM", 40.0),
       # COST "separate" (2026-10-16 fix, matches the written formulation ||delta||^2 + lambda (1 - sigma)^2): the path change
       # is measured on the bent path GEOMETRY before time-scaling, so slowing down is charged only by lambda (1 - sigma)^2.
       # "timed" = the old code: compared step-by-step in time, so a slower plan also counted as a large path change.
       # SIG0: sigma initialisations; each of the first STARTS / len(SIG0) starts is tried at every SIG0 (initialisations only).
       "COST": os.environ.get("OPT_COST", "separate"), "SIG0": [0.99, 0.55]}


def time_scale(plan, sigma):
    """plan (S,16,4) torch, sigma (S,) -> plan executed at speed sigma: step t goes to the plan's fractional index t*sigma."""
    import torch
    S = plan.shape[0]
    t = torch.arange(16, device=plan.device, dtype=plan.dtype)[None] * sigma[:, None]          # (S,16)
    lo = t.floor().clamp(0, 15).long(); hi = (lo + 1).clamp(max=15); w = (t - lo.float())[..., None]
    gather = lambda idx: torch.gather(plan[..., :3], 1, idx[..., None].expand(-1, -1, 3))
    xyz = (1 - w) * gather(lo) + w * gather(hi)
    grip = torch.gather(plan[..., 3], 1, t.round().clamp(0, 15).long())
    return torch.cat([xyz, grip[..., None]], -1)


def run(tag="opt"):
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    C, K = q.load_cache()
    rows = json.loads((q.OUT / "checks" / "plans.json").read_text())                          # 75 risky decisions
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    out, t0 = [], time.time()
    for n_, r in enumerate(rows):
        i = r["i"]; z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        plan, info = optimise(mon, cbf, sim, z0, C["site"][i], np.asarray(r["plans"], np.float32)[:CFG["STARTS"]], K[i]["h_now"], 1000 + i)
        out.append({"i": i, **info, "chunk": plan[1:9].astype(np.float32).tolist()})
        if n_ % 15 == 0:
            print(f"run {n_ + 1}/{len(rows)} {time.time() - t0:.0f}s start {info['start']} sigma {info['sigma']} change {info['change_mm']} mm feasible {info['feasible']}", flush=True)
    (q.OUT / f"cbfopt_{tag}.json").write_text(json.dumps({"cfg": CFG, "rows": out}))
    print("run done", len(out), f"{time.time() - t0:.0f}s", flush=True)


def optimise(mon, cbf, sim, z0, site, starts, h_now, seed, fork_check=None):
    """The single CBF optimisation for one decision. starts (S,16,4): the policy's plan first, then its own samples.
    sim must be at the decision state (for the exact fork check). Returns (plan (16,4) numpy, info)."""
    import torch
    fork_check = (os.environ.get("OPT_FORK_CHECK") == "1") if fork_check is None else fork_check
    M = torch.as_tensor(q.knot_matrix(), device="cuda", dtype=torch.float32)
    bar = q.Barrier(mon, z0, seed, CFG["N"], 1.0)
    P = torch.as_tensor(np.asarray(starts, np.float32), device="cuda")
    pol = P[:1]
    sig0 = CFG["SIG0"] if CFG["COST"] == "separate" else [0.99]
    if len(sig0) > 1:                                # same number of starts in total: fewer plans, each at every sigma init
        P = P[:max(1, len(P) // len(sig0))].repeat(len(sig0), 1, 1)
    s_init = torch.tensor([v for v in sig0 for _ in range(len(P) // len(sig0))], device="cuda")
    s_init = torch.logit(((s_init - 0.3) / 0.7).clamp(1e-3, 1 - 1e-3))
    with torch.no_grad():
        o_pol = float(q.mean_o(bar, pol, z0, site)[0])
    tau = (1 - CFG["ALPHA"]) * o_pol
    S = len(P)
    knots = torch.zeros(S, len(q.QP["KNOTS"]), 3, device="cuda", requires_grad=True)
    s_raw = s_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [knots], "lr": CFG["LR_MM"]}, {"params": [s_raw], "lr": 0.3}])

    def plans_of(knots, s_raw):
        sig = 0.3 + 0.7 * torch.sigmoid(s_raw)
        off = torch.einsum("tk,skd->std", M, knots) / 1000
        if CFG["COST"] == "separate":                # bend the path, then execute it at speed sigma
            geo = torch.cat([P[..., :3] + off, P[..., 3:]], -1)
            return time_scale(geo, sig), sig, geo
        p = time_scale(P, sig)
        p = torch.cat([p[..., :3] + off, p[..., 3:]], -1)
        return p, sig, p

    def change(geo):                                 # mm^2 from the policy's plan (path geometry, or timed for "timed")
        return ((geo[:, 1:9, :3] - pol[:, 1:9, :3]) * 1000).pow(2).sum((1, 2))
    for _ in range(CFG["STEPS"]):
        p, sig, geo = plans_of(knots, s_raw)
        ob = q.mean_o(bar, p, z0, site)
        dist = change(geo)
        loss = (dist / 100 + CFG["LAMBDA"] * (1 - sig) ** 2 / 100 + CFG["MU"] * torch.relu(ob / max(tau, 1e-9) - 1) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            nrm = knots.norm(dim=(1, 2), keepdim=True).clamp_min(1e-9)
            knots.mul_(torch.clamp(CFG["MAX_MM"] / nrm, max=1.0))
    with torch.no_grad():
        p, sig, geo = plans_of(knots, s_raw)
        bar32 = q.Barrier(mon, z0, seed + 1000, 32, 1.0)                                       # fresh draws for the choice
        ob = q.mean_o(bar32, p, z0, site).cpu().numpy()
        tau32 = (1 - CFG["ALPHA"]) * float(q.mean_o(bar32, pol, z0, site)[0])
        dist = change(geo).sqrt().cpu().numpy()
        cost = (change(geo) / 100 + CFG["LAMBDA"] * (1 - sig) ** 2 / 100).cpu().numpy()     # the objective, for the choice
    feas = ob <= tau32
    pall = p.cpu().numpy()
    if fork_check:                                   # second safety condition: the exact fork barrier rule
        hh, oo = cbf.score(sim, z0, pall, 64)
        okf = cbf.ok(hh, oo, h_now)
        both = feas & okf
        pool_ = both if both.any() else (okf if okf.any() else feas)
    else:
        pool_ = feas
    key = cost if CFG["COST"] == "separate" else dist
    j = int(np.flatnonzero(pool_)[np.argmin(key[pool_])]) if pool_.any() else int(np.argmin(ob))
    plan = pall[j]
    h, o = cbf.score(sim, z0, plan[None], 64)
    info = {"start": j, "feasible": bool(feas[j]), "n_feasible": int(feas.sum()), "sigma": round(float(sig[j]), 3),
            "change_mm": round(float(dist[j]), 1), "obar_policy": round(o_pol, 4), "obar_final": round(float(ob[j]), 4),
            "verified": bool(cbf.ok(h, o, h_now)[0])}
    return plan, info


def truth(tag="opt"):
    C, K = q.load_cache()
    os.environ["TRUTH_N"] = "32"
    cache_p = q.OUT / "truth_32.json"; cache = json.loads(cache_p.read_text())
    jobs = []
    for r in json.loads((q.OUT / f"cbfopt_{tag}.json").read_text())["rows"]:
        i = r["i"]; ch = np.asarray(r["chunk"], np.float32); h = q._hash(ch)
        if h not in cache:
            jobs.append((h, ((i, 7, 0), C["state"][i], C["ctrl"][i], C["target"][i], C["gripper"][i], C["tilt"][i], ch))); cache[h] = None
    print(len(jobs), "new truth jobs", flush=True)
    with mp.get_context("spawn").Pool(30) as pool:
        for h, flags in pool.imap_unordered(q._truth_one, jobs):
            cache[h] = sum(flags) / len(flags)
    cache_p.write_text(json.dumps(cache))


def report(tag="opt"):
    C, K = q.load_cache()
    T = json.loads((q.OUT / "truth_32.json").read_text())
    rows = json.loads((q.OUT / f"cbfopt_{tag}.json").read_text())["rows"]
    hyb = {r["i"]: r for r in json.loads((q.OUT / "opt_hybrid.json").read_text())["rows"]}
    eps = 2 / 32
    rp = np.array([T[q._hash(C["nominal"][r["i"]])] for r in rows]); rs = np.array([T[q._hash(C["executed"][r["i"]])] for r in rows])
    ro = np.array([T[q._hash(np.asarray(r["chunk"], np.float32))] for r in rows])
    rh = np.array([T[q._hash(np.asarray(hyb[r["i"]]["chunk"], np.float32))] for r in rows])
    res = {"tag": tag, "cfg": json.loads((q.OUT / f"cbfopt_{tag}.json").read_text())["cfg"], "decisions": len(rows),
           "real_risk": {"policy": round(float(rp.mean()), 4), "selection": round(float(rs.mean()), 4), "hybrid": round(float(rh.mean()), 4),
                         "cbf_opt": round(float(ro.mean()), 4)},
           "cbf_opt_vs_policy": {"helped": int((ro < rp - eps).sum()), "hurt": int((ro > rp + eps).sum())},
           "cbf_opt_vs_selection": {"better": int((ro < rs - eps).sum()), "worse": int((ro > rs + eps).sum())},
           "feasible_rate": round(float(np.mean([r["feasible"] for r in rows])), 3),
           "verified_rate": round(float(np.mean([r["verified"] for r in rows])), 3),
           "median_change_mm": float(np.median([r["change_mm"] for r in rows])),
           "slowed_down (sigma<0.9)": int(sum(r["sigma"] < 0.9 for r in rows)),
           "start_used": {str(k): int(sum(r["start"] == k for r in rows)) for k in range(CFG["STARTS"])}}
    (q.OUT / f"cbfopt_report_{tag}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"run": run, "truth": truth, "report": report}[sys.argv[1]](*sys.argv[2:])
