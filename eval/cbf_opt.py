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
       # SOLVER: "grad" = penalty method on the mean disturbance (dev8); "cem" = cross-entropy method on the EXACT accepted
       # rule (fork barrier h >= min(h_now, h_safe), o <= O_MAX) with the same objective; "al" = augmented-Lagrangian
       # gradient solver on a CVaR disturbance barrier (risk-aware), more steps, early stop.
       "SOLVER": os.environ.get("OPT_SOLVER", "grad"), "CEM_POP": E("OPT_CEM_POP", 32), "CEM_ITERS": E("OPT_CEM_ITERS", 3),
       "CEM_ELITES": E("OPT_CEM_ELITES", 6), "CEM_SD_MM": E("OPT_CEM_SD_MM", 8.0), "CEM_N": E("OPT_CEM_N", 32),
       # CEM v2 (2026-10-16, after CEM v1 returned the flagged policy plan unchanged in 45/75 risky decisions because a
       # re-test with new random draws passed it): CRN = same draws for every candidate of a decision; NO_NOMINAL =
       # exclude the untouched flagged plan; VERIFY = check the answer on fresh draws, fall back to the next candidate.
       "CEM_CRN": E("OPT_CEM_CRN", 0), "CEM_NO_NOMINAL": E("OPT_CEM_NO_NOMINAL", 0), "CEM_VERIFY": E("OPT_CEM_VERIFY", 1), "CVAR": E("OPT_CVAR", 0.25),
       "AL_STEPS": E("OPT_AL_STEPS", 20), "AL_N": E("OPT_AL_N", 32),
       # SOLVER "chunk" (2026-10-16): decision variable = the executed action chunk itself (steps 1-8, xyz; gripper
       # fixed), lookahead 9-15 = the policy's own continuation shifted by the step-8 change; cost = weighted change +
       # ETA * second differences (smoothness, incl. the current target); ONE learned barrier B (eval/learned_barrier.py,
       # predicted real topple fraction) both as the constraint B <= EPS and for acceptance (fresh draws); augmented
       # Lagrangian; starts = the policy's plan and its own samples, as initial chunks.
       "EPS": E("OPT_EPS", 0.10), "ETA": E("OPT_ETA", 4.0), "CH_STEPS": E("OPT_CH_STEPS", 20),
       "COST": os.environ.get("OPT_COST", "separate"), "SIG0": [float(v) for v in os.environ.get("OPT_SIG0", "0.99,0.55").split(",")],
       # world model (2026-10-16: runs after b50 silently used the cem_filter default threeblock_v3 while b50 used model_CF)
       "WM": os.environ.get("CBF_WM", "threeblock_v3 (default)"),
       # PLAUS_W (2026-10-16, step 2): policy-plausibility cost (eval/plausibility.py) in the grad solver:
       # + PLAUS_W * relu(err(p) - err(p_pi)) / err(p_pi). Fixed a priori: doubling the policy's own denoising error costs
       # as much as a 15 mm change (15^2 / 100 = 2.25). 0 = off.
       "PLAUS_W": E("OPT_PLAUS_W", 0.0)}


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
    rows = rows[:int(os.environ.get("OPT_LIMIT", len(rows)))]
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    plaus_of = _plaus_setup() if CFG["PLAUS_W"] > 0 else None
    out, t0 = [], time.time()
    for n_, r in enumerate(rows):
        i = r["i"]; z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        t1 = time.time(); pl = plaus_of(K[i]) if plaus_of else None
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        plan, info = optimise(mon, cbf, sim, z0, C["site"][i], np.asarray(r["plans"], np.float32)[:CFG["STARTS"]], K[i]["h_now"], 1000 + i,
                              decision=i, plaus=pl)
        out.append({"i": i, **info, "seconds": round(time.time() - t1, 1), "chunk": plan[1:9].astype(np.float32).tolist()})
        if n_ % 15 == 0:
            print(f"run {n_ + 1}/{len(rows)} {time.time() - t0:.0f}s start {info['start']} sigma {info['sigma']} change {info['change_mm']} mm feasible {info['feasible']}", flush=True)
    (q.OUT / f"cbfopt_{tag}.json").write_text(json.dumps({"cfg": CFG, "rows": out}))
    print("run done", len(out), f"{time.time() - t0:.0f}s", flush=True)


def _plaus_setup():
    """Offline: key {ep, d} of a cached dev3A decision -> plaus(plans) (differentiable policy denoising error, conditioned
    on that decision's re-rendered 2-frame observation). Needs the diffusion policy (run under dp_venv)."""
    import inspector_data as insp
    import plausibility as pl
    import policy_rollout as pr
    P = pl.Plausibility(pr.load_policy()); sim2 = insp._sim(); cams = pr.Cameras(sim2)

    def plaus_of(key):
        with np.load(q.RUN / f"ep{key['ep']:03d}.npz") as f:
            S = {k: f[k] for k in f.files}
        d = int(np.flatnonzero(S["chunk_index"] == key["d"])[0])
        gc = P.cond(pl.obs_hist(sim2, cams, S, d, pl.episode_noises(key["ep"], int(S["chunk_index"][-1]) + 1)))
        return lambda plans: P.error(gc, plans)
    return plaus_of


def plaus_tag(tag):
    """Relative excess plausibility error of an existing run's chosen chunks (as 16-step plans) -> cbfopt_plaus_<tag>.json."""
    import torch
    import plausibility as pl
    C, K = q.load_cache(); plaus_of = _plaus_setup()
    rows = json.loads((q.OUT / f"cbfopt_{tag}.json").read_text())["rows"]; out = {}
    for r in rows:
        i = r["i"]; f = plaus_of(K[i])
        with torch.no_grad():
            e = f(np.stack([C["plan16"][i], pl.as_plan(C["plan16"][i], r["chunk"])])).cpu().numpy()
        out[i] = round(float(max(e[1] - e[0], 0) / e[0]), 4)
    (q.OUT / f"cbfopt_plaus_{tag}.json").write_text(json.dumps(out))
    print(tag, "median relative excess error", float(np.median(list(out.values()))), flush=True)


def optimise(mon, cbf, sim, z0, site, starts, h_now, seed, fork_check=None, decision=None, return_all=False, plaus=None):
    """The single CBF optimisation for one decision. starts (S,16,4): the policy's plan first, then its own samples.
    sim must be at the decision state (for the exact fork check). Returns (plan (16,4) numpy, info)."""
    import torch
    fork_check = (os.environ.get("OPT_FORK_CHECK") == "1") if fork_check is None else fork_check
    if CFG["SOLVER"] == "cem":
        return optimise_cem(mon, cbf, sim, z0, site, starts, h_now, seed)
    if CFG["SOLVER"] == "chunk":
        return optimise_chunk(mon, cbf, sim, z0, site, starts, h_now, seed, decision)
    if CFG["SOLVER"] == "al":
        return optimise_al(mon, cbf, sim, z0, site, starts, h_now, seed)
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
        e_pol = float(plaus(pol)[0]) if plaus is not None else None
    tau = (1 - CFG["ALPHA"]) * o_pol
    W = CFG["PLAUS_W"] if plaus is not None else 0.0
    prel = lambda p: torch.relu(plaus(p) - e_pol) / e_pol                # relative excess denoising error
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
        loss = (dist / 100 + CFG["LAMBDA"] * (1 - sig) ** 2 / 100 + CFG["MU"] * torch.relu(ob / max(tau, 1e-9) - 1) ** 2)
        if W > 0:
            loss = loss + W * prel(p)
        loss = loss.sum()
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
        pr_all = prel(p).cpu().numpy() if plaus is not None else np.zeros(len(p))
        cost = cost + W * pr_all
    feas = ob <= tau32
    pall = p.cpu().numpy()
    if fork_check:                                   # second safety condition: the exact fork barrier rule
        hh, oo = cbf.score(sim, z0, pall, 64)
        okf = cbf.ok(hh, oo, h_now)
        both = feas & okf
        pool_ = both if both.any() else (okf if okf.any() else feas)
    else:
        pool_ = feas
    key = cost if (CFG["COST"] == "separate" or W > 0) else dist
    j = int(np.flatnonzero(pool_)[np.argmin(key[pool_])]) if pool_.any() else int(np.argmin(ob))
    plan = pall[j]
    h, o = cbf.score(sim, z0, plan[None], 64)
    info = {"start": j, "feasible": bool(feas[j]), "n_feasible": int(feas.sum()), "sigma": round(float(sig[j]), 3),
            "change_mm": round(float(dist[j]), 1), "obar_policy": round(o_pol, 4), "obar_final": round(float(ob[j]), 4),
            "verified": bool(cbf.ok(h, o, h_now)[0])}
    if plaus is not None:
        info["plaus_rel"] = round(float(pr_all[j]), 4)
    if return_all:                                   # every optimised start (world-model data collection, eval/oil_data.py)
        info["all"], info["ob_all"] = pall, ob
    return plan, info


def _build(P, M, c, kn, sg):
    """Plans from (start index c (B,), knots kn (B,K,3) mm, speed sg (B,)): bend the start's path, then run it at sg."""
    import torch
    geo = P[c].clone()
    geo[..., :3] = geo[..., :3] + torch.einsum("tk,bkd->btd", M, kn) / 1000
    return time_scale(geo, sg), geo


def _objective(geo, sg, pol):
    """The objective of the formulation: path change (mm^2 / 100, geometry, steps 1:8) + lambda (1 - sigma)^2 / 100."""
    ch = ((geo[:, 1:9, :3] - pol[:, 1:9, :3]) * 1000).pow(2).sum((1, 2))
    return ch / 100 + CFG["LAMBDA"] * (1 - sg) ** 2 / 100, ch.sqrt()


def _rank(ok, J, h, o):
    """Feasible (exact rule) first by objective; infeasible by largest h, then smallest o."""
    return sorted(range(len(J)), key=lambda i: (0, J[i], 0) if ok[i] else (1, -h[i], o[i]))


def optimise_cem(mon, cbf, sim, z0, site, starts, h_now, seed):
    """Cross-entropy method on the exact accepted rule. Search space: which policy sample to bend (categorical; the
    policy's samples are only the initial distribution), 4 xyz knots (Gaussian, |.| <= MAX_MM) and sigma (Gaussian,
    clipped to [0.3, 1]). Every candidate gets the exact fork barrier (64 futures x 4 scales), as the selection filter's
    candidates do; the answer is the cheapest candidate satisfying the rule (else the largest h)."""
    import torch
    rng = np.random.default_rng(seed); rng_saved = mon.rng              # restored at the end (closed loop)
    M = torch.as_tensor(q.knot_matrix(), device="cuda", dtype=torch.float32)
    P = torch.as_tensor(np.asarray(starts, np.float32), device="cuda"); pol = P[:1]
    S, K, pop, ne = len(P), len(q.QP["KNOTS"]), CFG["CEM_POP"], CFG["CEM_ELITES"]
    probs = np.full(S, 1 / S); mu = np.zeros((K, 3)); sd = np.full((K, 3), CFG["CEM_SD_MM"]); smu, ssd = 0.85, 0.2
    C_, KN, SG, H, O, OK, J, CH = [], [], [], [], [], [], [], []
    for it in range(CFG["CEM_ITERS"]):
        c = np.repeat(np.arange(S), int(np.ceil(pop / S)))[:pop] if it == 0 else rng.choice(S, pop, p=probs)
        kn = rng.normal(mu, sd, (pop, K, 3)); sg = np.clip(rng.normal(smu, ssd, pop), 0.3, 1.0)
        if it == 0:                                  # include every start untouched (bend 0, full speed)
            first = np.r_[0, np.flatnonzero(np.diff(c)) + 1]; kn[first] = 0; sg[first] = 1.0
            if CFG["CEM_NO_NOMINAL"]:                # ... except the flagged policy plan itself
                kn[0] = rng.normal(0, CFG["CEM_SD_MM"], (K, 3)); sg[0] = np.clip(rng.normal(smu, ssd), 0.3, 1.0)
        nrm = np.linalg.norm(kn.reshape(pop, -1), axis=1); kn *= np.minimum(1, CFG["MAX_MM"] / np.maximum(nrm, 1e-9))[:, None, None]
        with torch.no_grad():
            pl, geo = _build(P, M, torch.as_tensor(c, device="cuda"), torch.as_tensor(kn, device="cuda", dtype=torch.float32),
                             torch.as_tensor(sg, device="cuda", dtype=torch.float32))
            j_, ch_ = _objective(geo, torch.as_tensor(sg, device="cuda", dtype=torch.float32), pol)
        if CFG["CEM_CRN"]:
            mon.rng = np.random.default_rng(seed + 7)    # common random numbers: same draws for every candidate
        h, o = cbf.score(sim, z0, pl.cpu().numpy(), CFG["CEM_N"]); ok = cbf.ok(h, o, h_now)
        if CFG["CEM_NO_NOMINAL"] and it == 0:
            ok[0] = False; h[0] = -np.inf                # never return the untouched flagged plan
        C_ += list(c); KN += list(kn); SG += list(sg); H += list(h); O += list(o); OK += list(ok)
        J += list(j_.cpu().numpy()); CH += list(ch_.cpu().numpy())
        el = _rank(OK, J, H, O)[:ne]                 # elites over everything seen so far
        cnt = np.bincount(np.asarray(C_)[el], minlength=S)
        probs = 0.8 * cnt / ne + 0.2 / S
        ek = np.asarray(KN)[el]; es = np.asarray(SG)[el]
        mu, sd = ek.mean(0), np.maximum(ek.std(0), 2.0); smu, ssd = es.mean(), max(es.std(), 0.05)
    order = _rank(OK, J, H, O)
    if CFG["CEM_CRN"]:
        mon.rng = np.random.default_rng(seed + 99)       # fresh draws for the verification
    for tries, b in enumerate(order[:max(1, CFG["CEM_VERIFY"])]):
        with torch.no_grad():
            pl, _ = _build(P, M, torch.as_tensor([C_[b]], device="cuda"), torch.as_tensor(KN[b][None], device="cuda", dtype=torch.float32),
                           torch.as_tensor([SG[b]], device="cuda", dtype=torch.float32))
        plan = pl[0].cpu().numpy()
        hv, ov = cbf.score(sim, z0, plan[None], 64)  # fresh draws
        if cbf.ok(hv, ov, h_now)[0] or not OK[b]:
            break
    info = {"start": int(C_[b]), "feasible": bool(OK[b]), "n_feasible": int(np.sum(OK)), "sigma": round(float(SG[b]), 3),
            "change_mm": round(float(CH[b]), 1), "h_choice": float(H[b]), "evaluated": len(J), "verify_tries": tries + 1,
            "verified": bool(cbf.ok(hv, ov, h_now)[0])}
    mon.rng = rng_saved
    return plan, info


def _o_futures(bar, plans, z0, site):
    """Per-future latent change away from the hand: (C, n)."""
    import torch
    import cbf_barrier as cb
    _, e30 = bar.endpoints(plans)
    out = []
    for c in range(len(plans)):
        pts = np.vstack([site[None], plans[c][1:16, :3].detach().cpu().numpy()])
        m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
        diff = (e30[c].reshape(-1, 28, 28, 8) - z0[None]).pow(2).mean(-1)
        out.append((diff * m).sum((1, 2)) / m.sum())
    return torch.stack(out)


def _cvar(x, beta):
    k = max(1, int(np.ceil(beta * x.shape[1])))
    return x.topk(k, dim=1).values.mean(1)


def optimise_al(mon, cbf, sim, z0, site, starts, h_now, seed):
    """Gradient solver, improved: risk-aware barrier g = CVaR_beta(outside-hand change) <= (1 - alpha) g(policy),
    augmented Lagrangian (multiplier + growing penalty) instead of a fixed penalty, AL_STEPS steps with early stop once
    every start is feasible, AL_N futures. The horizon already ends in 20 hold steps after the plan (backup: hold still).
    Choice: cheapest start satisfying the CVaR barrier (fresh draws) AND the exact fork rule (as cbfmo)."""
    import torch
    M = torch.as_tensor(q.knot_matrix(), device="cuda", dtype=torch.float32)
    P0 = torch.as_tensor(np.asarray(starts, np.float32), device="cuda"); pol = P0[:1]
    sig0 = CFG["SIG0"]
    P = P0[:max(1, len(P0) // len(sig0))].repeat(len(sig0), 1, 1); S = len(P)
    c = torch.arange(S, device="cuda")
    s_init = torch.tensor([v for v in sig0 for _ in range(S // len(sig0))], device="cuda")
    bar = q.Barrier(mon, z0, seed, CFG["AL_N"], 1.0)
    with torch.no_grad():
        tau = (1 - CFG["ALPHA"]) * float(_cvar(_o_futures(bar, pol, z0, site), CFG["CVAR"])[0])
    knots = torch.zeros(S, len(q.QP["KNOTS"]), 3, device="cuda", requires_grad=True)
    s_raw = torch.logit(((s_init - 0.3) / 0.7).clamp(1e-3, 1 - 1e-3)).clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [knots], "lr": CFG["LR_MM"]}, {"params": [s_raw], "lr": 0.3}])
    lam = torch.zeros(S, device="cuda"); rho = 10.0
    for it in range(CFG["AL_STEPS"]):
        sg = 0.3 + 0.7 * torch.sigmoid(s_raw)
        pl, geo = _build(P, M, c, knots, sg)
        g = _cvar(_o_futures(bar, pl, z0, site), CFG["CVAR"]) / max(tau, 1e-9) - 1         # <= 0 is feasible
        J, _ = _objective(geo, sg, pol)
        loss = (J + lam * g + 0.5 * rho * torch.relu(g + lam / rho) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            nrm = knots.norm(dim=(1, 2), keepdim=True).clamp_min(1e-9)
            knots.mul_(torch.clamp(CFG["MAX_MM"] / nrm, max=1.0))
            if (it + 1) % 5 == 0:
                lam = torch.clamp(lam + rho * g.detach(), min=0); rho = min(rho * 2, 1e4)
            if bool((g <= 0).all()) and it >= 4:
                break
    with torch.no_grad():
        sg = 0.3 + 0.7 * torch.sigmoid(s_raw)
        pl, geo = _build(P, M, c, knots, sg)
        bar2 = q.Barrier(mon, z0, seed + 1000, CFG["AL_N"], 1.0)
        tau2 = (1 - CFG["ALPHA"]) * float(_cvar(_o_futures(bar2, pol, z0, site), CFG["CVAR"])[0])
        g2 = _cvar(_o_futures(bar2, pl, z0, site), CFG["CVAR"]).cpu().numpy()
        J, ch = _objective(geo, sg, pol); J = J.cpu().numpy(); ch = ch.cpu().numpy()
    feas = g2 <= tau2
    pall = pl.cpu().numpy()
    hh, oo = cbf.score(sim, z0, pall, 64); okf = cbf.ok(hh, oo, h_now)
    both = feas & okf
    pool_ = both if both.any() else (okf if okf.any() else feas)
    j = int(np.flatnonzero(pool_)[np.argmin(J[pool_])]) if pool_.any() else int(np.argmax(hh))
    plan = pall[j]
    h, o = cbf.score(sim, z0, plan[None], 64)
    info = {"start": j, "feasible": bool(feas[j]), "n_feasible": int(feas.sum()), "fork_ok": int(okf.sum()),
            "sigma": round(float(sg[j]), 3), "change_mm": round(float(ch[j]), 1), "steps": it + 1,
            "verified": bool(cbf.ok(h, o, h_now)[0])}
    return plan, info


def _chunk_plans(pol, D):
    """pol (1,16,4), D (S,8,3) m: steps 1-8 = policy + D, steps 9-15 = policy continuation + D[step 8]; gripper kept."""
    import torch
    off = torch.cat([torch.zeros_like(D[:, :1]), D, D[:, -1:].expand(-1, 7, -1)], 1)          # (S,16,3)
    p = pol.expand(len(D), -1, -1).clone()
    p[..., :3] = p[..., :3] + off
    return p


def _chunk_cost(D):
    """mm^2 / 100: change of the executed steps (weights 1) + ETA x second differences incl. the fixed step 0."""
    import torch
    Dm = D * 1000
    full = torch.cat([torch.zeros_like(Dm[:, :1]), Dm], 1)
    d2 = full[:, 2:] - 2 * full[:, 1:-1] + full[:, :-2]
    return (Dm.pow(2).sum((1, 2)) + CFG["ETA"] * d2.pow(2).sum((1, 2))) / 100, Dm.pow(2).sum((1, 2)).sqrt()


def optimise_chunk(mon, cbf, sim, z0, site, starts, h_now, seed, decision=None):
    import torch
    import learned_barrier as lb
    B = lb.load(decision)
    P0 = torch.as_tensor(np.asarray(starts, np.float32), device="cuda"); pol = P0[:1]
    D = (P0[:, 1:9, :3] - pol[:, 1:9, :3]).clone().requires_grad_(True)                     # starts as initial chunks
    bar = q.Barrier(mon, z0, seed, lb.N_FUT, 1.0)
    opt = torch.optim.Adam([D], lr=CFG["LR_MM"] / 1000)
    lam = torch.zeros(len(D), device="cuda"); rho = 10.0; eps = CFG["EPS"]
    for it in range(CFG["CH_STEPS"]):
        pl = _chunk_plans(pol, D)
        b = B(lb.features(bar, pl, z0, site))
        g = b / eps - 1
        J, _ = _chunk_cost(D)
        loss = (J + lam * g + 0.5 * rho * torch.relu(g + lam / rho) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                        # trust region: |change| <= MAX_MM per step
            n = D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            D.mul_(torch.clamp(CFG["MAX_MM"] / 1000 / n, max=1.0))
            if (it + 1) % 5 == 0:
                lam = torch.clamp(lam + rho * g.detach(), min=0); rho = min(rho * 2, 1e4)
    with torch.no_grad():
        pl = _chunk_plans(pol, D)
        bar2 = q.Barrier(mon, z0, seed + 1000, lb.N_FUT, 1.0)                                 # fresh draws: acceptance
        b2 = B(lb.features(bar2, pl, z0, site)).cpu().numpy()
        J, ch = _chunk_cost(D); J = J.cpu().numpy(); ch = ch.cpu().numpy()
    feas = b2 <= eps
    j = int(np.flatnonzero(feas)[np.argmin(J[feas])]) if feas.any() else int(np.argmin(b2))
    plan = pl[j].cpu().numpy()
    h, o = cbf.score(sim, z0, plan[None], 64)
    info = {"start": j, "feasible": bool(feas[j]), "n_feasible": int(feas.sum()), "sigma": 1.0,
            "change_mm": round(float(ch[j]), 1), "B_choice": round(float(b2[j]), 4), "B_policy": None,
            "verified": bool(cbf.ok(h, o, h_now)[0])}
    with torch.no_grad():
        info["B_policy"] = round(float(B(lb.features(bar2, pol, z0, site))[0]), 4)
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
    {"run": run, "truth": truth, "report": report, "plaus": plaus_tag}[sys.argv[1]](*sys.argv[2:])
