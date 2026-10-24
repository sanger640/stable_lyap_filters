"""Formulation G (docs/OPT_FILTER_REDESIGN.md, pre-registered 2026-10-17): optimise 4 coarse, universal transformations
of the ONE given chunk instead of free mm offsets. theta = (v speed in [0,1], k pause steps in [0,8], s amplitude in [0,1],
b back-off mm in [0,30] along the reverse of the chunk's own net direction). Gripper untouched (OPT3_GRIP=shift adds a 5th knob,
d in [-4, 8]: the policy's gripper sequence shifted by round(d) steps, cost GRIP_COST d^2; docs/LITERATURE_PLAN.md step 6). Certificate, cost and
fallback as formulation F (eval/opt2.py). Solvers: cem | grad (separate).

usage: python3 eval/opt3.py run <cem|grad> <dev|holdout> <tag> ; truth <set> <tag> ; report <set> <tag>
env: OPT2_CERT = OIL+ENS (pre-registered) | OIL, OPT2_LIMIT, OPT3_ABS=1 (absolute requirement h >= h_safe)
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import opt2  # noqa: E402  (sets env, CBF_WM / CBF_CALIB defaults)
import bench2 as b2  # noqa: E402

CFG = {**opt2.CFG, "CEM_POP": 16, "CEM_ITERS": 3, "CEM_ELITES": 6, "CEM_FINAL": 4, "CEM_SD": [0.4, 3.0, 0.4, 12.0],
       "G_STARTS": 6, "G_STEPS": 15, "G_LR": [0.05, 0.4, 0.05, 1.5], "B_MIN_MOVE_MM": 2.0,
       "GRIP": os.environ.get("OPT3_GRIP", "fixed"),                              # fixed = G, warp = G2, shift = G-grip
       "GRIP_COST": 2.0, "GRIP_SD": 3.0,                                          # shift: cost per step^2 (= a 5 mm offset over 8 steps)
       "OBJ": os.environ.get("OPT3_OBJ", "cost"),     # cost = G (cheapest certified); o = GO: min o + LAMBDA_O * cost over certified + hold
       "LAMBDA_O": float(os.environ.get("OPT3_LAMBDA_O", 0.003)),                 # (docs/FILTER_DEEP_DIVE.md 5f; = O-CEM's dev choice)
       "MARGIN": os.environ.get("OPT3_MARGIN", "2")}                              # 3 = no fork up to 3x (as closed-loop cbfg3m)
LO, HI = np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 8.0, 1.0, 30.0])
IDENTITY = np.array([1.0, 0.0, 1.0, 0.0])
if CFG["GRIP"] == "shift":                                                        # 5th knob: gripper timing d in [-4, 8] steps
    LO, HI, IDENTITY = np.r_[LO, -4.0], np.r_[HI, 8.0], np.r_[IDENTITY, 0.0]      # (> 0 delays the policy's open/close)
    CFG["CEM_SD"] = CFG["CEM_SD"] + [CFG["GRIP_SD"]]


def transform(p16, theta):
    """p16 (16, 4) torch, theta (S, 4) torch -> plans (S, 16, 4). Differentiable in theta."""
    import torch
    v, k, s, b = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3], theta[:, 3:4] / 1000
    t = torch.arange(16, device=theta.device, dtype=theta.dtype)[None]
    tau = (v * torch.relu(t - k)).clamp(0, 15)                                   # (S, 16)
    i = tau.detach().floor().clamp(0, 14).long(); w = (tau - i)[..., None]
    xyz = p16[:, :3]
    P = xyz[i] * (1 - w) + xyz[i + 1] * w                                         # (S, 16, 3)
    q = xyz[0] + s[..., None] * (P - xyz[0])
    net = xyz[15] - xyz[0]; nrm = net.norm()
    if float(nrm) * 1000 >= CFG["B_MIN_MOVE_MM"]:
        ramp = (t.clamp(max=8) / 8)[..., None]                                    # (1, 16, 1)
        q = q - b[..., None] * ramp * (net / nrm)
    if CFG["GRIP"] == "warp":                                                     # G2: gripper follows the arm's plan time
        g = p16[:, 3][tau.detach().round().long()][..., None]
    elif CFG["GRIP"] == "shift":                                                  # G-grip: the policy's gripper sequence shifted by d
        d = theta[:, 4:5].detach().round()
        g = p16[:, 3][(t - d).clamp(0, 15).long()][..., None]
    else:                                                                         # G: gripper on the policy's clock
        g = p16[None, :, 3:].expand(len(theta), -1, -1)
    return torch.cat([q, g], -1)


class Problem(opt2.Problem):
    """Same certificate as F; the candidate is theta instead of D."""

    def __init__(self, *a):
        import torch
        super().__init__(*a)
        self.p0t = torch.as_tensor(self.p0)

    def plan(self, th):
        import torch
        return transform(self.p0t.double(), torch.as_tensor(np.asarray(th, np.float64))[None])[0].numpy().astype(np.float32)

    def cost(self, th):
        c = super().cost(self.plan(th)[1:9, :3] - self.p0[1:9, :3])
        return c + (CFG["GRIP_COST"] * round(float(th[4])) ** 2 if CFG["GRIP"] == "shift" else 0.0)


def solve_cem(P, seed):
    rng = np.random.default_rng(seed); sd0 = np.array(CFG["CEM_SD"])
    mu, sd = IDENTITY.copy(), sd0.copy(); pool = []
    for it in range(CFG["CEM_ITERS"]):
        T = np.clip(mu + sd * rng.standard_normal((CFG["CEM_POP"], len(LO))), LO, HI)
        state = P.mon.rng.bit_generator.state
        h, o = P.score([P.plan(th) for th in T], CFG["N_SEARCH"])
        P.mon.rng.bit_generator.state = state
        ok = P.cbf.ok(h, o, P.h_hold); cost = np.array([P.cost(th) for th in T])
        obj = cost if CFG["OBJ"] == "cost" else o + CFG["LAMBDA_O"] * cost
        key = [(0 if ok[c] else 1, obj[c] if ok[c] else -h[c], 0 if ok[c] else o[c]) for c in range(len(T))]
        order = sorted(range(len(T)), key=lambda c: key[c])
        el = T[order[:CFG["CEM_ELITES"]]]
        mu, sd = el.mean(0), np.maximum(el.std(0), 0.02 * (HI - LO))
        pool += [(key[c], T[c]) for c in order[:CFG["CEM_FINAL"]]]
    pool.sort(key=lambda x: x[0])
    finals = [th for _, th in pool[:CFG["CEM_FINAL"]]]
    if CFG["OBJ"] == "o":
        out, info = finish_o(P, finals, {"search_certified": int(sum(k[0] == 0 for k, _ in pool))})
    else:
        out, info = opt2.finish(P, finals, {"search_certified": int(sum(k[0] == 0 for k, _ in pool))})
    return out, {**info, "theta": _chosen(P, out, finals)}


def finish_o(P, finals, extra):
    """GO: certify the finals AND hold (gripper held at its current command) with fresh draws on every certifier model;
    among the certified, minimise the pessimistic imagined disturbance (max o over models) + LAMBDA_O * change cost.
    None certified -> as opt2.finish (best (h, -o) among the finals)."""
    plans = [P.plan(th) for th in finals]
    hold = np.repeat(np.r_[P.sim.target, P.p0[0, 3]][None], 16, 0).astype(np.float32)
    ok, H, Os = None, None, []
    for k, m in enumerate(P.cert_models):
        if m is not None:
            saved, P.mon.model = P.mon.model, m
        h, o = P.score(plans + [hold], opt2.CFG["N_CERT"])
        if m is not None:
            P.mon.model = saved
        okk = P.cbf.ok(h, o, P.h_hold); ok = okk if ok is None else ok & okk; Os.append(o)
        if k == 0:
            H = h
    O = np.max(np.stack(Os), 0)
    costs = np.array([P.cost(th) for th in finals] + [opt2.Problem.cost(P, hold[1:9, :3] - P.p0[1:9, :3])])
    J = O + CFG["LAMBDA_O"] * costs
    if ok.any():
        j = int(np.flatnonzero(ok)[np.argmin(J[ok])])
        out, kind = (plans + [hold])[j], ("certified_hold" if j == len(finals) else "certified")
    else:
        j = int(np.lexsort((O[:-1], -H[:-1]))[0]); out, kind = plans[j], "fallback_best"
    change = float(np.abs(out[1:9, :3] - P.p0[1:9, :3]).max() * 1000)
    return out, {"kind": kind, "n_certified": int(ok[:-1].sum()), "hold_certified": bool(ok[-1]), "n_final": len(finals),
                 "change_mm": round(change, 1), "cost": round(float(costs[j]), 2), "O": round(float(O[j]), 4), **extra}


def solve_grad(P, seed):
    import torch
    import cbf_barrier as cb
    import cbf_qp as q
    mon, z0 = P.mon, P.z0
    req = [s for s in (0.5, 1.0, 2.0) if np.log2(s) < min(P.h_hold, opt2.HS) - 1e-9]
    bars = {s: q.Barrier(mon, z0, seed + int(s * 10), CFG["G_N"], s) for s in set(req) | {1.0}}
    rng = np.random.default_rng(seed)
    starts = np.vstack([IDENTITY] + [LO + (HI - LO) * rng.random(4) for _ in range(CFG["G_STARTS"] - 1)])
    lo, hi = torch.as_tensor(LO, device="cuda").float(), torch.as_tensor(HI, device="cuda").float()
    th = torch.as_tensor(starts, device="cuda", dtype=torch.float32).requires_grad_(True)
    p0 = torch.as_tensor(P.p0, device="cuda")
    O_MAX = float(P.cbf_O_MAX); site = P.sim.data.site_xpos[P.sim.site].copy()
    lam = torch.zeros(len(starts), 2, device="cuda"); mu_al = 10.0
    lr = torch.as_tensor(CFG["G_LR"], device="cuda").float()
    opt = torch.optim.Adam([th], lr=1.0)                                       # per-parameter step sizes via lr scaling below

    def cons(p):                                                                # as opt2.solve_grad
        g = []
        ends = {s: bars[s].endpoints(p) for s in set(req) | {1.0}}
        e30 = ends[1.0][1]
        fm = [torch.minimum(bars[s].margin(ends[s][0]), bars[s].margin(ends[s][1])) for s in req]
        g.append(torch.stack(fm).max(0).values + CFG["KAPPA"] if fm else torch.full((len(p),), -1.0, device="cuda"))
        ob = []
        for c in range(len(p)):
            pts = np.vstack([site[None], p[c, 1:16, :3].detach().cpu().numpy()])
            m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
            o_f = ((e30[c].reshape(-1, 28, 28, 8) - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum()
            ob.append(o_f.topk(max(1, len(o_f) // 4)).values.mean())
        g.append(torch.stack(ob) / O_MAX - 1 + CFG["KAPPA_O"])
        return torch.stack(g, 1)

    def cost_t(p):
        D = (p[:, 1:9, :3] - p0[1:9, :3]) * 1000
        sd = torch.diff(torch.cat([torch.zeros_like(D[:, :1]), D], 1), 2, dim=1)
        return ((D ** 2).sum((1, 2)) + CFG["ETA"] * (sd ** 2).sum((1, 2))) / 100

    for it in range(CFG["G_STEPS"]):
        p = transform(p0, th)
        g = cons(p)
        al = (torch.relu(lam + mu_al * g) ** 2 - lam ** 2).sum(1) / (2 * mu_al)
        loss = (cost_t(p) + al).sum()
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            th.grad.mul_(lr)
        opt.step()
        with torch.no_grad():
            th.copy_(torch.maximum(torch.minimum(th, hi), lo))
            if it % 5 == 4:
                lam = torch.relu(lam + mu_al * g.detach())
    finals = list(th.detach().cpu().numpy().astype(np.float64))
    out, info = opt2.finish(P, finals, {"required_scales": req})
    return out, {**info, "theta": _chosen(P, out, finals)}


def _chosen(P, out, finals):
    for th in finals:
        if np.allclose(P.plan(th), out):
            return [round(float(x), 3) for x in th]
    return None                                                                 # fallback hold


def progress(p0, out):
    a = np.linalg.norm(p0[8, :3] - p0[0, :3]); b = np.linalg.norm(out[8, :3] - p0[0, :3])
    return float(b / a) if a > 1e-4 else 1.0


def run(solver, name, tag):
    import torch
    import cbf_filter as cbff
    import cem_filter as cf
    import inspector_data as insp
    import step3_world_model as wm
    if CFG["MARGIN"] == "3":                                                    # as policy_rollout cbfg3m
        import cbf_barrier
        cbf_barrier.SCALES = (0.5, 1.0, 2.0, 3.0, 4.0); opt2.HS = float(np.log2(4.0))
    cbff.CFG["H_SAFE"] = opt2.HS
    C, R = b2.load(name); n = int(os.environ.get("OPT2_LIMIT", len(R)))
    si, sn = (int(x) for x in os.environ.get("OPT3_SHARD", os.environ.get("TASK", "0/1")).split("/"))   # cluster shard (py_array TASK)
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    certs = [None] + ([wm.WorldModel.load(ROOT / f"results/program_b/ens_v1/model_ENS{k}.pt") for k in range(4)] if CFG["CERT"] == "OIL+ENS" else [])
    out, t0 = [], time.time()
    for i in range(si, n, sn):
        t1 = time.time()
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        hold = np.repeat(np.r_[C["target"][i], C["plan16"][i][0, 3]][None], 16, 0).astype(np.float32)
        hh, _ = cbf.score(sim, z0, hold[None], 64)
        p16 = C["plan16"][i].copy(); p16[0, :3] = C["target"][i]                   # anchor at the current target (A4)
        h_req = opt2.HS if os.environ.get("OPT3_ABS") == "1" else float(hh[0])       # ABS: require h >= h_safe, not >= h_hold
        P = Problem(mon, cbf, sim, z0, p16, h_req, certs); P.cbf_O_MAX = cbff.CFG["O_MAX"]
        plan, info = (solve_cem if solver == "cem" else solve_grad)(P, 7000 + i)
        out.append({"i": i, **info, "progress": round(progress(P.p0, plan), 3), "seconds": round(time.time() - t1, 1),
                    "chunk": plan[1:9].astype(np.float32).tolist()})
        if i % 10 == 0:
            print(f"G-{solver} {i + 1}/{n} {time.time() - t0:.0f}s {info['kind']} theta {info['theta']} {out[-1]['seconds']} s", flush=True)
    (b2.OUT / (f"opt3_{name}_{tag}.json" if sn == 1 else f"opt3_{name}_{tag}.part{si}.json")).write_text(
        json.dumps({"cfg": CFG, "solver": solver, "rows": out}))
    print("done", len(out), f"{time.time() - t0:.0f}s", flush=True)


def truth(name, tag):
    rows = json.loads((b2.OUT / f"opt3_{name}_{tag}.json").read_text())["rows"]
    b2.truth_chunks(name, [(r["i"], np.asarray(r["chunk"], np.float32)) for r in rows])


def report(name, tag):
    src, dst = b2.OUT / f"opt3_{name}_{tag}.json", b2.OUT / f"opt2_{name}_g_{tag}.json"
    dst.write_text(src.read_text())                                             # reuse F's report (same metrics and A2 gate)
    opt2.report(name, f"g_{tag}"); dst.unlink()
    rep = b2.OUT / f"report_{name}_g_{tag}.json"; res = json.loads(rep.read_text())
    rows = json.loads(src.read_text())["rows"]; pr = np.array([r["progress"] for r in rows])
    res["progress_kept_median"] = float(np.median(pr)); res["hold_equivalent_share"] = round(float((pr < 0.1).mean()), 3)
    rep.write_text(json.dumps(res, indent=1)); print("progress kept (median)", res["progress_kept_median"], "hold-equivalent", res["hold_equivalent_share"])


if __name__ == "__main__":
    {"run": run, "truth": truth, "report": report}[sys.argv[1]](*sys.argv[2:])
