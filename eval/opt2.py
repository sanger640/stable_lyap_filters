"""Optimisation-only safety filter, redesign (docs/OPT_FILTER_REDESIGN.md): ONE formulation, two separate solvers.

Formulation F: input = one chunk (16-step plan: steps 1-8 executed, 9-15 look-ahead); decision variable D = xyz offsets of
steps 1-8 (look-ahead shifted by D[7]; gripper fixed); minimise cost(D) = (||D||^2 + ETA ||second differences||^2) / 100
(mm) subject to the certificate C: h >= min(h_hold, log2 3) and o <= O_MAX on FRESH futures with the certifier model(s);
fallback: hold still if certified, else the best (h, -o) candidate.
Solvers: cem (exact rule on search draws, smooth knot basis) | grad (augmented Lagrangian on a smooth surrogate of C).

usage: python3 eval/opt2.py run <cem|grad> <dev|holdout> <tag> ; truth <set> <tag> ; report <set> <tag>
env: OPT2_CERT = OIL (default) or OIL+ENS (all must pass), OPT2_LIMIT
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ.setdefault("EXEC_NOISE_SCALE", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
os.environ.setdefault("CBF_WM", str(ROOT / "results/program_b/oil_v1/model/model_OIL.pt"))
os.environ.setdefault("CBF_CALIB", "results/jenga/cbf_calibration_OIL.json")
import bench2 as b2  # noqa: E402

E = lambda k, d: type(d)(os.environ.get(k, d))
CFG = {"ETA": E("OPT2_ETA", 4.0), "N_SEARCH": E("OPT2_N_SEARCH", 32), "N_CERT": 64, "CERT": os.environ.get("OPT2_CERT", "OIL"),
       "CEM_POP": E("OPT2_CEM_POP", 24), "CEM_ITERS": E("OPT2_CEM_ITERS", 3), "CEM_ELITES": 6, "CEM_SD_MM": 8.0, "CEM_FINAL": 6,
       "G_STARTS": E("OPT2_G_STARTS", 6), "G_STEPS": E("OPT2_G_STEPS", 15), "G_N": E("OPT2_G_N", 12), "G_LR_MM": 1.5,
       "G_MAX_MM": 40.0, "KAPPA": 0.1, "KAPPA_O": 0.05}
KNOTS = (1, 3, 5, 8)
HS = float(np.log2(3.0))


def knot_basis():
    """(8, 4): offsets of steps 1-8 from 4 knots (linear interpolation)."""
    B = np.zeros((8, len(KNOTS)))
    for t in range(1, 9):
        k = int(np.searchsorted(KNOTS, t))
        if KNOTS[min(k, len(KNOTS) - 1)] == t:
            B[t - 1, k] = 1
        else:
            w = (t - KNOTS[k - 1]) / (KNOTS[k] - KNOTS[k - 1]); B[t - 1, k - 1], B[t - 1, k] = 1 - w, w
    return B


class Problem:
    """One decision: the given plan, the world model(s), the certificate and the cost."""

    def __init__(self, mon, cbf, sim, z0, plan16, h_hold, cert_models):
        self.mon, self.cbf, self.sim, self.z0, self.p0 = mon, cbf, sim, z0, np.asarray(plan16, np.float32)
        self.h_hold, self.cert_models = h_hold, cert_models

    def plan(self, D):
        """D (8, 3) metres -> 16-step plan."""
        p = self.p0.copy(); p[1:9, :3] += D; p[9:, :3] += D[-1]
        return p

    def cost(self, D):
        mm = np.asarray(D) * 1000
        sd = np.diff(np.vstack([np.zeros((1, 3)), mm]), 2, axis=0)
        return float((mm ** 2).sum() + CFG["ETA"] * (sd ** 2).sum()) / 100

    def score(self, plans, n):
        return self.cbf.score(self.sim, self.z0, np.stack(plans), n)

    def certify(self, plans):
        """Fresh draws, every certifier model must pass. -> ok (C,), h, o (with the first model)."""
        import cbf_filter as cbff
        ok, H, O = None, None, None
        spread = cbff.CFG["BARRIER"] == "spread"            # spread level is model-specific: each certifier model is
        if spread:                                          # compared with ITS OWN hold (as the trigger's hold: gripper 0)
            plans = list(plans) + [np.repeat(np.r_[self.sim.target, 0.0][None], 16, 0).astype(np.float32)]
        for k, m in enumerate(self.cert_models):
            if m is not None:
                saved, self.mon.model = self.mon.model, m
            h, o = self.score(plans, CFG["N_CERT"])
            if m is not None:
                self.mon.model = saved
            if spread:
                okk = self.cbf.ok(h[:-1], o[:-1], float(h[-1])); h, o = h[:-1], o[:-1]
            else:
                okk = self.cbf.ok(h, o, self.h_hold)
            ok = okk if ok is None else ok & okk
            if k == 0:
                H, O = h, o
        return ok, H, O


def finish(P, cands, extra):
    """Certify the solver's final candidates; cheapest certified, else fallback (hold if certified, else best (h, -o))."""
    plans = [P.plan(D) for D in cands]
    hold = np.repeat(np.r_[P.sim.target, P.p0[0, 3]][None], 16, 0).astype(np.float32)
    ok, h, o = P.certify(plans + [hold])
    costs = np.array([P.cost(D) for D in cands])
    okc = ok[:-1]
    if okc.any():
        j = int(np.flatnonzero(okc)[np.argmin(costs[okc])]); out, kind = plans[j], "certified"
    elif ok[-1]:
        j, out, kind = -1, hold, "fallback_hold"
    else:
        j = int(np.lexsort((o[:-1], -h[:-1]))[0]); out, kind = plans[j], "fallback_best"
    change = float(np.abs(out[1:9, :3] - P.p0[1:9, :3]).max() * 1000)
    return out, {"kind": kind, "n_certified": int(okc.sum()), "n_final": len(cands), "change_mm": round(change, 1),
                 "cost": round(float(costs[j]), 2) if j >= 0 else None, **extra}


def solve_cem(P, seed):
    B = knot_basis(); rng = np.random.default_rng(seed)
    mu = np.zeros((len(KNOTS), 3)); sd = np.tile(np.array([1, 1, 0.5]) * CFG["CEM_SD_MM"] / 1000, (len(KNOTS), 1))
    pool = []                                                       # (rank key, D)
    for it in range(CFG["CEM_ITERS"]):
        K = mu[None] + sd[None] * rng.standard_normal((CFG["CEM_POP"], len(KNOTS), 3))
        Ds = [B @ k for k in K]
        state = P.mon.rng.bit_generator.state                       # common random numbers across iterations
        h, o = P.score([P.plan(D) for D in Ds], CFG["N_SEARCH"])
        P.mon.rng.bit_generator.state = state
        ok = P.cbf.ok(h, o, P.h_hold); cost = np.array([P.cost(D) for D in Ds])
        key = [(0 if ok[c] else 1, cost[c] if ok[c] else -h[c], 0 if ok[c] else o[c]) for c in range(len(Ds))]
        order = sorted(range(len(Ds)), key=lambda c: key[c])
        el = K[order[:CFG["CEM_ELITES"]]]
        mu, sd = el.mean(0), np.maximum(el.std(0), 1e-3)
        pool += [(key[c], Ds[c]) for c in order[:CFG["CEM_FINAL"]]]
    pool.sort(key=lambda x: x[0])
    finals = [D for _, D in pool[:CFG["CEM_FINAL"]]]
    return finish(P, finals, {"search_certified": int(sum(k[0] == 0 for k, _ in pool))})


def solve_grad(P, seed):
    import torch
    import cbf_qp as q
    import cbf_barrier as cb
    mon = P.mon; z0 = P.z0
    req = [s for s in (0.5, 1.0, 2.0) if np.log2(s) < min(P.h_hold, HS) - 1e-9]       # scales that must not fork
    bars = {s: q.Barrier(mon, z0, seed + int(s * 10), CFG["G_N"], s) for s in set(req) | {1.0}}
    rng = np.random.default_rng(seed); B = knot_basis()
    starts = [np.zeros((8, 3))] + [B @ (rng.standard_normal((len(KNOTS), 3)) * np.array([1, 1, 0.5]) * CFG["CEM_SD_MM"] / 1000)
                                   for _ in range(CFG["G_STARTS"] - 1)]
    D = torch.as_tensor(np.stack(starts) * 1000, device="cuda", dtype=torch.float32).requires_grad_(True)    # mm
    p0 = torch.as_tensor(P.p0, device="cuda")
    O_MAX = float(P.cbf_O_MAX)
    lam = torch.zeros(len(starts), 2, device="cuda"); mu_al = 10.0
    opt = torch.optim.Adam([D], lr=CFG["G_LR_MM"])
    site = P.sim.data.site_xpos[P.sim.site].copy()

    def plans_of(Dmm):
        p = p0[None].repeat(len(Dmm), 1, 1).clone()
        p[:, 1:9, :3] = p[:, 1:9, :3] + Dmm / 1000; p[:, 9:, :3] = p[:, 9:, :3] + Dmm[:, -1:] / 1000
        return p

    def cons(p):
        g = []
        ends = {s: bars[s].endpoints(p) for s in set(req) | {1.0}}
        e30 = ends[1.0][1]
        # fork at scale s in the exact rule = split at BOTH endpoints -> smooth fork-ness = min(margin10, margin30) (> 0 = fork)
        fm = [torch.minimum(bars[s].margin(ends[s][0]), bars[s].margin(ends[s][1])) for s in req]
        g.append(torch.stack(fm).max(0).values + CFG["KAPPA"] if fm else torch.full((len(p),), -1.0, device="cuda"))
        ob = []
        for c in range(len(p)):
            pts = np.vstack([site[None], p[c, 1:16, :3].detach().cpu().numpy()])
            m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
            o_f = ((e30[c].reshape(-1, 28, 28, 8) - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum()
            ob.append(o_f.topk(max(1, len(o_f) // 4)).values.mean())
        g.append(torch.stack(ob) / O_MAX - 1 + CFG["KAPPA_O"])
        return torch.stack(g, 1)                                     # (S, 2), <= 0 = satisfied

    def cost_t(Dmm):
        sd = torch.diff(torch.cat([torch.zeros_like(Dmm[:, :1]), Dmm], 1), 2, dim=1)
        return ((Dmm ** 2).sum((1, 2)) + CFG["ETA"] * (sd ** 2).sum((1, 2))) / 100
    for it in range(CFG["G_STEPS"]):
        g = cons(plans_of(D))
        al = (torch.relu(lam + mu_al * g) ** 2 - lam ** 2).sum(1) / (2 * mu_al)
        loss = (cost_t(D) + al).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            nrm = D.norm(dim=(1, 2), keepdim=True).clamp_min(1e-9); D.mul_(torch.clamp(CFG["G_MAX_MM"] / nrm, max=1.0))
            if it % 5 == 4:
                lam = torch.relu(lam + mu_al * g.detach())
    finals = [d for d in (D.detach().cpu().numpy() / 1000)]
    return finish(P, finals, {"required_scales": req})


def run(solver, name, tag):
    import torch
    import cbf_filter as cbff
    import cem_filter as cf
    import inspector_data as insp
    import step3_world_model as wm
    cbff.CFG["H_SAFE"] = HS
    C, R = b2.load(name); n = int(os.environ.get("OPT2_LIMIT", len(R)))
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    certs = [None] + ([wm.WorldModel.load(ROOT / f"results/program_b/ens_v1/model_ENS{k}.pt") for k in range(4)] if CFG["CERT"] == "OIL+ENS" else [])
    out, t0 = [], time.time()
    for i in range(n):
        t1 = time.time()
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        hold = np.repeat(np.r_[C["target"][i], C["plan16"][i][0, 3]][None], 16, 0).astype(np.float32)
        hh, _ = cbf.score(sim, z0, hold[None], 64)
        P = Problem(mon, cbf, sim, z0, C["plan16"][i], float(hh[0]), certs); P.cbf_O_MAX = cbff.CFG["O_MAX"]
        plan, info = (solve_cem if solver == "cem" else solve_grad)(P, 7000 + i)
        out.append({"i": i, **info, "seconds": round(time.time() - t1, 1), "chunk": plan[1:9].astype(np.float32).tolist()})
        if i % 10 == 0:
            print(f"{solver} {i + 1}/{n} {time.time() - t0:.0f}s {info['kind']} change {info['change_mm']} mm {out[-1]['seconds']} s", flush=True)
    (b2.OUT / f"opt2_{name}_{tag}.json").write_text(json.dumps({"cfg": CFG, "solver": solver, "rows": out}))
    print("done", len(out), f"{time.time() - t0:.0f}s", flush=True)


def truth(name, tag):
    rows = json.loads((b2.OUT / f"opt2_{name}_{tag}.json").read_text())["rows"]
    b2.truth_chunks(name, [(r["i"], np.asarray(r["chunk"], np.float32)) for r in rows])


def report(name, tag):
    import collections
    C, R = b2.load(name)
    d = json.loads((b2.OUT / f"opt2_{name}_{tag}.json").read_text()); rows = d["rows"]; I = [r["i"] for r in rows]
    T = lambda chs: np.array(b2.truth_chunks(name, list(zip(I, chs))))
    s = T([np.asarray(r["chunk"], np.float32) for r in rows]); sel = T([C["executed"][i] for i in I])
    pol = T([C["nominal"][i] for i in I]); hold = T([b2.hold_chunk(C, i) for i in I])
    dlt = s[:, 0] - sel[:, 0]; se = dlt.std(ddof=1) / np.sqrt(len(dlt))
    res = {"tag": tag, "solver": d["solver"], "cfg": d["cfg"], "set": name, "decisions": len(rows),
           "neighbour_risk": {"solver": round(float(s[:, 0].mean()), 4), "selection": round(float(sel[:, 0].mean()), 4),
                              "policy": round(float(pol[:, 0].mean()), 4), "hold": round(float(hold[:, 0].mean()), 4)},
           "red_fall_risk": {"solver": round(float(s[:, 1].mean()), 4), "selection": round(float(sel[:, 1].mean()), 4),
                             "policy": round(float(pol[:, 1].mean()), 4), "hold": round(float(hold[:, 1].mean()), 4)},
           "solver_minus_selection_neighbour": {"mean": round(float(dlt.mean()), 4), "one_sided_95_upper": round(float(dlt.mean() + 1.645 * se), 4),
                                                "better": int((dlt < 0).sum()), "worse": int((dlt > 0).sum())},
           "kinds": dict(collections.Counter(r["kind"] for r in rows)),
           "median_change_mm": float(np.median([r["change_mm"] for r in rows])), "median_seconds": float(np.median([r["seconds"] for r in rows]))}
    ns, nsel, rs, rsel = res["neighbour_risk"]["solver"], res["neighbour_risk"]["selection"], res["red_fall_risk"]["solver"], res["red_fall_risk"]["selection"]
    res["gate_A2"] = {"neighbour <= sel + 0.5pt": ns <= nsel + 0.005, "upper bound <= +1.5pt": res["solver_minus_selection_neighbour"]["one_sided_95_upper"] <= 0.015,
                      "red <= sel + 1pt": rs <= rsel + 0.01, "seconds <= 30": res["median_seconds"] <= 30}
    res["gate_A2"]["PASS"] = all(res["gate_A2"].values())
    (b2.OUT / f"report_{name}_{tag}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"run": run, "truth": truth, "report": report}[sys.argv[1]](*sys.argv[2:])
