"""Gradient-based CBF-QP safety filter, validated offline (no closed-loop runs) for fast iteration.

Test bed: decisions recorded in the dev3 runs (episodes 2000-2149, arm A = final-test filter): every decision where the
filter intervened (+ a random sample of untouched ones). For each, the camera latent is cached once.

Smooth fork barrier at error scale s (fixed random draws -> deterministic, differentiable in the plan):
  endpoints e (n futures) at 1 s and 3 s -> projection on the main axis -> soft two-group fit (EM from the hard best cut)
  margin  m = min( dBIC / n , D - 2 )       (> 0  <=>  fork, matching the discrete test: BIC_2 < BIC_1 and D > 2)
  B(p)    = -min(m_1s, m_3s)                (> 0  <=>  no fork at scale s; the discrete rule needs both endpoints)
CBF-QP (SQP, closed-form min-norm step per iteration, trust region):
  min_theta ||theta||^2   s.t.   B(p_pi + P theta) >= tau,  tau = min(B(hold), 0) + kappa
  theta = xyz offsets at K knots, linearly interpolated over plan steps 1..15 (gripper untouched)
Verification: the result is re-scored with the exact discrete 4-scale test (CBF.score); accepted only if Safe, else
the current selection is used.  Ground truth: real topple risk of each executed chunk (exact simulator, n futures).

stages:  prep | agree | grad | opt [tag] | truth [tag] | report [tag]
env: QP_N (futures in the barrier, 32), QP_SCALE (2), QP_KAPPA (0.05), QP_ITERS (6), QP_STEP_MM (8), QP_MAX_MM (40),
     QP_SUBSET (hard|all), TRUTH_N (32 screening / 64 confirm)
"""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EXEC_NOISE_SCALE", "2")
os.environ.setdefault("HARD_MASS", "0")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/program_b/cbf_qp"
RUN = ROOT / "results/program_b/policy_rollouts/paired_cbf_dev3A"
LOG = ROOT / "results/jenga/paired_cbf_dev3A.jsonl"
E = lambda k, d: type(d)(os.environ.get(k, d))
QP = {"N": E("QP_N", 32), "SCALE": E("QP_SCALE", 2.0), "KAPPA": E("QP_KAPPA", 0.05), "ITERS": E("QP_ITERS", 6),
      "STEP_MM": E("QP_STEP_MM", 8.0), "MAX_MM": E("QP_MAX_MM", 40.0), "KNOTS": (1, 5, 10, 15),
      "VERSION": E("QP_VERSION", 0), "KAPPA_O": E("QP_KAPPA_O", 0.02)}
# v1: the barrier is enforced at the scales the rule actually needs (h >= min(h_now, 1) <=> no fork at every
# s < 2**min(h_now, 1): {1/2, 1} or {1/2}), plus the regime constraint o(p) <= O_max (differentiable, mask fixed per
# iteration); the min-norm step for the linearised constraints is solved exactly (Dykstra projection).


def decisions():
    rows = [json.loads(l) for l in LOG.read_text().splitlines()]
    out = []
    rng = np.random.default_rng(7)
    for r in rows:
        acts = [c["chunk"] for c in r["log"] if c.get("intervened")]
        calm = [c["chunk"] for c in r["log"] if not c.get("intervened")]
        out += [(r["episode"], d, True) for d in acts]
        out += [(r["episode"], int(d), False) for d in rng.choice(calm, size=min(1, len(calm)), replace=False)]
    return out


def prep():
    """Cache per decision: latent z0, saved sim state, plans, the logged filter info."""
    import cem_filter as cf
    import inspector_data as insp
    OUT.mkdir(parents=True, exist_ok=True)
    info = {}
    for l in LOG.read_text().splitlines():
        r = json.loads(l)
        for c in r["log"]:
            info[(r["episode"], c["chunk"])] = c
    mon, sim = cf.Monitor(), insp._sim()
    keys, Z, rec = [], [], {k: [] for k in ("state", "ctrl", "target", "gripper", "tilt", "plan16", "nominal", "executed", "site")}
    for ep, d, acted in decisions():
        with np.load(RUN / f"ep{ep:03d}.npz") as S:
            i = int(np.flatnonzero(S["chunk_index"] == d)[0])
            snap = (S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i]))
            sim.restore(snap)
            Z.append(mon.latent(mon.frame(sim)[None])[0].astype(np.float16))
            for k, v in (("state", S["state"][i]), ("ctrl", S["ctrl"][i]), ("target", S["target"][i]), ("gripper", S["gripper"][i]),
                         ("tilt", S["start_tilt"][i]), ("plan16", S["plan16"][i]), ("nominal", S["nominal"][i]),
                         ("executed", S["executed"][i]), ("site", sim.data.site_xpos[sim.site].copy())):
                rec[k].append(v)
        c = info[(ep, d)]
        keys.append({"ep": ep, "d": d, "acted": acted, "h_nom": c["h_nom"], "o_nom": c["o_nom"], "h_now": c["h_now"],
                     "choice": c.get("choice"), "change_mm": c.get("change_mm")})
    np.savez_compressed(OUT / "cache.npz", z0=np.stack(Z), **{k: np.stack(v) for k, v in rec.items()})
    (OUT / "keys.json").write_text(json.dumps(keys))
    print("cached", len(keys), "decisions,", sum(k["acted"] for k in keys), "with an intervention", flush=True)


class Barrier:
    """Differentiable smooth fork barrier for a batch of plans from ONE decision (fixed draws)."""

    def __init__(self, mon, z0, seed, n, scale):
        import torch
        import cbf_filter as cbff
        import step3_world_model as wm
        self.torch, self.wm, self.mon, self.n, self.s, self.cbff = torch, wm, mon, n, scale, cbff
        d_ = np.stack([np.concatenate([(p * [1, 1, 0] * scale).reshape(9), (r * [0, 0, 1] * scale).reshape(9)])
                       for p, r in (mon.errors.sample(seed, 0, 0, j) for j in range(n))])
        with torch.no_grad():
            self.st = mon.g(z0[None].expand(n, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32)).float()
        nl = cbff.CFG["N_LOOK"]
        self.noise = torch.as_tensor(np.stack([np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:nl] * scale
                                               for j in range(n)]), device="cuda", dtype=torch.float32)  # (n, nl, 3)
        m = mon.model
        self.net, self.a_mean, self.a_std, self.res_std = m.net, m.stats.a_mean.cuda(), m.stats.a_std.cuda(), m.stats.res_std
        for p_ in self.net.parameters():
            p_.requires_grad_(False)

    def horizon(self, plans):
        """plans (C, 16, 4) torch -> commands (C, 38, 4), as cbf_filter.horizon."""
        nl, ps = self.cbff.CFG["N_LOOK"], self.cbff.CFG["PLAN_START"]
        a = plans[:, ps:ps + nl]
        return self.torch.cat([a, a[:, -1:].expand(-1, 38 - nl, -1)], 1)

    def endpoints(self, plans):
        torch, wm, n = self.torch, self.wm, self.n
        from torch.utils.checkpoint import checkpoint
        C = len(plans)
        cmd = self.horizon(plans)[:, None].expand(-1, n, -1, -1).clone()             # (C, n, 38, 4)
        nl = self.noise.shape[1]
        cmd[:, :, :nl, :3] = cmd[:, :, :nl, :3] - self.noise[None]
        pairs = cmd.reshape(C * n, wm.STEPS, -1)
        z = self.st.repeat(C, 1, 1, 1)

        def step(z, a):
            a = (a - self.a_mean) / self.a_std
            r = self.net(z, torch.zeros_like(z), a, torch.zeros(len(z), device=z.device))
            return z + r * self.res_std
        ends = {}
        for t in range(wm.STEPS):
            z = checkpoint(step, z, pairs[:, t], use_reentrant=False)
            if t in (wm.HOLD10, wm.HOLD30):
                ends[t] = z.reshape(C, n, -1)
        return ends[wm.HOLD10], ends[wm.HOLD30]

    def margin(self, e):
        """Smooth fork margin per plan: e (C, n, D) -> m (C,) (> 0 = fork)."""
        torch = self.torch
        import cem_filter as cf
        C, n, D = e.shape
        x = e.double()
        X = x - x.mean(1, keepdim=True)
        w, V = torch.linalg.eigh(X @ X.transpose(1, 2))
        y = V[:, :, -1] * w[:, -1:].clamp_min(1e-12).sqrt()
        with torch.no_grad():
            _, lab, _ = cf.split_test_gpu(e.detach())
        r = lab.double() * 0.9 + 0.05
        for _ in range(3):
            na, nb = (1 - r).sum(1), r.sum(1)
            ma, mb = ((1 - r) * y).sum(1) / na, (r * y).sum(1) / nb
            va = ((1 - r) * (y - ma[:, None]) ** 2).sum(1) / na + 1e-9
            vb = (r * (y - mb[:, None]) ** 2).sum(1) / nb + 1e-9
            la = torch.log(na / n)[:, None] - 0.5 * torch.log(va)[:, None] - (y - ma[:, None]) ** 2 / (2 * va[:, None])
            lb = torch.log(nb / n)[:, None] - 0.5 * torch.log(vb)[:, None] - (y - mb[:, None]) ** 2 / (2 * vb[:, None])
            r = torch.sigmoid(lb - la)
        na, nb = (1 - r).sum(1), r.sum(1)
        n2ll = lambda m, v: m * (torch.log(2 * np.pi * v) + 1)
        bic1 = n2ll(torch.full_like(na, n), y.var(1, unbiased=False) + 1e-9) + 2 * np.log(n)
        bic2 = n2ll(na, va) + n2ll(nb, vb) - 2 * (na * torch.log(na / n) + nb * torch.log(nb / n)) + 5 * np.log(n)
        dD = np.sqrt(2) * (ma - mb).abs() / (va + vb).sqrt()
        return torch.minimum((bic1 - bic2) / n, dD - 2).float()

    def __call__(self, plans):
        e10, e30 = self.endpoints(plans)
        return -torch_min(self.margin(e10), self.margin(e30))


def torch_min(a, b):
    import torch
    return torch.minimum(a, b)


def knot_matrix(knots=QP["KNOTS"]):
    """(16, K) linear interpolation of K knot offsets onto plan steps; step 0 (not executed) fixed."""
    M = np.zeros((16, len(knots)))
    for t in range(1, 16):
        k = np.searchsorted(knots, t)
        if k == 0:
            M[t, 0] = 1
        elif k >= len(knots):
            M[t, -1] = 1
        elif knots[k] == t:
            M[t, k] = 1
        else:
            w = (t - knots[k - 1]) / (knots[k] - knots[k - 1]); M[t, k - 1], M[t, k] = 1 - w, w
    return M


def load_cache():
    C = np.load(OUT / "cache.npz")
    return {k: C[k] for k in C.files}, json.loads((OUT / "keys.json").read_text())


def agree():
    """Does the smooth barrier agree with the exact discrete fork test (scale QP_SCALE, both endpoints)?"""
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    C, K = load_cache()
    mon = cf.Monitor()
    rows = []
    for i, k in enumerate(K):
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        bar = Barrier(mon, z0, 1000 + i, QP["N"], QP["SCALE"])
        plans = torch.as_tensor(np.stack([C["plan16"][i], np.repeat(np.r_[C["target"][i], 0.0][None], 16, 0)]), device="cuda", dtype=torch.float32)
        with torch.no_grad():
            e10, e30 = bar.endpoints(plans)
            B = bar(plans).cpu().numpy()
            ok10, _, _ = cf.split_test_gpu(e10); ok30, _, _ = cf.split_test_gpu(e30)
        fork = (ok10 & ok30).cpu().numpy()
        rows += [{"i": i, "which": w, "B": float(B[w]), "fork": bool(fork[w])} for w in (0, 1)]
    f = np.array([r["fork"] for r in rows]); b = np.array([r["B"] for r in rows])
    agree_ = float(np.mean((b <= 0) == f))
    from sklearn.metrics import roc_auc_score
    res = {"n": len(rows), "forks": int(f.sum()), "sign_agreement": round(agree_, 3),
           "auc_minusB_vs_fork": round(float(roc_auc_score(f, -b)), 3) if 0 < f.sum() < len(f) else None}
    (OUT / "agree.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res), flush=True)


def grad():
    """Autograd vs finite differences of the barrier w.r.t. the knot offsets, on a few decisions."""
    import torch
    import cem_filter as cf
    C, K = load_cache()
    mon = cf.Monitor(); M = torch.as_tensor(knot_matrix(), device="cuda", dtype=torch.float32)
    out = []
    for i in [j for j, k in enumerate(K) if k["acted"]][:5]:
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        bar = Barrier(mon, z0, 1000 + i, QP["N"], QP["SCALE"])
        base = torch.as_tensor(C["plan16"][i], device="cuda", dtype=torch.float32)
        th = torch.zeros(len(QP["KNOTS"]), 3, device="cuda", requires_grad=True)
        p = base.clone(); p[:, :3] = p[:, :3] + M @ th / 1000
        B = bar(p[None])[0]; B.backward(); g = th.grad.flatten().cpu().numpy()
        fd = []
        for j in range(th.numel()):
            dv = torch.zeros_like(th).flatten(); dv[j] = 1.0; dv = dv.view_as(th)
            with torch.no_grad():
                vals = []
                for sgn in (1, -1):
                    q = base.clone(); q[:, :3] = q[:, :3] + M @ (sgn * dv) / 1000; vals.append(float(bar(q[None])[0]))
            fd.append((vals[0] - vals[1]) / 2)
        fd = np.array(fd)
        cos = float(g @ fd / (np.linalg.norm(g) * np.linalg.norm(fd) + 1e-12))
        out.append({"i": i, "B": float(B), "cos_autograd_fd": round(cos, 3), "norm_ratio": round(float(np.linalg.norm(g) / (np.linalg.norm(fd) + 1e-12)), 3)})
        print(out[-1], flush=True)
    (OUT / "grad.json").write_text(json.dumps(out, indent=1))


def min_norm(gs, cs, x0, iters=200):
    """argmin ||x||^2 s.t. g_i . x >= c_i (linear), by Dykstra's projection of 0 onto the intersection of half-spaces."""
    x = np.zeros_like(x0); P = [np.zeros_like(x0) for _ in gs]
    for _ in range(iters):
        for k, (g, c) in enumerate(zip(gs, cs)):
            y = x + P[k]
            viol = c - g @ y
            x_new = y + (max(viol, 0.0) / (g @ g + 1e-12)) * g
            P[k] = y - x_new; x = x_new
    return x


def regime_o(e30, z0, mask):
    """Differentiable o: max over futures of the channel-mean squared latent change outside the hand mask."""
    import torch
    n = e30.shape[0]
    diff = (e30.reshape(n, 28, 28, 8) - z0[None]).pow(2).mean(-1)
    m = (~mask).float()
    return ((diff * m).sum((1, 2)) / m.sum().clamp_min(1)).max()


def opt(tag="v0"):
    if QP["VERSION"] >= 1:
        return opt_v1(tag)
    """CBF-QP by SQP for every intervened decision -> optimized plan, barrier trace, exact verification."""
    import torch
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    C, K = load_cache()
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    M = torch.as_tensor(knot_matrix(), device="cuda", dtype=torch.float32)
    sel = [i for i, k in enumerate(K) if k["acted"]]
    if os.environ.get("QP_SUBSET", "all") == "hard":
        sel = sel[::2]
    res = []
    t0 = time.time()
    for n_, i in enumerate(sel):
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        bar = Barrier(mon, z0, 1000 + i, QP["N"], QP["SCALE"])
        base = torch.as_tensor(C["plan16"][i], device="cuda", dtype=torch.float32)
        hold = torch.as_tensor(np.repeat(np.r_[C["target"][i], 0.0][None], 16, 0), device="cuda", dtype=torch.float32)
        with torch.no_grad():
            B_hold = float(bar(hold[None])[0])
        tau = min(B_hold, 0.0) + QP["KAPPA"]
        th = torch.zeros(len(QP["KNOTS"]), 3, device="cuda")
        trace = []
        for it in range(QP["ITERS"] + 1):
            thv = th.clone().requires_grad_(True)
            p = base.clone(); p[:, :3] = p[:, :3] + M @ thv / 1000
            B = bar(p[None])[0]
            trace.append(round(float(B), 4))
            if it == QP["ITERS"] or float(B) >= tau and it > 0:
                break
            B.backward()
            g = thv.grad.flatten(); x = th.flatten()
            c = tau - float(B) + float(g @ x)
            target = torch.zeros_like(x) if c <= 0 else c * g / (g @ g + 1e-12)
            stepv = target - x
            nrm = float(stepv.norm())
            if nrm > QP["STEP_MM"]:
                stepv = stepv * QP["STEP_MM"] / nrm
            x = x + stepv
            if float(x.norm()) > QP["MAX_MM"]:
                x = x * QP["MAX_MM"] / float(x.norm())
            th = x.view_as(th).detach()
        with torch.no_grad():
            p = base.clone(); p[:, :3] = p[:, :3] + M @ th / 1000
        plan = p.cpu().numpy()
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        h, o = cbf.score(sim, z0, np.stack([plan]), 64)
        ok = bool(cbf.ok(h, o, K[i]["h_now"])[0])
        chunk = plan[cbff.CFG["PLAN_START"]:cbff.CFG["PLAN_START"] + 8].astype(np.float32)
        res.append({"i": i, "ep": K[i]["ep"], "d": K[i]["d"], "B_hold": round(B_hold, 4), "tau": round(tau, 4), "trace": trace,
                    "change_mm": round(float(np.linalg.norm(plan[1:9, :3] - C["plan16"][i][1:9, :3]) * 1000), 1),
                    "knots_mm": np.round(th.cpu().numpy(), 1).tolist(), "h": float(h[0]), "o": round(float(o[0]), 4),
                    "verified": ok, "chunk": chunk.tolist()})
        if n_ % 10 == 0:
            print(f"opt {n_ + 1}/{len(sel)} {time.time() - t0:.0f}s B {trace[0]:.3f}->{trace[-1]:.3f} tau {tau:.3f} change {res[-1]['change_mm']} mm verified {ok}", flush=True)
    (OUT / f"opt_{tag}.json").write_text(json.dumps({"cfg": QP, "rows": res}))
    print("opt done", len(res), "verified", sum(r["verified"] for r in res), f"{time.time() - t0:.0f}s", flush=True)


def _hash(chunk):
    return hashlib.sha1(np.asarray(chunk, np.float32).round(5).tobytes()).hexdigest()[:16]


def opt_v1(tag):
    import torch
    import cbf_barrier as cb
    import cem_filter as cf
    import cbf_filter as cbff
    import inspector_data as insp
    C, K = load_cache()
    mon = cf.Monitor(); cbf = cbff.CBF(mon); sim = insp._sim()
    O_MAX = cbff.CFG["O_MAX"]
    M = torch.as_tensor(knot_matrix(), device="cuda", dtype=torch.float32)
    sel = [i for i, k in enumerate(K) if k["acted"]]
    res, t0 = [], time.time()
    for n_, i in enumerate(sel):
        z0 = torch.as_tensor(C["z0"][i].astype(np.float32), device="cuda")
        hreq = min(K[i]["h_now"], 1.0)
        scales = [s_ for s_ in (0.5, 1.0) if np.log2(s_) < hreq - 1e-9]          # must not fork at these scales
        bars = {s_: Barrier(mon, z0, 1000 + i, QP["N"], s_) for s_ in set(scales) | {1.0}}
        base = torch.as_tensor(C["plan16"][i], device="cuda", dtype=torch.float32)
        th = np.zeros(len(QP["KNOTS"]) * 3); trace = []
        for it in range(QP["ITERS"] + 1):
            thv = torch.as_tensor(th, device="cuda", dtype=torch.float32).view(-1, 3).requires_grad_(True)
            p = base.clone(); p[:, :3] = p[:, :3] + M @ thv / 1000
            pts = np.vstack([C["site"][i][None], p[1:16, :3].detach().cpu().numpy()])
            mask = torch.as_tensor(cb.hand_mask(pts), device="cuda")
            cons, vals = [], []
            if scales:
                Bs = torch.stack([bars[s_](p[None])[0] for s_ in scales]); Bv = Bs.min()
                cons.append(Bv); vals.append(("B", float(Bv), QP["KAPPA"]))
            _, e30 = bars[1.0].endpoints(p[None])
            ov = regime_o(e30[0], z0, mask)
            cons.append(-ov); vals.append(("o", float(ov), None))
            trace.append({k: round(v, 4) for k, v, _ in vals})
            okB = (not scales) or float(Bv) >= QP["KAPPA"]
            okO = float(ov) <= O_MAX - QP["KAPPA_O"]
            if it == QP["ITERS"] or (okB and okO and it > 0) or (okB and okO and it == 0):
                break
            gs, cs = [], []
            for c_, (name, v, _) in zip(cons, vals):
                g = torch.autograd.grad(c_, thv, retain_graph=True)[0].flatten().cpu().numpy().astype(np.float64)
                if name == "B":
                    cs.append(QP["KAPPA"] - v + g @ th)
                else:
                    cs.append((-(O_MAX - QP["KAPPA_O"])) - (-v) + g @ th)
                gs.append(g)
            target = min_norm(gs, cs, th)
            stepv = target - th; nrm = np.linalg.norm(stepv)
            if nrm > QP["STEP_MM"]:
                stepv *= QP["STEP_MM"] / nrm
            th = th + stepv
            if np.linalg.norm(th) > QP["MAX_MM"]:
                th *= QP["MAX_MM"] / np.linalg.norm(th)
        with torch.no_grad():
            p = base.clone(); p[:, :3] = p[:, :3] + M @ torch.as_tensor(th, device="cuda", dtype=torch.float32).view(-1, 3) / 1000
        plan = p.cpu().numpy()
        sim.restore((C["state"][i], C["ctrl"][i], C["target"][i], float(C["gripper"][i])))
        h, o = cbf.score(sim, z0, np.stack([plan]), 64)
        ok = bool(cbf.ok(h, o, K[i]["h_now"])[0])
        chunk = plan[cbff.CFG["PLAN_START"]:cbff.CFG["PLAN_START"] + 8].astype(np.float32)
        res.append({"i": i, "ep": K[i]["ep"], "d": K[i]["d"], "scales": scales, "trace": trace,
                    "change_mm": round(float(np.linalg.norm(plan[1:9, :3] - C["plan16"][i][1:9, :3]) * 1000), 1),
                    "knots_mm": np.round(th.reshape(-1, 3), 1).tolist(), "h": float(h[0]), "o": round(float(o[0]), 4),
                    "verified": ok, "chunk": chunk.tolist()})
        if n_ % 20 == 0:
            print(f"opt {n_ + 1}/{len(sel)} {time.time() - t0:.0f}s {trace[0]} -> {trace[-1]} change {res[-1]['change_mm']} mm verified {ok}", flush=True)
    (OUT / f"opt_{tag}.json").write_text(json.dumps({"cfg": QP, "rows": res}))
    print("opt done", len(res), "verified", sum(r["verified"] for r in res), f"{time.time() - t0:.0f}s", flush=True)


def _truth_one(job):
    import inspector_data as insp
    h, args = job
    return h, insp.truth_job(args)[1]


def truth(tag="v0"):
    """Real topple risk (exact simulator, TRUTH_N futures) of: policy chunk, current selection, CBF-QP chunk. Cached."""
    import inspector_data as insp
    C, K = load_cache()
    n_truth = int(os.environ.get("TRUTH_N", 32))
    cache_p = OUT / f"truth_{n_truth}.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    rows = json.loads((OUT / f"opt_{tag}.json").read_text())["rows"]
    jobs = []
    for r in rows:
        i = r["i"]
        base = (C["state"][i], C["ctrl"][i], C["target"][i], C["gripper"][i], C["tilt"][i])
        for ch in (C["nominal"][i], C["executed"][i], np.asarray(r["chunk"], np.float32)):
            h = _hash(ch)
            if h not in cache:
                jobs.append((h, ((i, 7, 0), *base, ch))); cache[h] = None     # same draws for every plan of decision i
    jobs = [j for j in jobs]
    print(len(jobs), "new truth jobs", flush=True)
    os.environ["TRUTH_N"] = str(n_truth)
    with mp.get_context("spawn").Pool(int(os.environ.get("TRUTH_WORKERS", 30))) as pool:
        for n_, (h, flags) in enumerate(pool.imap_unordered(_truth_one, jobs), 1):
            cache[h] = sum(flags) / len(flags)
            if n_ % 50 == 0 or n_ == len(jobs):
                cache_p.write_text(json.dumps(cache)); print(f"truth {n_}/{len(jobs)}", flush=True)
    cache_p.write_text(json.dumps(cache))


def report(tag="v0"):
    C, K = load_cache()
    n_truth = int(os.environ.get("TRUTH_N", 32))
    T = json.loads((OUT / f"truth_{n_truth}.json").read_text())
    rows = json.loads((OUT / f"opt_{tag}.json").read_text())["rows"]
    eps = 2 / n_truth
    out = []
    for r in rows:
        i = r["i"]
        rp, rs, rq = T[_hash(C["nominal"][i])], T[_hash(C["executed"][i])], T[_hash(np.asarray(r["chunk"], np.float32))]
        rv = rq if r["verified"] else rs
        out.append({"policy": rp, "selection": rs, "qp": rq, "qp_verified": rv, "change_qp": r["change_mm"],
                    "change_sel": K[i]["change_mm"], "verified": r["verified"]})
    A = lambda k: np.array([o[k] for o in out])
    rp, rs, rq, rv = A("policy"), A("selection"), A("qp"), A("qp_verified")
    risky = rp >= eps
    summ = lambda x: {"mean_risk": round(float(x.mean()), 4), "helped": int(((x < rp - eps)).sum()), "hurt": int(((x > rp + eps)).sum()),
                      "mean_risk_when_risky": round(float(x[risky].mean()), 4) if risky.any() else None}
    res = {"tag": tag, "decisions": len(out), "truth_n": n_truth, "risky_decisions": int(risky.sum()),
           "policy": {"mean_risk": round(float(rp.mean()), 4), "mean_risk_when_risky": round(float(rp[risky].mean()), 4)},
           "selection": summ(rs), "qp_raw": summ(rq), "qp_verified_else_selection": summ(rv),
           "qp_better_than_selection": int((rq < rs - eps).sum()), "qp_worse_than_selection": int((rq > rs + eps).sum()),
           "verified_rate": round(float(A("verified").mean()), 3),
           "median_change_mm": {"selection": float(np.median(A("change_sel"))), "qp": float(np.median(A("change_qp")))}}
    (OUT / f"report_{tag}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]; arg = sys.argv[2:]
    {"prep": prep, "agree": agree, "grad": grad, "opt": opt, "truth": truth, "report": report}[stage](*arg)
