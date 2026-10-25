"""Fork-boundary CBF safety filter, purely in the world model's latent space (overnight plan 2026-10-15, step 3).

Safe set: the robot's current regime (the scene now, assumed safe). Boundary: the fork, where imagined futures split.
For a plan p (the policy's remaining 15 steps, then hold) the world model imagines 64 futures under the robot's
perception and execution uncertainty scaled by s in SCALES:
  h(p)    barrier = log2 of the smallest uncertainty scale at which the futures split (capped when they never do)
  o(p)    regime change = largest latent change, over the futures, outside the robot's own hand path
Discrete-time CBF condition, relative to doing nothing (h_now = barrier of holding still):
  accept p  iff  h(p) >= min(h_now, H_SAFE) - GAMMA  and  o(p) <= O_MAX
Filter: execute the policy's plan if it satisfies the condition. Otherwise
  minimise  || p - p_policy ||     over candidates = 15 more plans sampled from the policy + 3 simple moves
  subject to the condition;        none feasible -> the candidate with the largest barrier (lowest o as tie-break).
No object state, no failure labels, no task goal. O_MAX is calibrated label-free on recorded decisions (99th percentile
of o for the policy's own plans).
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
CFG = {"TREND_MAX": 2.0, "O4_LAMBDA": 0.003, "H_SAFE": 1.0, "GAMMA": 0.0, "O_MAX": None, "N_SAMPLES": 15, "FUT_CAND": 32, "PLAN_START": 1, "SKIP_TOP": os.environ.get("CBF_SKIP_TOP", "1") == "1", "N_LOOK": 15,
       # variants (2026-10-15 diagnosis): TRIGGER "both" = barrier or regime check can trigger (final-test filter),
       # "barrier" = only the barrier triggers (regime check still screens candidates); SELECT "closest" = closest feasible
       # candidate (final-test filter), "margin" = largest barrier among feasible candidates, then closest
       "TRIGGER": os.environ.get("CBF_TRIGGER", "both"), "SELECT": os.environ.get("CBF_SELECT", "closest"),
       # FIX "select" = choose among policy samples + 3 primitives (final-test filter); "opt" = the single CBF optimisation
       # (eval/cbf_opt.py: path bends + speed, multi-start from the policy plan + 7 policy samples, relative disturbance
       # barrier, exact fork check in the choice; no primitives)
       "FIX": os.environ.get("CBF_FIX", "select"), "OPT_STARTS": int(os.environ.get("OPT_STARTS", 8)),
       # "qfix" (2026-10-15 night): pick the lowest fixed-readout risk among the policy plan + 15 policy samples, no
       # primitives. Fixed readout: fork -> share of futures in the disturbed group; no fork -> 1 if the futures' mean
       # change away from the hand > QFIX_TAU x O_MAX (everything toppling is NOT safe), else 0.
       "QFIX_TAU": float(os.environ.get("QFIX_TAU", 0.5)), "QFIX_TIE": float(os.environ.get("QFIX_TIE", 0.02)),
       # VETO (2026-10-16): final-test selection, but candidates whose fixed readout exceeds the policy plan's are excluded
       # (unless that leaves none). Targets the filter's harmful choices (offline: 9 -> 6 hurt, 36 helped unchanged).
       "VETO": os.environ.get("CBF_VETO") == "1",
       # matched-rate baseline (2026-10-16): TRIGGER "random" acts at a random RAND_P share of decisions (the final filter
       # acted on 7.1% in final test 2), seeded per episode via self.trig_rng
       "RAND_P": float(os.environ.get("CBF_RAND_P", 0.071)),
       # START "G" = object-centric start sampler (per-block pose noise, 3 blocks; all tests so far); "gte" = object-free
       # Gauss-TE field fitted on closed-loop latents (eval/gte_check.py; Phase 2 P2-0b estimator)
       "START": os.environ.get("CBF_START", "G"),
       # BARRIER "fork" = h = log2 of the first splitting scale (all tests so far); "spread" (results/jenga/spread_protocol.json):
       # continuous h = -log(principal-axis variance of the imagined HOLD30 latents at SPREAD_SCALE x the errors), accepted
       # iff h >= h_hold - TAU_S (the plan may raise outcome sensitivity over stopping by at most e^TAU_S); o barrier unchanged
       "BARRIER": os.environ.get("CBF_BARRIER", "fork"), "SPREAD_SCALE": 3.0,
       "TAU_S": float(os.environ["CBF_TAU_S"]) if os.environ.get("CBF_TAU_S") else None,
       # GATE_TV (results/jenga/gtv_protocol.json): act only if the trigger fires AND the relative total variance of the
       # plan's imagined HOLD30 futures at 4x the errors (log tvar plan - log tvar hold, as eval/spread_check.py tvar30)
       # exceeds GATE_TV; same futures as the trigger's fork test (the 4x scale is added; h unchanged, exact)
       "GATE_TV": float(os.environ["CBF_GATE_TV"]) if os.environ.get("CBF_GATE_TV") else None}
CALIB = ROOT / os.environ.get("CBF_CALIB", "results/jenga/cbf_calibration.json")


def horizon(plan, n_look):
    a = np.asarray(plan[CFG["PLAN_START"]:CFG["PLAN_START"] + n_look], np.float32)
    return np.concatenate([a, np.repeat(a[-1:], 38 - n_look, 0)])


class CBF:
    def __init__(self, mon):
        self.trig_rng = np.random.default_rng(0)
        if CFG["START"] == "gte":
            import gte_check
            self.gte = gte_check.GaussStarts()
        import cbf_barrier as cb
        self.mon, self.cb = mon, cb
        if CFG["O_MAX"] is None and CALIB.exists():
            CFG["O_MAX"] = json.loads(CALIB.read_text())["O_MAX"]

    def score(self, sim, frame_z0, plans, n_fut):
        """plans (C, 16, 4) -> h (C,), o (C,) under SCALES, n_fut futures each."""
        import cem_filter as cf
        import step3_world_model as wm
        mon, cb, torch = self.mon, self.cb, self.mon.torch
        z0 = frame_z0
        C = len(plans)
        split = np.zeros((C, len(cb.SCALES)), bool); o = np.zeros(C); hs = np.zeros(C)
        seed = int(mon.rng.integers(2 ** 31))
        # the largest scale never changes h (split only there and no split both give log2(SCALES[-1])): skip it (exact)
        scales = cb.SCALES[:-1] if CFG["SKIP_TOP"] else cb.SCALES
        want_tv = getattr(self, "_want_tv", False)
        if want_tv:                                  # the trigger's gate needs tvar at 4x: keep the top scale (h unchanged)
            assert cb.SCALES[-1] == 4.0
            scales = cb.SCALES; self._tv = None
        if CFG["BARRIER"] == "spread":
            scales = (1.0, CFG["SPREAD_SCALE"])
        for si, s in enumerate(scales):
            d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                           for p, r in (mon.errors.sample(seed, 0, 0, j) for j in range(n_fut))])
            if CFG["START"] == "gte":            # object-free starts: stationary Gaussian field in the latent (no objects)
                st = self.gte(z0, n_fut, s, seed + si)
            else:
                st = mon.g(z0[None].expand(n_fut, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
            acts = []
            for p in plans:
                base = horizon(p, CFG["N_LOOK"])
                for j in range(n_fut):
                    a = base.copy()
                    noise = np.concatenate([mon.snippets[j % 64], mon.snippets[(j + 1) % 64]])[:CFG["N_LOOK"]] * s
                    a[:CFG["N_LOOK"], :3] -= noise
                    acts.append(a)
            pairs = torch.as_tensor(np.stack(acts), device="cuda").reshape(C * n_fut, wm.STEPS, -1)
            z = st.float().repeat(C, 1, 1, 1); ends = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(wm.STEPS):
                    z = mon.model.step(z, pairs[:, t]).float()
                    if t in (wm.HOLD10, wm.HOLD30):
                        ends[t] = z.reshape(C, n_fut, -1)
            if want_tv and s == 4.0:                                          # as eval/spread_check.py (tvar30)
                x = ends[wm.HOLD30].double(); X = x - x.mean(1, keepdim=True)
                self._tv = ((X ** 2).sum((1, 2)) / n_fut / x.shape[2]).float().cpu().numpy()
            if CFG["BARRIER"] == "spread":
                if s == CFG["SPREAD_SCALE"]:                                  # as eval/spread_check.py (spread30)
                    x = ends[wm.HOLD30].double(); X = x - x.mean(1, keepdim=True)
                    w = torch.linalg.eigvalsh(X @ X.transpose(1, 2))
                    hs = -np.log((w[:, -1] / n_fut / x.shape[2]).float().cpu().numpy() + 1e-12)
            else:
                ok10, _, _ = cf.split_test_gpu(ends[wm.HOLD10]); ok30, _, _ = cf.split_test_gpu(ends[wm.HOLD30])
                split[:, si] = (ok10 & ok30).cpu().numpy()
            if s == 1.0:
                for c in range(C):
                    pts = np.vstack([sim.data.site_xpos[sim.site][None], np.asarray(plans[c])[CFG["PLAN_START"]:CFG["PLAN_START"] + CFG["N_LOOK"], :3]])
                    m = torch.as_tensor(~cb.hand_mask(pts), device="cuda")
                    diff = (ends[wm.HOLD30][c].reshape(n_fut, 28, 28, 8) - z0[None]).pow(2).mean(-1)
                    o[c] = float(((diff * m).sum((1, 2)) / m.sum().clamp_min(1)).max())
        if CFG["BARRIER"] == "spread":
            return hs.astype(float), o
        h = np.full(C, np.log2(cb.SCALES[-1]))
        for c in range(C):
            hit = np.flatnonzero(split[c])
            if len(hit):
                h[c] = np.log2(cb.SCALES[hit[0]])
        return h, o

    def qfix(self, sim, z0, plans, n=64):
        """Fixed readout per plan (scale 1, 64 futures): fork -> disturbed-group share; no fork -> mean outside-hand change
        above QFIX_TAU * O_MAX -> 1, else 0."""
        import cem_filter as cf
        import cbf_barrier as cb
        import cbf_qp as q
        torch = self.mon.torch
        bar = q.Barrier(self.mon, z0, int(self.mon.rng.integers(2 ** 31)), n, 1.0)
        with torch.no_grad():
            e10, e30 = bar.endpoints(torch.as_tensor(np.asarray(plans, np.float32), device="cuda"))
        ok10, _, _ = cf.split_test_gpu(e10); ok30, lab, _ = cf.split_test_gpu(e30)
        tau = CFG["QFIX_TAU"] * CFG["O_MAX"]
        site = sim.data.site_xpos[sim.site].copy(); out = []
        for c in range(len(plans)):
            pts = np.vstack([site[None], np.asarray(plans[c])[CFG["PLAN_START"]:CFG["PLAN_START"] + CFG["N_LOOK"], :3]])
            m = torch.as_tensor(~cb.hand_mask(pts), device="cuda").float()
            o = (((e30[c].reshape(n, 28, 28, 8) - z0[None]).pow(2).mean(-1) * m).sum((1, 2)) / m.sum()).cpu().numpy()
            L = lab[c].cpu().numpy().astype(bool)
            if bool(ok10[c] & ok30[c]) and not (L.all() or (~L).all()):
                bad = L if o[L].mean() > o[~L].mean() else ~L; out.append(float(bad.mean()))
            else:
                out.append(float(o.mean() > tau))
        return np.array(out)

    def ok(self, h, o, h_now):
        if CFG["BARRIER"] == "spread":
            return (h >= h_now - CFG["TAU_S"]) & (o <= (CFG["O_MAX"] if CFG["O_MAX"] is not None else np.inf))
        return (h >= min(h_now, CFG["H_SAFE"]) - CFG["GAMMA"]) & (o <= (CFG["O_MAX"] if CFG["O_MAX"] is not None else np.inf))

    def primitives(self, sim, plan):
        """Simple alternative 16-step plans: half speed, vertical only, back off (up 3 cm), as absolute targets."""
        p = np.asarray(plan, np.float32); cur = np.r_[sim.target, 0.0].astype(np.float32)
        slow = p.copy(); w = np.linspace(0, 1, 16)[:, None] * 0.5
        slow[:, :3] = cur[:3] + (p[:, :3] - cur[:3]) * (0.5 + 0 * w); slow[:, 3] = p[:, 3]
        vert = p.copy(); vert[:, :2] = cur[:2]
        back = np.repeat(cur[None], 16, 0); back[:, 2] += np.linspace(0, 0.03, 16); back[:, 3] = 0.0
        return [slow, vert, back]

    def lsf(self, sim, z0, plan, t0):
        """Least-restrictive latent safety filter (Nakamura et al. 2025) at chunk level: imagine the executed segment of
        the policy's plan (8 control steps = 4 model steps) in the world model; execute it if V = Q(z, g, pi(z, g)) > eps
        after every imagined step, else the safety actor's 4-model-step plan imagined from the current real latent."""
        import lsf as L
        torch = self.mon.torch
        if getattr(self, "_lsf", None) is None:
            self._lsf = (L.ValueFn(os.environ.get("LSF_VALUE_NAME", "value.pt")), float(os.environ.get("LSF_EPS", "0.3")))
        vf, eps = self._lsf
        S = CFG["PLAN_START"]; seg = np.asarray(plan, np.float32)[S:S + 8]
        g = torch.as_tensor(np.r_[sim.target, plan[0][3]][None], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            z, gg, vals = z0[None], g, []
            for m in range(4):
                a = torch.as_tensor(seg[2 * m:2 * m + 2].reshape(1, 8), device="cuda")
                z = self.mon.model.step(z, a).float(); gg = torch.cat([a[:, 4:7], a[:, 7:8]], 1)
                vals.append(float(vf.V(z, gg)))
            v_min = min(vals)
            if v_min > eps:
                return seg.copy(), {"intervened": False, "lsf_vmin": v_min, "seconds": round(time.time() - t0, 2)}
            if os.environ.get("LSF_FALLBACK") == "hold":   # LSF-hold: same switch, Stage 4 hold-and-replan fix
                out = seg.copy(); out[:, :3] = sim.target
                return out, {"intervened": True, "abstained": False, "kind": "lsf_hold", "lsf_vmin": v_min,
                             "seconds": round(time.time() - t0, 2)}
            z, gg, out = z0[None], g, []
            for m in range(4):
                a = vf.act(z, gg); out.append(a.view(2, 4).cpu().numpy())
                z = self.mon.model.step(z, a).float(); gg = torch.cat([a[:, 4:7], a[:, 7:8]], 1)
        out = np.concatenate(out).astype(np.float32)
        return out, {"intervened": True, "abstained": False, "kind": "lsf", "lsf_vmin": v_min,
                     "change_mm": float(np.abs(out[:, :3] - seg[:, :3]).max() * 1000), "seconds": round(time.time() - t0, 2)}

    def lsf_step(self, sim, cmds, k, j, info):
        """LSF-step (docs/LATENT_SAFETY_FILTER.md audit 3, stage 1): the least-restrictive switch at every model step (2 control
        steps): encode the real frame now, imagine the policy's next 2 commands one model step ahead, keep them if
        V(z', g') > eps, else the safety actor's 2-step action. Updates the chunk's log entry (info) in place."""
        import lsf as L
        torch = self.mon.torch; t0 = time.time()
        if getattr(self, "_lsf", None) is None:
            self._lsf = (L.ValueFn(os.environ.get("LSF_VALUE_NAME", "value.pt")), float(os.environ.get("LSF_EPS", "0.3")))
        vf, eps = self._lsf
        cmds = np.asarray(cmds, np.float32)
        z = torch.as_tensor(self.mon.latent(self.mon.frame(sim)[None])[0], device="cuda").float()[None]
        g = torch.as_tensor(np.r_[sim.target, cmds[0][3]][None], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            a = torch.as_tensor(cmds.reshape(1, 8), device="cuda"); g2 = torch.cat([a[:, 4:7], a[:, 7:8]], 1)
            if L.ALEA:                         # stage 2: V after each of ALEA noisy one-step imaginations (perceived start and
                if getattr(self, "_noise", None) is None:      # execution noise, the monitor's noise model); the minimum
                    self._noise = L.Noise()
                self._noise.rng = self.mon.rng                 # per-episode seeded (policy_rollout resets mon.rng per episode)
                zs = self._noise.starts(z.expand(L.ALEA, -1, -1, -1).contiguous())
                v = float(vf.V(self.mon.model.step(zs, self._noise.actions(a, L.ALEA)[:, 0]).float(), g2.expand(L.ALEA, -1)).min())
            else:
                v = float(vf.V(self.mon.model.step(z, a).float(), g2))
            out = cmds if v > eps else vf.act(z, g).view(2, 4).cpu().numpy().astype(np.float32)
        info["lsf_vmin"] = min(v, info.get("lsf_vmin", np.inf)); info["lsf_steps"] = info.get("lsf_steps", 0) + 1
        if v <= eps:
            info["intervened"] = True; info["lsf_overrides"] = info.get("lsf_overrides", 0) + 1
        info["seconds"] = round(info.get("seconds", 0.0) + time.time() - t0, 2)
        return out

    def filter(self, sim, chunk, k, plan=None, hist=None, pol=None):
        import policy_rollout as pr
        torch = self.mon.torch
        t0 = time.time()
        if CFG["FIX"] == "lsf_step":           # all switching happens per model step in lsf_step (hooked by policy_rollout)
            return chunk, {"intervened": False, "abstained": False, "kind": "lsf_step", "seconds": 0.0}
        frame = self.mon.frame(sim)
        z0 = torch.as_tensor(self.mon.latent(frame[None])[0], device="cuda").float()
        if CFG["FIX"] == "lsf":                # latent safety filter baseline (docs/LATENT_SAFETY_FILTER.md); no fork score
            return self.lsf(sim, z0, plan, t0)
        hold = np.repeat(np.r_[sim.target, 0.0][None].astype(np.float32), 16, 0)
        self._want_tv = CFG["GATE_TV"] is not None
        try:
            h, o = self.score(sim, z0, np.stack([plan, hold]), 64)
        finally:
            self._want_tv = False
        h_nom, o_nom, h_now = h[0], o[0], h[1]
        info = {"h_nom": float(h_nom), "o_nom": float(o_nom), "h_now": float(h_now)}
        gated = False
        if CFG["GATE_TV"] is not None:
            tv_rel = float(np.log(self._tv[0] + 1e-12) - np.log(self._tv[1] + 1e-12)); info["tv_rel"] = round(tv_rel, 4)
            gated = tv_rel <= CFG["GATE_TV"]
        o_trig = o_nom if CFG["TRIGGER"] == "both" else -np.inf
        if CFG["TRIGGER"] == "random":       # matched-rate baseline: act at random decisions (same fix procedure)
            if not (self.trig_rng.random() < CFG["RAND_P"]):
                return chunk, {**info, "intervened": False, "seconds": round(time.time() - t0, 2)}
        elif CFG["TRIGGER"] == "trend":      # fork-distance CBF with a decrease condition (docs/LITERATURE_PLAN.md step 2):
            if k <= getattr(self, "_last_k", -1):     # act if h < min(h_hold, H_SAFE) (level), OR h fell since the previous
                self._prev_h = None                   # decision while h <= TREND_MAX and h < h_hold (early-warning trend);
            self._last_k = k                          # the fix must then restore h >= min(h_hold, max(H_SAFE, h_prev))
            prev, self._prev_h = getattr(self, "_prev_h", None), float(h_nom)
            lvl = h_nom < min(h_now, CFG["H_SAFE"]) - 1e-9
            trd = prev is not None and h_nom < prev - 1e-9 and h_nom <= CFG["TREND_MAX"] + 1e-9 and h_nom < h_now - 1e-9
            info.update({"h_prev": prev, "trend": bool(trd and not lvl)})
            if not (lvl or trd or o_nom > (CFG["O_MAX"] if CFG["O_MAX"] is not None else np.inf)):
                return chunk, {**info, "intervened": False, "seconds": round(time.time() - t0, 2)}
            self._req = max(CFG["H_SAFE"], prev) if (trd and not lvl) else CFG["H_SAFE"]
        elif CFG["TRIGGER"] == "abs":        # absolute trigger (O-CEM closed loop): also act at relative passes
            if self.ok(np.array([h_nom]), np.array([o_nom]), CFG["H_SAFE"])[0]:
                return chunk, {**info, "intervened": False, "seconds": round(time.time() - t0, 2)}
        elif self.ok(np.array([h_nom]), np.array([o_trig]), h_now)[0]:
            return chunk, {**info, "intervened": False, "seconds": round(time.time() - t0, 2)}
        elif gated:                          # the fork trigger fired but the plan does not spread the futures more than holding
            return chunk, {**info, "intervened": False, "gated": True, "seconds": round(time.time() - t0, 2)}
        if CFG["FIX"] == "hold":               # hold and re-plan: keep position for one chunk (gripper column untouched)
            out = np.asarray(plan, np.float32)[CFG["PLAN_START"]:CFG["PLAN_START"] + 8].copy(); out[:, :3] = sim.target
            if CFG.get("HOLD_GRIP"):           # hold-and-replan 2: gripper also held at the policy's step-0 command
                out[:, 3] = np.asarray(plan, np.float32)[0, 3]
            return out, {**info, "intervened": True, "abstained": False, "kind": "hold", "seconds": round(time.time() - t0, 2)}
        if CFG["FIX"] == "o4":                 # O-CEM (docs/FILTER_DEEP_DIVE.md section 3, eval/opt4.py): minimise imagined disturbance
            import opt4
            import opt3
            import step3_world_model as wm
            if getattr(self, "_certs", None) is None:
                self._certs = [None] + [wm.WorldModel.load(Path(__file__).resolve().parent.parent / f"results/program_b/ens_v1/model_ENS{e}.pt")
                                        for e in range(4)]
            p16 = np.asarray(plan, np.float32).copy(); p16[0, :3] = sim.target
            P = opt3.Problem(self.mon, self, sim, z0, p16, float(h_now), self._certs); P.cbf_O_MAX = CFG["O_MAX"]
            finals, O, costs = opt4.solve(P, int(self.mon.rng.integers(2 ** 31)), CFG["O4_LAMBDA"])
            j = int(np.argmin(O + CFG["O4_LAMBDA"] * costs)); best = P.plan(finals[j])
            out = np.asarray(best, np.float32)[CFG["PLAN_START"]:CFG["PLAN_START"] + 8].copy()
            return out, {**info, "intervened": True, "abstained": False, "kind": "o4", "choice": j, "unchanged": j == 0,
                         "theta": [round(float(x), 4) for x in finals[j]], "O": [round(float(x), 4) for x in O],
                         "cost": round(float(costs[j]), 3), "progress": round(opt3.progress(p16, best), 3),
                         "seconds": round(time.time() - t0, 2)}
        if CFG["FIX"] == "g3":                 # formulation G, G-CEM: coarse transformations of THIS chunk only (eval/opt3.py)
            import opt3
            import step3_world_model as wm
            if getattr(self, "_certs", None) is None:
                self._certs = [None] + [wm.WorldModel.load(Path(__file__).resolve().parent.parent / f"results/program_b/ens_v1/model_ENS{e}.pt")
                                        for e in range(4)]
            p16 = np.asarray(plan, np.float32).copy(); p16[0, :3] = sim.target
            P = opt3.Problem(self.mon, self, sim, z0, p16, float(h_now), self._certs); P.cbf_O_MAX = CFG["O_MAX"]
            saved = CFG["H_SAFE"]
            if CFG["TRIGGER"] == "trend":
                CFG["H_SAFE"] = self._req                 # certificate level for this decision (restored below)
            try:
                best, oi = opt3.solve_cem(P, int(self.mon.rng.integers(2 ** 31)))
            finally:
                CFG["H_SAFE"] = saved
            out = np.asarray(best, np.float32)[CFG["PLAN_START"]:CFG["PLAN_START"] + 8].copy()
            return out, {**info, "intervened": True, "abstained": False, **oi, "fix_kind": oi["kind"], "kind": "g3",
                         "progress": round(opt3.progress(p16, best), 3), "seconds": round(time.time() - t0, 2)}
        if CFG["FIX"] == "psel":               # policy-sample selection (results/jenga/psel_protocol.json; fixbench "sel"):
            import opt2                         # 16 of the policy's own plans, G-CEM's certificate (OIL+ENS), the certified
            import opt3                         # one closest to the policy's plan; none -> hold if certified, else best (h, -o)
            import step3_world_model as wm
            if getattr(self, "_certs", None) is None:
                self._certs = [None] + [wm.WorldModel.load(Path(__file__).resolve().parent.parent / f"results/program_b/ens_v1/model_ENS{e}.pt")
                                        for e in range(4)]
            p16 = np.asarray(plan, np.float32).copy(); p16[0, :3] = sim.target
            P = opt3.Problem(self.mon, self, sim, z0, p16, float(h_now), self._certs); P.cbf_O_MAX = CFG["O_MAX"]
            rs = (torch.get_rng_state(), torch.cuda.get_rng_state())      # sampling candidates must not shift the policy's RNG
            try:
                samples = np.asarray(pr.policy_chunk(pol, hist, n=16), np.float32)
            finally:
                torch.set_rng_state(rs[0]); torch.cuda.set_rng_state(rs[1])
            hold16 = np.repeat(np.r_[sim.target, p16[0, 3]][None], 16, 0).astype(np.float32)
            ok, hc, oc = P.certify(list(samples) + [hold16])
            costs = np.array([opt2.Problem.cost(P, c[1:9, :3] - p16[1:9, :3]) for c in samples]); okc = ok[:-1]
            if okc.any():
                j = int(np.flatnonzero(okc)[np.argmin(costs[okc])]); best, kind = samples[j], "certified"
            elif ok[-1]:
                j, best, kind = -1, hold16, "fallback_hold"
            else:
                j = int(np.lexsort((oc[:-1], -hc[:-1]))[0]); best, kind = samples[j], "fallback_best"
            out = best[CFG["PLAN_START"]:CFG["PLAN_START"] + 8].copy()
            return out, {**info, "intervened": True, "abstained": False, "kind": "psel", "fix_kind": kind, "choice": j,
                         "n_certified": int(okc.sum()), "change_mm": round(float(np.abs(best[1:9, :3] - p16[1:9, :3]).max() * 1000), 1),
                         "progress": round(opt3.progress(p16, best), 3), "seconds": round(time.time() - t0, 2)}
        rs = (torch.get_rng_state(), torch.cuda.get_rng_state())          # sampling candidates must not shift the policy's RNG
        try:
            samples = pr.policy_chunk(pol, hist, n=CFG["OPT_STARTS"] - 1 if CFG["FIX"] == "opt" else CFG["N_SAMPLES"])
        finally:
            torch.set_rng_state(rs[0]); torch.cuda.set_rng_state(rs[1])
        if CFG["FIX"] == "qfix":
            cands = np.concatenate([np.asarray(plan, np.float32)[None], np.asarray(samples, np.float32)])
            qv = self.qfix(sim, z0, cands)
            dist_c = np.array([np.linalg.norm(np.asarray(c)[1:9, :3] - np.asarray(plan)[1:9, :3]) for c in cands])
            near = np.flatnonzero(qv <= qv.min() + CFG["QFIX_TIE"])
            j = int(near[np.argmin(dist_c[near])])                          # lowest readout risk; ties -> closest to the policy
            out = cands[j][CFG["PLAN_START"]:CFG["PLAN_START"] + 8].copy()
            return out, {**info, "intervened": j != 0, "abstained": False, "kind": "qfix", "choice": j,
                         "q_policy": float(qv[0]), "q_choice": float(qv[j]), "change_mm": float(dist_c[j] * 1000),
                         "seconds": round(time.time() - t0, 2)}
        if CFG["FIX"] == "opt":
            import cbf_opt
            starts = np.concatenate([np.asarray(plan, np.float32)[None], np.asarray(samples, np.float32)])
            best, oi = cbf_opt.optimise(self.mon, self, sim, z0, sim.data.site_xpos[sim.site].copy(), starts, h_now,
                                        int(self.mon.rng.integers(2 ** 31)), fork_check=True)
            out = np.asarray(best, np.float32)[CFG["PLAN_START"]:CFG["PLAN_START"] + 8].copy()
            return out, {**info, "intervened": True, "abstained": False, "kind": "opt", **oi,
                         "seconds": round(time.time() - t0, 2)}
        cands = list(samples) + self.primitives(sim, plan)
        hc, oc = self.score(sim, z0, np.stack(cands), CFG["FUT_CAND"])
        feas = self.ok(hc, oc, h_now)
        veto = {}
        if CFG["VETO"]:                      # readout veto: never pick a candidate the fixed readout rates as more disturbed
            qa = self.qfix(sim, z0, np.concatenate([np.asarray(plan, np.float32)[None], np.stack(cands).astype(np.float32)]))
            keep = qa[1:] <= qa[0] + 1e-9
            veto = {"q_policy": float(qa[0]), "vetoed": int((feas & ~keep).sum())}
            if (feas & keep).any():
                feas = feas & keep
        dist = np.array([np.linalg.norm(np.asarray(c)[1:9, :3] - np.asarray(plan)[1:9, :3]) for c in cands])
        if feas.any() and CFG["SELECT"] == "margin":
            f_ = np.flatnonzero(feas); j = int(f_[np.lexsort((dist[f_], -hc[f_]))[0]]); kind = "feasible"
        elif feas.any():
            j = int(np.flatnonzero(feas)[np.argmin(dist[feas])]); kind = "feasible"
        else:
            j = int(np.lexsort((oc, -hc))[0]); kind = "best_effort"
        out = np.asarray(cands[j], np.float32)[CFG["PLAN_START"]:CFG["PLAN_START"] + 8].copy()
        return out, {**info, **veto, "intervened": True, "abstained": False, "choice": j, "kind": kind,
                     "h_choice": float(hc[j]), "o_choice": float(oc[j]), "change_mm": float(dist[j] * 1000),
                     "h_cands": [float(x) for x in hc], "o_cands": [round(float(x), 4) for x in oc],
                     "seconds": round(time.time() - t0, 2)}
