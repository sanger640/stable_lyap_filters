"""Task-agnostic world-model monitor and CEM safety filter for closed-loop control (three-block task first).

Monitor: close-camera frame of the current scene -> DINOv2/PCA latent -> v3 start sampler G (64 planar perception-error
draws) -> v3 plain dynamics US with the candidate chunk under 64 execution-noise snippets -> endpoints at hold 10/30 ->
BOTH split test (the clean-trained plain model's readout; final test on panels 6/7: 86% caught / 12% false alarms).

Filter (minimal-change safety filter, no task knowledge):
    minimise  ||delta||^2 / s_a^2  +  lam * risk(a_nom + delta)  +  mu * intent(a_nom + delta)
    delta: xyz offsets at the chunk's start and end, linearly interpolated over its 8 commands (6 numbers, |.| <= 10 mm);
           the gripper command is never changed.
    risk:   continuous fork score of the 64 predicted futures at hold 30: (minority fraction) x (group separation / floor),
            zero when the BOTH test does not fire.
    intent: distance of the candidate's predicted majority outcome from the nominal chunk's predicted majority outcome
            (latent, / floor) -- the policy's own action defines what it intends.
Solved with CEM (population 16, 3 iterations, 4 elites); every candidate gets the exact 64-future BOTH test (GPU). If the best candidate is still flagged the filter ABSTAINS:
the robot holds its current target for the chunk and the policy re-plans from the new observation.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
V3 = ROOT / "results/program_b/threeblock_v3"
CEM = {"pop": 16, "iters": 3, "elites": 4, "max_mm": 10.0, "init_sd_mm": 5.0, "s_a_mm": 6.0, "lam": 20.0, "mu": 1.0}
# Profile (3090): frame 0.2 s, encode + G 0.4 s, 64-future rollout 0.36 s; the BOTH test runs exactly on the GPU
# (split_test_gpu, 80/80 agreement with outcome_modes.two_mode_test, 0.08 s for 32 candidates).


class Monitor:
    def __init__(self, seed=20261501):
        import torch
        import mujoco  # noqa: F401
        import step3_world_model as wm
        import step3b_models as sm
        import uncertainty_monitor_dev as um
        import visual_pipeline_panel3 as p3
        from step3_data import Latent
        self.torch, self.wm, self.sm = torch, wm, sm
        self.model = wm.WorldModel.load(V3 / "model_US.pt"); self.model.net.eval()
        self.g = sm.G.load(V3 / "G.pt")
        self.tau = json.loads((V3 / "floor_US.json").read_text())["tau"]
        self.latent, self.errors = Latent(), p3.ErrorModel()
        self.snippets = np.load(um.BENCH)["snippets"]
        self.rng = np.random.default_rng(seed)
        self.renderer = None

    def frame(self, sim):
        import mujoco
        from jenga_render_frames import close_camera
        if self.renderer is None:
            self.renderer = mujoco.Renderer(sim.model, height=384, width=384); self.cam = close_camera(mujoco)
        self.renderer.update_scene(sim.data, camera=self.cam)
        return self.renderer.render().copy()

    def starts(self, frame):
        """64 sampled perceived starts for the current camera frame."""
        torch = self.torch
        z0 = torch.as_tensor(self.latent(frame[None])[0], device="cuda").float()
        key = int(self.rng.integers(2 ** 31))
        d = np.stack([np.concatenate([(p * [1, 1, 0]).reshape(9), (r * [0, 0, 1]).reshape(9)])
                      for p, r in (self.errors.sample(key, 0, 0, j) for j in range(64))])
        return self.g(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d, dtype=torch.float32))

    def endpoints(self, starts, chunks, idx=None):
        """starts (n, ...) ; chunks (C, 8, 4) candidate chunks -> e10, e30 each (C, n, 6272) on GPU. Future j uses perceived
        start j and execution-noise snippet idx[j] (idx = all 64 by default)."""
        torch, wm = self.torch, self.wm
        C, n = len(chunks), len(starts)
        snip = self.snippets if idx is None else self.snippets[idx]
        acts = np.stack([wm.panel_actions(c, snip) for c in chunks])                      # (C, n, 38, 4)
        pairs = torch.as_tensor(acts, device="cuda").reshape(C * n, wm.STEPS, -1)
        z = starts.float().repeat(C, 1, 1, 1)
        ends = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(wm.STEPS):
                z = self.model.step(z, pairs[:, t]).float()
                if t in (wm.HOLD10, wm.HOLD30):
                    ends[t] = z.reshape(C, n, -1)
        return ends[wm.HOLD10], ends[wm.HOLD30]

    def assess(self, e10, e30):
        """Per candidate (exact GPU BOTH test): alarm, continuous risk (alarmed: 1 + minority fraction x separation / floor;
        else 0), majority centroid at hold 30."""
        ok10, _, _ = split_test_gpu(e10)
        ok30, lab, sep = split_test_gpu(e30)
        alarm = (ok10 & ok30)
        n = lab.shape[1]
        ones = lab.sum(1)
        frac = self.torch.minimum(ones, n - ones).double() / n
        major = (ones > n / 2).long()[:, None]
        w = (lab == major).double()
        centroid = (w[..., None] * e30.double()).sum(1) / w.sum(1, keepdim=True).clamp_min(1)
        risk = alarm.double() * (1 + frac * sep / self.tau)
        return alarm.cpu().numpy(), risk.cpu().numpy(), centroid

    def check(self, sim, chunk):
        t0 = time.time()
        s = self.starts(self.frame(sim))
        alarm, _, _ = self.assess(*self.endpoints(s, chunk[None]))
        return bool(alarm[0]), time.time() - t0, s


def shift(chunk, p):
    """p (6,) mm: start/end xyz offsets, interpolated over the chunk; gripper untouched."""
    w = np.linspace(0, 1, len(chunk))[:, None]
    out = np.asarray(chunk, np.float32).copy()
    out[:, :3] += ((1 - w) * p[:3] + w * p[3:]) / 1000.0
    return out


def split_test_gpu(e):
    """Exact batched GPU version of outcome_modes.two_mode_test (perturbation 0) for e (C, n, D):
    principal axis of the centred endings, best two-way split of the 1-D projection (>= 2 per side), BIC 1 vs 2 Gaussians
    and Ashman's D. -> ok (C,) bool = BIC prefers two & D > 2 & minority >= 2 (the alarm rule of phase2_amend4.alarm),
    labels (C, n), separation (C,) in RMS units."""
    torch = __import__("torch")
    x = e.double()
    C, n, D = x.shape
    X = x - x.mean(1, keepdim=True)
    w, V = torch.linalg.eigh(X @ X.transpose(1, 2))
    y = V[:, :, -1] * w[:, -1:].clamp_min(0).sqrt()                      # = centred @ top right-singular vector (sign free)
    ys, order = y.sort(1)
    i = torch.arange(1, n, device=x.device, dtype=x.dtype)[None]
    cs = ys.cumsum(1); tot = cs[:, -1:]
    m1 = cs[:, :-1] / i; m2 = (tot - cs[:, :-1]) / (n - i)
    between = i * (n - i) / n * (m1 - m2) ** 2
    valid = (i >= 2) & (n - i >= 2)
    between = torch.where(valid, between, torch.full_like(between, -1.0))
    split = between.argmax(1) + 1                                          # size of the low group
    rank = torch.arange(n, device=x.device)[None]
    low_sorted = rank < split[:, None]
    lab = torch.zeros(C, n, device=x.device, dtype=torch.long)
    lab.scatter_(1, order, (~low_sorted).long())
    ya = torch.where(lab == 0, y, torch.nan); yb = torch.where(lab == 1, y, torch.nan)
    na = (lab == 0).sum(1).double(); nb = (lab == 1).sum(1).double()
    ma, mb = ya.nanmean(1), yb.nanmean(1)
    va = ((ya - ma[:, None]) ** 2).nansum(1) / na; vb = ((yb - mb[:, None]) ** 2).nansum(1) / nb
    pooled = (va + vb).sqrt()
    d = np.sqrt(2) * (ma - mb).abs() / pooled
    n2ll = lambda m, v: m * (torch.log(2 * np.pi * v.clamp_min(1e-12)) + 1)
    bic1 = n2ll(torch.full_like(na, n), y.var(1, unbiased=False)) + 2 * np.log(n)
    bic2 = n2ll(na, va) + n2ll(nb, vb) - 2 * (na * torch.log(na / n) + nb * torch.log(nb / n)) + 5 * np.log(n)
    ok = (bic2 < bic1) & (d > 2) & (torch.minimum(na, nb) >= 2)
    xa = (x * (lab == 0)[..., None]).sum(1) / na[:, None]; xb = (x * (lab == 1)[..., None]).sum(1) / nb[:, None]
    sep = (xa - xb).norm(dim=1) / np.sqrt(D // 3)
    return ok, lab, sep


def proxy(e30, scale):
    """GPU fork score per candidate: project hold-30 endpoints (C, n, D) on their principal direction, best two-way split
    of the 1-D projection (max between-group variance) -> minority fraction x separation (RMS units) / scale; and the
    majority-group centroid (C, D)."""
    torch = __import__("torch")
    C, n, D = e30.shape
    X = e30.float() - e30.float().mean(1, keepdim=True)
    G = X @ X.transpose(1, 2)                                            # (C, n, n)
    w, V = torch.linalg.eigh(G)
    p = V[:, :, -1] * w[:, -1:].clamp_min(0).sqrt()                      # projection on the top direction (C, n)
    ps, idx = p.sort(1)
    cs = ps.cumsum(1); tot = cs[:, -1:]
    k = torch.arange(1, n, device=e30.device).float()[None]
    m1 = cs[:, :-1] / k; m2 = (tot - cs[:, :-1]) / (n - k)
    bvar = k * (n - k) / n ** 2 * (m1 - m2) ** 2
    bvar[:, :1] = 0; bvar[:, -1:] = 0                                    # need >= 2 futures per group
    j = bvar.argmax(1)
    kk = (j + 1).float()
    sep = (m2.gather(1, j[:, None]) - m1.gather(1, j[:, None])).abs()[:, 0] / np.sqrt(D / 3)
    frac = torch.minimum(kk, n - kk) / n
    lab = torch.zeros(C, n, device=e30.device)
    lab.scatter_(1, idx, (torch.arange(n, device=e30.device)[None] > j[:, None]).float())
    major = (lab.sum(1) > n / 2).float()[:, None]
    wmaj = (lab == major).float()
    centroid = (wmaj[:, :, None] * e30.float()).sum(1) / wmaj.sum(1, keepdim=True).clamp_min(1)
    return (frac * sep / scale), centroid


def cem_filter(mon, sim, chunk, starts=None, rng=None):
    """-> (chunk to execute, or None to abstain; info). Every candidate is scored with the monitor's own exact BOTH test
    on 64 futures; the cheapest unflagged candidate (action change + intent) is executed."""
    t0 = time.time()
    rng = rng or np.random.default_rng(0)
    s = starts if starts is not None else mon.starts(mon.frame(sim))
    alarm0, _, cen0 = mon.assess(*mon.endpoints(s, chunk[None]))
    if not alarm0[0]:
        return chunk, {"intervened": False, "seconds": time.time() - t0}
    D = cen0.shape[1]
    mean, sd = np.zeros(6), np.full(6, CEM["init_sd_mm"])
    best = None
    for it in range(CEM["iters"]):
        P = np.clip(rng.normal(mean, sd, (CEM["pop"], 6)), -CEM["max_mm"], CEM["max_mm"])
        alarm, risk, cen = mon.assess(*mon.endpoints(s, np.stack([shift(chunk, p) for p in P])))
        intent = ((cen - cen0).norm(dim=1) / np.sqrt(D // 3) / mon.tau).cpu().numpy()
        change = (P ** 2).sum(1) / CEM["s_a_mm"] ** 2
        cost = change + CEM["lam"] * risk + CEM["mu"] * intent
        for j in np.flatnonzero(~alarm):
            if best is None or cost[j] < best[0]:
                best = (float(cost[j]), P[j], float(change[j]), float(intent[j]))
        order = np.argsort(cost)
        el = P[order[:CEM["elites"]]]
        mean, sd = el.mean(0), el.std(0) + 0.5
    if best is None:
        return None, {"intervened": True, "abstained": True, "seconds": time.time() - t0}
    return shift(chunk, best[1]), {"intervened": True, "abstained": False, "seconds": time.time() - t0,
                                   "offset_mm": best[1].round(2).tolist(), "intent": round(best[3], 3)}
