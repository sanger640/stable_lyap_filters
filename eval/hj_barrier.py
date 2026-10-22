"""Idea 4 (docs/HJ_REACHABILITY.md): the learned viability value V (eval/hj_value.py) as the CBF barrier, gradient solver
over formulation G's theta. Constraint V(z_next) >= (1 - ALPHA) * V(z_now), z_next = world-model mean after the
transformed chunk (4 model steps = 8 actions); cost and final certificate exactly as G-CEM (opt3 / opt2.finish), so the
only change is the search. Draft: not run until its spec is written (after gate 3a).
usage: python3 eval/hj_barrier.py run <dev|holdout> <tag> ; then opt3.py truth / report with the same tag
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import opt3  # noqa: E402

CFG = {"ALPHA": 0.5, "STARTS": 6, "STEPS": 30, "LR": [0.05, 0.4, 0.05, 1.5], "MU": 10.0}
_V = None


def value_net():
    global _V
    if _V is None:
        import torch
        import hj_value as hv
        _V = hv.net().cuda(); _V.load_state_dict(torch.load(hv.OUT / "value_V.pt")); _V.eval()
        for q in _V.parameters():
            q.requires_grad_(False)
    return _V


def v_next(P, plans):
    """plans (S, 16, 4) torch -> V of the imagined state after the chunk (S,), differentiable in plans."""
    import torch
    V = value_net(); S = len(plans)
    a = plans[:, 1:16]; acts = torch.cat([a, a[:, -1:].expand(-1, 38 - 15, -1)], 1)          # cbf_filter.horizon
    z = P.z0[None].expand(S, -1, -1, -1).contiguous()
    for t in range(4):
        z = P.mon.model.step(z, acts[:, 2 * t:2 * t + 2].reshape(S, 8)).float()
    pr = torch.cat([plans[:, 8, :3], torch.where(plans[:, 8, 3:4] > 0, 1.0, -1.0)], 1)
    return torch.sigmoid(V(z, pr))


def solve_v(P, seed):
    import torch
    V = value_net()
    site = P.sim.data.site_xpos[P.sim.site].copy()
    with torch.no_grad():
        pr0 = torch.as_tensor(np.r_[site, 1.0 if P.sim.gripper < 50 else -1.0][None], device="cuda").float()
        v0 = float(torch.sigmoid(V(P.z0[None], pr0))[0])
    rng = np.random.default_rng(seed)
    starts = np.vstack([opt3.IDENTITY] + [opt3.LO + (opt3.HI - opt3.LO) * rng.random(4) for _ in range(CFG["STARTS"] - 1)])
    lo, hi = (torch.as_tensor(x, device="cuda").float() for x in (opt3.LO, opt3.HI))
    th = torch.as_tensor(starts, device="cuda", dtype=torch.float32).requires_grad_(True)
    p0 = torch.as_tensor(P.p0, device="cuda"); lr = torch.as_tensor(CFG["LR"], device="cuda").float()
    opt = torch.optim.Adam([th], lr=1.0); lam = torch.zeros(len(starts), device="cuda"); mu = CFG["MU"]

    def cost_t(p):
        D = (p[:, 1:9, :3] - p0[1:9, :3]) * 1000
        sd = torch.diff(torch.cat([torch.zeros_like(D[:, :1]), D], 1), 2, dim=1)
        return ((D ** 2).sum((1, 2)) + opt3.CFG["ETA"] * (sd ** 2).sum((1, 2))) / 100

    for it in range(CFG["STEPS"]):
        p = opt3.transform(p0, th)
        g = (1 - CFG["ALPHA"]) * v0 - v_next(P, p)                                   # <= 0 when the barrier holds
        al = (torch.relu(lam + mu * g) ** 2 - lam ** 2) / (2 * mu)
        loss = (cost_t(p) + 100 * al).sum()
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            th.grad.mul_(lr)
        opt.step()
        with torch.no_grad():
            th.copy_(torch.maximum(torch.minimum(th, hi), lo))
            if it % 5 == 4:
                lam = torch.relu(lam + mu * g.detach())
    with torch.no_grad():
        vf = v_next(P, opt3.transform(p0, th)).cpu().numpy()
    finals = list(th.detach().cpu().numpy().astype(np.float64))
    out, info = opt3.opt2.finish(P, finals, {"v0": round(v0, 3), "v_final": [round(float(x), 3) for x in vf]})
    return out, {**info, "theta": opt3._chosen(P, out, finals)}


if __name__ == "__main__":
    assert sys.argv[1] == "run"
    opt3.solve_grad = solve_v
    opt3.run("grad", sys.argv[2], sys.argv[3])
