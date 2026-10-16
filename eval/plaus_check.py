"""Step 1 diagnostic (results/jenga/plaus_step1_protocol.json): the diffusion policy's own plausibility of the filters'
executed chunks in dev10, and whether it separates the optimiser's harmful changes (block dropped, no topple) better
than the size of the change in mm.

plausibility error = denoising loss of the policy (epsilon MSE on normalised actions) at fixed timesteps with fixed
noise draws (common random numbers), conditioned on the decision's 2-frame observation re-rendered from saved states.
Delta = error(executed, as a 16-step plan) - error(the policy's own plan).

usage: DIFFUSION_POLICY_DIR=... dp_venv/bin/python eval/plaus_check.py   -> results/jenga/plaus_step1_result.json
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["HARD_MASS"] = "0"; os.environ["EXEC_NOISE_SCALE"] = "2"
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
RUNS = ROOT / "results/program_b/policy_rollouts"
TS, N_EPS = (2, 5, 10, 20, 35), 8


class Plausibility:
    def __init__(self, pol):
        import torch
        self.torch, self.pol = torch, pol
        g = torch.Generator(device="cuda").manual_seed(0)
        self.eps = torch.randn(len(TS) * N_EPS, pol.horizon, pol.action_dim, device="cuda", generator=g)
        self.t = torch.tensor([t for t in TS for _ in range(N_EPS)], device="cuda").long()

    def cond(self, hist):
        """hist: [(wrist, fixed, agent_pos)] x 2 -> global condition (1, D)."""
        torch, pol = self.torch, self.pol
        from diffusion_policy.common.pytorch_util import dict_apply
        w = np.stack([h[0] for h in hist]); f = np.stack([h[1] for h in hist]); s = np.stack([h[2] for h in hist])
        t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2)[None].float().cuda() / 255.0
        obs = {"camera_1": t(w), "camera_2": t(f), "agent_pos": torch.from_numpy(s)[None].cuda()}
        nobs = pol.normalizer.normalize(obs)
        this = dict_apply(nobs, lambda x: x[:, :pol.n_obs_steps].reshape(-1, *x.shape[2:]))
        return pol.obs_encoder(this).reshape(1, -1)

    def error(self, gc, plans):
        """plans (C, 16, 4) numpy -> (C,) mean denoising error over the fixed timesteps and noise draws."""
        torch, pol = self.torch, self.pol
        na = pol.normalizer["action"].normalize(torch.as_tensor(plans, device="cuda").float())        # (C,16,4)
        C, K = len(na), len(self.t)
        x0 = na[:, None].expand(C, K, -1, -1).reshape(C * K, *na.shape[1:])
        eps = self.eps[None].expand(C, -1, -1, -1).reshape(C * K, *na.shape[1:]); t = self.t.repeat(C)
        noisy = pol.noise_scheduler.add_noise(x0, eps, t)
        with torch.no_grad():
            pred = pol.model(noisy, t, local_cond=None, global_cond=gc.expand(C * K, -1))
        return ((pred - eps) ** 2).mean((1, 2)).reshape(C, K).mean(1).cpu().numpy()


def as_plan(plan16, executed):
    """The executed 8-step chunk as a 16-step plan: steps 1-8 replaced, 9-15 shifted by the step-8 change."""
    p = np.asarray(plan16, np.float32).copy(); e = np.asarray(executed, np.float32)
    shift = e[-1, :3] - p[8, :3]
    p[1:9] = e; p[9:, :3] += shift
    return p


def main():
    import policy_rollout as pr
    import inspector_data as insp
    import uncertainty_monitor_dev as um
    from sklearn.metrics import roc_auc_score
    pol = pr.load_policy(); P = Plausibility(pol)
    sim = insp._sim(); cams = pr.Cameras(sim); snippets = np.load(um.BENCH)["snippets"]
    rows = []
    for arm in ("cbfmoil", "cbfmsoil"):
        res = {(r := json.loads(l))["episode"]: r for l in (ROOT / f"results/jenga/paired_{arm}_dev10.jsonl").read_text().splitlines()}
        for ep, r in sorted(res.items()):
            with np.load(RUNS / f"paired_{arm}_dev10" / f"ep{ep:03d}.npz") as f:
                S = {k: f[k] for k in f.files}
            nrng = np.random.default_rng([20261502, ep])
            noises = [snippets[nrng.integers(len(snippets))] * 2.0 for _ in range(int(S["chunk_index"][-1]) + 1)]
            for d in np.flatnonzero(S["intervened"]):
                snap = lambda i: (S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i]))
                sim.restore(snap(d)); cur = cams.obs()
                if d > 0 and S["chunk_index"][d - 1] == S["chunk_index"][d] - 1:     # obs after step 7 of the previous chunk
                    sim.restore(snap(d - 1)); k = int(S["chunk_index"][d - 1])
                    for j, a in enumerate(S["executed"][d - 1][:7]):
                        a = np.asarray(a, np.float32).copy(); a[:3] -= noises[k][j]; sim.execute(a)
                    prev = cams.obs()
                else:
                    prev = cur
                gc = P.cond([prev, cur])
                e = P.error(gc, np.stack([S["plan16"][d], as_plan(S["plan16"][d], S["executed"][d])]))
                rows.append({"arm": arm, "ep": int(ep), "d": int(d), "err_policy": float(e[0]), "err_exec": float(e[1]),
                             "delta": float(e[1] - e[0]),
                             "change_mm": float(np.abs(S["executed"][d][:, :3] - S["nominal"][d][:, :3]).max() * 1000)})
            print(arm, ep, flush=True)
    out = {"interventions": {a: sum(r["arm"] == a for r in rows) for a in ("cbfmoil", "cbfmsoil")}}
    for a in ("cbfmoil", "cbfmsoil"):
        D = np.array([r["delta"] for r in rows if r["arm"] == a]); M = np.array([r["change_mm"] for r in rows if r["arm"] == a])
        out[a] = {"median_delta": round(float(np.median(D)), 4), "q75_delta": round(float(np.quantile(D, 0.75)), 4),
                  "median_err_policy": round(float(np.median([r["err_policy"] for r in rows if r["arm"] == a])), 4),
                  "median_change_mm": round(float(np.median(M)), 1),
                  "spearman_delta_vs_mm": round(float(__import__("scipy.stats").stats.spearmanr(D, M).correlation), 3)}
    # optimiser episodes without a topple: max Delta / max mm per episode -> 'not lifted' vs 'success'
    res = {(r := json.loads(l))["episode"]: r for l in (ROOT / "results/jenga/paired_cbfmoil_dev10.jsonl").read_text().splitlines()}
    y, sd, sm = [], [], []
    for ep, r in res.items():
        R = [x for x in rows if x["arm"] == "cbfmoil" and x["ep"] == ep]
        if r["neighbour_toppled"] or not R:
            continue
        y.append(r.get("seconds_to_3cm") is None); sd.append(max(x["delta"] for x in R)); sm.append(max(x["change_mm"] for x in R))
    y = np.array(y)
    out["opt_no_topple_episodes"] = {"n": int(len(y)), "not_lifted": int(y.sum()),
                                     "auc_max_delta": round(float(roc_auc_score(y, sd)), 3), "auc_max_mm": round(float(roc_auc_score(y, sm)), 3)}
    a, b = out["cbfmoil"]["median_delta"], out["cbfmsoil"]["median_delta"]
    out["criterion_a (opt delta clearly above selection)"] = bool(a > b and out["cbfmoil"]["q75_delta"] > out["cbfmsoil"]["q75_delta"])
    out["criterion_b (AUC delta >= AUC mm + 0.05)"] = bool(out["opt_no_topple_episodes"]["auc_max_delta"] >= out["opt_no_topple_episodes"]["auc_max_mm"] + 0.05)
    (ROOT / "results/jenga/plaus_step1_result.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
