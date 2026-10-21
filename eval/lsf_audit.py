"""LSF re-audit (docs/LATENT_SAFETY_FILTER.md "Open item"): is the DDPG safety actor exploiting the world model?
UNISafe (Seo et al. 2025, CMU-IntentLab/UNISafe) adds the world model's epistemic uncertainty (ensemble disagreement) as a
failure margin precisely because a reachability actor otherwise drives into states the model cannot predict. Here, on the
729 held-out decision points of eval/lsf_diag.py, 4 imagined model steps under three plans:
  actor  = the trained safety actor (deployed fallback), policy = the policy's own next 8 control steps, hold = stay put.
Reports: actor saturation (|tanh| > 0.95), its mean target motion per control step, ensemble disagreement (std over
model_OIL + ENS0-3 of the predicted latent) per plan, and imagined V after each plan.
usage: LSF_VALUE_NAME=value_g0.95.pt python3 eval/lsf_audit.py
"""
import glob
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]


def main():
    import torch
    import lsf as L
    import step3_world_model as wm
    vf = L.ValueFn(os.environ.get("LSF_VALUE_NAME", "value_g0.95.pt"))
    models = [wm.WorldModel.load(L.WM_PATH)] + [wm.WorldModel.load(ROOT / f"results/program_b/ens_v1/model_ENS{e}.pt") for e in range(4)]
    for m in models:
        m.net.eval()
    files = [f for src in ("cf_v1", "cf_v2", "closedloop_v1", "oil_v1")
             for f in sorted(glob.glob(str(ROOT / f"results/program_b/{src}/val/ep*.npz"))) if not f.endswith(".tmp.npz")]
    rz, rg, ra = [], [], []
    for f in files:
        d = np.load(f); rz.append(d["z0"]); rg.append(d["actions"][0, 0]); ra.append(d["actions"][0, :8])
    z0 = torch.as_tensor(np.stack(rz), device="cuda").float(); g0 = torch.as_tensor(np.stack(rg), device="cuda").float()
    pol = torch.as_tensor(np.stack(ra), device="cuda").float()                           # (N, 8, 4) policy's next 8 steps
    N = len(z0); out = {"decision_points": N}

    def run(plan_fn, name):
        dis, vals, z, g = [], [], z0, g0
        sat, dz = [], []
        with torch.no_grad():
            for m in range(4):
                a = plan_fn(m, z, g)
                if name == "actor":
                    raw = (a.view(-1, 2, 4)[:, :, :3] - torch.stack([g[:, :3], a[:, :3]], 1)) / L.MAX_STEP
                    sat.append(float((raw.abs() > 0.95).float().mean())); dz.append(raw[:, :, 2].mean().item() * L.MAX_STEP * 1000)
                preds = torch.stack([mm.step(z, a).float() for mm in models])               # (5, N, 28, 28, 8)
                dis.append(preds.std(0).mean((1, 2, 3)).cpu().numpy())
                z = preds[0]; g = torch.cat([a[:, 4:7], a[:, 7:8]], 1)
                vals.append(vf.V(z, g).cpu().numpy())
        r = {"disagreement_mean_per_step": [float(x.mean()) for x in dis], "disagreement_final_median": float(np.median(dis[-1])),
             "V_after_plan_median": float(np.median(vals[-1]))}
        if sat:
            r["actor_saturated_share"] = float(np.mean(sat)); r["actor_mean_dz_mm_per_step"] = float(np.mean(dz))
        return r, dis[-1]

    out["actor"], d_act = run(lambda m, z, g: vf.act(z, g), "actor")
    out["policy"], d_pol = run(lambda m, z, g: pol[:, 2 * m:2 * m + 2].reshape(N, 8), "policy")
    out["hold"], d_hold = run(lambda m, z, g: torch.cat([g, g], 1), "hold")
    out["share_actor_disagreement_above_policy_p95"] = float((d_act > np.quantile(d_pol, 0.95)).mean())
    (ROOT / f"results/jenga/lsf_audit_{os.environ.get('LSF_VALUE_NAME', 'value_g0.95.pt')[:-3]}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
