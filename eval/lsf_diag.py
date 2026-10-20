"""LSF sanity check (diagnostic only, no tuning): on held-out validation decision points, does the learned safety value
V(z) = Q(z, g, pi(z, g)) separate states whose REAL futures topple a neighbour from those whose futures do not?
Start latents and labels are read from the same validation files (no reliance on stored array order).
usage: python3 eval/lsf_diag.py
"""
import glob
import os
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]


def main():
    import torch
    import lsf as L
    from wm_libero import auroc
    from fresh_fragility_panel import neighbour_tilts, TOPPLE_DEG
    _, Critic, Actor = L.nets()
    w = torch.load(L.OUT / os.environ.get("LSF_VALUE_NAME", "value.pt"), map_location="cuda")
    Q, A = Critic().cuda().eval(), Actor().cuda().eval(); Q.load_state_dict(w["critic"]); A.load_state_dict(w["actor"])
    Clf, _, _ = L.nets(); clf = Clf().cuda().eval(); clf.load_state_dict(torch.load(L.OUT / "classifier.pt", map_location="cuda"))
    files = [f for src in ("cf_v1", "cf_v2", "closedloop_v1", "oil_v1")
             for f in sorted(glob.glob(str(ROOT / f"results/program_b/{src}/val/ep*.npz"))) if not f.endswith(".tmp.npz")]
    share, anyf, rz, rp = [], [], [], []
    for f in files:
        d = np.load(f); elig = d["start_tilt"] < TOPPLE_DEG
        rz.append(d["z0"]); rp.append(d["actions"][0, 0])              # start latent and target, read from the same file
        top = [bool(np.any((neighbour_tilts(t[-1]) >= TOPPLE_DEG) & elig)) for t in d["traces"]]
        share.append(float(np.mean(top))); anyf.append(any(top))
    share, anyf, rz, rp = np.array(share), np.array(anyf), np.stack(rz), np.stack(rp)
    with torch.no_grad():
        z = torch.as_tensor(rz, device="cuda").float(); g = torch.as_tensor(rp, device="cuda").float()
        V = np.concatenate([Q(z[i:i + 256], g[i:i + 256], A(z[i:i + 256], g[i:i + 256])).cpu().numpy() for i in range(0, len(z), 256)])
        l0 = np.concatenate([clf(z[i:i + 256], g[i:i + 256]).cpu().numpy() for i in range(0, len(z), 256)])
    res = {"decision_points": int(len(V)), "with_any_toppling_future": int(anyf.sum()),
           "AUROC_minusV_vs_any_topple": auroc(-V, anyf), "spearman_minusV_vs_topple_share": float(np.corrcoef(np.argsort(np.argsort(-V)), np.argsort(np.argsort(share)))[0, 1]),
           "AUROC_minus_l_now_vs_any_topple": auroc(-l0, anyf),
           "V_quantiles": np.quantile(V, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist(),
           "share_V_below_0.3": float((V < 0.3).mean()), "share_V_below_0": float((V < 0).mean())}
    (ROOT / f"results/jenga/lsf_value_diag_{os.environ.get('LSF_VALUE_NAME', 'value.pt')[:-3]}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
