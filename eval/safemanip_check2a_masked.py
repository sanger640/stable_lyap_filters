"""SafeManip check 2a-masked (docs/BENCHMARK_DD.md, spec fixed before running): does masking the robot's own pixels let
the split test see real drop / tip / knock forks?

Inputs: ~/wksp/safemanip/dd_masked_s{31,32}/<task>.json (labels), <task>_s<seed>_e<ep>_t<t>.npy (9 replays x 3 views x
256 x 256 x 3; views: agentview_left (gate), monitor camera, eye_in_hand) and ..._mask.npy (robot masks, packbits).
Per view: channel PCA 8 of DINOv2-392 tokens fitted on exact-replay frames; per decision point, latent patches (28x28)
with > 30% robot pixels in any of the 8 noisy replays are excluded; two-mode split test on the remaining latent.
usage: python3 eval/safemanip_check2a_masked.py
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
import os
DIRS = {int(k): Path.home() / f"wksp/safemanip/{v}" for k, v in (x.split(":") for x in os.environ.get("SM_DIRS", "31:dd_masked_s31,32:dd_masked_s32").split(","))}
FORK_LO = int(os.environ.get("SM_FORK_LO", 2))
OUT_NAME = os.environ.get("SM_OUT_NAME", "safemanip_check2a_masked_result.json")
VIEWS = ("agentview_left", "monitor_cam", "eye_in_hand")
OUT = ROOT / "results/benchmark_dd"


def patch_mask(m):
    """(256, 256) bool -> (28, 28) robot fraction."""
    edges = np.linspace(0, 256, 29).astype(int)
    return np.array([[m[edges[i]:edges[i + 1], edges[j]:edges[j + 1]].mean() for j in range(28)] for i in range(28)])


def main():
    import torch
    import cem_filter as cf
    from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
    from jenga_vjepa_encode import dino_tokens
    from phase2_encoder_diag import fit_channel_pca
    from wm_libero import auroc
    enc = load_world_model(DEFAULT_CHECKPOINT, "cuda").encoder
    pts = []
    for seed, d in DIRS.items():
        for jf in sorted(d.glob("*.json")):
            for r in json.loads(jf.read_text())["rows"]:
                f = d / f"{r['task']}_s{seed}_e{r['episode']}_t{r['t']}.npy"
                if f.exists():
                    pts.append((f, f.with_name(f.stem + "_mask.npy"), r))
    nf = len(np.load(pts[0][0], mmap_mode="r")) - 1                     # noisy replays per point
    fork = np.array([FORK_LO <= r["n_bad"] <= nf - FORK_LO for *_, r in pts])
    res = {"saved_points": len(pts), "real_forks": int(fork.sum()), "points_with_any_bad": int(sum(r["n_bad"] > 0 for *_, r in pts))}
    for v, name in enumerate(VIEWS):
        rng = np.random.default_rng(0)
        fit = np.stack([np.load(pts[i][0])[0, v] for i in rng.choice(len(pts), min(150, len(pts)), replace=False)])
        t = dino_tokens(enc, fit, "cuda").reshape(-1, 384)
        mean, basis, zm, zs = fit_channel_pca(t[torch.as_tensor(rng.choice(len(t), min(60000, len(t)), replace=False), device="cuda")])
        sep_m, ok_m, sep_u, ok_u, kept = [], [], [], [], []
        for f, mf, _ in pts:
            fr = np.load(f)[1:, v]
            mk = np.unpackbits(np.load(mf), axis=-1)[1:, v, :, :256].astype(bool)
            z = (((dino_tokens(enc, fr, "cuda") - mean) @ basis - zm) / zs).reshape(len(fr), 28, 28, 8)
            keep = np.max(np.stack([patch_mask(m) for m in mk]), 0) <= 0.3
            kept.append(float(keep.mean()))
            zm_ = z[:, torch.as_tensor(keep, device="cuda")].reshape(1, len(fr), -1)
            o, _, s = cf.split_test_gpu(zm_); ok_m.append(bool(o[0])); sep_m.append(float(s[0]))
            o, _, s = cf.split_test_gpu(z.reshape(1, len(fr), -1)); ok_u.append(bool(o[0])); sep_u.append(float(s[0]))
        ok_m, ok_u = np.array(ok_m), np.array(ok_u)
        res[name] = {"masked": {"AUROC_separation_vs_fork": auroc(np.array(sep_m), fork), "split_rate_forks": float(ok_m[fork].mean()),
                                "split_rate_nonforks": float(ok_m[~fork].mean())},
                     "unmasked": {"AUROC_separation_vs_fork": auroc(np.array(sep_u), fork), "split_rate_forks": float(ok_u[fork].mean()),
                                  "split_rate_nonforks": float(ok_u[~fork].mean())},
                     "share_of_patches_kept_median": float(np.median(kept))}
        print(name, json.dumps(res[name]), flush=True)
    gm = res["monitor_cam"]["masked"]
    res["gate_pass_monitor_cam_masked"] = bool(gm["AUROC_separation_vs_fork"] >= 0.7 and gm["split_rate_forks"] >= 2 * max(gm["split_rate_nonforks"], 1e-9))
    g = res["agentview_left"]["masked"]
    res["gate_pass_agentview_left_masked"] = bool(g["AUROC_separation_vs_fork"] >= 0.7 and g["split_rate_forks"] >= 2 * max(g["split_rate_nonforks"], 1e-9))
    (OUT / OUT_NAME).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
