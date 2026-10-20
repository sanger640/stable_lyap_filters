"""OopsieBench check 2a (docs/BENCHMARK_DD.md, spec fixed before rendering): are real egg-damage forks visible as a
split of the REAL futures' end latents?

Inputs: ~/wksp/oopsie/dd/ends_{without,with}_0.05.npz (end frames of the 9 replays per decision point, 2 cameras:
0 = robot0_agentview_right (gate), 1 = robot0_eye_in_hand (secondary)) and fork_{split}_0.05_render.json (labels).
Per camera: channel PCA 8 of DINOv2-392 tokens fitted on the exact-replay end frames; two-mode split test on the 8 noisy
end latents -> split flag and separation; AUROC(separation, real fork) and split rates. Also a damage readout (linear,
pooled latents, demos split by index) as a visibility ceiling.
usage: python3 eval/oopsie_check2a.py
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
DD = Path.home() / "wksp/oopsie/dd"
OUT = ROOT / "results/benchmark_dd"


def main():
    import torch
    import cem_filter as cf
    from jenga_runtime import DEFAULT_CHECKPOINT, load_world_model
    from jenga_vjepa_encode import dino_tokens
    from phase2_encoder_diag import fit_channel_pca
    from wm_libero import auroc
    from sklearn.linear_model import LogisticRegression
    enc = load_world_model(DEFAULT_CHECKPOINT, "cuda").encoder
    F, rows = [], []
    for sp in ("without", "with"):
        e = np.load(DD / f"ends_{sp}_0.05.npz"); r = json.load(open(DD / f"fork_{sp}_0.05_render.json"))["rows"]
        assert len(r) == len(e["frames"])
        F.append(e["frames"]); rows += [{**x, "split": sp} for x in r]
    F = np.concatenate(F)                                     # (P, 9, 2, 256, 256, 3)
    nd = np.array([x["n_damaged"] for x in rows]); fork = (nd >= 2) & (nd <= 6)
    dmg = np.array([np.array(x["drops"]) >= 10 for x in rows])          # (P, 8) per noisy replay
    res = {"decision_points": len(rows), "real_forks": int(fork.sum())}
    for cam, name in ((0, "agentview_right"), (1, "eye_in_hand")):
        rng = np.random.default_rng(0)
        fit = F[rng.choice(len(F), min(150, len(F)), replace=False), 0, cam]
        t = dino_tokens(enc, fit, "cuda").reshape(-1, 384)
        mean, basis, zm, zs = fit_channel_pca(t[torch.as_tensor(rng.choice(len(t), 60000, replace=False), device="cuda")])
        Z = []
        for p in range(len(F)):
            tok = dino_tokens(enc, F[p, 1:, cam], "cuda")
            Z.append((((tok - mean) @ basis - zm) / zs).reshape(8, -1))
        Z = torch.stack(Z)                                     # (P, 8, 6272)
        ok, lab, sep = cf.split_test_gpu(Z)
        ok, sep = ok.cpu().numpy(), sep.cpu().numpy()
        # does the latent split separate damaged from undamaged replays at real forks? (agreement of split labels)
        agree = [max((lab[p].cpu().numpy() == dmg[p]).mean(), (lab[p].cpu().numpy() != dmg[p]).mean()) for p in np.flatnonzero(fork)]
        # damage readout ceiling: pooled end latents of single replays, train on even demos, test on odd
        X = Z.reshape(len(F) * 8, 28, 28, 8).cpu().numpy().reshape(-1, 7, 4, 7, 4, 8).mean((2, 4)).reshape(len(F) * 8, -1)
        y = dmg.reshape(-1); demo_id = np.repeat([int(x["demo"].split("_")[1]) + (0 if x["split"] == "without" else 100) for x in rows], 8)
        trn = demo_id % 2 == 0
        ro = auroc(LogisticRegression(C=0.1, max_iter=3000).fit(X[trn], y[trn]).predict_proba(X[~trn])[:, 1], y[~trn]) if y[~trn].any() else None
        res[name] = {"AUROC_separation_vs_fork": auroc(sep, fork), "split_rate_forks": float(ok[fork].mean()),
                     "split_rate_nonforks": float(ok[~fork].mean()),
                     "split_label_agreement_with_damage_at_forks_median": float(np.median(agree)) if agree else None,
                     "damage_readout_AUROC_heldout_demos": ro}
    res["gate_pass_agentview"] = bool(res["agentview_right"]["AUROC_separation_vs_fork"] >= 0.7 and
                                      res["agentview_right"]["split_rate_forks"] >= 2 * max(res["agentview_right"]["split_rate_nonforks"], 1e-9))
    (OUT / "oopsie_check2a_result.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
