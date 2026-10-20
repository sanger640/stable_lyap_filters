"""SafeManip check 2a (docs/BENCHMARK_DD.md, spec fixed before running): are real drop / tip / knock forks visible as
a split of the REAL futures' end latents? Same scoring as eval/oopsie_check2a.py.

Inputs: ~/wksp/safemanip/dd_check2a{,_s12}/<task>.json (labels) and <task>_s<seed>_e<ep>_t<t>.npy (9 replays x 2
cameras x 256 x 256 x 3: 0 = robot0_agentview_left (gate), 1 = robot0_eye_in_hand). Saved points: all with any bad
replay + a random 20% of the others.
usage: python3 eval/safemanip_check2a.py
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src"), str(ROOT / "libero")]
DIRS = {11: Path.home() / "wksp/safemanip/dd_check2a", 12: Path.home() / "wksp/safemanip/dd_check2a_s12"}
OUT = ROOT / "results/benchmark_dd"


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
            j = json.loads(jf.read_text())
            for r in j["rows"]:
                f = d / f"{r['task']}_s{seed}_e{r['episode']}_t{r['t']}.npy"
                if f.exists():
                    pts.append((f, r))
    fork = np.array([2 <= r["n_bad"] <= 6 for _, r in pts]); anybad = np.array([r["n_bad"] > 0 for _, r in pts])
    res = {"saved_points": len(pts), "real_forks": int(fork.sum()), "points_with_any_bad": int(anybad.sum())}
    for cam, name in ((0, "agentview_left"), (1, "eye_in_hand")):
        rng = np.random.default_rng(0)
        fit = np.stack([np.load(pts[i][0])[0, cam] for i in rng.choice(len(pts), min(150, len(pts)), replace=False)])
        t = dino_tokens(enc, fit, "cuda").reshape(-1, 384)
        mean, basis, zm, zs = fit_channel_pca(t[torch.as_tensor(rng.choice(len(t), min(60000, len(t)), replace=False), device="cuda")])
        Z = []
        for f, _ in pts:
            tok = dino_tokens(enc, np.load(f)[1:, cam], "cuda")
            Z.append((((tok - mean) @ basis - zm) / zs).reshape(8, -1))
        ok, lab, sep = cf.split_test_gpu(torch.stack(Z))
        ok, sep = ok.cpu().numpy(), sep.cpu().numpy()
        res[name] = {"AUROC_separation_vs_fork": auroc(sep, fork), "split_rate_forks": float(ok[fork].mean()),
                     "split_rate_nonforks": float(ok[~fork].mean()),
                     "AUROC_separation_vs_fork_among_points_with_any_bad": auroc(sep[anybad], fork[anybad]) if 0 < fork[anybad].sum() < anybad.sum() else None}
    a = res["agentview_left"]
    res["gate_pass_agentview_left"] = bool(a["AUROC_separation_vs_fork"] >= 0.7 and a["split_rate_forks"] >= 2 * max(a["split_rate_nonforks"], 1e-9))
    (OUT / "safemanip_check2a_result.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
