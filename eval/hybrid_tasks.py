"""Amendment 13: per-task hybrids. HYB_T1 (three-block) and HYB_T2 (push past tall objects), amendment-10 recipe.

Stages: t1 | t2  (each trains if needed, then grades on its task's selection set)
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_hybrid as hy  # noqa: E402
import phase2_models as pm  # noqa: E402

T1DIR = ROOT / "results/program_b/t1_hybrid"
T2DIR = ROOT / "results/program_b/t2"


def t1():
    import torch
    import phase1_models as p1
    import phase1b_models as p1b
    import phase1d_models as p1d
    import step3_world_model as wm
    from fresh_fragility_panel import topple_count
    from outcome_modes import two_mode_test
    from phase2_amend4 import alarm, score
    T1DIR.mkdir(parents=True, exist_ok=True)
    src = ROOT / "results/program_b/phase1e"
    if not (T1DIR / "modes.json").exists():
        (T1DIR / "modes.json").write_text((src / "modes.json").read_text())
    if not (T1DIR / "model_US.pt").exists():
        (T1DIR / "model_US.pt").symlink_to(src / "model_US.pt")
    names = [str(p) for f in p1b.FOLDERS_ENRICHED for p in sorted(Path(f).glob("ep*.npz")) if not p.name.endswith(".tmp.npz")]
    pm.load_train = lambda: (p1d.load(p1b.FOLDERS_ENRICHED, proprio=False), names)    # three-block data, Phase-1e order
    hy.OUT = T1DIR
    hy.train(name="HYB_T1", det_name="US")
    s = torch.load(T1DIR / "model_HYB_T1.pt", map_location="cuda")
    head = hy.build_head(s["action_dim"]).cuda(); head.load_state_dict(s["state_dict"]); head.eval()
    det = wm.WorldModel.load(T1DIR / "model_US.pt"); det.net.eval()
    fork, ends, errs = [], [], []
    for n, path in enumerate(sorted((p1.OUT / "h").glob("ep*.npz"))):
        with np.load(path) as f:
            k = topple_count(f["traces"], f["start_tilt"])
            z0 = torch.as_tensor(f["z0"], device="cuda").float()[None].expand(64, -1, -1, -1).contiguous()
            act = det._act(torch.as_tensor(p1d.augment(f["actions"], None, False), device="cuda"))
            real = f["z"][:, 1].reshape(64, -1).astype(np.float64)
        g = torch.Generator(device="cuda").manual_seed(hy.MONITOR_SEED + n)
        fut = hy.sample(head, det, z0, act, s["residual_scale"], g)
        e10 = fut[:, 0].reshape(64, -1).double().cpu().numpy(); e30 = fut[:, 1].reshape(64, -1).double().cpu().numpy()
        fork.append(2 <= k <= 62); ends.append((e10, e30))
        errs.extend(np.linalg.norm(e30 - real, axis=1) / np.sqrt(real.shape[1] // 3))
        torch.cuda.empty_cache()
    fork = np.array(fork); tau = float(np.mean(errs))
    out = {"model": "HYB_T1", "tau_H": tau}
    for r in ("BOTH", "H30", "H30_floor"):
        out[r] = score(np.array([alarm(a, b, r, tau) for a, b in ends]), fork)
    out["reference_US"] = {"BOTH": 0.715, "H30_floor_with_validation_floor": 0.808}
    out["helps"] = bool(out["BOTH"]["score"] >= 0.715 + 0.05)
    (ROOT / "results/jenga/hybrid_T1_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


def t2():
    pm.OUT = T2DIR; hy.OUT = T2DIR
    hy.train(name="HYB_T2", det_name="US2_long")
    hy.grade("HYB_T2", ROOT / "results/jenga/hybrid_T2_result.json")


if __name__ == "__main__":
    {"t1": t1, "t2": t2}[sys.argv[1]]()
