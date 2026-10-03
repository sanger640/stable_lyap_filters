"""Forks caught / false alarms on the 60 held-out D0 states against the physical fork label (2-62 of 64 real
futures topple a neighbour): the split test on real latent endings, and the iteration-2 pipeline (G starts,
and exactly rendered starts) with its full deterministic dynamics. Descriptive; reproduces P1's decisions."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]


def rates(decisions, label):
    d, y = np.asarray(decisions, bool), np.asarray(label, bool)
    return {"caught": float((d & y).sum() / y.sum()), "false_alarm": float((d & ~y).sum() / (~y).sum()),
            "alarms": int(d.sum()), "forks": int(y.sum()), "states": len(y)}


def main():
    import step3b_models as sm
    import step3_world_model as wm
    from counterfactual_monitor import persistent_mode_alarm
    from fresh_fragility_panel import topple_count
    d0 = sm.load("d0")
    g = sm.G.load(sm.DATA / "G.pt")
    full = wm.WorldModel.load(sm.DATA / "dyn_full.pt")
    label, latent, with_g, with_rendered = [], [], [], []
    for n, path in enumerate(sorted((sm.DATA / "d0").glob("ep*.npz"))):
        with np.load(path) as f:
            label.append(2 <= topple_count(f["traces"], f["start_tilt"]) <= 62)
        real = d0["z"][n].double().numpy()
        latent.append(bool(persistent_mode_alarm(real[:, 0].reshape(64, -1), real[:, 1].reshape(64, -1)).alarm))
        z0 = d0["z0"][n].to(sm.DEVICE).float()
        starts_g = g(z0[None].expand(64, -1, -1, -1), d0["d"][n])
        rendered = d0["z0k"][n].to(sm.DEVICE).float()
        acts = d0["actions"][n].numpy()
        with_g.append(bool(persistent_mode_alarm(*sm.roll(full, starts_g, acts)).alarm))
        with_rendered.append(bool(persistent_mode_alarm(*sm.roll(full, rendered, acts)).alarm))
    out = {"latent_split_test": rates(latent, label), "iter2_G_starts": rates(with_g, label),
           "iter2_rendered_starts": rates(with_rendered, label)}
    p1 = json.loads((ROOT / "results/jenga/step3b_p1_result.json").read_text())["summary"]
    assert sum(with_g) == p1["full_G"]["alarms"] and sum(with_rendered) == p1["full_oracle"]["alarms"], "P1 not reproduced"
    (ROOT / "results/jenga/d0_fork_rates.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
