"""Label-free O_MAX for the CBF filter with a given world model (CBF_WM): 99th percentile of the largest outside-hand
latent change over the policy's own 15-step plans on the 300 random development decisions of the look-ahead check."""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]


def main():
    import cbf_filter as cbff
    import cem_filter as cf
    import lookahead_check as la
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    data = json.loads(la.OUT.read_text())
    mon = cf.Monitor(seed=20261620); cbf = cbff.CBF(mon); torch = mon.torch
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="cal_"))))
    o = []
    for e, c in data["background"]:
        S, i = la.load_state((e, c))
        sim.restore((S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i])))
        z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
        _, oo = cbf.score(sim, z0, S["plan16"][i][None], 64)
        o.append(float(oo[0])); torch.cuda.empty_cache()
    q = float(np.quantile(o, 0.99))
    out = {"O_MAX": q, "world_model": os.environ.get("CBF_WM", "threeblock_v3"), "n": len(o),
           "source": "99th percentile of o over the policy's own H15 plans, 300 random development decisions (no labels)"}
    (ROOT / os.environ.get("CBF_CALIB_OUT", "results/jenga/cbf_calibration.json")).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
