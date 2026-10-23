"""G-grip analysis as pre-registered (results/jenga/grip_protocol.json): G-CEM + gripper-timing knob vs G-CEM, paired per
decision on benchmark v2 real risk (neighbour topple, red block fallen).
usage: python3 eval/grip_report.py <dev|holdout|relpass2>   (after opt3.py run/truth with OPT3_GRIP=shift, tag grip)
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval")]
import bench2 as b2  # noqa: E402


def grip_event(p16, g0):
    """First plan step (1..15) where the effective gripper state changes; None if it doesn't (state 0 = closed)."""
    st = s0 = 1 if g0 < 50 else 0
    for t in range(1, 16):
        st = 1 if p16[t, 3] > 0.9 else 0 if p16[t, 3] < -0.9 else st
        if st != s0:
            return t
    return None


def main(name):
    C, _ = b2.load(name)
    G = json.loads((b2.OUT / f"opt3_{name}_g_cem.json").read_text())["rows"]
    S = json.loads((b2.OUT / f"opt3_{name}_grip.json").read_text())["rows"]; n = len(S)
    tr = lambda rows: np.array(b2.truth_chunks(name, [(r["i"], np.asarray(r["chunk"], np.float32)) for r in rows]))
    g, s = tr(G[:n]), tr(S)
    ev = np.array([grip_event(C["plan16"][i], C["gripper"][i]) is not None for i in range(n)])
    d = np.array([0 if r["theta"] is None else round(r["theta"][4]) for r in S])
    rng = np.random.default_rng(20261110)
    out = {"set": name, "decisions": n, "plan_has_gripper_event": int(ev.sum()), "d_nonzero_share": round(float((d != 0).mean()), 3),
           "d_hist": {int(v): int(c) for v, c in zip(*np.unique(d, return_counts=True))}}
    for col, lab in ((0, "neighbour"), (1, "red_fall")):
        for sub, m in (("all", np.ones(n, bool)), ("gripper_event", ev), ("no_event", ~ev)):
            x = s[m, col] - g[m, col]
            if not len(x):
                continue
            bs = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(10000)])
            out[f"{lab}_{sub}"] = {"g_cem": round(float(g[m, col].mean()), 4), "g_grip": round(float(s[m, col].mean()), 4),
                                   "diff_pp": round(100 * float(x.mean()), 3), "upper95_pp": round(100 * float(np.percentile(bs, 95)), 3),
                                   "better": int((x < 0).sum()), "worse": int((x > 0).sum())}
    if name == "holdout":
        out["PRIMARY_pass"] = bool(out["neighbour_all"]["upper95_pp"] <= 0.25 and out["red_fall_all"]["upper95_pp"] <= 1.0)
    (ROOT / f"results/jenga/grip_{name}_result.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
