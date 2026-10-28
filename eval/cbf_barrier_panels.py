"""In-distribution check of the fork-boundary barrier: panels 6 and 7 (normal mass, corrected ground truth), where the
world model was trained. Same two latent signals as eval/cbf_barrier.py (distance to the fork; latent change outside the
robot's own hand path), scored against real outcomes (labels only for evaluation)."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]


def main():
    import cem_filter as cf
    import cbf_barrier as cb
    from fresh_fragility_panel import topple_count
    from regrade_any_outcome import events_three_block_v2, labels
    mon = cf.Monitor(seed=20261570)
    torch = mon.torch
    PANELS = {"panel6": ROOT / "results/program_b/panel6", "panel7": ROOT / "results/program_b/panel7"}
    out = []
    for panel, d in PANELS.items():
        for n, path in enumerate(sorted((ROOT / f"results/program_b/{panel}_v2").glob("ep*.npz"))):
            with np.load(path) as t, np.load(d / path.name) as f:
                if not bool(t["at_rest"]):
                    continue
                trA, tilt = t["truth_A"], t["start_tilt"]
                frame, chunk = f["close_frame"], f["chunk"]
            af, asafe = labels(events_three_block_v2(trA)); k = topple_count(trA, tilt)
            ee0 = trA[0, 0, 57:60]
            pts = np.vstack([ee0[None], chunk[:, :3]])
            z0 = torch.as_tensor(mon.latent(frame[None])[0], device="cuda").float()
            split = []
            outside = None
            for s in cb.SCALES:
                d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                               for p, r in (mon.errors.sample(20261571 + n, 0, 0, j) for j in range(64))])
                st = mon.g(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
                keep = mon.snippets; mon.snippets = keep * s
                try:
                    e10, e30 = mon.endpoints(st, chunk[None])
                finally:
                    mon.snippets = keep
                ok10, _, _ = cf.split_test_gpu(e10); ok30, _, _ = cf.split_test_gpu(e30)
                split.append(bool((ok10 & ok30).item()))
                if s == 1.0:
                    m = torch.as_tensor(~cb.hand_mask(pts), device="cuda")
                    diff = (e30[0].reshape(64, 28, 28, 8) - z0[None]).pow(2).mean(-1)
                    outside = ((diff * m).sum((1, 2)) / m.sum().clamp_min(1)).cpu().numpy()
            hit = [i for i, v in enumerate(split) if v]
            h = float(np.log2(cb.SCALES[hit[0]])) if hit else float(np.log2(cb.SCALES[-1]))
            out.append({"panel": panel, "state": path.stem, "harm_fork": bool(2 <= k <= 62), "harm_safe": k == 0,
                        "topples": int(k), "any_fork": af, "any_safe": asafe, "h": h, "split": split,
                        "outside_mean": float(outside.mean()), "outside_q90": float(np.quantile(outside, 0.9)),
                        "outside_max": float(outside.max())})
            torch.cuda.empty_cache()
    (ROOT / "results/jenga/cbf_barrier_panels.json").write_text(json.dumps(out) + "\n")
    from sklearn.metrics import roc_auc_score
    res = {}
    for lab, pk, nk in (("neighbour topple", "harm_fork", "harm_safe"), ("any outcome", "any_fork", "any_safe")):
        sub = [r for r in out if r[pk] or r[nk]]
        y = np.array([r[pk] for r in sub])
        res[lab] = {"forks": int(y.sum()), "safe": int((~y).sum()),
                    "auc_barrier": round(roc_auc_score(y, [-r["h"] for r in sub]), 3),
                    "auc_split_at_1x": round(roc_auc_score(y, [r["split"][1] for r in sub]), 3),
                    "auc_outside_q90": round(roc_auc_score(y, [r["outside_q90"] for r in sub]), 3),
                    "auc_outside_max": round(roc_auc_score(y, [r["outside_max"] for r in sub]), 3)}
    # confident topples (all 64 futures topple): the case the split test cannot see
    conf = [r for r in out if r["topples"] >= 62]; safe = [r for r in out if r["harm_safe"]]
    if conf and safe:
        res["confident_topples_vs_safe"] = {"n_confident": len(conf),
                                            "auc_outside_q90": round(roc_auc_score([1] * len(conf) + [0] * len(safe), [r["outside_q90"] for r in conf + safe]), 3)}
    (ROOT / "results/jenga/cbf_barrier_panels_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
