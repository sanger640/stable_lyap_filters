"""Which forks does cell E miss, and why? (descriptive diagnostic on H, checked on fresh panel 7)

For every real fork: real topple count k/64, which neighbour topples, episode step; cell E's 64 futures projected
on the real fork direction (real hold-30 latents' first principal axis, H only) and the fraction of them beyond the
real threshold ('model topple fraction'); per-state side agreement; and, for misses, which split-test criterion
failed (no BIC preference, Ashman D <= 2, minority < 2, or partition not persistent from hold 10 to hold 30).
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
OUT = ROOT / "results/jenga/cellE_miss_analysis.json"


def criterion(e10, e30):
    from outcome_modes import same_partition, two_mode_test
    t10, t30 = two_mode_test(e10, 0.0), two_mode_test(e30, 0.0)
    ok = lambda t: t.bic_prefers_two and t.ashman_d > 2 and t.minority >= 2
    if not ok(t30) or not ok(t10):
        t = t30 if not ok(t30) else t10
        which = "hold30" if not ok(t30) else "hold10"
        if not t.bic_prefers_two:
            return f"no_two_groups_{which}", t
        if t.ashman_d <= 2:
            return f"groups_not_separated_{which}", t
        return f"minority_lt2_{which}", t
    if not same_partition(t10.labels, t30.labels):
        return "not_persistent", t30
    return "alarm", t30


def which_neighbour(traces, tilt):
    from fresh_fragility_panel import TOPPLE_DEG, neighbour_tilts
    hits = np.array([neighbour_tilts(t[-1]) for t in traces]) >= TOPPLE_DEG
    hits &= (np.asarray(tilt) < TOPPLE_DEG)[None]
    return {"left": int(hits[:, 0].sum()), "right": int(hits[:, 1].sum())}


def main():
    import torch
    import phase1_models as p1
    import step3_world_model as wm
    import step3b_models as sm
    from fresh_fragility_panel import topple_count
    gop = sm.G.load(sm.DATA / "G.pt")
    model = wm.WorldModel.load(p1.OUT / "model_E.pt")
    rows = []
    for path in sorted((p1.OUT / "h").glob("ep*.npz")):
        with np.load(path) as f:
            k = topple_count(f["traces"], f["start_tilt"])
            if not 2 <= k <= 62:
                continue
            d = np.concatenate([f["d_pos"].reshape(64, 9), f["d_rot"].reshape(64, 9)], 1)
            z0 = torch.as_tensor(f["z0"], device=wm.DEVICE).float()
            e10, e30 = sm.roll(model, gop(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d).float()), f["actions"])
            real = f["z"][:, 1].reshape(64, -1).astype(np.float64)
            from fresh_fragility_panel import neighbour_tilts, TOPPLE_DEG
            top = np.array([np.any((np.array(neighbour_tilts(t[-1])) >= TOPPLE_DEG) & (f["start_tilt"] < TOPPLE_DEG))
                            for t in f["traces"]])
            mean = real.mean(0)
            u = np.linalg.svd(real - mean, full_matrices=False)[2][0]
            pr, pm = (real - mean) @ u, (e30 - mean) @ u
            if pr[top].mean() < pr[~top].mean():
                pr, pm = -pr, -pm
            thr = (pr[top].mean() + pr[~top].mean()) / 2
            verdict, t = criterion(e10, e30)
            rows.append({"state": path.stem, "step": int(path.stem.split("_s")[-1]), "real_topples": int(k),
                         "neighbour": which_neighbour(f["traces"], f["start_tilt"]),
                         "caught": verdict == "alarm", "verdict": verdict,
                         "model_topple_fraction": float(np.mean(pm > thr)),
                         "side_agreement": float(np.mean((pr > thr) == (pm > thr))),
                         "model_spread_on_axis_vs_real": float(pm.std() / max(pr.std(), 1e-9)),
                         "model_minority": int(t.minority), "model_ashman_d": None if not np.isfinite(t.ashman_d) else float(t.ashman_d)})
        torch.cuda.empty_cache()
    caught = [r for r in rows if r["caught"]]
    missed = [r for r in rows if not r["caught"]]

    def bins(group):
        edges = [(2, 8), (9, 24), (25, 39), (40, 55), (56, 62)]
        return {f"{a}-{b}": f"{sum(a <= r['real_topples'] <= b and r['caught'] for r in group)}/"
                            f"{sum(a <= r['real_topples'] <= b for r in group)}" for a, b in edges}
    verdicts = {}
    for r in missed:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    shift = {"model_all_stay(<=2/64 beyond real threshold)": sum(r["model_topple_fraction"] <= 2 / 64 for r in missed),
             "model_all_topple(>=62/64)": sum(r["model_topple_fraction"] >= 62 / 64 for r in missed),
             "model_mixed": sum(2 / 64 < r["model_topple_fraction"] < 62 / 64 for r in missed)}
    # fresh panel 7 (rows already graded there): catch rate by real topple count
    p7 = json.loads((ROOT / "results/jenga/wm_readout_fix_panel7_result.json").read_text())["rows"]
    screen = json.loads((ROOT / "results/program_b/panel7/screen.json").read_text())
    p7rows = []
    for r in p7:
        if not r["fork"]:
            continue
        e, i = r["state"].split(":")
        with np.load(ROOT / f"results/program_b/panel7/ep{e}_s{i}.npz") as f:
            k = topple_count(f["truth_A"], np.array(screen[e][i]["start_tilt"]))
        p7rows.append({"real_topples": int(k), "caught": r["R0"]})
    result = {"note": __doc__.strip().splitlines()[0], "H_forks": len(rows), "H_caught": len(caught),
              "H_catch_by_real_topples": bins(rows), "panel7_catch_by_real_topples": bins(p7rows),
              "H_missed_verdicts": verdicts, "H_missed_model_side": shift,
              "median_model_topple_fraction": {"caught": float(np.median([r["model_topple_fraction"] for r in caught])),
                                               "missed": float(np.median([r["model_topple_fraction"] for r in missed]))},
              "median_side_agreement": {"caught": float(np.median([r["side_agreement"] for r in caught])),
                                        "missed": float(np.median([r["side_agreement"] for r in missed]))},
              "median_axis_spread_ratio": {"caught": float(np.median([r["model_spread_on_axis_vs_real"] for r in caught])),
                                           "missed": float(np.median([r["model_spread_on_axis_vs_real"] for r in missed]))},
              "neighbour": {"caught_left_dominant": sum(r["neighbour"]["left"] > r["neighbour"]["right"] for r in caught),
                            "caught_right_dominant": sum(r["neighbour"]["left"] < r["neighbour"]["right"] for r in caught),
                            "missed_left_dominant": sum(r["neighbour"]["left"] > r["neighbour"]["right"] for r in missed),
                            "missed_right_dominant": sum(r["neighbour"]["left"] < r["neighbour"]["right"] for r in missed)},
              "median_step": {"caught": float(np.median([r["step"] for r in caught])),
                              "missed": float(np.median([r["step"] for r in missed]))},
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
