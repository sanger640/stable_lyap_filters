"""KNOCK stage 2 / 3 / 4 analysis (results/jenga/knock_protocol.json stage2, stage3, stage4): per-arm rates, paired
differences vs the policy (paired bootstrap), H_block and H_frag (stage 3: on the task-trained values; arms stopped at their
gate are reported as not run); stage 4: H_nsi_safe / H_nsi_red for the irreversible-fragility arm and refusal targeting.
usage: python3 eval/knock_analysis.py [stage3 | stage4]"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results/program_b/policy_rollouts"
STAGES = {"stage2": {"policy": "paired_record_knock2", "ours90": "paired_lfc000_knock2_o90", "frag": "paired_lfc000_knock2_frag"},
          "stage3": {"policy": "paired_record_knock2", "z_o90": "paired_lfc000_knock3_zo90", "z_frag": "paired_lfc000_knock3_zfrag",
                     "k_o90": "paired_lfc000_knock3_ko90", "k_frag": "paired_lfc000_knock3_kfrag"},
          "stage4": {"policy": "paired_record_knock2", "z_o90": "paired_lfc000_knock3_zo90", "z_fx": "paired_lfc000_knock4_zfx"}}
HYP = {"stage2": ("ours90", "frag"), "stage3": ("k_o90", "k_frag")}          # (H_block arm, H_frag arm)
EPS = range(45000, 45200)


def load(tag):
    out = {}
    for f in (RUNS / tag).glob("ep*.json"):
        r = json.loads(f.read_text()); out[int(r["episode"])] = r
    return out


def boot(x, y, seed=0, n=10_000):
    rng = np.random.default_rng(seed); d = x.astype(float) - y.astype(float); idx = rng.integers(len(d), size=(n, len(d)))
    m = d[idx].mean(1); return [round(100 * float(np.quantile(m, q)), 1) for q in (0.025, 0.975)]


def mcnemar_less(a, b):
    """one-sided exact p that arm a has fewer events than b (paired binary)"""
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum()); n = n01 + n10
    return sum(comb(n, k) for k in range(0, n01 + 1)) / 2 ** n if n else 1.0, n01, n10


def main(stage="stage2"):
    data = {k: load(v) for k, v in STAGES[stage].items()}
    data = {k: d for k, d in data.items() if len(d) >= len(EPS)}            # stage 3: arms stopped at their gate have no episodes
    eps = [e for e in EPS if all(e in d for d in data.values())]
    col = lambda arm, key: np.array([bool(data[arm][e][key]) for e in eps])
    out = {"protocol": f"results/jenga/knock_protocol.json {stage}", "paired_episodes": len(eps), "arms": {},
           "not_run": [k for k in STAGES[stage] if k not in data]}
    for arm in data:
        log = [data[arm][e].get("log", []) for e in eps]
        out["arms"][arm] = {k: int(col(arm, k).sum()) for k in ("success", "red_down", "neighbour_toppled")} | {
            "edits_per_episode": round(float(np.mean([sum(bool(c.get("intervened")) for c in l) for l in log])), 2),
            "chunk0_edited": int(sum(bool(l and l[0].get("intervened")) for l in log))}
    diff = lambda a, b, k: {"diff": round(100 * (col(a, k).mean() - col(b, k).mean()), 1), "ci95": boot(col(a, k), col(b, k))}
    for arm in data:
        if arm != "policy":
            out[f"{arm}_minus_policy_points"] = {k: diff(arm, "policy", k) for k in ("success", "red_down", "neighbour_toppled")}
    for arm in data:
        if arm != "policy":
            log = [c for e in eps for c in data[arm][e].get("log", []) if c.get("intervened")]
            out["arms"][arm]["edited_chunks_over_15mm_share"] = round(float(np.mean([c.get("lsf_edit_mm", 0) > 15 for c in log])), 3) if log else 0.0
    if stage == "stage4":
        return stage4(out, data, eps, col, diff)
    hb, hf = HYP[stage]
    if stage == "stage2":
        out["frag_minus_ours90_success"] = diff("frag", "ours90", "success")
    else:                                                                     # task-trained vs zero-shot, same world model
        for k_, z_ in (("k_o90", "z_o90"), ("k_frag", "z_frag")):
            if k_ in data and z_ in data:
                out[f"{k_}_minus_{z_}_points"] = {k: diff(k_, z_, k) for k in ("success", "red_down", "neighbour_toppled")}
    out["H_block"] = bool(out[f"{hb}_minus_policy_points"]["red_down"]["diff"] <= -10) if hb in data else "not run (gate)"
    if hf in data:
        p, a, b = mcnemar_less(col(hf, "neighbour_toppled"), col("policy", "neighbour_toppled"))
        out["H_frag"] = {"pass": bool(p < 0.05 and out[f"{hf}_minus_policy_points"]["red_down"]["diff"] > -10),
                         "mcnemar_p_one_sided": round(p, 5), "frag_only_topples": a, "policy_only_topples": b}
    else:
        out["H_frag"] = "not run (gate)"
    (ROOT / f"results/knock/{stage}_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def stage4(out, data, eps, col, diff):
    from scipy.stats import fisher_exact
    if "z_fx" not in data:
        out["H_nsi"] = "not run"
    else:
        out["z_fx_minus_z_o90_points"] = {k: diff("z_fx", "z_o90", k) for k in ("success", "red_down", "neighbour_toppled")}
        p_s, a_s, b_s = mcnemar_less(col("z_fx", "neighbour_toppled"), col("policy", "neighbour_toppled"))
        p_r, a_r, b_r = mcnemar_less(col("z_o90", "red_down"), col("z_fx", "red_down"))
        out["H_nsi_safe"] = {"pass": bool(p_s < 0.05), "mcnemar_p_one_sided": round(p_s, 6), "z_fx_only_topples": a_s, "policy_only_topples": b_s}
        out["H_nsi_red"] = {"pass": bool(p_r < 0.05), "mcnemar_p_one_sided": round(p_r, 6), "z_o90_only_red_down": a_r, "z_fx_only_red_down": b_r}
        out["H_nsi"] = bool(out["H_nsi_safe"]["pass"] and out["H_nsi_red"]["pass"])
    pt = col("policy", "neighbour_toppled")
    for arm in [a for a in ("z_o90", "z_fx") if a in data]:
        red = col(arm, "red_down"); ref = ~red
        tab = [[int((ref & pt).sum()), int((ref & ~pt).sum())], [int((red & pt).sum()), int((red & ~pt).sum())]]
        out[f"{arm}_refusal_targeting"] = {"refused": int(ref.sum()), "policy_topple_share_refused": round(float(pt[ref].mean()), 3) if ref.any() else None,
                                           "policy_topple_share_allowed": round(float(pt[red].mean()), 3) if red.any() else None,
                                           "fisher_p_two_sided": round(float(fisher_exact(tab)[1]), 4)}
        out[f"{arm}_on_policy_topple_episodes"] = {"clean_knockdown": int((pt & col(arm, "success")).sum()), "refused": int((pt & ref).sum()),
                                                   "toppled": int((pt & col(arm, "neighbour_toppled")).sum())}
    (ROOT / "results/knock/stage4_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
