"""Regenerate every table in the README from the raw result files in results/."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def h(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def masking():
    d = load("pc1_mask_full_corpus")
    if not d:
        return
    h(f"1. Patch masking -- full corpus, {d['n_chunks']} chunks, held-out episode splits")
    print(f"{'metric':<16}{'row mask only':>16}{'+ low-norm k=30':>18}{'+ PC1 75% keep':>18}")
    print("-" * 68)
    for metric, runs in d["results"].items():
        um = np.mean([r["unmasked"] for r in runs])
        ln = np.mean([r["low_norm"] for r in runs])
        pc = np.mean([r["pc1"] for r in runs])
        print(f"{metric:<16}{um:>16.3f}{ln:>18.3f}{pc:>18.3f}")
    print(f"\n  (mean held-out AUC over {len(next(iter(d['results'].values())))} random "
          f"episode splits; hyperparameters chosen on the train half only)")


def sign():
    d = load("pc1_sign_sweep_compare")
    if not d:
        return
    h("2. Which side of PC1? -- keep-% swept per sign, held-out selection")
    print(f"{'configuration':<34}{'held-out AUC':>14}{'% chosen':>26}")
    print("-" * 74)
    for k, v in d.items():
        if "ensemble" in k:
            continue
        chosen = ", ".join(f"{p}%x{n}" for p, n in sorted(v["pct_chosen"].items(),
                                                          key=lambda x: -x[1]))
        print(f"{k:<34}{v['mean']:>14.3f}{chosen:>26}")
    print("\n  background (low-motion patches) wins decisively and picks 75% in every split;")
    print("  the foreground direction's own optimum is 'apply no PC1 filtering at all'.")


def confound():
    d = load("phase_confound_test")
    if not d:
        return
    h("3. WHY background beats foreground -- motion/phase confound")
    print(f"{'group':<34}{'corr(motion,score)':>20}{'safe only':>12}{'unsafe only':>13}")
    print("-" * 79)
    for k, v in d.items():
        cs = "n/a" if not np.isfinite(v["corr_safe"]) else f"{v['corr_safe']:+.3f}"
        print(f"{k:<34}{v['corr_all']:>+20.3f}{cs:>12}{v['corr_unsafe']:>+13.3f}")
    print("\n  Foreground divergence correlates with motion just as strongly in SAFE chunks")
    print("  as unsafe ones -> it is a motion confound from the always-moving arm, not a")
    print("  failure signal. Background patches have ~zero baseline motion, so divergence")
    print("  there actually means something.")


def probe():
    d = load("probe_vs_divergence")
    if not d:
        return
    h(f"4. Divergence vs a supervised tilt probe -- same {d['n_chunks']} chunks, "
      f"{d['n_unsafe']} unsafe")
    print(f"{'method':<34}{'AUC':>8}{'best F1':>10}{'recall@p95':>12}")
    print("-" * 64)
    for name, v in d["results"].items():
        pts = v.get("points", {})
        f1 = max((p["f1"] for p in pts.values()), default=float("nan"))
        r95 = pts.get("p95", {}).get("recall", float("nan"))
        print(f"{name:<34}{v['auc']:>8.3f}{f1:>10.3f}{r95:>12.2f}")
    print("\n  The probe is better, but it needs tilt labels from sim physics -- it is NOT")
    print("  zero-shot. Divergence is the honest zero-shot number.")


def negatives():
    h("5. Things that did NOT help (documented so nobody redoes them)")
    for name, label, ref in [
        ("combined_mask_sweep", "low-norm mask + PC1 mask together", "PC1 alone = 0.894/0.896"),
        ("temporal_agg_test", "temporal aggregation over chunk scores", "no aggregation = 0.895/0.897"),
    ]:
        d = load(name)
        if not d:
            continue
        print(f"\n  {label}")
        for metric, v in d.items():
            chosen = max(v["cfg_chosen"].items(), key=lambda x: x[1])
            print(f"    {metric:<16} held-out AUC {v['mean']:.3f}   "
                  f"selector picked {chosen[0]} in {chosen[1]}/20 splits")
        print(f"    -> no gain vs {ref}")


def main():
    masking(); sign(); confound(); probe(); negatives()
    print("\n" + "=" * 78)
    print("Headline: d_end / ftle_variance + row mask + PC1 background mask, p90 over")
    print("patches, threshold at p95 of the SAFE distribution. AUC 0.894 / 0.896.")
    print("=" * 78)


if __name__ == "__main__":
    main()
