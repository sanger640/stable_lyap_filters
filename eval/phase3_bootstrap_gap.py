"""Paired bootstrap on the AUC gap: is METHOD > best action statistic, or is n=100 too small?"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/sanger/wksp/stable_lyap_filters")
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval")]
import tipping_block as tb
from run_phase3_monitor import T_ON, T_OFF
sys.path.insert(0, str(Path(__file__).parent))
from impulse_check import auc          # reuse the average-rank AUC

n = 450
thr = tb.topple_threshold(n, T_ON, T_OFF)
erng = np.random.default_rng(777)
acts = np.stack([tb.random_push(erng, n, thr) for _ in range(100)])
rows = sorted(json.load(open(ROOT / "results/phase3/eval100/results.json"))["rows"],
              key=lambda r: r["i"])
near  = np.array([r["near"] for r in rows]); k_max = np.array([r["k_max"] for r in rows], float)

L2 = np.linalg.norm(acts, axis=1)
Sstar = json.load(open(ROOT / "results/phase3/eval100/action_shortcut.json"))["L2 norm ||a||"]["Sstar"]
base = -np.abs(Sstar / L2 - 1.0)

rng = np.random.default_rng(0)
d, am, ab = [], [], []
for _ in range(4000):
    idx = rng.integers(0, 100, 100)
    if near[idx].sum() in (0, 100): continue
    a1, a2 = auc(k_max[idx], near[idx]), auc(base[idx], near[idx])
    am.append(a1); ab.append(a2); d.append(a1 - a2)
d, am, ab = np.array(d), np.array(am), np.array(ab)
print(f"METHOD    AUC {np.mean(am):.3f}  95% CI [{np.percentile(am,2.5):.3f}, {np.percentile(am,97.5):.3f}]")
print(f"L2 (fitted) AUC {np.mean(ab):.3f}  95% CI [{np.percentile(ab,2.5):.3f}, {np.percentile(ab,97.5):.3f}]")
print(f"\ngap       {np.mean(d):+.3f}  95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")
print(f"P(method > fitted baseline) = {(d > 0).mean():.2f}")
