"""Does a one-line action statistic shortcut the PROXIMITY question on the tipping block?

If toppling were purely 'impulse exceeds I*', then because impulse is LINEAR in the action,
scaling a by s scales I by s, so the outcome flips at s* = I*/I(a) and

    margin = |I*/I(a) - 1|

i.e. the margin computable in closed form from ONE scalar. The margin oracle scales the whole
action, the same axis impulse is linear along, so the shortcut would be near-total.

Every baseline is given the maximum unfair advantage: S* is fitted by grid search to MAXIMISE
its own AUC on the very labels it is scored against. If it still loses, the shortcut is not there.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/sanger/wksp/stable_lyap_filters")
sys.path[:0] = [str(ROOT / "src" / "systems"), str(ROOT / "eval")]
import tipping_block as tb
from run_phase3_monitor import T_ON, T_OFF

def auc(score, lab):
    lab = np.asarray(lab, bool); s = np.asarray(score, float)
    p, q = s[lab], s[~lab]
    if len(p) == 0 or len(q) == 0: return float("nan")
    x = np.concatenate([p, q])
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float); r[order] = np.arange(1, len(x) + 1)
    xs = x[order]                                   # AVERAGE ranks for ties -- k is an integer
    i = 0                                           # with many ties, so this matters
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]: j += 1
        if j > i: r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q))

n = 450
thr = tb.topple_threshold(n, T_ON, T_OFF)
erng = np.random.default_rng(777)                      # SAME seed as phase3_eval100
acts = np.stack([tb.random_push(erng, n, thr) for _ in range(100)])

res  = json.load(open(ROOT / "results/phase3/eval100/results.json"))
rows = sorted(res["rows"], key=lambda r: r["i"])
margin = np.array([r["margin"] for r in rows])
near   = np.array([r["near"] for r in rows])
k_max  = np.array([r["k_max"] for r in rows], float)
fell   = np.array([r["fell"] for r in rows])

fell_re = np.array([bool(tb.simulate(a, dt=tb.DT_DEFAULT)[1][-1]) for a in acts])
print(f"seed check: regenerated outcomes match stored: {(fell_re == fell).mean():.0%}")
assert (fell_re == fell).mean() > 0.99, "action regeneration does not match the stored run"

stats = {
    "net impulse |sum a|": np.abs(acts.sum(1)),
    "abs impulse sum|a|" : np.abs(acts).sum(1),
    "peak force max|a|"  : np.abs(acts).max(1),
    "L2 norm ||a||"      : np.linalg.norm(acts, axis=1),
}

print(f"\nground truth: {near.sum()} near (margin<0.23), {fell.sum()} topple\n")
print(f"{'statistic':<22}{'AUC outcome':>12}{'AUC prox':>10}{'corr w/ margin':>16}")
print("-" * 60)
out = {}
for name, S in stats.items():
    best = (-1, None)
    for Sstar in np.linspace(S.min() * 0.5, S.max() * 1.5, 4000):
        pm = np.abs(Sstar / S - 1.0)
        a_ = auc(-pm, near)
        if a_ > best[0]: best = (a_, Sstar)
    a_prox, Sstar = best
    pm = np.abs(Sstar / S - 1.0)
    a_out = max(auc(S, fell), auc(-S, fell))
    c = np.corrcoef(np.minimum(pm, 0.5), margin)[0, 1]
    out[name] = dict(auc_outcome=a_out, auc_prox=a_prox, corr=c, Sstar=float(Sstar))
    print(f"{name:<22}{a_out:>12.3f}{a_prox:>10.3f}{c:>16.3f}")

print("-" * 60)
m_out, m_prox = max(auc(k_max, fell), auc(-k_max, fell)), auc(k_max, near)
print(f"{'METHOD (dissent k)':<22}{m_out:>12.3f}{m_prox:>10.3f}"
      f"{np.corrcoef(k_max, margin)[0,1]:>16.3f}")
out["METHOD"] = dict(auc_outcome=m_out, auc_prox=m_prox,
                     corr=float(np.corrcoef(k_max, margin)[0, 1]))
print("\nevery baseline row had S* fitted against the labels it is scored on;")
print("the method row is fitted to nothing at all.")
json.dump(out, open(ROOT / "results/phase3/eval100/action_shortcut.json", "w"), indent=1)
