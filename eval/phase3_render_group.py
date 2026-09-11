"""Render a specific diagnostic group of episodes, and audit WHY they fired.

Default group D: alarmed, margin beyond the probe's reach (>23%), and the block survived --
the genuinely spurious alarms, 16 of 100. Half sit at the 0.50 search cap, i.e. no scaling
within +-50% flips the outcome, so these are maximally SAFE actions being flagged. That makes
them the most tractable remaining fault: an alarm here is model error or clustering error, not
a label artefact.

The audit checks which: if the model's own tilt readout says the block topples when it does not,
the fault is the model's dynamics. If the readout is right but the probe lands in the wrong
basin, the fault is the attractor assignment.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src" / p) for p in ("systems", "models", "training")]
sys.path.insert(0, str(ROOT / "eval"))
import tipping_block as tb                                                # noqa: E402
from shplrnn import ShPLRNN                                               # noqa: E402
from run_phase3_monitor import ACT_DIM, OBS_DIM, T_OFF, T_ON, make_data   # noqa: E402
from phase3_pipeline import find_attractors, settle_latents               # noqa: E402
from phase3_live_demo import Monitor                                      # noqa: E402
from phase3_eval100 import render                                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--group", default="D", choices=["A", "B", "C", "D"])
ap.add_argument("--max-videos", type=int, default=16)
ap.add_argument("--every", type=int, default=15)
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--n-probe", type=int, default=32)
a = ap.parse_args()

n, SET, ALPHA = 450, 150, tb.ALPHA_DEFAULT
OUT = ROOT / "results" / "phase3" / f"group{a.group}"
OUT.mkdir(parents=True, exist_ok=True)
thr = tb.topple_threshold(n, T_ON, T_OFF)
_, _, mu, sd, amu, asd = make_data(40, thr, n, seed=0)
model = ShPLRNN(d=4, H=128, action_dim=ACT_DIM, obs_dim=OBS_DIM).double()
model.load_state_dict(torch.load(ROOT / "results/phase3/monitor_model_T450.pt"))
s0n = ((torch.zeros(1, OBS_DIM).double() - mu) / sd)
rng = np.random.default_rng(21)
off = np.stack([tb.random_push(rng, n, thr) for _ in range(600)])
E0, st_ = settle_latents(model, off, mu, sd, amu, asd, s0n, n, 400)
scale = np.linalg.norm(E0 - E0.mean(0), axis=1).mean()
C, _, _, _ = find_attractors(E0[st_ < 0.01 * scale], scale)
basis = np.linalg.svd(E0 - E0.mean(0), full_matrices=False)[2][:2]
proj = lambda X: (X - E0.mean(0)) @ basis.T                               # noqa: E731
with torch.no_grad():
    cth = (model.observe(torch.from_numpy(C).double())[:, 0] * sd[0] + mu[0]).numpy()
order = np.argsort(cth)
cname = {order[0]: "falls left", order[1]: "stays up", order[2]: "falls right"}
upright = order[1]
mon = Monitor(model, C, mu, sd, amu, asd, thr, SET, 0.10, a.n_probe, rng)

rows = json.load(open(ROOT / "results/phase3/eval100/results.json"))["rows"]
m = np.array([r["margin"] for r in rows]); f = np.array([r["fell"] for r in rows])
al = np.array([r["alarm"] for r in rows]); reach = m < 0.23
sel = {"A": al & (m >= .10) & reach & f, "B": al & (m >= .10) & reach & ~f,
       "C": al & (m >= .10) & ~reach & f, "D": al & (m >= .10) & ~reach & ~f}[a.group]
ids = np.where(sel)[0]
print(f"group {a.group}: {len(ids)} episodes -> {list(ids)}\n", flush=True)

erng = np.random.default_rng(777)
acts = np.stack([tb.random_push(erng, n, thr) for _ in range(100)])
times = np.arange(0, n, a.every)
allp, Cp = proj(E0), proj(C)
print(f"{'ep':>4}{'margin':>8}{'peak k':>8}{'@t':>6}{'model peak|th|':>16}"
      f"{'true peak|th|':>15}   diagnosis")
for i in ids[:a.max_videos]:
    act = acts[i]; traj, _ = tb.simulate(act, dt=tb.DT_DEFAULT)
    S, EE, LL, TH = [], [], [], []
    for t in times:
        s_, e_, l_, t_ = mon.score(traj[t], act[t:])
        S.append(s_); EE.append(e_); LL.append(l_); TH.append(t_)
    S = np.array(S)
    K = np.array([0 if l is None else int(a.n_probe - max((l == j).sum() for j in range(len(C))))
                  for l in LL])
    j = int(np.argmax(K))
    mth = np.nanmax([np.abs(t_).max() for t_ in TH if t_ is not None])
    tth = np.abs(traj[:, 0]).max()
    # which fault? model thinks it topples (dynamics) vs lands off-attractor (assignment)
    diag = "model predicts a topple that never happens" if mth >= ALPHA \
        else "readout stays safe -> attractor ASSIGNMENT error"
    print(f"{i:>4}{m[i]:>7.1%}{K[j]:>8}{times[j]:>6}{mth:>16.3f}{tth:>15.3f}   {diag}",
          flush=True)
    title = (f"[SPURIOUS]   marginal? no   topples? no   margin {m[i]:.1%}   |   "
             f"alarm fired, peak dissent {K[j]}/{a.n_probe} at t={times[j]}")
    render(OUT / f"ep{i:03d}_spurious.mp4", title, act, m[i], False, S, times, EE, LL, TH,
           C, proj, Cp, allp, cname, n, a.stride, 25, ALPHA)
print(f"\n-> {OUT}")
