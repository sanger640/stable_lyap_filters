"""Retrain the plain world model (US recipe, unchanged) on the clean three-block data (threeblock_v3) PLUS the policy's
closed-loop states (closedloop_v1/train), then compare old vs new world model on held-out validation episodes
(dev episodes with e % 5 == 0) of the look-ahead check: imagined split / barrier vs real 15-step outcomes.

stages: modes | us | eval
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
V3 = ROOT / "results/program_b/threeblock_v3"
CL = ROOT / "results/program_b/closedloop_v1"
OUT = CL / "model"


def _patch():
    import phase1b_models as p1b
    import step3b_models as sm
    sm.DATA = V3                                      # G.pt and validation data stay the clean three-block ones
    sel = json.loads((CL / "selection.json").read_text())
    keep = {f"ep{k.split(':')[0]}_s{k.split(':')[1]}.npz" for k, v in sel.items() if v[0] == "train"}
    sub = CL / "train_selected"
    sub.mkdir(exist_ok=True)
    for p in (CL / "train").glob("ep*.npz"):          # only states in the shared selection (symlinks)
        if p.name in keep and not (sub / p.name).exists():
            (sub / p.name).symlink_to(p)
            st = p.with_name(p.stem + ".start.npy")
            (sub / st.name).symlink_to(st)
    p1 = V3 / "p1train"
    if os.environ.get("CL_DROP_T8") == "1":            # memory (60 GB machine): keep the fork-rich phase-1 states only
        roles = json.loads((V3 / "selection_phase1.json").read_text())["roles"]
        t16 = {f"ep{ch.split(':')[0]}_seed{ch.split(':')[1]}_s{i}.npz" for ch, sel_ in roles.items() for i, r in sel_.items() if r == "t16"}
        p1 = V3 / "p1train_t16"; p1.mkdir(exist_ok=True)
        for q in (V3 / "p1train").glob("ep*.npz"):
            if q.name in t16 and not (p1 / q.name).exists():
                (p1 / q.name).symlink_to(q); st = q.with_name(q.stem + ".start.npy"); (p1 / st.name).symlink_to(st)
    p1b.FOLDERS_ENRICHED = [V3 / "train", p1, sub]


def _names_states():
    import phase1b_models as p1b
    import phase1d_models as p1d
    names = [str(p) for f in p1b.FOLDERS_ENRICHED for p in sorted(Path(f).glob("ep*.npz")) if not p.name.endswith(".tmp.npz")]
    states = p1d.load(p1b.FOLDERS_ENRICHED, proprio=False)
    return names, states


def stage_modes():
    _patch()
    import phase1e_models as p1e
    OUT.mkdir(parents=True, exist_ok=True)
    names, states = _names_states()
    (OUT / "modes.json").write_text(json.dumps(p1e.detect_modes(states, names)) + "\n")
    print("modes done", len(states), flush=True)


def stage_us():
    _patch()
    import fast_train as ft
    import phase1d_models as p1d
    names, states = _names_states()
    modes = json.loads((OUT / "modes.json").read_text())
    rare = np.array([k in set(modes[n]) for n, s in zip(names, states) for k in range(len(s["z"]))])
    has_rare = np.array([bool(modes[n]) for n, s in zip(names, states) for _ in range(len(s["z"]))])
    pools = (np.flatnonzero(rare), np.flatnonzero(~rare & has_rare))
    sampler = lambda rng, b: np.concatenate([rng.choice(pools[0], b // 2), rng.choice(pools[1], b - b // 2)])
    model = ft.train("US", OUT, states, sampler)
    tau, n_val = p1d.floor(model, False)
    (OUT / "floor_US.json").write_text(json.dumps({"tau": tau, "validation_futures": n_val}) + "\n")
    print("US done, floor", tau, flush=True)


def stage_eval():
    """Old (threeblock_v3) vs new world model on validation-episode decisions of the look-ahead check."""
    import tempfile
    import cem_filter as cf
    import cbf_barrier as cb
    import lookahead_check as la
    import step3_world_model as wm
    import uncertainty_monitor_dev as um
    from jenga_short_held_tails import DirectJengaSim, extract_sim
    from sklearn.metrics import roc_auc_score
    data = json.loads(la.OUT.read_text())
    T = data["truth"]
    keys = [k for k in T if int(k.split(":")[0]) % 5 == 0]
    sim = DirectJengaSim(str(extract_sim(um.SIM_ARCHIVE, tempfile.mkdtemp(prefix="cle_"))))
    res = {"decisions": len(keys)}
    for name, mpath in (("old", V3 / "model_US.pt"), ("new", OUT / "model_US.pt")):
        mon = cf.Monitor(seed=20261610); mon.model = wm.WorldModel.load(mpath); mon.model.net.eval()
        torch = mon.torch
        out = {}
        for k in keys:
            e, c = map(int, k.split(":"))
            S, i = la.load_state((e, c))
            sim.restore((S["state"][i], S["ctrl"][i], S["target"][i], float(S["gripper"][i])))
            z0 = torch.as_tensor(mon.latent(mon.frame(sim)[None])[0], device="cuda").float()
            row = {}
            for hz in ("H8", "H15"):
                split = []
                for s in cb.SCALES:
                    d_ = np.stack([np.concatenate([(p * [1, 1, 0] * s).reshape(9), (r * [0, 0, 1] * s).reshape(9)])
                                   for p, r in (mon.errors.sample(la.SEED + 2, e, c, j) for j in range(64))])
                    st = mon.g(z0[None].expand(64, -1, -1, -1), torch.as_tensor(d_, dtype=torch.float32))
                    acts = np.stack([la.horizon_actions(S["plan16"][i], hz, la.noise_for(mon.snippets, j) * s) for j in range(64)])
                    pairs = torch.as_tensor(acts, device="cuda").reshape(64, wm.STEPS, -1)
                    z = st.float(); ends = {}
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for t in range(wm.STEPS):
                            z = mon.model.step(z, pairs[:, t]).float()
                            if t in (wm.HOLD10, wm.HOLD30):
                                ends[t] = z.reshape(1, 64, -1)
                    ok10, _, _ = cf.split_test_gpu(ends[wm.HOLD10]); ok30, _, _ = cf.split_test_gpu(ends[wm.HOLD30])
                    split.append(bool((ok10 & ok30).item()))
                hit = [q for q, v in enumerate(split) if v]
                row[hz] = {"h": float(np.log2(cb.SCALES[hit[0]])) if hit else float(np.log2(cb.SCALES[-1])), "split1": split[1]}
            out[k] = row
            torch.cuda.empty_cache()
        r = {}
        for hz in ("H8", "H15"):
            y = np.array([T[k][hz]["topples"] / 64 >= 0.03 for k in keys])
            fk = [k for k in keys if 0.1 <= T[k][hz]["topples"] / 64 <= 0.9]; sk = [k for k in keys if T[k][hz]["topples"] == 0]
            r[hz] = {"risky": int(y.sum()),
                     "auc_barrier": round(roc_auc_score(y, [-out[k][hz]["h"] for k in keys]), 3) if 0 < y.sum() < len(y) else None,
                     "split_caught_forks": round(np.mean([out[k][hz]["split1"] for k in fk]), 3) if fk else None,
                     "split_false_on_safe": round(np.mean([out[k][hz]["split1"] for k in sk]), 3) if sk else None}
        res[name] = r
    (ROOT / "results/jenga/closedloop_train_eval.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    {"modes": stage_modes, "us": stage_us, "eval": stage_eval}[sys.argv[1]]()
