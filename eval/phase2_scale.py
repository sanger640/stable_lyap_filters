"""Phase 2 amendment 11: 3x training-data scale-up (US2 recipe, xlong budget) -> US2_x3data.

Frozen protocol: results/jenga/phase2_generality_protocol.json (amendment_11).
  data   (3090 machine) screen scene seeds 1500-6499 per family, select 4,800 new states per family, render/encode
  pack   (5080 machine) write all training futures (old + new) into disk memmaps
  train  (5080 machine) US2 recipe, 67,500 steps, reading batches from the memmaps
  grade  H2, BOTH readout, beside US2_xlong
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase2_data as pdata  # noqa: E402
import phase2_models as pm  # noqa: E402

OUT = pm.OUT
XDIR = OUT / "x"                              # new data: XDIR/screen, XDIR/train
PACK = OUT / "pack_x3"
NEW_SEEDS = range(1500, 6500)
PER_FAMILY, CAP = 4_800, 2_400
SELECT_SEED = 20261260
RESULT = ROOT / "results/jenga/phase2_scale_result.json"
STEPS = 67_500


def _patch():
    pdata.OUT = XDIR


def screen_job(job):
    _patch(); return pdata.screen_job(job)


def render_job(job):
    _patch(); return pdata.render_job(job)


def data(screen_workers=16, render_workers=5):
    _patch()
    for d in ("screen", "train"):
        (XDIR / d).mkdir(parents=True, exist_ok=True)
    screens = pdata.pool_run(screen_job, [(f, s) for f in ("F1", "F2") for s in NEW_SEEDS], screen_workers, "screen")
    pdata.SEEDS = dict(pdata.SEEDS, select=SELECT_SEED)
    pdata.PER_FAMILY, pdata.DISAGREE_CAP = PER_FAMILY, CAP
    roles, summary = pdata.select(screens)
    (XDIR / "selection.json").write_text(json.dumps({"summary": summary, "roles": {f"{f}:{s}": v for (f, s), v in roles.items()}}) + "\n")
    print(json.dumps(summary), flush=True)
    jobs = [(f, s, r) for (f, s), r in sorted(roles.items())]
    pdata.pool_run(render_job, jobs, render_workers, "render")
    print("data done", flush=True)


def pack():
    """Memmaps of all training futures (old 4,800 + new states), in a fixed sorted order."""
    import phase1e_models as p1e
    import phase1d_models as p1d
    import torch
    PACK.mkdir(parents=True, exist_ok=True)
    files = sorted((OUT / "train").glob("*.npz")) + sorted((XDIR / "train").glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    counts = []
    for f in files:
        with np.load(f) as d:
            counts.append(len(d["z"]))
    F, S = int(sum(counts)), len(files)
    z = np.lib.format.open_memmap(PACK / "z.npy", "w+", np.float16, (F, 19, 28, 28, 8))
    z0k = np.lib.format.open_memmap(PACK / "z0k.npy", "w+", np.float16, (F, 28, 28, 8))
    act = np.lib.format.open_memmap(PACK / "actions.npy", "w+", np.float32, (F, 19, 8))
    z0 = np.zeros((S, 28, 28, 8), np.float16); state_of = np.zeros(F, np.int64)
    rare, has_rare = np.zeros(F, bool), np.zeros(F, bool)
    i = 0
    for si, f in enumerate(files):
        with np.load(f) as d:
            n = len(d["z"])
            z[i:i + n] = d["z"]; z0k[i:i + n] = d["z0k"]; act[i:i + n] = p1d.augment(d["actions"], None, False)
            z0[si] = d["z0"]; state_of[i:i + n] = si
            st = {"z": torch.from_numpy(d["z"])}
        m = p1e.detect_modes([st], [str(f)])[str(f)]
        rare[i + np.array(m, int)] = True if m else False
        has_rare[i:i + n] = bool(m)
        i += n
        if si % 500 == 0:
            print(f"pack {si}/{S}", flush=True)
    z.flush(); z0k.flush(); act.flush()
    np.save(PACK / "z0.npy", z0); np.save(PACK / "state_of.npy", state_of)
    np.save(PACK / "rare.npy", rare); np.save(PACK / "has_rare.npy", has_rare)
    (PACK / "files.json").write_text(json.dumps([f.name for f in files]) + "\n")
    print(json.dumps({"states": S, "futures": F, "rare_futures": int(rare.sum()), "states_with_rare": int(has_rare.sum())}), flush=True)


def train(seed=0, log_every=250):
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    import fast_train as ft
    import step3_world_model as wm
    import step3b_models as sm
    path = OUT / "model_US2_x3data.pt"
    if path.exists():
        return
    z = np.load(PACK / "z.npy", mmap_mode="r"); z0k = np.load(PACK / "z0k.npy", mmap_mode="r")
    act_all = np.load(PACK / "actions.npy", mmap_mode="r")
    z0 = torch.from_numpy(np.load(PACK / "z0.npy")).cuda(); state_of = np.load(PACK / "state_of.npy")
    rare, has_rare = np.load(PACK / "rare.npy"), np.load(PACK / "has_rare.npy")
    pools = (np.flatnonzero(rare), np.flatnonzero(~rare & has_rare))
    te = pm.GaussTE(torch.load(OUT / "Gauss_TE_phase2.pt"))
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    # stats from the same first 400 ORIGINAL training states as US2 (identical normalisation)
    old = pm.load_train()[0][:400]
    seq = torch.cat([torch.cat([s["z0k"][:, None], s["z"]], 1) for s in old]); pairs = torch.cat([s["actions"] for s in old])
    stats = wm.Stats(seq, pairs); del old, seq
    model = wm.WorldModel("deterministic", stats, pairs.shape[-1])
    net = torch.compile(model.net)
    opt = torch.optim.AdamW(model.net.parameters(), lr=ft.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    wts = torch.ones(wm.STEPS, device="cuda"); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT
    zeros_tau = torch.zeros(ft.BATCH, device="cuda")
    log, t0 = [{"batch": ft.BATCH, "steps": STEPS, "lr": ft.LR, "futures": int(len(state_of))}], time.time()

    def step_fn(inp, a):
        return net(inp, torch.zeros_like(inp), a, zeros_tau)
    for step in range(STEPS):
        idx = np.sort(np.concatenate([rng.choice(pools[0], ft.BATCH // 2), rng.choice(pools[1], ft.BATCH - ft.BATCH // 2)]))
        target = torch.from_numpy(np.ascontiguousarray(z[idx])).cuda(non_blocking=True).float()
        rend = torch.from_numpy(np.ascontiguousarray(z0k[idx])).cuda().float()
        a = model._act(torch.from_numpy(np.ascontiguousarray(act_all[idx])).cuda())
        zs = z0[torch.as_tensor(state_of[idx], device="cuda")].float()
        use_g = torch.rand(ft.BATCH, device="cuda") < 0.5
        with torch.no_grad():
            start = torch.where(use_g[:, None, None, None], te(zs), rend)
        model.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, loss = start, 0.0
            for t in range(wm.STEPS):
                inp = cur + torch.randn_like(cur) * sm.INPUT_NOISE * stats.res_std
                r = checkpoint(step_fn, inp, a[:, t], use_reentrant=False)
                cur = cur + r.float() * stats.res_std
                loss = loss + wts[t] * ((cur - target[:, t]) ** 2).mean()
            loss = loss / wts.sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % log_every == 0:
            log.append({"step": step + 1, "loss": float(loss), "seconds": round(time.time() - t0, 1)})
            print(f"US2_x3data step {step + 1}: loss {float(loss):.5f} ({time.time() - t0:.0f}s)", flush=True)
    model.net.eval()
    model.save(path)
    (OUT / "model_US2_x3data_log.json").write_text(json.dumps(log) + "\n")
    print("trained US2_x3data", flush=True)


def grade():
    import torch
    import fast_train as ft
    import step3_world_model as wm
    import step3b_models as sm
    from phase2_amend4 import alarm, score
    te = pm.GaussTE(torch.load(OUT / "Gauss_TE_phase2.pt"))
    models = {a: wm.WorldModel.load(OUT / f"model_{a}.pt") for a in ("US2_xlong", "US2_x3data")}
    fork, fam, ends, spread = [], [], {a: [] for a in models}, {a: [] for a in models}
    with torch.no_grad():
        for n_state, path in enumerate(sorted((OUT / "h2").glob("*.npz"))):
            with np.load(path) as f:
                k = int(f["harm"].sum())
                z0 = torch.as_tensor(f["z0"], device="cuda").float()
                g = torch.Generator(device="cuda").manual_seed(pm.SEEDS["monitor"] + n_state)
                starts = te(z0[None].expand(64, -1, -1, -1).contiguous(), g)
                acts = np.asarray(f["actions"], np.float32).reshape(64, wm.STEPS, 8)
                real30 = f["z"][:, 1].reshape(64, -1).astype(np.float64)
            fork.append(2 <= k <= 62); fam.append(path.stem.split("_")[0])
            for a, m in models.items():
                e10, e30 = sm.roll(m, starts, acts)
                ends[a].append(alarm(e10, e30, "BOTH"))
                if 2 <= k <= 62:
                    spread[a].append(float(e30.std(0).sum() / real30.std(0).sum()))
            torch.cuda.empty_cache()
    fork, fam = np.array(fork), np.array(fam)
    out = {"readout": "BOTH"}
    for a in models:
        al = np.array(ends[a])
        out[a] = {"all": score(al, fork), "F1": score(al[fam == "F1"], fork[fam == "F1"]),
                  "F2": score(al[fam == "F2"], fork[fam == "F2"]), "fork_spread_ratio_median": float(np.median(spread[a]))}
    L = json.loads((OUT / "model_US2_x3data_log.json").read_text())
    out["US2_x3data"]["possibly_undertrained"] = ft.undertrained(L, STEPS)
    out["data_helps"] = bool(out["US2_x3data"]["all"]["score"] >= out["US2_xlong"]["all"]["score"] + 0.10)
    out["passes_in_family_bar_on_H2"] = bool(out["US2_x3data"]["all"]["caught"] >= 0.80 and out["US2_x3data"]["all"]["false_alarm"] <= 0.20)
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("data", "pack", "train", "grade"))
    args = parser.parse_args()
    if not json.loads((ROOT / "results/jenga/phase2_generality_protocol.json").read_text())["amendment_11"]["status"].startswith("frozen"):
        raise SystemExit("amendment 11 not frozen")
    {"data": data, "pack": pack, "train": train, "grade": grade}[args.stage]()


if __name__ == "__main__":
    main()
