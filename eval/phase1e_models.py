"""Branching-WM Phase 1e: label-free rebalancing by each training state's rare outcome mode (U, US).

Frozen protocol: results/jenga/phase1e_protocol.json. Rare modes come from the goal-free split test on the real
futures' latents; topple labels are used only to describe the modes afterwards.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]

import phase1b_models as p1b  # noqa: E402
import phase1d_models as p1d  # noqa: E402
import step3_world_model as wm  # noqa: E402

PROTOCOL = ROOT / "results/jenga/phase1e_protocol.json"
OUT = ROOT / "results/program_b/phase1e"
MODES = OUT / "modes.json"
RESULT = ROOT / "results/jenga/phase1e_result.json"
DEVICE, BATCH, STEPS = "cuda", 32, 30_000


def detect_modes(states, names):
    from counterfactual_monitor import persistent_mode_alarm
    from outcome_modes import two_mode_test
    modes = {}
    for name, s in zip(names, states):
        z = s["z"].float().numpy()
        e10 = z[:, wm.HOLD10].reshape(len(z), -1).astype(np.float64)
        e30 = z[:, wm.HOLD30].reshape(len(z), -1).astype(np.float64)
        rare = []
        if persistent_mode_alarm(e10, e30).alarm:
            lab = two_mode_test(e30, 0.0).labels
            minority = 1 if lab.sum() <= len(lab) / 2 else 0
            rare = np.flatnonzero(lab == minority).tolist()
        modes[name] = rare
    return modes


def describe(states, names, modes):
    rare_top = rare_n = top_n = top_in_rare = split_states = dis_states = dis_and_split = 0
    for name, s in zip(names, states):
        r = set(modes[name]); t = s["top"]
        split_states += bool(r); dis = bool(t.any() and (~t).any()); dis_states += dis; dis_and_split += dis and bool(r)
        rare_n += len(r); rare_top += sum(bool(t[k]) for k in r)
        top_n += int(t.sum()); top_in_rare += sum(bool(t[k]) for k in r)
    return {"states": len(states), "states_with_rare_mode": split_states, "states_with_both_outcomes": dis_states,
            "both_outcomes_and_rare_mode": dis_and_split, "rare_futures": rare_n,
            "share_of_rare_futures_that_topple": rare_top / max(rare_n, 1),
            "share_of_toppling_futures_in_a_rare_mode": top_in_rare / max(top_n, 1)}


def train(name, siblings, states, modes_list, seed=0):
    import step3b_models as sm
    path = OUT / f"model_{name}.pt"
    if path.exists():
        return wm.WorldModel.load(path)
    torch.manual_seed(seed); np.random.seed(seed)
    g = sm.G.load(sm.DATA / "G.pt")
    rows = [(si, k) for si, s in enumerate(states) for k in range(len(s["z"]))]
    rare = np.array([k in modes_list[si] for si, k in rows])
    has_rare = np.array([bool(modes_list[si]) for si, _ in rows])
    idx_rare = np.flatnonzero(rare)
    idx_other = np.flatnonzero(~rare & has_rare) if siblings else np.flatnonzero(~rare)
    seq = torch.cat([torch.cat([s["z0k"][:, None], s["z"]], 1) for s in states[:400]])
    pairs = torch.cat([s["actions"] for s in states[:400]])
    stats = wm.Stats(seq, pairs)
    model = wm.WorldModel("deterministic", stats, pairs.shape[-1])
    opt = torch.optim.AdamW(model.net.parameters(), lr=wm.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    wts = torch.ones(wm.STEPS, device=DEVICE); wts[[wm.HOLD10, wm.HOLD30]] = sm.HOLD_WEIGHT
    log = [{"rare_futures": int(len(idx_rare)), "other_pool": int(len(idx_other))}]
    for step in range(STEPS):
        sel = np.concatenate([np.random.choice(idx_rare, BATCH // 2), np.random.choice(idx_other, BATCH // 2)])
        pick = [rows[i] for i in sel]
        z0 = torch.stack([states[i]["z0"] for i, _ in pick]).to(DEVICE).float()
        rend = torch.stack([states[i]["z0k"][k] for i, k in pick]).to(DEVICE).float()
        dd = torch.stack([states[i]["d"][k] for i, k in pick]).to(DEVICE)
        target = torch.stack([states[i]["z"][k] for i, k in pick]).to(DEVICE).float()
        act = model._act(torch.stack([states[i]["actions"][k] for i, k in pick]).to(DEVICE))
        use_g = torch.rand(BATCH, device=DEVICE) < 0.5
        start = rend.clone()
        if use_g.any():
            start[use_g] = g(z0[use_g], dd[use_g])
        model.net.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, loss = start, 0.0
            for t in range(wm.STEPS):
                inp = cur + torch.randn_like(cur) * sm.INPUT_NOISE * stats.res_std
                r = model.net(inp, torch.zeros_like(inp), act[:, t], torch.zeros(BATCH, device=DEVICE))
                cur = cur + r.float() * stats.res_std
                loss = loss + wts[t] * ((cur - target[:, t]) ** 2).mean()
            loss = loss / wts.sum()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            log.append({"step": step + 1, "loss": float(loss)})
            print(f"{name} step {step + 1}: loss {float(loss):.5f}", flush=True)
    model.net.eval()
    model.save(path)
    (OUT / f"model_{name}_log.json").write_text(json.dumps(log) + "\n")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("modes", "U", "US"))
    args = parser.parse_args()
    if not json.loads(PROTOCOL.read_text())["status"].startswith("frozen"):
        raise SystemExit("protocol not frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    names = [str(p) for f in p1b.FOLDERS_ENRICHED for p in sorted(Path(f).glob("ep*.npz")) if not p.name.endswith(".tmp.npz")]
    states = p1d.load(p1b.FOLDERS_ENRICHED, proprio=False)
    assert len(names) == len(states)
    if args.stage == "modes":
        modes = detect_modes(states, names)
        MODES.write_text(json.dumps(modes) + "\n")
        summary = describe(states, names, modes)
        (OUT / "modes_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1), flush=True)
        return
    modes = json.loads(MODES.read_text())
    model = train(args.stage, args.stage == "US", states, [set(modes[n]) for n in names])
    tau, n_val = p1d.floor(model, False)
    results = json.loads(RESULT.read_text()) if RESULT.exists() else {
        k: v for k, v in json.loads(p1d.RESULT.read_text()).items() if k in ("E_reference", "R")}
    results[args.stage] = p1d.evaluate(model, False, tau) | {"validation_futures": n_val}
    log = [r for r in json.loads((OUT / f"model_{args.stage}_log.json").read_text()) if "step" in r]
    last = np.mean([r["loss"] for r in log if STEPS - 3000 < r["step"] <= STEPS])
    before = np.mean([r["loss"] for r in log if STEPS - 6000 < r["step"] <= STEPS - 3000])
    results[args.stage]["possibly_undertrained"] = bool(last < before - 0.02 * abs(before))
    score = lambda a: results[a]["R0"]["caught"] - results[a]["R0"]["false_alarm"]
    arms = [a for a in ("U", "US") if a in results]
    results["ranking_R0"] = sorted(arms, key=score, reverse=True)
    results["matches_R_within_0.03"] = {a: bool(abs(score(a) - score("R")) <= 0.03 or score(a) > score("R")) for a in arms}
    best = results[results["ranking_R0"][0]]["R0"]
    results["gate_for_panel_6"] = bool(best["caught"] >= 0.80 and best["false_alarm"] <= 0.20)
    RESULT.write_text(json.dumps(results, indent=1) + "\n")
    print(args.stage, json.dumps(results[args.stage]), flush=True)


if __name__ == "__main__":
    main()
