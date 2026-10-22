"""LiftPegUpright fork check on a policy's OWN closed-loop rollouts (docs/BENCHMARK_DD.md "LiftPegUpright policies").
E episodes run in parallel on the GPU sim (nominal execution). Every 8 steps (a DP chunk boundary) the state and the
next 8 executed actions are recorded (DP: its chunk; PPO: the 8 actions it actually took next, i.e. the same open-loop
replay of closed-loop actions as the quick check on the demos). Episodes run the task's default 50 steps (success = peg upright at the END; the demos also hold it upright to the end).
Each point is then branched 33 ways: exact + 32 with
~2.5 mm smoothed noise (sigma 0.025 xyz / 0.0083 rot, normalised ee-delta) + 20 hold steps.
Strict label (as the quick-check correction): a replay STANDS if the peg ends < 20 deg from vertical, is FLAT if > 60 deg.
Fork = >= 2 standing AND >= 2 flat among the 32 noisy replays. Bars: >= 15 forks, forks >= 25% of points where any
replay is flat, forks >= 5% of all points.
Runs in ~/wksp/maniskill_bench/.venv.  usage: python eval/maniskill_policy_fork_check.py ppo|dp [CKPT] --out DIR [--episodes 16]
"""
import argparse
import json
import pathlib
import time

import numpy as np

import maniskill_policies as P

H, HOLD, N = 8, 20, 33
SIG = np.array([0.025] * 3 + [0.025 / 3] * 3)


def tilt_t(q):
    """(B, 4) wxyz -> (B,) deg of the peg's long axis (local x) from vertical."""
    import torch
    w, x, y, z = q.unbind(1)
    return torch.rad2deg(torch.arccos(torch.clamp((2 * (x * z - w * y)).abs(), max=1.0)))


def sel(d, e, n):
    if isinstance(d, dict):
        return {k: sel(v, e, n) for k, v in d.items()}
    return d[e:e + 1].expand(n, *d.shape[1:]).contiguous()


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("policy"); ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--out", required=True); ap.add_argument("--episodes", type=int, default=32); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=P.MAX_STEPS, help="episode length (planner-demo DP: 250)")
    ap.add_argument("--act-h", type=int, default=8, help="DP chunk length; branch points stay every 8 steps")
    ap.add_argument("--scale", type=float, default=1.0, help="noise multiplier")
    ap.add_argument("--to-end", action="store_true", help="after the noisy chunk run the policy to the episode end; fork = final success splits")
    ap.add_argument("--branch-h", type=int, default=H, help="actions replayed per branch before the hold (1 = per-step filter)")
    a = ap.parse_args(); P.MAX_STEPS = a.steps; BH = a.branch_h
    pol = P.PPO() if a.policy == "ppo" else P.DP(a.ckpt, act_h=a.act_h)
    E = a.episodes; t0 = time.time()
    env = P.make_env(E); obs, _ = env.reset(seed=a.seed); u = env.unwrapped; pol.reset(E)
    pts, acts, succ_once = [], [], torch.zeros(E, dtype=torch.bool, device=obs.device)
    first = torch.full((E,), 10 ** 6, device=obs.device)                     # step of first success (reported only)
    t = 0
    while t < P.MAX_STEPS:
        st = u.get_state_dict(); st = {g: {k: v.clone() for k, v in d.items()} if isinstance(d, dict) else d.clone() for g, d in st.items()}
        if pol.chunked:
            seq = []
            while len(seq) < H:                                                # chunks of act_h (2, 4 or 8) until 8 steps ran
                ch = pol.chunk(obs)
                for k in range(ch.shape[1]):
                    seq.append(ch[:, k]); obs, _, _, _, info = env.step(ch[:, k]); succ_once |= info["success"]; t += 1
                    first = torch.where(info["success"] & (first > t), torch.full_like(first, t), first)
                    if k < ch.shape[1] - 1:
                        pol.push(obs)
        else:
            seq = []
            for k in range(H):
                ac = pol.act(obs); seq.append(ac); obs, _, _, _, info = env.step(ac); succ_once |= info["success"]; t += 1
                first = torch.where(info["success"] & (first > t), torch.full_like(first, t), first)
        pts.append((t - H, st)); acts.append(torch.stack(seq, 1))                     # (E, H, 7)
    final_tilt = tilt_t(u.peg.pose.q).cpu().numpy(); succ_end = info["success"].cpu().numpy()
    env.close()
    print("rollouts done", round(time.time() - t0, 1), "s; success_once", float(succ_once.float().mean()), "upright at end", float(succ_end.mean()), flush=True)
    first = first.cpu().numpy()
    benv = P.make_env(N * E); benv.reset(seed=0); bu = benv.unwrapped          # all episodes' branches of one time step at once
    rng = np.random.default_rng(20261021); rows = []
    rep = lambda d: {k: rep(v) for k, v in d.items()} if isinstance(d, dict) else d.repeat_interleave(N, 0).contiguous()
    for (tt, st), ac in zip(pts, acts):
        live = list(range(E))
        if not live:
            continue
        bu.set_state_dict(rep(st))
        base = ac[:, :BH].repeat_interleave(N, 0).double().cpu().numpy()      # (E*N, BH, 7): first BH executed actions
        nz = rng.normal(size=(E * N, BH + 2, 6)) * SIG * a.scale; nz = (nz[:, :-2] + nz[:, 1:-1] + nz[:, 2:]) / 3; nz[::N] = 0
        base[:, :, :6] = np.clip(base[:, :, :6] + nz, -1, 1); hold = base[:, -1].copy(); hold[:, :6] = 0
        if a.to_end:                                                           # noisy chunk, then the policy closed loop to the episode's end
            ob = None
            for k in range(BH):
                ob, _, _, _, info = benv.step(torch.as_tensor(base[:, k], device=bu.device).float())
                if k == BH - 1:
                    tcp = bu.agent.tcp.pose.p.reshape(E, N, 3); tdev = (tcp - tcp[:, :1]).norm(dim=-1)[:, 1:].cpu().numpy() * 1000
            pol.reset(E * N); pol.group = N; t2 = tt + BH
            while t2 < P.MAX_STEPS:
                if pol.chunked:
                    ch = pol.chunk(ob)
                    for k in range(ch.shape[1]):
                        ob, _, _, _, info = benv.step(ch[:, k]); t2 += 1
                        if k < ch.shape[1] - 1:
                            pol.push(ob)
                else:
                    ob, _, _, _, info = benv.step(pol.act(ob)); t2 += 1
            pol.group = 1
            se = info["success"].cpu().numpy().reshape(E, N); angs = tilt_t(bu.peg.pose.q).cpu().numpy().reshape(E, N)
            for e in live:
                ns = int(se[e, 1:].sum())
                rows.append({"episode": e, "t": tt, "n_succ": ns, "exact_succ": bool(se[e, 0]), "fork": 2 <= ns <= 30,
                             "n_flat": int((angs[e, 1:] > 60).sum()), "tcp_dev_mm_median": float(np.median(tdev[e])), "tilts": angs[e].round(1).tolist()})
            continue
        for k in range(BH + HOLD):
            benv.step(torch.as_tensor(base[:, k] if k < BH else hold, device=bu.device).float())
            if k == BH - 1:                                                    # realized gripper deviation at the end of the chunk
                tcp = bu.agent.tcp.pose.p.reshape(E, N, 3); tdev = (tcp - tcp[:, :1]).norm(dim=-1)[:, 1:].cpu().numpy() * 1000
        angs = tilt_t(bu.peg.pose.q).cpu().numpy().reshape(E, N)
        for e in live:
            ang = angs[e]; ns, nf = int((ang[1:] < 20).sum()), int((ang[1:] > 60).sum())
            rows.append({"episode": e, "t": tt, "n_stand": ns, "n_flat": nf, "exact_tilt": float(ang[0]), "fork": ns >= 2 and nf >= 2, "tcp_dev_mm_median": float(np.median(tdev[e])),
                         "tilts": ang.round(1).tolist()})
    benv.close()
    fork = sum(r["fork"] for r in rows); anyflat = sum(r["n_flat"] > 0 for r in rows)
    if a.to_end:                                                               # bad = episode ends unsuccessful
        anyflat = sum(r["n_succ"] < 32 for r in rows)
    res = {"policy": a.policy, "noise_scale": a.scale, "ckpt": a.ckpt, "episodes": E, "success_once": float(succ_once.float().mean()), "upright_at_end": float(succ_end.mean()),
           "first_success_step": first.tolist(), "points": len(rows), "forks": fork, "points_any_flat": anyflat, "all_or_nearly_all_flat": sum(r["n_flat"] >= 31 for r in rows),
           "tcp_dev_mm_median": float(np.median([r["tcp_dev_mm_median"] for r in rows])),
           "pass": bool(fork >= 15 and fork >= 0.25 * max(anyflat, 1) and fork >= 0.05 * len(rows)), "seconds": round(time.time() - t0, 1)}
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.policy}{'' if BH == H else f'_bh{BH}'}{'' if a.scale == 1 else f'_x{a.scale:g}'}{'_toend' if a.to_end else ''}_rows.json").write_text(json.dumps(rows)); (out / f"{a.policy}{'' if BH == H else f'_bh{BH}'}{'' if a.scale == 1 else f'_x{a.scale:g}'}{'_toend' if a.to_end else ''}_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
