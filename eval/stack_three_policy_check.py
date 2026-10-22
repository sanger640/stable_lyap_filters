"""StackThree_D1 fork check on our Diffusion Policy's OWN closed-loop rollouts (docs/BENCHMARK_DD.md; same branching and
labels as eval/toolhang_recheck.py): at every 2nd decision (the policy's chunk of 8), 33 replays of the chunk (exact + 32
with the robosuite noise, x scale) + 20 hold steps. Generic label vs the exact replay's end: BAD = a cube ends > 2 cm
lower or > 45 deg more tilted (fell / stack collapsed); GOOD = all cubes within 1 cm. Fork = >= 2 BAD and >= 2 GOOD.
Restore keeps robosuite 1.4's hidden state (robot, controller, gripper ramp, warm start, step counter).
Runs in ~/wksp/mimicgen_bench/.venv.  usage: python eval/stack_three_policy_check.py --ckpt CKPT --seeds 1 2 --out DIR [--scale 1.4]
"""
import argparse
import copy
import json
import pathlib
import time

import numpy as np

H, HOLD, N, EVERY_DEC, HORIZON = 8, 20, 33, 2, 400
CUBES = ("cubeA", "cubeB", "cubeC")


def _stateful(rs):
    r = rs.robots[0]
    return [r, r.controller, r.gripper]                                   # robosuite 1.4 API


def _attrs(o):
    return {k: copy.deepcopy(v) for k, v in vars(o).items()
            if isinstance(v, (np.ndarray, float, int, bool, np.floating, np.integer)) or v is None}


def snapshot(rs):
    return (rs.sim.get_state().flatten().copy(), rs.sim.data.qacc_warmstart.copy(), [_attrs(o) for o in _stateful(rs)], rs.timestep, rs.done)


def restore(rs, s):
    rs.sim.set_state_from_flattened(s[0]); rs.sim.forward(); rs.sim.data.qacc_warmstart[:] = s[1]
    for o, a in zip(_stateful(rs), s[2]):
        for k, v in a.items():
            setattr(o, k, copy.deepcopy(v))
    rs.timestep, rs.done = s[3], s[4]


def cubes(rs):
    out = []
    for c in CUBES:
        bid = rs.sim.model.body_name2id(getattr(rs, c).root_body)
        R = rs.sim.data.body_xmat[bid].reshape(3, 3)
        out.append((rs.sim.data.body_xpos[bid].copy(), float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))))
    return out


def branch(rs, chunk, rng, scale):
    s = snapshot(rs)
    base = np.asarray(chunk[:H], np.float64); hold = base[-1].copy(); hold[:6] = 0
    ends, succ, eef = [], [], []
    for k in range(N):
        restore(rs, s); a = base.copy()
        if k > 0:
            n = rng.normal(size=(len(a) + 2, 6)) * np.r_[[0.05] * 3, [0.05 / 3] * 3] * scale
            a[:, :6] = np.clip(a[:, :6] + (n[:-2] + n[1:-1] + n[2:]) / 3, -1, 1)
        for i, act in enumerate(np.vstack([a, np.tile(hold, (HOLD, 1))])):
            rs.step(act)
            if i == H - 1:
                eef.append(rs.sim.data.site_xpos[rs.robots[0].eef_site_id].copy())
        ends.append(cubes(rs)); succ.append(bool(rs._check_success()))
    restore(rs, s)
    ex = ends[0]
    bad = [any(ex[c][0][2] - e[c][0][2] > 0.02 or e[c][1] - ex[c][1] > 45 for c in range(3)) for e in ends[1:]]
    good = [all(np.linalg.norm(e[c][0] - ex[c][0]) < 0.01 for c in range(3)) for e in ends[1:]]
    nb, ng, ns = int(sum(bad)), int(sum(good)), int(sum(succ[1:]))
    return {"n_bad": nb, "n_good": ng, "fork": nb >= 2 and ng >= 2, "succ_exact": succ[0], "n_succ_noisy": ns,
            "eef_dev_mm_median": float(np.median(np.linalg.norm(np.array(eef[1:]) - eef[0], axis=1)) * 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--scale", type=float, default=1.4)
    ap.add_argument("--no-branch", action="store_true", help="success rate only")
    a = ap.parse_args()
    import mimicgen  # noqa: F401
    import torch
    import robomimic.utils.file_utils as FU
    policy, ckpt = FU.policy_from_checkpoint(ckpt_path=a.ckpt, device="cuda", verbose=False)
    env, _ = FU.env_from_checkpoint(ckpt_dict=ckpt, render=False, render_offscreen=False, verbose=False)
    rs = env
    while not hasattr(rs, "sim"):
        rs = rs.env
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); tag = f"_x{a.scale:g}" + ("_nb" if a.no_branch else "")
    for seed in a.seeds:
        f = out / f"StackThree_s{seed}{tag}.json"
        if f.exists():
            continue
        np.random.seed(seed); torch.manual_seed(seed); rng = np.random.default_rng(20261022 + seed)
        t0 = time.time(); policy.start_episode(); obs = env.reset()
        rows, t, dec, done = [], 0, 0, False
        while t < HORIZON and not done:
            new = len(policy.policy.action_queue) == 0
            act = policy(ob=obs)
            if new:
                chunk = np.vstack([act] + [x.detach().cpu().numpy().reshape(-1) for x in policy.policy.action_queue])
                if not a.no_branch and dec % EVERY_DEC == 0:
                    rows.append({"t": t, **branch(rs, chunk, rng, a.scale)})
                dec += 1
            obs, r, _, _ = env.step(act); t += 1
            done = bool(env.is_success()["task"])
        ep = {"seed": seed, "success": done, "steps": t, "points": len(rows), "seconds": round(time.time() - t0, 1)}
        print(json.dumps(ep), flush=True)
        f.write_text(json.dumps({"episode": ep, "rows": rows}))


if __name__ == "__main__":
    main()
