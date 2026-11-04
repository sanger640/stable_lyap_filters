"""KNOCK closed loop (results/jenga/knock_protocol.json): eval/policy_rollout.py's paired runner (same noise, seeds, saved
decision states and filters) with the knock-down expert (eval/knock_task.py) in place of the diffusion policy.
The expert's 16-target plan is returned in the policy's layout: row 0 = the current target, rows 1-8 = the executed chunk.
Each episode's result adds red_down (red block tilted >= 45 deg at the end) and success = red_down and no neighbour topple.
usage: <policy_rollout env: PAIRED_TAG, EP_OFFSET, SHARD, EXEC_NOISE_SCALE, HARD_OFF=1, PAIRED_MAX_CHUNKS=15, ...>
       python3 eval/knock_rollout.py <mode: record | lfc000 | ...> <n>"""
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "src")]
import jenga_short_held_tails as J  # noqa: E402
import knock_task as K  # noqa: E402
import policy_rollout as PR  # noqa: E402

STATE = {}
_reset, _close, _episode = J.DirectJengaSim.reset, J.DirectJengaSim.close, PR.episode


def reset(self, seed):
    _reset(self, seed); STATE.update(sim=self, exp=None)


def close(self):
    STATE["final"] = K.tilts(self); _close(self)


def policy_chunk(pol, hist, full=False, n=1):
    sim = STATE["sim"]
    if STATE["exp"] is None:                                  # first decision comes after the settle
        sim._z0 = float(sim.data.xpos[sim.tracked_block_ids[0]][2]); STATE["exp"] = K.Expert(STATE["ep"])
    plan = STATE["exp"].plan(sim)
    p16 = np.vstack([np.r_[sim.target, 1.0][None], plan[:15]]).astype(np.float32)
    return p16 if (full or n == 1) and n == 1 else np.repeat(p16[None], n, 0)


def episode(pol, i, *a, **kw):
    STATE["ep"] = int(os.environ.get("KNOCK_SEED_BASE", 0)) + i
    res, frames, chunks = _episode(pol, i, *a, **kw)
    t = STATE["final"]
    res["red_down"] = bool(t[0] >= K.TOPPLE_DEG); res["red_final_tilt"] = round(float(t[0]), 1)
    res["success"] = res["red_down"] and not res["neighbour_toppled"]
    return res, frames, chunks


J.DirectJengaSim.reset, J.DirectJengaSim.close = reset, close
PR.policy_chunk, PR.episode, PR.load_policy = policy_chunk, episode, (lambda steps=8: object())

if __name__ == "__main__":
    PR.paired(sys.argv[1], int(sys.argv[2]))
