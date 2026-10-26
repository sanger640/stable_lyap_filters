"""Filter latency, each filter ALONE on the local RTX 3090 (5 dev episodes 15000-15004, closed loop, tag _lat; launched as
scratchpad lat_chain.sh: irr_rollout lfck000 / policy_rollout paired lfc000 / lsfa030, SHARD=0/1). Per-move compute from the
logged seconds (gate + edits); the first move of the first episode is excluded (loads networks).
usage: python3 eval/latency_summary.py"""
import json, glob, numpy as np
R = "/home/corey/wksp/stable_lyap_filters/results/program_b/policy_rollouts/paired_{}_lat/ep*.json"
for arm in ("lfck000", "lfc000", "lsfa030"):
    fs = sorted(glob.glob(R.format(arm)))
    if not fs:
        print(arm, "no episodes yet"); continue
    sec, cl, op = [], [], []
    for i, f in enumerate(fs):
        for c in json.load(open(f))["log"]:
            if i == 0 and c["chunk"] == 0:
                continue                                   # first call loads the value net / ensembles
            s = c.get("seconds", 0) or 0; sec.append(s)
            if "irr_gate" in c:
                (op if c["irr_gate"] else cl).append(s)
    sec = np.array(sec)
    out = f"{arm:8s} {len(fs)} eps, {len(sec)} moves | per 0.8 s move: mean {sec.mean():.2f} median {np.median(sec):.2f} p95 {np.percentile(sec, 95):.2f} max {sec.max():.2f} s | real-time factor {sec.mean() / 0.8:.2f}x | share of moves over 0.8 s {100 * (sec > 0.8).mean():.0f}%"
    if cl:
        out += f" | gate closed {np.mean(cl):.2f} s (p95 {np.percentile(cl, 95):.2f}), gate open {np.mean(op):.2f} s (p95 {np.percentile(op, 95):.2f}), open {100 * len(op) / (len(op) + len(cl)):.0f}% of moves"
    print(out)
