"""XFER1 (results/jenga/xfer1_protocol.json): the outcome audit replayed in the SHIFTED physics (neighbours' mass and inertia
x0.3, as MASS_ONLY=0.3 in eval/policy_rollout.py). eval/outcome_audit.py forces normal physics (HARD_MASS=0); this sets
HARD_MASS=0.3 after its import, which eval/inspector_data._sim applies to the same two bodies (mass and inertia).
usage: AUDIT_OUT=... AUDIT_NOISE_SCALE=2 python3 eval/xfer1_audit.py <run> [<run> ...]"""
import os
import sys

import outcome_audit as oa

os.environ["HARD_MASS"] = "0.3"
oa.audit_runs(*sys.argv[1:])
