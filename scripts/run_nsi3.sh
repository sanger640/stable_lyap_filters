#!/usr/bin/env bash
# NSI3 (results/jenga/nsi3_protocol.json): 1.6 s NSI labels -> margin (MN16) -> value (AN16) -> dev n16g20 at 2x then 1x
# -> audits -> eval/nsi3_dev_analysis.py. Waits for the H2H6 chain. One GPU pool at a time.
# usage (repo root): nohup bash scripts/run_nsi3.sh <h2h6 chain pid> > results/jenga/nsi3_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
while kill -0 "$1" 2>/dev/null; do sleep 120; done
echo "labels start $(date)"
for i in 0 1; do IRRM_NSI=1 IRRM_NL=16 SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/irrm2.py labels > logs/irrm2_labels_nsi16_$i.out 2>&1 & done
wait
IRRM_NSI=1 IRRM_NL=16 PYTHONUNBUFFERED=1 $PY eval/irrm2.py held > logs/irrm2_held_nsi16.out 2>&1
ls results/jenga/irrm2_labels_nsi16_0.npz results/jenga/irrm2_labels_nsi16_1.npz results/jenga/irrm2_held_nsi16.npz || { echo "labels missing, stop $(date)"; exit 1; }
$PY - <<'PYEOF' 2>&1 | tee results/jenga/nsi16_labels_repro.log
import numpy as np
for i in (0, 1):
    a, b = np.load(f"results/jenga/irrm2_labels_nl16_{i}.npz")["r"], np.load(f"results/jenga/irrm2_labels_nsi16_{i}.npz")["r"]
    print(i, len(a), len(b), "max |dr|", float(np.abs(a - b).max()) if len(a) == len(b) else "length mismatch")
PYEOF
echo "margin start $(date)"
IRRM_NSI=1 IRRM_NL=16 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_nsi16.log 2>&1
$PY eval/irrm_gate.py MN16 > results/jenga/irrm_MN16_gate.log 2>&1
grep -q '"M_pass": true' results/jenga/irrm_MN16_gate.json || { echo "MN16 gate FAIL, stop $(date)"; exit 0; }
echo "value start $(date)"
LSF_VALUE_NAME=value_nsi16_alea.pt LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_nsi_nl16.pt LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
  $PY eval/lsf.py sac > results/program_b/lsf_v1/value_nsi16_alea_train.log 2>&1
ls -la results/program_b/lsf_v1/value_nsi16_alea.pt || { echo "value missing, stop $(date)"; exit 1; }
export LSF_VALUE_NAME=value_nsi16_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8
$PY eval/lsf_audit.py > results/jenga/irrm_AN16_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/irrm_AN16_diag.log 2>&1
$PY eval/irrm_gate.py AN16 > results/jenga/irrm_AN16_gate.log 2>&1
unset LSF_VALUE_NAME LSF_CLF LSF_ALEA
grep -q '"A_pass": true' results/jenga/irrm_AN16_gate.json || { echo "AN16 gate FAIL, stop $(date)"; exit 0; }
echo "value gated $(date)"
for nz in 2 1; do
  tag=_n16g20dev${nz}x; echo "$tag start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      LSF_ALEA_SCALE=$nz CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_nsi16_alea.pt LSF_CBF_GAMMA=0.2 \
      $PY eval/policy_rollout.py paired lfc000 100 >> logs/nsi3${tag}_$i.out 2>&1 &
  done
  wait
  $PY cluster/merge_paired.py paired_lfc000$tag
  AUDIT_OUT=results/jenga/outcome_audit$tag.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs paired_lfc000$tag \
    > results/jenga/outcome_audit$tag.log 2>&1
  echo "$tag audited $(date)"; tail -1 results/jenga/outcome_audit$tag.log
done
$PY eval/nsi3_dev_analysis.py > results/jenga/nsi3_dev_analysis.log 2>&1
echo "NSI3 analysed $(date)"
