#!/usr/bin/env bash
# H2H5-NSI (results/jenga/nsi_cl_protocol.json): irreversible-fragility labels -> margin -> value -> gates -> closed loop on
# H2H5's episodes (arm nsi80 = ours80 with the NSI value) -> audit -> eval/h2h5_nsi_analysis.py. One GPU pool at a time.
# usage (repo root): nohup bash scripts/run_h2h5_nsi.sh <h2h5 chain pid> > results/jenga/h2h5_nsi_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
while kill -0 "$1" 2>/dev/null; do sleep 120; done
echo "labels start $(date)"
for i in 0 1; do IRRM_NSI=1 IRRM_NL=8 SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/irrm2.py labels > logs/irrm2_labels_nsi8_$i.out 2>&1 & done
wait
IRRM_NSI=1 IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py held > logs/irrm2_held_nsi8.out 2>&1
ls results/jenga/irrm2_labels_nsi8_0.npz results/jenga/irrm2_labels_nsi8_1.npz results/jenga/irrm2_held_nsi8.npz || { echo "labels missing, stop $(date)"; exit 1; }
$PY - <<'PYEOF' 2>&1 | tee results/jenga/nsi_labels_repro.log
import numpy as np
for i in (0, 1):
    a, b = np.load(f"results/jenga/irrm2_labels_nl8_{i}.npz")["r"], np.load(f"results/jenga/irrm2_labels_nsi8_{i}.npz")["r"]
    print(i, len(a), len(b), "max |dr|", float(np.abs(a - b).max()) if len(a) == len(b) else "length mismatch")
PYEOF
echo "margin start $(date)"
IRRM_NSI=1 IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_nsi8.log 2>&1
$PY eval/irrm_gate.py MN > results/jenga/irrm_MN_gate.log 2>&1
grep -q '"M_pass": true' results/jenga/irrm_MN_gate.json || { echo "MN gate FAIL, stop $(date)"; exit 0; }
echo "value start $(date)"
LSF_VALUE_NAME=value_nsi8_alea.pt LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_nsi_nl8.pt LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
  $PY eval/lsf.py sac > results/program_b/lsf_v1/value_nsi8_alea_train.log 2>&1
ls -la results/program_b/lsf_v1/value_nsi8_alea.pt || { echo "value missing, stop $(date)"; exit 1; }
export LSF_VALUE_NAME=value_nsi8_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8
$PY eval/lsf_audit.py > results/jenga/irrm_AN_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/irrm_AN_diag.log 2>&1
$PY eval/irrm_gate.py AN > results/jenga/irrm_AN_gate.log 2>&1
unset LSF_VALUE_NAME LSF_CLF LSF_ALEA
grep -q '"A_pass": true' results/jenga/irrm_AN_gate.json || { echo "AN gate FAIL, stop $(date)"; exit 0; }
echo "value gated $(date)"
for nz in 2 1; do
  echo "nsi80 ${nz}x start $(date)"
  for i in 0 1 2 3 4 5; do
    env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=_h5n80_${nz}x EP_OFFSET=26200 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
      PAIRED_MAX_CHUNKS=120 CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
      LSF_VALUE_NAME=value_nsi8_alea.pt LSF_CBF_GAMMA=0.2 LSF_ALEA_SCALE=$nz \
      $PY eval/policy_rollout.py paired lfc000 600 >> logs/h2h5_h5n80_${nz}x_$i.out 2>&1 &
  done
  wait
  echo "nsi80 ${nz}x done $(ls results/program_b/policy_rollouts/paired_lfc000_h5n80_${nz}x/ep*.json | wc -l) episodes $(date)"
  $PY cluster/merge_paired.py paired_lfc000_h5n80_${nz}x
  AUDIT_OUT=results/jenga/outcome_audit_h5nsi_${nz}x.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs paired_lfc000_h5n80_${nz}x \
    > results/jenga/outcome_audit_h5nsi_${nz}x.log 2>&1
  echo "${nz}x audited $(date)"
done
$PY eval/h2h5_nsi_analysis.py > results/jenga/h2h5_nsi_analysis.log 2>&1
echo "H2H5-NSI analysed $(date)"
