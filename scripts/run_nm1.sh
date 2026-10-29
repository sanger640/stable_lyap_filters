#!/usr/bin/env bash
# NM1 (results/jenga/nm1_protocol.json): no hand mask. S0 CF3 rescoring (report) -> labels -> margin (MNM) -> value (ANM)
# -> dev nm10 / nm20 at 2x then 1x -> audits -> eval/nm1_dev_analysis.py. Waits for the NSI3 chain. One GPU pool at a time.
# usage (repo root): nohup bash scripts/run_nm1.sh <nsi3 chain pid> > results/jenga/nm1_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt
while kill -0 "$1" 2>/dev/null; do sleep 120; done
echo "S0 start $(date)"
for i in 0 1; do CF_MASK=none CF_TAG=cf3nm SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/cf_conf.py run > results/jenga/cf3nm_run_$i.log 2>&1 & done
wait
$PY eval/redm1.py nm > results/jenga/nm1_stage0.log 2>&1; echo "S0 done $(date)"
echo "labels start $(date)"
for i in 0 1; do IRRM_MASK=none IRRM_NL=8 SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/irrm2.py labels > logs/irrm2_labels_nm8_$i.out 2>&1 & done
wait
IRRM_MASK=none IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py held > logs/irrm2_held_nm8.out 2>&1
ls results/jenga/irrm2_labels_nm8_0.npz results/jenga/irrm2_labels_nm8_1.npz results/jenga/irrm2_held_nm8.npz || { echo "labels missing, stop $(date)"; exit 1; }
echo "margin start $(date)"
IRRM_MASK=none IRRM_NL=8 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_nm8.log 2>&1
$PY eval/irrm_gate.py MNM > results/jenga/irrm_MNM_gate.log 2>&1
grep -q '"M_pass": true' results/jenga/irrm_MNM_gate.json || { echo "MNM gate FAIL, stop $(date)"; exit 0; }
echo "value start $(date)"
LSF_VALUE_NAME=value_irrnm8_alea.pt LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_irrnm_nl8.pt LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
  $PY eval/lsf.py sac > results/program_b/lsf_v1/value_irrnm8_alea_train.log 2>&1
ls -la results/program_b/lsf_v1/value_irrnm8_alea.pt || { echo "value missing, stop $(date)"; exit 1; }
export LSF_VALUE_NAME=value_irrnm8_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8
$PY eval/lsf_audit.py > results/jenga/irrm_ANM_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/irrm_ANM_diag.log 2>&1
$PY eval/irrm_gate.py ANM > results/jenga/irrm_ANM_gate.log 2>&1
unset LSF_VALUE_NAME LSF_CLF LSF_ALEA
$PY eval/irrm_gate.py AREF > results/jenga/irrm_AREF_gate.log 2>&1
grep -q '"A_pass": true' results/jenga/irrm_ANM_gate.json || { echo "ANM gate FAIL, stop $(date)"; exit 0; }
echo "value gated $(date)"
for nz in 2 1; do
  for g in 10 20; do
    tag=_nm${g}dev${nz}x; echo "$tag start $(date)"
    for i in 0 1 2 3 4 5; do
      env MUJOCO_GL=egl PER_EPISODE_SEEDS=1 PAIRED_PER_EP=1 PAIRED_TAG=$tag EP_OFFSET=15000 SHARD=$i/6 EXEC_NOISE_SCALE=$nz HARD_OFF=1 \
        LSF_ALEA_SCALE=$nz CBF_WM=$WM CBF_CALIB=results/jenga/cbf_calibration_OIL.json PYTHONUNBUFFERED=1 \
        LSF_VALUE_NAME=value_irrnm8_alea.pt LSF_CBF_GAMMA=0.$g \
        $PY eval/policy_rollout.py paired lfc000 100 >> logs/nm1${tag}_$i.out 2>&1 &
    done
    wait
    $PY cluster/merge_paired.py paired_lfc000$tag
    AUDIT_OUT=results/jenga/outcome_audit$tag.json MUJOCO_GL=egl AUDIT_NOISE_SCALE=$nz $PY eval/outcome_audit.py runs paired_lfc000$tag \
      > results/jenga/outcome_audit$tag.log 2>&1
    echo "$tag audited $(date)"; tail -1 results/jenga/outcome_audit$tag.log
  done
done
$PY eval/nm1_dev_analysis.py > results/jenga/nm1_dev_analysis.log 2>&1
echo "NM1 analysed $(date)"
