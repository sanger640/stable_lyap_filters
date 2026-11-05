#!/usr/bin/env bash
# KNOCK stage 5 / HZN1 values (results/jenga/knock_protocol.json stage5, hzn1_protocol.json), single GPU processes, run alongside
# the pool chain (scripts/run_knock5_hzn1.sh waits for the files): fx seed 2 (value_nsif16s2_alea.pt) -> flat NSI margin at
# 0.8 s (gate MNF8; fail -> results/jenga/hzn1_f8_skipped) -> value_nsif8_alea.pt. Resumable.
# usage (repo root): nohup bash scripts/run_train_k5.sh > results/jenga/train_k5_chain.log 2>&1 &
cd "$(dirname "$0")/.."
PY=/home/corey/wksp/dp_venv/bin/python
V1=results/program_b/lsf_v1
echo "start $(date)"
value() {  # value-name margin-name gate extra-env...   (GFX1's recipe for value_nsif16_alea.pt)
  local v=$1 m=$2 gate=$3; shift 3
  if [ ! -f $V1/$v ]; then
    env "$@" LSF_VALUE_NAME=$v LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=$m LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 \
      $PY eval/lsf.py sac > $V1/${v%.pt}_train.log 2>&1
  fi
  ls -la $V1/$v || { echo "$v missing, stop $(date)"; exit 1; }
  LSF_VALUE_NAME=$v LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_audit.py > results/jenga/irrm_${gate}_audit.log 2>&1
  LSF_VALUE_NAME=$v LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/lsf_diag.py > results/jenga/irrm_${gate}_diag.log 2>&1
  LSF_VALUE_NAME=$v LSF_CLF=margin_lf.pt LSF_ALEA=8 $PY eval/irrm_gate.py $gate > results/jenga/irrm_${gate}_gate.log 2>&1
  echo "$v graded $(date)"; grep -E '"(LF2_pass|decisions_pass|A_pass)"' results/jenga/irrm_${gate}_gate.json
}
value value_nsif16s2_alea.pt margin_nsif_nl16.pt ANF2 LSF_SEED=20261105
if [ ! -f $V1/margin_nsif_nl8.pt ]; then
  IRRM_NSI=1 IRRM_NL=8 IRRM_NSI_MODE=flat PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_flat8.log 2>&1
fi
ls -la $V1/margin_nsif_nl8.pt || { echo "margin missing, stop $(date)"; touch results/jenga/hzn1_f8_skipped; exit 1; }
$PY eval/irrm_gate.py MNF8 > results/jenga/irrm_MNF8_gate.log 2>&1; cat results/jenga/irrm_MNF8_gate.json
grep -q '"M_pass": true' results/jenga/irrm_MNF8_gate.json || { echo "MNF8 FAIL, f8 not run $(date)"; touch results/jenga/hzn1_f8_skipped; exit 0; }
value value_nsif8_alea.pt margin_nsif_nl8.pt ANF8
echo "done $(date)"
