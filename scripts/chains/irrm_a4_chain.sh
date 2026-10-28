# IRRM amendment 4: 1.6 s irreversibility labels -> margin -> value -> gates -> closed loop (after amendment 3)
source scripts/chains/irrm_lib.sh
A3=$1
echo "a4 labels start $(date)"
for i in 0 1; do IRRM_NL=16 SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/irrm2.py labels > logs/irrm2_labels_nl16_$i.out 2>&1 & done
wait
IRRM_NL=16 PYTHONUNBUFFERED=1 $PY eval/irrm2.py held > logs/irrm2_held_nl16.out 2>&1
echo "a4 labels done $(date)"
IRRM_NL=16 PYTHONUNBUFFERED=1 $PY eval/irrm2.py margin > results/jenga/irrm2_margin_nl16.log 2>&1
$PY eval/irrm_gate.py M16 > results/jenga/irrm_M16_gate.log 2>&1
grep -q '"M_pass": true' results/jenga/irrm_M16_gate.json || { echo "a4 margin gate FAIL, stop $(date)"; exit 0; }
while kill -0 $A3 2>/dev/null; do sleep 60; done                     # one training / closed-loop job at a time
echo "a4 train start $(date)"
LSF_VALUE_NAME=value_irr16_alea.pt LSF_CLF=margin_lf.pt LSF_IRR_MARGIN=margin_irr_nl16.pt LSF_MARGIN_MODE=irr LSF_ALEA=8 PYTHONUNBUFFERED=1 $PY eval/lsf.py sac > results/program_b/lsf_v1/value_irr16_alea_train.log 2>&1
ls -la results/program_b/lsf_v1/value_irr16_alea.pt || exit 1
export LSF_VALUE_NAME=value_irr16_alea.pt LSF_CLF=margin_lf.pt LSF_ALEA=8
$PY eval/lsf_audit.py > results/jenga/irrm_A16_audit.log 2>&1
$PY eval/lsf_diag.py > results/jenga/irrm_A16_diag.log 2>&1
$PY eval/irrm_gate.py A16 > results/jenga/irrm_A16_gate.log 2>&1
unset LSF_VALUE_NAME LSF_CLF LSF_ALEA
echo "a4 gated $(date)"
grep -q '"A_pass": true' results/jenga/irrm_A16_gate.json || { echo "a4 value gate FAIL: no closed loop $(date)"; exit 0; }
G=$(python3 -c "import json;print(json.load(open('results/jenga/irrm_a3_select.json'))['selected_gamma'] or 0.5)")
cl _j0dev2x 2 2 value_irr16_alea.pt 0.5
cl _j0dev1x 1 1 value_irr16_alea.pt 0.5
if [ "$G" != "0.5" ]; then cl _jsdev2x 2 2 value_irr16_alea.pt $G; cl _jsdev1x 1 1 value_irr16_alea.pt $G; fi
$PY eval/irrm_select.py i0 g20 g10 j0 js > results/jenga/irrm_a4_select.json
echo "a4 selected $(date)"; tail -3 results/jenga/irrm_a4_select.json
