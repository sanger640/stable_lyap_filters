# IRRM amendment 3: barrier strength for the irreversibility value
source scripts/chains/irrm_lib.sh
cl _g20dev2x 2 2 value_irr8_alea.pt 0.2
cl _g10dev2x 2 2 value_irr8_alea.pt 0.1
cl _g20dev1x 1 1 value_irr8_alea.pt 0.2
cl _g10dev1x 1 1 value_irr8_alea.pt 0.1
$PY eval/irrm_select.py i0 g20 g10 > results/jenga/irrm_a3_select.json
echo "a3 selected $(date)"; cat results/jenga/irrm_a3_select.json | tail -3
