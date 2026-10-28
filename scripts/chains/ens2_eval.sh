cd /home/corey/wksp/stable_lyap_filters
until grep -q "ens2 done" results/program_b/ens2_0.log; do sleep 120; done
RF_ENS=results/program_b/ens_v2 RF_TAG=_ensv2 python3 eval/readout_fix_imagined.py > results/program_b/rf_ensv2.log 2>&1
python3 eval/ens_check.py results/program_b/ens_v2 > results/program_b/ens_check_v2.log 2>&1
echo "ens2 eval done $(date)" >> results/program_b/rf_ensv2.log
