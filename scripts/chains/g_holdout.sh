set -e
cd /home/corey/wksp/stable_lyap_filters
export OPT2_CERT=OIL+ENS CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json
python3 eval/bench2.py baseline holdout
python3 eval/opt3.py run cem holdout g_cem
python3 eval/opt3.py truth holdout g_cem
python3 eval/opt3.py report holdout g_cem
