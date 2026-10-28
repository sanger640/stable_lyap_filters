set -e
cd /home/corey/wksp/stable_lyap_filters
export OPT2_CERT=OIL+ENS CBF_WM=$PWD/results/program_b/oil_v1/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json
python3 eval/opt3.py run grad dev g_grad
python3 eval/opt3.py truth dev g_grad
python3 eval/opt3.py report dev g_grad
