cd /home/corey/wksp/stable_lyap_filters
OIL=$PWD/results/program_b/oil_v1/model/model_OIL.pt
export DIFFUSION_POLICY_DIR=/home/corey/wksp/diffusion_policy CBF_WM=$OIL CBF_CALIB=results/jenga/cbf_calibration_OIL.json OPT_FORK_CHECK=1 OPT_ALPHA=0.5 OPT_STARTS=8 OPT_STEPS=10 OPT_COST=timed OPT_SIG0=0.987410
PY=/home/corey/wksp/dp_venv/bin/python
OPT_PLAUS_W=2.25 $PY eval/cbf_opt.py run b50oilp || exit 1
python3 eval/cbf_opt.py truth b50oilp && python3 eval/cbf_opt.py report b50oilp || exit 1
$PY eval/cbf_opt.py plaus b50oil && $PY eval/cbf_opt.py plaus b50oilp
echo STEP2_DONE
