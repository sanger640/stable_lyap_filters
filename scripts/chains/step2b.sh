cd /home/corey/wksp/stable_lyap_filters
OIL=$PWD/results/program_b/oil_v1/model/model_OIL.pt
export DIFFUSION_POLICY_DIR=/home/corey/wksp/diffusion_policy CBF_WM=$OIL CBF_CALIB=results/jenga/cbf_calibration_OIL.json OPT_FORK_CHECK=1 OPT_ALPHA=0.0 OPT_STARTS=8 OPT_STEPS=10 OPT_COST=timed OPT_SIG0=0.987410 OPT_PLAUS_W=2.25
PY=/home/corey/wksp/dp_venv/bin/python
$PY eval/cbf_opt.py run a0p || exit 1
python3 eval/cbf_opt.py truth a0p && python3 eval/cbf_opt.py report a0p || exit 1
$PY eval/cbf_opt.py plaus a0p
echo STEP2B_DONE
