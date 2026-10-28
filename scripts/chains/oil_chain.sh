cd /home/corey/wksp/stable_lyap_filters
until grep -q LADDER_DONE results/program_b/wm_ladder.log; do sleep 60; done
CF=$PWD/results/program_b/cf_v1/model/model_CF.pt; OIL=$PWD/results/program_b/oil_v1
B="OPT_ALPHA=0.5 OPT_STARTS=8 OPT_STEPS=10 OPT_COST=timed OPT_SIG0=0.987410"
env $B CBF_WM=$CF CBF_CALIB=results/jenga/cbf_calibration_CF.json OIL_N=200 python3 eval/oil_data.py propose || exit 1
CF_OUT=results/program_b/oil_v1 CF_PLANS=$OIL/proposals.json CF_WORKERS=6 python3 eval/cf_data.py || exit 1
python3 eval/oil_train.py || exit 1
CBF_WM=$OIL/model/model_OIL.pt CBF_CALIB_OUT=results/jenga/cbf_calibration_OIL.json python3 eval/cbf_calibrate.py || exit 1
env $B OPT_FORK_CHECK=1 CBF_WM=$OIL/model/model_OIL.pt CBF_CALIB=results/jenga/cbf_calibration_OIL.json sh -c "python3 eval/cbf_opt.py run b50oil && python3 eval/cbf_opt.py truth b50oil && python3 eval/cbf_opt.py report b50oil" > results/program_b/cbf_opt_b50oil.log 2>&1
echo OIL_DONE
