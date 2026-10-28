cd /home/corey/wksp/stable_lyap_filters
until [ -f results/program_b/cbf_qp/cbfopt_report_b50rep.json ] || grep -q Traceback results/program_b/cbf_opt_b50rep.log; do sleep 60; done
B="OPT_FORK_CHECK=1 OPT_ALPHA=0.5 OPT_STARTS=8 OPT_STEPS=10 OPT_COST=timed OPT_SIG0=0.987410"
CL=$PWD/results/program_b/closedloop_v1/model/model_US.pt
env $B CBF_WM=$CL CBF_CALIB=results/jenga/cbf_calibration.json sh -c "python3 eval/cbf_opt.py run b50cl && python3 eval/cbf_opt.py truth b50cl && python3 eval/cbf_opt.py report b50cl" > results/program_b/cbf_opt_b50cl.log 2>&1
echo LADDER_DONE
