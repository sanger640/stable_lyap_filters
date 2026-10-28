cd /home/corey/wksp/stable_lyap_filters
until [ -f results/program_b/cbf_qp/cbfopt_report_cem2.json ] || grep -q Traceback results/program_b/cbf_opt_cem2.log; do sleep 60; done
B="OPT_FORK_CHECK=1 OPT_ALPHA=0.5 OPT_STARTS=8 OPT_STEPS=10"
env $B OPT_COST=timed OPT_SIG0=0.987410 sh -c "python3 eval/cbf_opt.py run b50rep && python3 eval/cbf_opt.py truth b50rep && python3 eval/cbf_opt.py report b50rep" > results/program_b/cbf_opt_b50rep.log 2>&1
env $B OPT_SIG0=0.99 sh -c "python3 eval/cbf_opt.py run sep50s1fc && python3 eval/cbf_opt.py truth sep50s1fc && python3 eval/cbf_opt.py report sep50s1fc" > results/program_b/cbf_opt_sep50s1fc.log 2>&1
echo CHAIN4_DONE
