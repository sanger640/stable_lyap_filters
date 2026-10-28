cd /home/corey/wksp/stable_lyap_filters
until [ -f results/program_b/cbf_qp/cbfopt_report_sep50fc.json ]; do sleep 30; done
B="OPT_FORK_CHECK=1 OPT_ALPHA=0.5 OPT_STARTS=8 OPT_SIG0=0.99"
env $B OPT_SOLVER=cem OPT_CEM_POP=24 OPT_CEM_N=32 sh -c "python3 eval/cbf_opt.py run cem && python3 eval/cbf_opt.py truth cem && python3 eval/cbf_opt.py report cem" > results/program_b/cbf_opt_cem.log 2>&1
env $B OPT_SOLVER=al OPT_AL_N=16 OPT_AL_STEPS=15 sh -c "python3 eval/cbf_opt.py run al && python3 eval/cbf_opt.py truth al && python3 eval/cbf_opt.py report al" > results/program_b/cbf_opt_al.log 2>&1
echo CHAIN_DONE
