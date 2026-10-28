cd /home/corey/wksp/stable_lyap_filters
until grep -q CHAIN_DONE results/program_b/opt_chain.log; do sleep 60; done
B="OPT_FORK_CHECK=1 OPT_ALPHA=0.5 OPT_STARTS=8 OPT_SIG0=0.99"
env $B OPT_SOLVER=cem OPT_CEM_CRN=1 OPT_CEM_NO_NOMINAL=1 OPT_CEM_VERIFY=3 OPT_CEM_N=64 OPT_CEM_POP=24 sh -c "python3 eval/cbf_opt.py run cem2 && python3 eval/cbf_opt.py truth cem2 && python3 eval/cbf_opt.py report cem2" > results/program_b/cbf_opt_cem2.log 2>&1
env $B sh -c "python3 eval/cbf_opt.py run sep50s1fc && python3 eval/cbf_opt.py truth sep50s1fc && python3 eval/cbf_opt.py report sep50s1fc" > results/program_b/cbf_opt_sep50s1fc.log 2>&1
echo CHAIN3_DONE
