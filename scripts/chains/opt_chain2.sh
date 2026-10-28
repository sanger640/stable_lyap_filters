cd /home/corey/wksp/stable_lyap_filters
until grep -q CHAIN_DONE results/program_b/opt_chain.log; do sleep 60; done
env OPT_FORK_CHECK=1 OPT_ALPHA=0.5 OPT_STARTS=8 OPT_SIG0=0.99 sh -c "python3 eval/cbf_opt.py run sep50s1fc && python3 eval/cbf_opt.py truth sep50s1fc && python3 eval/cbf_opt.py report sep50s1fc" > results/program_b/cbf_opt_sep50s1fc.log 2>&1
echo CHAIN2_DONE
