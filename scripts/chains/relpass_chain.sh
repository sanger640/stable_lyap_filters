cd /home/corey/wksp/stable_lyap_filters
OPT3_ABS=1 OPT2_CERT=OIL+ENS python3 eval/opt3.py run cem relpass abs > results/program_b/bench2/relpass_run.log 2>&1
for p in $(cat scripts/chains/rb1_pids.txt); do kill -CONT $p; done
echo resumed $(date)
python3 eval/opt3.py truth relpass abs > results/program_b/bench2/relpass_truth.log 2>&1
python3 eval/relpass.py report abs > results/program_b/bench2/relpass_report.log 2>&1
echo report-done $(date)
