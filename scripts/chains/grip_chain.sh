cd /home/corey/wksp/stable_lyap_filters
export OPT2_CERT=OIL+ENS OPT3_GRIP=shift
for s in dev holdout; do
  python3 eval/opt3.py run cem $s grip > results/program_b/bench2/grip_${s}_run.log 2>&1
  python3 eval/opt3.py truth $s grip > results/program_b/bench2/grip_${s}_truth.log 2>&1
  python3 eval/opt3.py report $s grip > results/program_b/bench2/grip_${s}_report.log 2>&1
done
echo done $(date)
