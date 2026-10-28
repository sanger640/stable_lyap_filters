cd /home/corey/wksp/stable_lyap_filters
for s in dev holdout relpass2; do
  OPT2_CERT=OIL+ENS python3 eval/opt4.py run $s o1 > results/program_b/bench2/opt4_${s}_run.log 2>&1
  python3 eval/opt4.py truth $s o1 > results/program_b/bench2/opt4_${s}_truth.log 2>&1
  python3 eval/opt4.py report $s o1 > results/program_b/bench2/opt4_${s}_report.log 2>&1
  echo $s done $(date)
done
