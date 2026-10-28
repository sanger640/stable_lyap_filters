cd /home/corey/wksp/stable_lyap_filters
for sv in cem grad; do
  python3 eval/opt2.py run $sv dev s2_$sv > results/program_b/stage2_$sv.log 2>&1 || exit 1
  python3 eval/opt2.py report dev s2_$sv >> results/program_b/stage2_$sv.log 2>&1
done
echo STAGE2_DONE
