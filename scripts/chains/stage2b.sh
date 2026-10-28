cd /home/corey/wksp/stable_lyap_filters
until grep -q "^done" results/program_b/stage2_cem.log; do sleep 30; done
python3 eval/opt2.py report dev s2_cem >> results/program_b/stage2_cem.log 2>&1
for sv in cem grad; do
  OPT2_CERT=OIL+ENS python3 eval/opt2.py run $sv dev s2e_$sv > results/program_b/stage2e_$sv.log 2>&1 || exit 1
  python3 eval/opt2.py report dev s2e_$sv >> results/program_b/stage2e_$sv.log 2>&1
done
echo STAGE2E_DONE
