cd /home/corey/wksp/stable_lyap_filters
python3 eval/threeblock_v2.py sim > results/program_b/h_v2.log 2>&1 || exit 1
while pgrep -f "hybrid_tasks.py t1" > /dev/null; do sleep 30; done
python3 eval/threeblock_v2.py encode >> results/program_b/h_v2.log 2>&1 || exit 1
python3 eval/threeblock_v2.py grade >> results/program_b/h_v2.log 2>&1
