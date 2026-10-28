cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -j $1 | wc -l" 2>/dev/null || echo 1; }
until [ "$(n 257435,257436,257437,261225)" = "0" ]; do sleep 300; done
rsync -a fir:/scratch/ali313/slf/repo/results/jenga/deep_dive/fixbench/ results/jenga/deep_dive/fixbench/
python3 eval/fix_bench.py analyse > results/jenga/fixbench_analysis.log 2>&1; echo "2x analysed $(date)"
until [ "$(n 261310,261311,261312,261313)" = "0" ]; do sleep 300; done
mkdir -p results/jenga/deep_dive/fixbench_n1
rsync -a fir:/scratch/ali313/slf/repo/results/jenga/deep_dive/fixbench_n1/ results/jenga/deep_dive/fixbench_n1/
BRANCH_NOISE_SCALE=1 python3 eval/fix_bench.py analyse > results/jenga/fixbench_n1_analysis.log 2>&1; echo "1x analysed $(date)"
