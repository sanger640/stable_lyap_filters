cd /home/corey/wksp/stable_lyap_filters
sleep 600
until [ "$(timeout 30 ssh -o BatchMode=yes fir 'squeue -u ali313 -h -n fixcf | wc -l' 2>/dev/null)" = "0" ]; do sleep 300; done
mkdir -p results/jenga/deep_dive/fixcf
rsync -a fir:/scratch/ali313/slf/repo/results/jenga/deep_dive/fixcf/ results/jenga/deep_dive/fixcf/
cat results/jenga/deep_dive/fixcf/rows_*.jsonl | wc -l
/home/corey/wksp/dp_venv/bin/python eval/fix_counterfactual.py analyse
