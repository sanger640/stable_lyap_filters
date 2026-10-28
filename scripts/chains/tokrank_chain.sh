cd /home/corey/wksp/stable_lyap_filters
while ps -p 1515230 >/dev/null; do sleep 60; done
tail -2 results/program_b/tok_v1/model/train.log
for s in dev relpass; do
  python3 eval/tok_rank.py $s results/program_b/tok_v1/model/tokwm_v1.pt v1 > results/jenga/deep_dive/tokrank_${s}_v1.log 2>&1
  echo $s done $(date)
done
