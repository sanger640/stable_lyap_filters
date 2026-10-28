cd /home/corey/wksp/stable_lyap_filters
while pgrep -f "^python3 eval/threeblock_retrain.py data" > /dev/null; do
  for d in train val p1train; do rsync -a --ignore-existing --exclude '*.tmp.npz' gpu5090:wksp/stable_lyap_filters/results/program_b/threeblock_v2/$d/ results/program_b/threeblock_v2/$d/; done
  sleep 300
done
ssh gpu5090 'pkill -f "threeblock_retrain.py data"' < /dev/null
for d in train val p1train; do rsync -a --ignore-existing --exclude '*.tmp.npz' gpu5090:wksp/stable_lyap_filters/results/program_b/threeblock_v2/$d/ results/program_b/threeblock_v2/$d/; done
