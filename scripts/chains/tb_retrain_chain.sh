cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/threeblock_retrain.log
python3 eval/threeblock_retrain.py data > $L 2>&1 || exit 1
# CPU free for T2: resume its paused training-data screen
P=$(pgrep -f "t2_data as t;t.train_data"); for p in $P $(ps -eo pid,ppid | awk -v P="$P" '$2==P{print $1}'); do kill -CONT $p; done
echo "T2 resumed $(date)" >> $L
# models on the 5080 (keeps the 3090 GPU free for T2's render pool)
rsync -a eval src results/jenga gpu5090:wksp/stable_lyap_filters/ --relative >> $L 2>&1
rsync -a --relative results/program_b/threeblock_v2 results/program_b/phase1/h_v2 gpu5090:wksp/stable_lyap_filters/ >> $L 2>&1 || exit 1
ssh gpu5090 'cd ~/wksp/stable_lyap_filters && . .venv/bin/activate && export FAST_TRAIN_Z_ON_CPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && for s in g modes us hybrid grade; do python eval/threeblock_retrain.py $s || exit 1; done' < /dev/null >> $L 2>&1
rsync -a gpu5090:wksp/stable_lyap_filters/results/jenga/threeblock_retrain_result.json results/jenga/ >> $L 2>&1
rsync -a --exclude 'train' --exclude 'val' --exclude 'p1train' gpu5090:wksp/stable_lyap_filters/results/program_b/threeblock_v2/ results/program_b/threeblock_v2/ >> $L 2>&1
echo "chain done $(date)" >> $L
