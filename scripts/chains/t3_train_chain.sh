cd /home/corey/wksp/stable_lyap_filters
L=results/program_b/t3_models.log
rsync -a --relative eval src results/jenga results/program_b/t3 gpu5090:wksp/stable_lyap_filters/ > $L 2>&1 || exit 1
ssh gpu5090 'cd ~/wksp/stable_lyap_filters && . .venv/bin/activate && export TASK=T3 T2_DIR=results/program_b/t3 FAST_TRAIN_Z_ON_CPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python eval/t2_data.py model && python eval/hybrid_tasks.py t2' < /dev/null >> $L 2>&1
rsync -a --include '*.pt' --include '*.json' --exclude '*/' gpu5090:wksp/stable_lyap_filters/results/program_b/t3/ results/program_b/t3/ >> $L 2>&1
echo "chain done $(date)" >> $L
