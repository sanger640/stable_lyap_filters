cd /home/corey/wksp/stable_lyap_filters
export TB_DIR=results/program_b/threeblock_v3
L=results/program_b/threeblock_v3.log
R="~/wksp/stable_lyap_filters"
ssh gpu5090 "cd $R && (setsid nohup bash -c '. .venv/bin/activate && TB_DIR=$TB_DIR TB_REVERSE=1 TB_SCREEN_WORKERS=8 python eval/threeblock_retrain.py screen > results/program_b/tb_v3_screen_remote.log 2>&1' > /dev/null 2>&1 < /dev/null &)" < /dev/null
( while true; do for n in step3 phase1; do mkdir -p $TB_DIR/screen_$n; rsync -a --ignore-existing --exclude '*.tmp' gpu5090:wksp/stable_lyap_filters/$TB_DIR/screen_$n/ $TB_DIR/screen_$n/ 2>/dev/null; done; sleep 300; done ) &
SYNC=$!
TB_SCREEN_WORKERS=16 python3 eval/threeblock_retrain.py screen > $L 2>&1 || { kill $SYNC; exit 1; }
kill $SYNC
ssh gpu5090 'pkill -f "threeblock_retrain.py screen"' < /dev/null
rsync -a --relative eval $TB_DIR/selection_step3.json $TB_DIR/selection_phase1.json gpu5090:wksp/stable_lyap_filters/ >> $L 2>&1
ssh gpu5090 "cd $R && . .venv/bin/activate && export TB_DIR=$TB_DIR MUJOCO_GL=egl FAST_TRAIN_Z_ON_CPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TB_WORKERS=4 && for s in data data g modes us hybrid grade; do python eval/threeblock_retrain.py \$s || [ \$s = data ] || exit 1; done" < /dev/null >> $L 2>&1
rsync -a gpu5090:wksp/stable_lyap_filters/results/jenga/threeblock_v3_result.json results/jenga/ >> $L 2>&1
rsync -a --exclude train --exclude val --exclude p1train --exclude 'screen_*' gpu5090:wksp/stable_lyap_filters/$TB_DIR/ $TB_DIR/ >> $L 2>&1
echo "chain done $(date)" >> $L
