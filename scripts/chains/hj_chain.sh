cd /home/corey/wksp/stable_lyap_filters
n() { timeout 30 ssh -o BatchMode=yes fir "squeue -u ali313 -h -n hjd,hjd3,mcm3 | wc -l" 2>/dev/null || echo 1; }
until [ "$(n)" = "0" ]; do sleep 120; done
R=fir:/scratch/ali313/slf/repo/results/jenga
rsync -a --exclude '*.tmp.npz' $R/hj2/data/ results/jenga/hj2/data/
rsync -a $R/deep_dive/m3/ results/jenga/deep_dive/m3/
for d in results/jenga/hj2/data/*; do echo $d $(ls $d | wc -l); done; ls results/jenga/deep_dive/m3 | wc -l
PY=/home/corey/wksp/dp_venv/bin/python
for K in negG_max neg_log_sens_4; do
  HJ_MARGIN=$K $PY eval/hj_value2.py train paired_record_s4 paired_cbfg_s4 paired_record_h2h2 paired_cbfg_h2h2 > results/jenga/hj2/train_$K.log 2>&1
  grep saved results/jenga/hj2/train_$K.log
  HJ_MARGIN=$K $PY eval/hj_value2.py gate2 > results/jenga/hj2/gateB2_$K.log 2>&1
  grep -v "Exception ignored\|TypeError\|File \"/home\|^Traceback" results/jenga/hj2/gateB2_$K.log | tail -30
done
echo hj done $(date)
