# overnight: undo-test labels for a possible IRRM follow-up (not a registered test; labels only, no training)
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
echo "labels start $(date)"
for NL in 8 2; do
  for i in 0 1; do IRRM_NL=$NL SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/irrm2.py labels > logs/irrm2_labels_nl${NL}_$i.out 2>&1 & done
  wait
  IRRM_NL=$NL PYTHONUNBUFFERED=1 $PY eval/irrm2.py held > logs/irrm2_held_nl$NL.out 2>&1
  echo "nl$NL labels done $(date)"
done
