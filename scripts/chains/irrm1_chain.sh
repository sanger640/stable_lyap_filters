# IRRM stage 1 (results/jenga/irrm_protocol.json), after the N1M chain (one GPU job at a time)
cd /home/corey/wksp/stable_lyap_filters
PY=/home/corey/wksp/dp_venv/bin/python
while kill -0 2174036 2>/dev/null; do sleep 60; done
echo "irrm1 start $(date)"
for i in 0 1; do SHARD=$i/2 PYTHONUNBUFFERED=1 $PY eval/irrm1.py run > logs/irrm1_$i.out 2>&1 & done
wait
$PY eval/irrm1.py grade > results/jenga/irrm1_grade.log 2>&1
echo "irrm1 graded $(date)"
