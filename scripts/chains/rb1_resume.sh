while kill -0 1174184 2>/dev/null; do sleep 60; done
for p in $(cat scripts/chains/rb1_pids.txt); do kill -CONT $p; done
echo resumed2 $(date)
