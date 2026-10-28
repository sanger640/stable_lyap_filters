sleep 1800
for i in $(seq 1 18); do
  out=$(timeout 25 ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none -v ak-fir true 2>&1)
  if echo "$out" | grep -qiE "partial success|Authenticated to"; then echo "$(date) KEY ACCEPTED"; exit 0; fi
  if echo "$out" | grep -qi "can continue"; then echo "$(date) reachable, key not yet accepted"; else echo "$(date) unreachable"; fi
  sleep 600
done
