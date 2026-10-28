#!/bin/bash
# Archive finished result folders to Rorqual project space as tars; delete locally only after verification.
cd /home/corey/wksp/stable_lyap_filters/results
DST=/project/def-soojeon/ali313/archive_stable_lyap
LOG=/home/corey/wksp/stable_lyap_filters/results/archive_to_rorqual.log
for d in libero/l2_data program_b/visual_panel3 program_b/visual_pipeline jenga/bulk_data program_b/threeblock_v2 program_b/phase1 program_b/phase1b program_b/phase1f program_b/phase2; do
  name=$(echo $d | tr / _).tar
  nloc=$(tar cf - $d | tee >(ssh rorqual "cat > $DST/$name") | tar tf - | wc -l)
  sleep 5
  nrem=$(ssh rorqual "tar tf $DST/$name | wc -l")
  if [ "$nloc" -gt 0 ] && [ "$nloc" = "$nrem" ]; then
    echo "$(date +%T) OK $d entries=$nloc -> $DST/$name ; deleting local" >> $LOG; rm -rf "$d"
  else
    echo "$(date +%T) MISMATCH $d local=$nloc remote=$nrem ; kept local" >> $LOG
  fi
done
echo "$(date +%T) ARCHIVE DONE" >> $LOG
