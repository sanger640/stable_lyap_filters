#!/bin/bash
# After Rorqual queue empties: fetch Nibi (14400-14479) + Rorqual episodes, check completeness, build jsonl, audit, analyse.
cd /home/corey/wksp/stable_lyap_filters
P=results/program_b/policy_rollouts
for a in record cbfmsoil cbfh2 cbfg cbfg2; do
  mkdir -p $P/paired_${a}_s4b
  rsync -a --exclude '*.tmp' nibi:projects/def-soojeon/ali313/slf/repo/$P/paired_${a}_s4b/ $P/paired_${a}_s4b/ || { echo "nibi rsync failed $a"; exit 1; }
  rsync -a --exclude '*.tmp' rorqual:/scratch/ali313/slf/repo/$P/paired_${a}_s4b/ $P/paired_${a}_s4b/ || { echo "rorqual rsync failed $a"; exit 1; }
  n=$(ls $P/paired_${a}_s4b/*.json | wc -l); echo "$a episodes $n"
  python3 -c "
import json,glob
fs=sorted(glob.glob('$P/paired_${a}_s4b/ep*.json'), key=lambda f:int(f.split('ep')[-1][:-5]))
open('results/jenga/paired_${a}_s4b.jsonl','w').write(''.join(json.dumps(json.load(open(f)))+'\n' for f in fs))"
done
echo "missing per arm:"; for a in record cbfmsoil cbfh2 cbfg cbfg2; do python3 -c "
import os; m=[e for e in range(14300,14600) if not os.path.exists('$P/paired_${a}_s4b/ep%d.json'%e)]; print('$a', len(m), m[:20])"; done
AUDIT_OUT=results/jenga/outcome_audit_s4b.json python3 eval/outcome_audit.py runs paired_record_s4b paired_cbfmsoil_s4b paired_cbfh2_s4b paired_cbfg_s4b paired_cbfg2_s4b > results/jenga/outcome_audit_s4b.log 2>&1 || { echo "audit failed"; tail results/jenga/outcome_audit_s4b.log; exit 1; }
grep '^paired' results/jenga/outcome_audit_s4b.log
python3 eval/stage4_analysis.py s4b > results/jenga/stage4b_analysis.log && echo "ANALYSIS DONE"
