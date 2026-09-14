#!/usr/bin/env bash
# Bundle everything the Jenga work needs that is NOT in git, for transfer to another machine.
#
#   ./scripts/bundle_jenga.sh            # core bundle (~1.3 GB): offline monitor work
#   ./scripts/bundle_jenga.sh --live     # adds the diffusion policy (~2.1 GB): live rollout
#
# On the far side: git clone the repo, then untar this over it.
set -euo pipefail
W=${WKSP:-$HOME/wksp}
LIVE=0; OUT=jenga_bundle.tar
case "${1:-}" in
  --live) LIVE=1; OUT=jenga_bundle_live.tar ;;
  "")     ;;
  *)      OUT=$1 ;;
esac

STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/data/jenga" "$STAGE/vendor"

echo "staging world model (stripped, 136 MB) ..."
cp data/jenga/world_model.pt "$STAGE/data/jenga/"
cp data/jenga/labels_noise*.json "$STAGE/data/jenga/"

echo "staging eval LMDB (1.1 GB real, 20 GB apparent -- it is sparse) ..."
cp -r --sparse=always "$W/panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb" \
   "$STAGE/data/jenga/"
rm -f "$STAGE/data/jenga/jenga_single_100.lmdb/lock.mdb"   # regenerated on open

echo "staging panda_express sim (36 MB) ..."
# sim.py loads franka_emika_panda/panda_jenga_setup.xml. The MuJoCo assets live THERE --
# tasks/jenga_mujoco/ is 3.6 GB of episodes and a second LMDB with no scene files in it at all.
tar -C "$W" -cf "$STAGE/vendor/panda_express_sim.tar" \
    panda_express/sim.py panda_express/franka_emika_panda/ 2>/dev/null || \
    echo "  (sim assets not found -- skipping, only needed for J6)"

if [[ $LIVE == 1 ]]; then
  echo "staging diffusion policy (815 MB) ..."
  P="$W/diffusion_policy/data/outputs/2026.08.11/21.59.04_train_franka_dual_jenga_jenga_image_dual"
  mkdir -p "$STAGE/data/jenga/policy"
  cp "$P/checkpoints/latest.ckpt" "$STAGE/data/jenga/policy/"
  cp -r "$P/.hydra" "$STAGE/data/jenga/policy/" 2>/dev/null || true
fi

for req in data/jenga/world_model.pt data/jenga/labels_noise100.json; do
  [[ -e "$STAGE/$req" ]] || { echo "MISSING from bundle: $req" >&2; exit 1; }
done
echo "contents:"
du -sh "$STAGE"/data/jenga/* "$STAGE"/vendor/* 2>/dev/null | sed 's/^/   /'
# --sparse is REQUIRED: LMDB preallocates its map file, so data.mdb is 20 GB apparent against
# 1.1 GB of real blocks. Without -S, tar faithfully writes 19 GB of zeros and the bundle is 21 GB.
tar -C "$STAGE" -Scf "$OUT" .
echo
echo "-> $OUT  ($(du -h "$OUT" | cut -f1))"
echo
echo "On the target machine:"
echo "   git clone git@github.com:sanger640/stable_lyap_filters.git && cd stable_lyap_filters"
echo "   tar -Sxf /path/to/$OUT        # -S restores the sparse LMDB"
echo "   python -m pytest tests/ -q          # 23 tests"
echo "   python eval/jenga_basins.py --lmdb data/jenga/jenga_single_100.lmdb \\"
echo "                               --labels data/jenga/labels_noise100.json"
