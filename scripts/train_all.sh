#!/usr/bin/env bash
# Train every model, skipping finished runs and resuming interrupted ones.
set -euo pipefail
cd "$(dirname "$0")/.."
run() {  # run <name> [extra args...]
  local name=$1; shift
  if [ -f "results/train_${name}.json" ] && [ -f "artifacts/${name}.pt" ]; then echo "skip ${name}"; return; fi
  python scripts/train.py --run "$name" "$@" >> "artifacts/${name}.log" 2>&1
}
run target    --steps 4000
run reference --steps 4000
# the target recipe again, saving snapshots to follow memorisation during training
run dynamics --corpus target_train --steps 4000 --snapshot-epochs 0.5,1,2,4,8,16
for pos in learned sinusoidal none; do
  run "pos_${pos}" --corpus ref_train --pos "$pos" --steps 2000 $( [ "$pos" = sinusoidal ] && echo --scale-embeddings )
done
# the first sinusoidal run, without embedding scaling (kept as a documented failure)
run pos_sinusoidal_unscaled --corpus ref_train --pos sinusoidal --steps 2000
echo "all training done"
