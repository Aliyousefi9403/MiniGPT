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
for pos in learned sinusoidal none; do
  run "pos_${pos}" --corpus ref_train --pos "$pos" --steps 2000
done
echo "all training done"
