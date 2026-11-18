#!/usr/bin/env bash
# Reproduce every number in docs/RESULTS.md from scratch.
# Timings measured on 4 CPU cores (Intel Xeon, 2.1 GHz); no GPU needed.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare.py                                   # split, tokenizer, traps    ~10 s
python scripts/train.py --run target    --steps 4000        # A + traps                  ~40 min
python scripts/train.py --run reference --steps 4000        # B only                     ~40 min
for pos in learned sinusoidal none; do                      # positional-encoding ablation
  python scripts/train.py --run "pos_${pos}" --corpus ref_train --pos "$pos" --steps 2000   # ~20 min each
done
python scripts/evaluate_lm.py                               # bpc, rhyme, copying        ~10 min
python scripts/run_mia.py                                   # all privacy experiments    ~10 min
python scripts/train_sae.py                                 # sparse autoencoder         ~10 min
python scripts/make_figures.py                              # figures/                   ~10 s
python scripts/export_web.py                                # web/model/                 ~5 s
