#!/usr/bin/env bash
# Reproduce every number in docs/RESULTS.md from scratch.
# Timings measured on 4 CPU cores (Intel Xeon, 2.1 GHz); no GPU needed.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/prepare.py                                   # split, tokenizer, traps    ~10 s
scripts/train_all.sh                                        # target ~40 min, reference ~40 min,
                                                            # 3 positional-encoding ablations ~20 min each
                                                            # (resumes automatically if interrupted)
python scripts/evaluate_lm.py                               # bpc, rhyme, copying        ~10 min
python scripts/run_mia.py                                   # all privacy experiments    ~5 min
python scripts/run_dynamics.py                              # MIA across training        ~10 min
python scripts/analyze_attention.py                         # rhyme heads + ablation     ~2 min
python scripts/train_sae.py                                 # sparse autoencoder         ~10 min
python scripts/make_figures.py                              # figures/                   ~10 s
python scripts/export_web.py                                # web/model/                 ~5 s
