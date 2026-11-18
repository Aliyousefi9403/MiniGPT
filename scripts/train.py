"""Train one model.

    python scripts/train.py --run target      # A + traps   (the model under attack)
    python scripts/train.py --run reference   # B only      (the attacker's reference)
    python scripts/train.py --run pos_sinusoidal --corpus target_train --pos sinusoidal --steps 2000

Checkpoints and training curves go to artifacts/<run>.pt(.json); a copy of the
curve is kept in results/train_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigpt.data import ROOT  # noqa: E402
from minigpt.model import GPTConfig  # noqa: E402
from minigpt.train import TrainConfig, train_model  # noqa: E402

ART, RES = ROOT / "artifacts", ROOT / "results"
DEFAULT_CORPUS = {"target": "target_train", "reference": "ref_train"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpus", default=None, help="artifacts/<corpus>.npy (default from --run)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pos", default="learned")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot-epochs", default="", help="e.g. 0.5,1,2,4,8,16: also save weights at these epochs")
    args = ap.parse_args()

    corpus = args.corpus or DEFAULT_CORPUS[args.run]
    train = np.load(ART / f"{corpus}.npy")
    val = np.load(ART / "val.npy")
    vocab = len(json.loads((ART / "tokenizer.json").read_text(encoding="utf-8"))["vocab"])
    mcfg = GPTConfig(
        vocab_size=vocab,
        block_size=args.block,
        n_layer=args.layers,
        n_head=args.heads,
        d_model=args.width,
        dropout=args.dropout,
        pos_encoding=args.pos,
    )
    steps_per_epoch = len(train) / (args.batch * args.block)
    snaps = tuple(sorted({round(float(e) * steps_per_epoch) for e in args.snapshot_epochs.split(",") if e}))
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch, lr=args.lr, seed=args.seed, snapshots=snaps,
                       extra={"corpus": corpus, "steps_per_epoch": steps_per_epoch})
    out = ART / f"{args.run}.pt"
    train_model(mcfg, tcfg, train, val, out, log=lambda s: print(s, flush=True))
    RES.mkdir(exist_ok=True)
    shutil.copy(str(out) + ".json", RES / f"train_{args.run}.json")


if __name__ == "__main__":
    main()
