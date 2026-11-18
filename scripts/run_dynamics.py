"""When does memorisation appear?  Membership signals across training.

The final target model has seen every token ~28 times, and at that point every
trap is trivially detectable (results/mia.json).  Real LLMs are trained for
about one pass over their data, where Meeus et al. (2024) found short traps
hard to detect.  To connect the two regimes, the target recipe is re-run with
weight snapshots at 0.5, 1, 2, 4, 8 and 16 epochs (scripts/train_all.sh,
run "dynamics") and every snapshot is attacked here.

A trap inserted r times has been seen about r x epochs times by a snapshot,
which is the x-axis of the main figure.  Caveat: an early snapshot is not the
same as a model *trained* for that many epochs (its learning rate has not
decayed yet), so this measures memorisation along one trajectory.

    python scripts/run_dynamics.py       (~10 min)

Writes results/dynamics.json.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minigpt.data import ROOT, eval_windows, load_sections  # noqa: E402
from minigpt.model import load_model  # noqa: E402
from minigpt.privacy.attacks import scores, token_stats  # noqa: E402
from minigpt.privacy.metrics import auc  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402
from minigpt.train import mean_loss  # noqa: E402
from run_mia import canary_experiment  # noqa: E402

ART, RES = ROOT / "artifacts", ROOT / "results"
N_NATURAL = 1500


def main() -> None:
    torch.set_num_threads(4)
    t0 = time.time()
    tok = BPETokenizer.load(ART / "tokenizer.json")
    ref = load_model(str(ART / "reference.pt"))
    sections = load_sections()
    split = json.loads((RES / "split.json").read_text())
    traps = json.loads((RES / "traps.json").read_text(encoding="utf-8"))
    canaries = json.loads((RES / "canaries.json").read_text(encoding="utf-8"))
    train_info = json.loads((RES / "train_dynamics.json").read_text())
    spe = train_info["train_config"]["extra"]["steps_per_epoch"]

    rng = random.Random(0)
    mem = [c for _, c in rng.sample([(i, c) for i in split["A"] for c in sections[i].couplets], N_NATURAL)]
    in_a = {c for i in split["A"] for c in sections[i].couplets}
    non = [c for i in split["H"] for c in sections[i].couplets if c not in in_a][:N_NATURAL]
    trap_texts = [t["text"] for t in traps]
    texts = trap_texts + mem + non
    ref_stats = token_stats(ref, tok, texts)
    val_w = eval_windows(np.load(ART / "val.npy"), 128)
    tr_w = eval_windows(np.load(ART / "target_train.npy"), 128)
    g = torch.Generator().manual_seed(0)
    tr_w = tr_w[torch.randperm(len(tr_w), generator=g)[:200]]

    snaps = sorted(ART.glob("snapshots/dynamics_step*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    snaps.append(ART / "dynamics.pt")  # best-validation checkpoint, as for the target
    rows = []
    for path in snaps:
        model = load_model(str(path))
        step = torch.load(path, weights_only=False)["step"]
        st = token_stats(model, tok, texts)
        sc = scores(texts, st, ref_stats)
        row = {"step": step, "epochs": step / spe, "val_loss": mean_loss(model, val_w),
               "train_loss": mean_loss(model, tr_w), "traps": [], "natural": {}}
        nt = len(traps)
        for a in ("loss", "ratio"):
            m = sc[a][nt : nt + len(mem)]
            n = sc[a][nt + len(mem) :]
            row["natural"][a] = auc(m, n)
        for L in sorted({t["n_couplets"] for t in traps}):
            non_idx = [i for i, t in enumerate(traps) if t["reps"] == 0 and t["n_couplets"] == L]
            for kind in ("exact", "fuzzy"):
                for r in sorted({t["reps"] for t in traps if t["reps"] > 0}):
                    idx = [i for i, t in enumerate(traps) if t["n_couplets"] == L and t["kind"] == kind and t["reps"] == r]
                    if idx:
                        row["traps"].append({"n_couplets": L, "kind": kind, "reps": r,
                                             "exposures": r * step / spe,
                                             **{a: auc(sc[a][idx], sc[a][non_idx]) for a in ("loss", "ratio")}})
        row["canaries"] = [{"reps": c["reps"], "exposure": c["target_exposure"]}
                           for c in canary_experiment(canaries, model, ref, tok)]
        rows.append(row)
        cells = {(c["n_couplets"], c["kind"], c["reps"]): c["ratio"] for c in row["traps"]}
        print(f"epoch {row['epochs']:5.1f} | val {row['val_loss']:.3f} train {row['train_loss']:.3f} | "
              f"natural ratio AUC {row['natural']['ratio']:.3f} | 1-couplet exact traps r=1,2,4: "
              f"{cells[(1, 'exact', 1)]:.2f} {cells[(1, 'exact', 2)]:.2f} {cells[(1, 'exact', 4)]:.2f} | "
              f"4-couplet r=1: {cells[(4, 'exact', 1)]:.2f}", flush=True)
    (RES / "dynamics.json").write_text(json.dumps({"steps_per_epoch": spe, "snapshots": rows}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
