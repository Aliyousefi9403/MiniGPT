"""Is there a "rhyme head"?

To make the second hemistich rhyme, the model must, while writing its last
word, look back at the last word of the first hemistich.  For every real
held-out couplet this script takes the query position = the final token of the
couplet's second half (the token predicting "\\n" ... and, one step earlier,
the rhyme itself) and measures, for every head, the share of attention that
lands on the final word of the first hemistich.  Under uniform attention the
expected share is (#tokens in that word) / (#tokens visible).

    python scripts/analyze_attention.py      (~1 min)

Writes results/attention.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigpt.data import ROOT, load_sections  # noqa: E402
from minigpt.model import load_model  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402

ART, RES = ROOT / "artifacts", ROOT / "results"


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(4)
    tok = BPETokenizer.load(ART / "tokenizer.json")
    model = load_model(str(ART / "target.pt"))
    sections = load_sections()
    split = json.loads((RES / "split.json").read_text())
    L, H = model.cfg.n_layer, model.cfg.n_head
    share = np.zeros((L, H))
    uniform = 0.0
    n = 0
    sep = tok.encode(" /")
    for i in split["H"]:
        for c in sections[i].couplets:
            ids = tok.encode("\n" + c)
            # locate the separator " /" and the word just before it
            try:
                s = next(j for j in range(len(ids)) if ids[j : j + len(sep)] == sep)
            except StopIteration:
                continue
            w_start = s - 1
            while w_start > 0 and not tok.vocab[ids[w_start]].startswith(" "):
                w_start -= 1
            target_word = list(range(w_start, s))
            q = len(ids) - 1  # last token of the second hemistich
            att = model(torch.tensor([ids]), return_attn=True)["attn"][0, :, :, q, :]  # (L,H,T)
            share += att[:, :, target_word].sum(-1).numpy()
            uniform += len(target_word) / (q + 1)
            n += 1
    share /= n
    uniform /= n
    best = np.unravel_index(share.argmax(), share.shape)
    out = {
        "n_couplets": n,
        "share_on_first_rhyme_word": share.round(4).tolist(),
        "uniform_baseline": uniform,
        "best_head": {"layer": int(best[0]), "head": int(best[1]), "share": float(share[best])},
    }
    (RES / "attention.json").write_text(json.dumps(out, indent=1))
    print(f"uniform baseline {uniform:.3f}")
    for l in range(L):
        print(f"layer {l}: " + " ".join(f"{v:.2f}" for v in share[l]))
    print("best:", out["best_head"])


if __name__ == "__main__":
    main()
