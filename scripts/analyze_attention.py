"""Is there a "rhyme head"?

To make the second hemistich rhyme, the model must, when it is about to write
the last word, look back at the last word of the first hemistich.  For every
real held-out couplet, and for every head, this script measures the share of
attention that lands on the first hemistich's final word from three query
positions in the second hemistich:

  predict_rhyme  the token right before the final word (its output predicts
                 the rhyme word's first token)
  last           the couplet's final token (predicts the line break)
  middle         a token halfway through the second hemistich (control: a
                 head that always looks at the previous hemistich's end
                 would score high here too)

Under uniform attention the expected share is (#tokens in that word) /
(#tokens visible), reported as the baseline.

Attention weights show where a head looks, not whether that matters.  The
causal test: zero one head's output (ablation) and measure how much the loss
rises on the rhyme word's first token versus on every other token.

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
    queries = ["predict_rhyme", "last", "middle"]
    share = {q: np.zeros((L, H)) for q in queries}
    uniform = {q: 0.0 for q in queries}
    n = 0
    sep = tok.encode(" /")

    def word_start(ids, j):  # index of the first token of the word containing token j
        while j > 0 and not tok.vocab[ids[j]].startswith(" "):
            j -= 1
        return j

    for i in split["H"]:
        for c in sections[i].couplets:
            ids = tok.encode("\n" + c)
            try:
                s = next(j for j in range(len(ids)) if ids[j : j + len(sep)] == sep)
            except StopIteration:
                continue
            target_word = list(range(word_start(ids, s - 1), s))
            last = len(ids) - 1
            final_start = word_start(ids, last)
            qpos = {"predict_rhyme": final_start - 1, "last": last, "middle": (s + len(sep) + final_start) // 2}
            if not (s + len(sep) < qpos["middle"] < qpos["predict_rhyme"]):
                continue  # second hemistich too short to place all three queries
            att = model(torch.tensor([ids]), return_attn=True)["attn"][0]  # (L,H,T,T)
            for q, pos in qpos.items():
                share[q] += att[:, :, pos, target_word].sum(-1).numpy()
                uniform[q] += len(target_word) / (pos + 1)
            n += 1
    out = {"n_couplets": n, "queries": {}}
    for q in queries:
        m = share[q] / n
        best = np.unravel_index(m.argmax(), m.shape)
        out["queries"][q] = {
            "share_on_first_rhyme_word": m.round(4).tolist(),
            "uniform_baseline": uniform[q] / n,
            "best_head": {"layer": int(best[0]), "head": int(best[1]), "share": float(m[best])},
        }
        print(f"query = {q}   (uniform baseline {uniform[q] / n:.3f})")
        for l in range(L):
            print(f"  layer {l}: " + " ".join(f"{v:.2f}" for v in m[l]))
    out["ablation"] = ablation(model, tok, sections, split, sep, word_start)
    (RES / "attention.json").write_text(json.dumps(out, indent=1))


@torch.no_grad()
def ablation(model, tok, sections, split, sep, word_start) -> dict:
    """Mean NLL on rhyme tokens vs other tokens with each head zero-ablated."""
    L, H = model.cfg.n_layer, model.cfg.n_head
    D = model.cfg.d_model // H
    seqs, rhyme_pos = [], []
    for i in split["H"]:
        for c in sections[i].couplets:
            ids = tok.encode("\n" + c)
            if not any(ids[j : j + len(sep)] == sep for j in range(len(ids))):
                continue
            seqs.append(ids)
            rhyme_pos.append(word_start(ids, len(ids) - 1))  # first token of the final word
    def nll(mask_head=None):
        hook = None
        if mask_head is not None:
            l, h = mask_head
            def pre(mod, args):
                x = args[0].clone()
                x[..., h * D : (h + 1) * D] = 0  # this head's contribution to W_O input
                return (x,)
            hook = model.blocks[l].attn.proj.register_forward_pre_hook(pre)
        r_sum, r_n, o_sum, o_n = 0.0, 0, 0.0, 0
        for ids, rp in zip(seqs, rhyme_pos):
            x = torch.tensor([ids])
            lp = torch.log_softmax(model(x[:, :-1])["logits"][0], -1)
            tok_nll = -lp.gather(-1, x[0, 1:, None]).squeeze(-1)  # position t predicts token t+1
            r_sum += tok_nll[rp - 1].item(); r_n += 1
            o_sum += tok_nll.sum().item() - tok_nll[rp - 1].item(); o_n += len(tok_nll) - 1
        if hook:
            hook.remove()
        return r_sum / r_n, o_sum / o_n
    base_r, base_o = nll()
    rows = []
    for l in range(L):
        for h in range(H):
            r, o = nll((l, h))
            rows.append({"layer": l, "head": h, "rhyme_nll_increase": r - base_r, "other_nll_increase": o - base_o})
    rows.sort(key=lambda d: -d["rhyme_nll_increase"])
    print(f"ablation: baseline NLL rhyme token {base_r:.3f}, other tokens {base_o:.3f}")
    for d in rows[:6]:
        print(f"  L{d['layer']}h{d['head']}: rhyme +{d['rhyme_nll_increase']:.3f}  other +{d['other_nll_increase']:.3f}")
    return {"baseline_rhyme_nll": base_r, "baseline_other_nll": base_o, "n_couplets": len(seqs), "heads": rows}


if __name__ == "__main__":
    main()
