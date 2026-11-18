"""Language-model quality: does the transformer actually beat simple baselines?

    python scripts/evaluate_lm.py        (~8 min)

1. bits per character (bpc) on the validation split V for
   - character n-gram models of order 1..8 fitted on A (the target's real data)
   - every trained GPT in artifacts/ (target, reference, ablations)
2. verse-form statistics (rhyme, well-formedness, copying) for
   - real held-out couplets (H)                  -> the ceiling
   - the same couplets with hemistichs re-paired -> chance level
   - samples from the target GPT and from the best n-gram

Writes results/lm_eval.json and results/samples.json.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigpt.data import ROOT, load_sections  # noqa: E402
from minigpt.model import load_model  # noqa: E402
from minigpt.ngram import CharNGram  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402
from minigpt.verse import chance_rhyme, verse_stats  # noqa: E402

ART, RES = ROOT / "artifacts", ROOT / "results"
N_SAMPLES = int(__import__("os").environ.get("N_SAMPLES", 300))  # generated couplets per model


@torch.no_grad()
def gpt_bpc(model, ids: list[int], n_chars: int, stride: int = 64) -> float:
    """Sliding-window NLL: every token is scored with >= block-stride tokens of
    context (except at the very start), then converted to bits per character."""
    T = model.cfg.block_size
    x = torch.tensor(ids)
    nll, pos = 0.0, 0
    starts = list(range(0, len(ids) - 1, stride))
    for s in starts:
        e = min(s + T, len(ids) - 1)
        if e <= pos:
            continue
        w = x[s : e + 1][None]
        lp = torch.log_softmax(model(w[:, :-1], fast=True)["logits"], -1)
        tok_lp = lp.gather(-1, w[:, 1:, None]).squeeze()
        first = pos - s  # skip tokens already scored by a previous window
        nll -= tok_lp[first:].sum().item()
        pos = e
    return nll / math.log(2) / n_chars


def gpt_couplets(model, tok, n: int, seed: int, temperature=0.8, top_k=40) -> list[str]:
    """Sample couplets, each from a fresh "start of section" context."""
    g = torch.Generator().manual_seed(seed)
    nl = tok.encode("\n")[0]
    out = []
    while len(out) < n:
        ctx = torch.tensor([tok.encode("\n\n")] * 50)
        gen = model.generate(ctx, 160, temperature=temperature, top_k=top_k, generator=g)
        for row in gen[:, 2:].tolist():
            text = tok.decode(row)
            out.extend([ln for ln in text.split("\n")[:-1] if ln.strip()][:4])  # drop the unfinished last line
    return out[:n]


def main() -> None:
    global ART, RES
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ART), help="directory with target.pt / reference.pt")
    ap.add_argument("--results", default=str(RES))
    args = ap.parse_args()
    ART, RES = Path(args.models), Path(args.results)
    RES.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    t0 = time.time()
    tok = BPETokenizer.load(ART / "tokenizer.json")
    sections = load_sections()
    split = json.loads((RES / "split.json").read_text())
    train_text = "".join(sections[i].text for i in split["A"])
    val_text = "".join(sections[i].text for i in split["V"])
    res: dict = {"bpc": {}}

    print("n-gram baselines ...", flush=True)
    best = (None, 1e9)
    for n in range(1, 9):
        m = CharNGram(n).fit(train_text)
        b = m.bits_per_char(val_text)
        res["bpc"][f"char {n}-gram"] = b
        print(f"  {n}-gram: {b:.3f} bpc", flush=True)
        if b < best[1]:
            best = (m, b)

    val_ids = tok.encode(val_text)
    for ck in sorted(ART.glob("*.pt")):
        model = load_model(str(ck))
        b = gpt_bpc(model, val_ids, len(val_text))
        res["bpc"][f"GPT {ck.stem}"] = b
        print(f"  GPT {ck.stem}: {b:.3f} bpc", flush=True)

    print("verse statistics ...", flush=True)
    train_couplets = {c for i in split["A"] for c in sections[i].couplets}
    train_words = {w for c in train_couplets for w in c.replace(" / ", " ").split()}
    real = [c for i in split["H"] for c in sections[i].couplets]
    target = load_model(str(ART / "target.pt"))
    gpt_lines = gpt_couplets(target, tok, N_SAMPLES, seed=0)
    rng = random.Random(0)
    ngram_lines = []
    while len(ngram_lines) < N_SAMPLES:
        s = best[0].sample(rng, 400, context="\n\n")
        ngram_lines.extend([ln for ln in s.split("\n")[:-1] if ln.strip()][:4])
    ngram_lines = ngram_lines[:N_SAMPLES]
    res["verse"] = {
        "real held-out (H)": verse_stats(real, train_couplets, train_words),
        "chance (re-paired H)": {"rhyme@1": chance_rhyme(real, 1), "rhyme@2": chance_rhyme(real, 2)},
        "GPT target (T=0.8, top-k 40)": verse_stats(gpt_lines, train_couplets, train_words),
        f"best n-gram (order {best[0].order})": verse_stats(ngram_lines, train_couplets, train_words),
    }
    for k, v in res["verse"].items():
        print(f"  {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in v.items() if isinstance(b, float)))
    res["minutes"] = (time.time() - t0) / 60
    (RES / "lm_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (RES / "samples.json").write_text(
        json.dumps({"gpt": gpt_lines[:60], "ngram": ngram_lines[:60]}, indent=1, ensure_ascii=False)
    )
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
