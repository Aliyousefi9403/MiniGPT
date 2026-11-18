"""Byte-pair encoding (BPE) tokenizer, written from scratch.

Why BPE and not characters?  A character-level model must spend attention and
depth just to assemble letters into words; BPE hands it frequent words and
word pieces as single symbols, so the same context window covers ~3x more
text.  Why not an off-the-shelf tokenizer?  Persian is badly served by English-
centric byte-level vocabularies (a single Persian letter is 2 UTF-8 bytes and
rarely merged), and the point of this project is to build every part.

Design choices (see docs/DESIGN.md for the reasoning):

* **Characters, not bytes, are the base alphabet.**  Every Persian letter is
  one base symbol, so no token can ever split a letter in half.  Unknown
  characters at encode time map to a single ``<unk>`` id.
* **Pre-tokenisation on whitespace.**  Text is first cut into chunks of
  "optional leading space + run of non-space characters"; merges never cross
  a chunk boundary, so tokens never straddle two words.
* **Digits are never merged.**  Each digit (Latin or Persian) is its own
  chunk.  This matters for the Secret-Sharer canary experiment: a 6-digit
  secret is always exactly 6 tokens, so every candidate secret has the same
  length and the exposure metric is well defined.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

DIGITS = "0123456789۰۱۲۳۴۵۶۷۸۹"
# Whitespace is listed explicitly (not ``\s``) because Python and JavaScript
# disagree on which exotic Unicode characters count as whitespace; the browser
# tokenizer (web/js/tokenizer.js) uses the identical pattern.
WS = " \t\n\r\f\v"
# leading-space + word (no digits) | a single digit | any single whitespace char
_CHUNK = re.compile(rf" ?[^{WS}{DIGITS}]+|[{DIGITS}]|[{WS}]")

UNK = "<unk>"


def pretokenize(text: str) -> list[str]:
    chunks = _CHUNK.findall(text)
    assert "".join(chunks) == text, "pre-tokeniser must be lossless"
    return chunks


class BPETokenizer:
    def __init__(self, vocab: list[str], merges: list[tuple[str, str]]):
        self.vocab = vocab  # id -> string
        self.merges = merges  # in priority order
        self.stoi = {s: i for i, s in enumerate(vocab)}
        self.ranks = {pair: r for r, pair in enumerate(merges)}
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ train
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Learn ``vocab_size - len(alphabet) - 1`` merges from ``text``.

        Classic algorithm (Sennrich et al., 2016): repeatedly find the most
        frequent adjacent pair of symbols and fuse it into a new symbol.  We
        work on the multiset of distinct chunks (each with a count), so one
        pass costs O(total symbols in distinct chunks), not O(len(text)).
        """
        chunk_counts = Counter(pretokenize(text))
        # digits are always in the alphabet, even if the training text has none
        # (the Shahnameh has none: without this, every canary secret would
        # encode to the same <unk> tokens)
        alphabet = sorted({ch for chunk in chunk_counts for ch in chunk} | set(DIGITS))
        vocab = [UNK] + alphabet
        words = {w: list(w) for w in chunk_counts}  # chunk -> current symbols
        merges: list[tuple[str, str]] = []
        n_merges = vocab_size - len(vocab)
        if n_merges < 0:
            raise ValueError(f"vocab_size {vocab_size} < alphabet size {len(vocab)}")

        # pair statistics, maintained incrementally
        pair_counts: Counter = Counter()
        for w, syms in words.items():
            if w[0] in DIGITS:
                continue
            c = chunk_counts[w]
            for a, b in zip(syms, syms[1:]):
                pair_counts[(a, b)] += c

        for step in range(n_merges):
            if not pair_counts:
                break
            (a, b), freq = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if freq < 2:
                break
            new = a + b
            merges.append((a, b))
            vocab.append(new)
            # update only the chunks that contain the pair
            for w, syms in words.items():
                if len(syms) < 2 or a not in syms:
                    continue
                i, changed = 0, False
                out: list[str] = []
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(new)
                        i += 2
                        changed = True
                    else:
                        out.append(syms[i])
                        i += 1
                if changed:
                    c = chunk_counts[w]
                    for p in zip(syms, syms[1:]):
                        pair_counts[p] -= c
                        if pair_counts[p] <= 0:
                            del pair_counts[p]
                    for p in zip(out, out[1:]):
                        pair_counts[p] += c
                    words[w] = out
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{n_merges}: {new!r} (freq {freq})")
        return cls(vocab, merges)

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        syms = list(chunk)
        while len(syms) > 1:
            # apply the highest-priority (earliest learned) merge present
            best = min(
                ((self.ranks.get((x, y), 1 << 30), i) for i, (x, y) in enumerate(zip(syms, syms[1:]))),
            )
            rank, i = best
            if rank == 1 << 30:
                break
            syms[i : i + 2] = [syms[i] + syms[i + 1]]
        unk = self.stoi[UNK]
        ids = [self.stoi.get(s, unk) for s in syms]
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in pretokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    def __len__(self) -> int:
        return len(self.vocab)

    # -------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"vocab": self.vocab, "merges": self.merges}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])
