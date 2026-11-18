"""Measuring whether generated text is *Shahnameh-shaped*.

Bits-per-character says how well a model predicts text, not whether its own
samples look like the poem.  The Shahnameh has a rigid form that can be
checked mechanically:

* every line is a couplet of two hemistichs (here separated by " / "), each
  typically 4-7 words, in the mutaqarib metre;
* the two hemistichs of a couplet **rhyme** with each other (the masnavi form:
  aa, bb, cc, ...), e.g.  کنج / ترنج  (kunj / turunj).

Persian script omits most short vowels, so rhyme is approximated on the written
form: ``rhyme@k`` = the two hemistichs end in the same last k letters.  To know
what "chance" looks like, the same statistic is computed after pairing every
first hemistich with the second hemistich of a *random other* couplet.
"""

from __future__ import annotations

import random

_NORMALISE = str.maketrans({"ي": "ی", "ك": "ک", "ة": "ه", "‌": "", "ـ": ""})
_STRIP = " .,:;!?؟،«»\"'()[]-"


def normalise(s: str) -> str:
    return s.translate(_NORMALISE).strip(_STRIP)


def split_couplet(line: str) -> tuple[str, str] | None:
    parts = line.split(" / ")
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return None
    return normalise(parts[0]), normalise(parts[1])


def well_formed(line: str, min_words: int = 3, max_words: int = 9) -> bool:
    cp = split_couplet(line)
    return cp is not None and all(min_words <= len(h.split()) <= max_words for h in cp)


def rhymes(a: str, b: str, k: int) -> bool:
    a, b = a.replace(" ", ""), b.replace(" ", "")
    return len(a) >= k and len(b) >= k and a[-k:] == b[-k:]


def verse_stats(lines: list[str], train_couplets: set[str] | None = None, train_words: set[str] | None = None) -> dict:
    """Form statistics over a list of generated (or real) lines."""
    lines = [ln for ln in lines if ln.strip()]
    cps = [split_couplet(ln) for ln in lines]
    good = [c for c, ln in zip(cps, lines) if c is not None and well_formed(ln)]
    n = max(len(good), 1)
    out = {
        "n_lines": len(lines),
        "well_formed": len(good) / max(len(lines), 1),
        "rhyme@1": sum(rhymes(a, b, 1) for a, b in good) / n,
        "rhyme@2": sum(rhymes(a, b, 2) for a, b in good) / n,
        "same_last_word": sum(a.split()[-1] == b.split()[-1] for a, b in good) / n,
    }
    if train_couplets is not None:
        out["verbatim_couplet_copy"] = sum(ln.strip() in train_couplets for ln in lines) / max(len(lines), 1)
    if train_words is not None:
        words = [w for a, b in good for w in (a + " " + b).split()]
        out["novel_word_rate"] = sum(w not in train_words for w in words) / max(len(words), 1)
    return out


def chance_rhyme(lines: list[str], k: int, seed: int = 0) -> float:
    """rhyme@k after re-pairing first and second hemistichs at random."""
    cps = [c for c in (split_couplet(ln) for ln in lines) if c is not None]
    seconds = [b for _, b in cps]
    random.Random(seed).shuffle(seconds)
    return sum(rhymes(a, b, k) for (a, _), b in zip(cps, seconds)) / max(len(cps), 1)
