"""Character n-gram language models: the baselines the transformer must beat.

Interpolated Witten-Bell smoothing (Witten & Bell, 1991):

    P(c | h) = ( C(h c) + N1+(h .) * P(c | h') ) / ( C(h) + N1+(h .) )

where h' is h with its oldest character dropped and N1+(h .) is the number of
distinct characters ever seen after h.  The more varied the continuations of a
context, the more mass is passed down to the shorter context.  The recursion
bottoms out in an add-one unigram over the alphabet, so every probability is
non-zero and every distribution sums to 1 (checked in tests/test_ngram.py).

Comparisons with the GPT are made in **bits per character**, which does not
depend on the tokenizer.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict


class CharNGram:
    def __init__(self, order: int):
        self.order = order  # n: predicts a char from the previous n-1 chars

    def fit(self, text: str) -> "CharNGram":
        self.alphabet = sorted(set(text))
        self.counts: list[dict[str, Counter]] = [defaultdict(Counter) for _ in range(self.order)]
        for i, c in enumerate(text):
            for n in range(self.order):  # context length n
                if i - n < 0:
                    break
                self.counts[n][text[i - n : i]][c] += 1
        self.totals = [{h: sum(cnt.values()) for h, cnt in lvl.items()} for lvl in self.counts]
        self.uniq = [{h: len(cnt) for h, cnt in lvl.items()} for lvl in self.counts]
        uni = self.counts[0][""]
        V = len(self.alphabet) + 1  # +1: an unseen-character bucket
        self.p0 = {c: (uni[c] + 1) / (sum(uni.values()) + V) for c in self.alphabet}
        self.p_unseen = 1 / (sum(uni.values()) + V)
        return self

    def prob(self, history: str, c: str) -> float:
        p = self.p0.get(c, self.p_unseen)
        for n in range(1, self.order):
            if len(history) < n:
                break
            h = history[-n:]
            tot = self.totals[n].get(h)
            if tot is None:
                break  # longer contexts are unseen too
            u = self.uniq[n][h]
            p = (self.counts[n][h].get(c, 0) + u * p) / (tot + u)
        return p

    def bits_per_char(self, text: str, context: str = "") -> float:
        full = context + text
        k = len(context)
        nll = 0.0
        for i in range(k, len(full)):
            nll -= math.log2(self.prob(full[max(0, i - self.order + 1) : i], full[i]))
        return nll / len(text)

    def sample(self, rng: random.Random, n_chars: int, context: str = "\n", stop: str | None = None) -> str:
        out = context
        for _ in range(n_chars):
            h = out[-(self.order - 1) :] if self.order > 1 else ""
            ps = [self.prob(h, c) for c in self.alphabet]
            c = rng.choices(self.alphabet, weights=ps)[0]
            out += c
            if stop is not None and out.endswith(stop):
                break
        return out[len(context) :]
