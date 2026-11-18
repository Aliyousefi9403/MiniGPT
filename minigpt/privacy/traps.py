"""Copyright traps and Secret-Sharer canaries.

**Copyright traps** (Meeus, Shilov, Faysse & de Montjoye, ICML 2024) are
synthetic sequences a content owner injects into their own documents so that,
later, they can test whether a model was trained on those documents.  Here a
trap is a run of pseudo-couplets sampled from a word-level Markov chain fitted
on the corpus: Shahnameh vocabulary, plausible local word order, no meaning
(and no rhyme).  Because the trap text appears nowhere in the real epic, any
membership signal on it can only come from its insertion.

The experiment varies three factors:

* ``reps``   how many times the trap is inserted (1, 2, 4, 8, 16)
* ``length`` 1 couplet (~20 tokens) or 4 couplets (~80 tokens)
* ``kind``   ``exact``: identical copies;  ``fuzzy``: each copy has a fraction
             of its words replaced at random, in the spirit of "mosaic memory"
             (Shilov, Meeus & de Montjoye, 2024): the original sequence is never
             itself seen more than once, only near-duplicates of it.

For every trap that is inserted ("member") there is a matched pool of traps
from the same generator that are *never* inserted ("non-members").  Members and
non-members are i.i.d. before insertion, so a perfectly random detector scores
AUC 0.5 and any excess is caused by training.

**Canaries** (Carlini et al., "The Secret Sharer", USENIX Security 2019) are a
fixed template with a random 5-digit secret.  Their *exposure* measures how
much more likely the model finds the true secret than the other 99,999.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
CANARY_TEMPLATE = "رمز گنج {hero} را بدان "  # "Know the code of <hero>'s treasure: "
CANARY_DIGITS = 5
# each canary gets its own hero, so inserted secrets never share a prefix and
# compete with each other for probability mass
HEROES = ["کیخسرو", "رستم", "سهراب", "سیاوش", "اسفندیار", "گودرز", "گیو", "بیژن", "فریدون",
          "منوچهر", "جمشید", "زال", "تهمینه", "منیژه", "رودابه", "کاوه", "هوشنگ", "تهمورث"]


class WordMarkov:
    """Word-level bigram chain over hemistichs (start/end aware)."""

    BOS, EOS = "<s>", "</s>"

    def __init__(self, couplets: list[str]):
        self.next: dict[str, Counter] = defaultdict(Counter)
        self.vocab: Counter = Counter()
        for c in couplets:
            for half in c.split(" / "):
                words = half.split()
                self.vocab.update(words)
                for a, b in zip([self.BOS] + words, words + [self.EOS]):
                    self.next[a][b] += 1
        self.words = list(self.vocab)
        self.weights = [self.vocab[w] for w in self.words]

    def hemistich(self, rng: random.Random, min_words=4, max_words=7) -> str:
        for _ in range(100):
            out, w = [], self.BOS
            while len(out) < max_words + 1:
                cands = self.next[w]
                w = rng.choices(list(cands), weights=list(cands.values()))[0]
                if w == self.EOS:
                    break
                out.append(w)
            if min_words <= len(out) <= max_words:
                return " ".join(out)
        return " ".join(out[:max_words])

    def couplet(self, rng: random.Random) -> str:
        return f"{self.hemistich(rng)} / {self.hemistich(rng)}"

    def random_word(self, rng: random.Random) -> str:
        return rng.choices(self.words, weights=self.weights)[0]


def fuzz(text: str, frac: float, gen: WordMarkov, rng: random.Random) -> str:
    """Replace ``frac`` of the words (never the " / " separator) with words
    drawn from the corpus unigram distribution."""
    lines = []
    for line in text.split("\n"):
        halves = []
        for half in line.split(" / "):
            words = half.split()
            k = max(1, round(frac * len(words)))
            for i in rng.sample(range(len(words)), k):
                words[i] = gen.random_word(rng)
            halves.append(" ".join(words))
        lines.append(" / ".join(halves))
    return "\n".join(lines)


@dataclass
class Trap:
    trap_id: int
    text: str
    n_couplets: int
    reps: int  # 0 = non-member (never inserted)
    kind: str  # "exact" | "fuzzy" | "none"
    copies: list[str] | None = None  # what was actually inserted

    def to_dict(self) -> dict:
        d = asdict(self)
        d["copies"] = None  # copies are re-derivable; keep the manifest small
        return d


def make_traps(
    gen: WordMarkov,
    seed: int,
    reps_levels=(1, 2, 4, 8, 16),
    lengths=(1, 4),
    n_per_cell: int = 24,
    n_nonmembers: int = 160,
    fuzz_frac: float = 0.2,
    existing: set[str] | None = None,
) -> list[Trap]:
    rng = random.Random(seed)
    existing = existing or set()
    traps: list[Trap] = []
    seen: set[str] = set()

    def fresh(n_couplets: int) -> str:
        while True:
            cps = [gen.couplet(rng) for _ in range(n_couplets)]
            if any(c in existing for c in cps):  # never coincide with a real couplet
                continue
            t = "\n".join(cps)
            if t not in seen:
                seen.add(t)
                return t

    for L in lengths:
        for kind in ("exact", "fuzzy"):
            for r in reps_levels:
                if kind == "fuzzy" and r == 1:
                    continue  # a single fuzzy copy is just a different text
                for _ in range(n_per_cell):
                    t = fresh(L)
                    copies = [t] * r if kind == "exact" else [fuzz(t, fuzz_frac, gen, rng) for _ in range(r)]
                    traps.append(Trap(len(traps), t, L, r, kind, copies))
        for _ in range(n_nonmembers):
            traps.append(Trap(len(traps), fresh(L), L, 0, "none", []))
    return traps


def to_persian_digits(s: str) -> str:
    return "".join(PERSIAN_DIGITS[int(ch)] for ch in s)


@dataclass
class Canary:
    canary_id: int
    secret: str  # Latin digits, e.g. "48159"
    reps: int
    hero: str

    @property
    def prefix(self) -> str:
        return CANARY_TEMPLATE.format(hero=self.hero)

    @property
    def text(self) -> str:
        return canary_text(self.secret, self.hero)


def canary_text(secret: str, hero: str) -> str:
    return CANARY_TEMPLATE.format(hero=hero) + to_persian_digits(secret)


def make_canaries(seed: int, reps_levels=(1, 4, 16, 64), per_level: int = 3) -> list[Canary]:
    rng = random.Random(seed)
    heroes = list(HEROES)
    rng.shuffle(heroes)
    out, used = [], set()
    for r in [0] + list(reps_levels):  # reps 0 = control canaries never inserted
        for _ in range(per_level):
            while (s := f"{rng.randrange(10**CANARY_DIGITS):0{CANARY_DIGITS}d}") in used:
                pass
            used.add(s)
            out.append(Canary(len(out), s, r, heroes[len(out)]))
    return out


def insert_documents(docs: list[str], inserts: list[str], seed: int) -> list[str]:
    """Scatter ``inserts`` as standalone documents at random positions among
    ``docs`` (each insert becomes its own blank-line-separated document, the
    way a trap would sit inside a crawled web page or book)."""
    rng = random.Random(seed)
    out = list(docs)
    for ins in inserts:
        out.insert(rng.randrange(len(out) + 1), ins + "\n\n")
    return out
