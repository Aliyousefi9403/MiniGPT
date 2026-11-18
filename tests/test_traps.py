import random

from minigpt.privacy.traps import CANARY_DIGITS, WordMarkov, fuzz, insert_documents, make_canaries, make_traps
from minigpt.tokenizer import BPETokenizer

CORPUS = [
    "اگر تندبادی براید ز کنج / بخاک افگند نارسیده ترنج",
    "ستمکاره خوانیمش ار دادگر / هنرمند دانیمش ار بی‌هنر",
    "اگر مرگ دادست بیداد چیست / ز داد این همه بانگ و فریاد چیست",
    "ازین راز جان تو آگاه نیست / بدین پرده اندر ترا راه نیست",
]


def test_traps_design():
    gen = WordMarkov(CORPUS)
    traps = make_traps(gen, seed=0, reps_levels=(1, 2, 4), lengths=(1, 2), n_per_cell=3, n_nonmembers=5)
    members = [t for t in traps if t.reps > 0]
    # cells: 2 lengths x (3 exact + 2 fuzzy rep levels) x 3 traps
    assert len(members) == 2 * 5 * 3
    assert sum(t.reps == 0 for t in traps) == 10
    for t in members:
        assert len(t.copies) == t.reps
        assert all(c.count("\n") == t.n_couplets - 1 for c in t.copies)
        if t.kind == "exact":
            assert all(c == t.text for c in t.copies)
    assert len({t.text for t in traps}) == len(traps)  # all distinct


def test_fuzz_keeps_structure():
    gen = WordMarkov(CORPUS)
    rng = random.Random(0)
    t = CORPUS[0]
    f = fuzz(t, 0.2, gen, rng)
    assert f.count(" / ") == 1 and len(f.split()) == len(t.split())


def test_insertion_preserves_documents():
    docs = [f"d{i}\n\n" for i in range(10)]
    out = insert_documents(docs, ["x", "y", "x"], seed=0)
    assert len(out) == 13 and out.count("x\n\n") == 2
    assert [d for d in out if d.startswith("d")] == docs  # original order kept


def test_canaries_have_distinct_prefixes_and_equal_length_secrets():
    cs = make_canaries(seed=0)
    assert len({c.hero for c in cs}) == len(cs)
    tok = BPETokenizer.train("\n".join(CORPUS) * 5, vocab_size=120)
    for c in cs:
        ids = tok.encode(c.text)
        assert tok.stoi["<unk>"] not in ids[-CANARY_DIGITS:]  # digits always known
        assert len(ids) == len(tok.encode(c.prefix)) + CANARY_DIGITS
