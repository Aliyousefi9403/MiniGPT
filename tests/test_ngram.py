import random

from minigpt.ngram import CharNGram

TEXT = "اگر تندبادی براید ز کنج / بخاک افگند نارسیده ترنج\n" * 10


def test_distributions_sum_to_one():
    m = CharNGram(4).fit(TEXT)
    for h in ["", "ا", "اگ", "ز ک", "qqq"]:
        total = sum(m.prob(h, c) for c in m.alphabet) + m.prob(h, "\x00")  # unseen-char bucket
        assert abs(total - 1) < 1e-9, (h, total)


def test_higher_order_fits_training_text_better():
    bpc = [CharNGram(n).fit(TEXT).bits_per_char(TEXT) for n in (1, 2, 4)]
    assert bpc[0] > bpc[1] > bpc[2]


def test_sampling_uses_alphabet():
    m = CharNGram(3).fit(TEXT)
    s = m.sample(random.Random(0), 50)
    assert set(s) <= set(m.alphabet)
