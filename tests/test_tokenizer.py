from minigpt.tokenizer import BPETokenizer, pretokenize

TEXT = (
    "اگر تندبادی براید ز کنج / بخاک افگند نارسیده ترنج\n"
    "ستمکاره خوانیمش ار دادگر / هنرمند دانیمش ار بی‌هنر\n\n"
    "اگر مرگ دادست بیداد چیست / ز داد این همه بانگ و فریاد چیست\n"
) * 20


def test_pretokenize_is_lossless_and_splits_digits():
    s = "رمز ۴۸۱۵۹ و 123  دو فاصله\n\nپایان"
    chunks = pretokenize(s)
    assert "".join(chunks) == s
    for d in "۴۸۱۵۹123":
        assert d in chunks  # every digit is its own chunk


def test_roundtrip_and_compression():
    tok = BPETokenizer.train(TEXT, vocab_size=120)
    ids = tok.encode(TEXT)
    assert tok.decode(ids) == TEXT
    assert len(ids) < len(TEXT) / 2  # merges actually compress


def test_unknown_characters_map_to_unk():
    tok = BPETokenizer.train(TEXT, vocab_size=80)
    ids = tok.encode("Z")
    assert ids == [tok.stoi["<unk>"]]


def test_merges_never_cross_words_or_digits():
    tok = BPETokenizer.train(TEXT + " ۱۲ ۱۲ ۱۲ ۱۲" * 50, vocab_size=200)
    for piece in tok.vocab[1:]:
        assert " " not in piece[1:], piece  # a space may only lead a token
        assert not (len(piece) > 1 and any(c in "۰۱۲۳۴۵۶۷۸۹" for c in piece)), piece


def test_save_load(tmp_path):
    tok = BPETokenizer.train(TEXT, vocab_size=100)
    tok.save(tmp_path / "t.json")
    tok2 = BPETokenizer.load(tmp_path / "t.json")
    assert tok2.encode(TEXT) == tok.encode(TEXT)
