from minigpt.verse import chance_rhyme, rhymes, split_couplet, verse_stats, well_formed


def test_rhyme_detection():
    assert rhymes("براید ز کنج", "نارسیده ترنج", 2)
    assert rhymes("خوانیمش ار دادگر", "دانیمش ار بی‌هنر", 1)
    assert not rhymes("خوانیمش ار دادگر", "دانیمش ار بی‌هنر", 2)


def test_form():
    assert split_couplet("الف ب / ج د") == ("الف ب", "ج د")
    assert split_couplet("no separator") is None
    assert well_formed("یک دو سه چهار / پنج شش هفت")
    assert not well_formed("یک / دو")


def test_stats_on_real_couplets():
    lines = ["اگر تندبادی براید ز کنج / بخاک افگند نارسیده ترنج"] * 3
    st = verse_stats(lines, train_couplets=set(lines))
    assert st["rhyme@2"] == 1.0 and st["verbatim_couplet_copy"] == 1.0
    assert 0 <= chance_rhyme(lines, 1) <= 1
