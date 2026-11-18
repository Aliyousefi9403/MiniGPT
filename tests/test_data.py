from minigpt.data import SPLIT_FRACTIONS, load_sections, split_sections


def test_split_is_a_disjoint_partition():
    n = 857
    s = split_sections(n)
    all_ids = [i for v in s.values() for i in v]
    assert sorted(all_ids) == list(range(n))  # every section exactly once
    for k, frac in SPLIT_FRACTIONS.items():
        assert abs(len(s[k]) / n - frac) < 0.01


def test_split_is_deterministic_and_not_positional():
    a, b = split_sections(857), split_sections(857)
    assert a == b
    h = a["H"]
    # held-out sections are spread across the epic, not a contiguous block
    assert max(h) - min(h) > 600


def test_corpus_loads_with_couplet_structure():
    secs = load_sections()
    assert len(secs) == 857
    assert sum(len(s.couplets) for s in secs) == 49610
    assert all(c.count(" / ") == 1 for s in secs for c in s.couplets)
