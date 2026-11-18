import numpy as np

from minigpt.privacy.metrics import auc, bootstrap_auc, roc_curve, tpr_at_fpr


def brute_auc(m, n):
    return np.mean([(a > b) + 0.5 * (a == b) for a in m for b in n])


def test_auc_matches_bruteforce_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.integers(0, 6, 15).astype(float)
        n = rng.integers(0, 6, 22).astype(float)
        assert abs(auc(m, n) - brute_auc(m, n)) < 1e-12


def test_auc_extremes():
    assert auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([0.0]), np.array([1.0])) == 0.0


def test_roc_and_tpr():
    m, n = np.array([0.9, 0.8, 0.3]), np.array([0.7, 0.2, 0.1, 0.05])
    fpr, tpr = roc_curve(m, n)
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1
    assert tpr_at_fpr(m, n, 0.0) == 2 / 3
    assert tpr_at_fpr(m, n, 0.2) == 2 / 3
    assert tpr_at_fpr(m, n, 0.25) == 1.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    m, n = rng.normal(0.5, 1, 200), rng.normal(0, 1, 200)
    lo, hi = bootstrap_auc(m, n, n_boot=300)
    assert lo < auc(m, n) < hi
