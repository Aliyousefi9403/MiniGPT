"""Evaluation of membership scores: ROC AUC, TPR at low FPR, bootstrap CIs.

Implemented with NumPy only (no scikit-learn) and checked against brute-force
definitions in tests/test_metrics.py.
"""

from __future__ import annotations

import numpy as np


def auc(members: np.ndarray, nonmembers: np.ndarray) -> float:
    """P(score(member) > score(non-member)) + 0.5 P(tie)  (Mann-Whitney U)."""
    m, n = np.asarray(members, float), np.asarray(nonmembers, float)
    allv = np.concatenate([m, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    # average ranks over ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    r_m = ranks[: len(m)].sum()
    return float((r_m - len(m) * (len(m) + 1) / 2) / (len(m) * len(n)))


def roc_curve(members: np.ndarray, nonmembers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.concatenate([members, nonmembers])
    y = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    keep = np.r_[np.diff(s) != 0, True]  # one point per distinct threshold
    tpr = np.r_[0, tps[keep] / len(members)]
    fpr = np.r_[0, fps[keep] / len(nonmembers)]
    return fpr, tpr


def tpr_at_fpr(members: np.ndarray, nonmembers: np.ndarray, fpr_target: float) -> float:
    """Largest TPR whose threshold keeps FPR <= target (Carlini et al., 2022)."""
    fpr, tpr = roc_curve(members, nonmembers)
    ok = fpr <= fpr_target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def bootstrap_auc(members, nonmembers, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI, resampling members and non-members separately."""
    rng = np.random.default_rng(seed)
    m, n = np.asarray(members), np.asarray(nonmembers)
    vals = [auc(rng.choice(m, len(m)), rng.choice(n, len(n))) for _ in range(n_boot)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarise(members, nonmembers, n_boot: int = 1000, seed: int = 0) -> dict:
    lo, hi = bootstrap_auc(members, nonmembers, n_boot, seed)
    return {
        "auc": auc(members, nonmembers),
        "auc_lo": lo,
        "auc_hi": hi,
        "tpr@1%": tpr_at_fpr(members, nonmembers, 0.01),
        "tpr@10%": tpr_at_fpr(members, nonmembers, 0.10),
        "n_members": int(len(members)),
        "n_nonmembers": int(len(nonmembers)),
    }
