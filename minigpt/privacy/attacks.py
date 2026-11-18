"""Membership-inference scores.

Every attack maps a text x to a score s(x); a higher score means "more likely
a training member".  All are computed from per-token log-probabilities
log p(t_i | t_<i) under the target model (and, for the ratio attack, under a
reference model trained on disjoint data).

* ``loss``     -mean NLL                               (Yeom et al., 2018)
* ``zlib``     -NLL_total / zlib_bits(x)               (Carlini et al., 2021)
               Normalises by how compressible the text is in general, so
               simple, repetitive strings stop looking like members.
* ``min_k``    mean of the k% *lowest* token log-probs (Shi et al., 2024)
               Members rarely contain very surprising tokens; unseen text does.
* ``min_k_pp`` Min-K%++ (Zhang et al., 2024): each token's log-prob is first
               standardised by the mean and std of log p over the whole
               next-token distribution at that position, which removes the
               "this position is just hard" effect.
* ``ratio``    NLL_ref - NLL_target                    (Carlini et al., 2021)
               The reference model knows Persian and the Shahnameh's style but
               not the members, so the difference isolates what is specific to
               the target's training set.  This is the strongest attack in
               Meeus et al. (2024) and the one this repo leads with.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch

from ..model import GPT

ATTACKS = ["loss", "zlib", "min_k", "min_k_pp", "ratio"]
PREFIX = "\n"  # every scored text is read as if it started a new line


@dataclass
class TokenStats:
    logp: np.ndarray  # (n,) log p of the actual tokens
    mu: np.ndarray  # (n,) E_{z~p}[log p(z)]
    sigma: np.ndarray  # (n,) Std_{z~p}[log p(z)]


@torch.no_grad()
def token_stats(model: GPT, tokenizer, texts: list[str], batch: int = 64) -> list[TokenStats]:
    """Per-token statistics for each text, batched with right padding."""
    model.eval()
    pre = tokenizer.encode(PREFIX)
    seqs = [pre + tokenizer.encode(t) for t in texts]
    T = model.cfg.block_size
    out: list[TokenStats | None] = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        L = min(max(len(seqs[i]) for i in idx), T)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for j, i in enumerate(idx):
            q = seqs[i][:L]  # texts longer than the context are truncated
            x[j, : len(q)] = torch.tensor(q)
        logits = model(x[:, :-1], fast=True)["logits"]
        lp = torch.log_softmax(logits, dim=-1)  # (b, L-1, V)
        p = lp.exp()
        mu = (p * lp).sum(-1)
        sigma = ((p * lp.pow(2)).sum(-1) - mu.pow(2)).clamp_min(1e-12).sqrt()
        tok_lp = lp.gather(-1, x[:, 1:, None]).squeeze(-1)
        for j, i in enumerate(idx):
            n = min(len(seqs[i]), L) - 1
            out[i] = TokenStats(tok_lp[j, :n].numpy(), mu[j, :n].numpy(), sigma[j, :n].numpy())
    return out  # type: ignore[return-value]


def zlib_bits(text: str) -> float:
    return 8.0 * len(zlib.compress(text.encode("utf-8"), 9))


def min_k(values: np.ndarray, k: float) -> float:
    n = max(1, int(round(k * len(values))))
    return float(np.sort(values)[:n].mean())


def scores(
    texts: list[str],
    target: list[TokenStats],
    reference: list[TokenStats] | None = None,
    k: float = 0.2,
) -> dict[str, np.ndarray]:
    out = {a: np.zeros(len(texts)) for a in ATTACKS}
    for i, (t, st) in enumerate(zip(texts, target)):
        nll = -st.logp.mean()
        out["loss"][i] = -nll
        out["zlib"][i] = -(-st.logp.sum()) / zlib_bits(t)
        out["min_k"][i] = min_k(st.logp, k)
        out["min_k_pp"][i] = min_k((st.logp - st.mu) / st.sigma, k)
        out["ratio"][i] = (-reference[i].logp.mean() - nll) if reference is not None else np.nan
    return out
