"""Sparse autoencoder (SAE) for dictionary learning on the residual stream.

The residual stream of a transformer is a d-dimensional vector per token, but
the model plausibly represents many more than d concepts in *superposition*:
as nearly-orthogonal directions that are each active only rarely (Elhage et
al., 2022).  An SAE tries to recover those directions:

    f(x)  = ReLU( W_enc (x - b_dec) + b_enc )      (m >> d features, mostly 0)
    x_hat = W_dec f(x) + b_dec

trained to minimise  ||x - x_hat||^2 + lambda * sum_i f_i(x) * ||W_dec[:, i]||.
The L1 term pushes most features to exactly zero on any given token, so each
feature that survives tends to fire for one recognisable thing.  This follows
the recipe in Bricken et al. (2023, "Towards Monosemanticity") and its
update with the decoder-norm-weighted penalty (Conerly et al., 2024), scaled
down to a 4-layer model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, d: int, m: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(m, d))
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_dec = nn.Parameter(torch.empty(d, m))
        self.b_dec = nn.Parameter(torch.zeros(d))
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec /= self.W_dec.norm(dim=0, keepdim=True) * 10  # small initial decoder columns
            self.W_enc.copy_(self.W_dec.T)  # encoder initialised as the transpose

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = f @ self.W_dec.T + self.b_dec
        return x_hat, f

    def loss(self, x: torch.Tensor, l1: float):
        x_hat, f = self(x)
        mse = (x - x_hat).pow(2).sum(-1).mean()
        sparsity = (f * self.W_dec.norm(dim=0)).sum(-1).mean()
        return mse + l1 * sparsity, mse, f


def fraction_variance_explained(sae: SparseAutoencoder, x: torch.Tensor) -> float:
    with torch.no_grad():
        x_hat, _ = sae(x)
        return float(1 - (x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum())
