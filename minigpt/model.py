"""A decoder-only transformer (GPT), written out in full.

Shapes used throughout:  B = batch, T = sequence length, C = model width
(d_model), H = number of heads, D = C // H = width of one head.

The forward pass, top to bottom:

    token ids (B,T)
      -> token embedding (B,T,C)  [+ positional encoding (T,C)]
      -> n_layer x Block:         x = x + Attn(LN(x));  x = x + MLP(LN(x))
      -> final LayerNorm
      -> unembedding (tied to the token embedding) -> logits (B,T,vocab)

Every piece below is plain tensor algebra; no ``nn.Transformer``, no
``nn.MultiheadAttention``.  ``scaled_dot_product_attention`` is used only as an
optional fast path during training, and a unit test checks that it agrees with
the explicit implementation (tests/test_model.py).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128  # maximum context length, in tokens
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 192
    dropout: float = 0.1
    pos_encoding: str = "learned"  # "learned" | "sinusoidal" | "none"

    def to_dict(self) -> dict:
        return asdict(self)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU, tanh approximation (Hendrycks & Gimpel 2016), as in GPT-2.

    The tanh form is used (rather than the exact erf form) because the browser
    engine re-implements it in JavaScript, which has no built-in ``erf``; using
    the same formula on both sides keeps the two implementations bit-close.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta, over the last dim."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


def sinusoidal_table(T: int, C: int) -> torch.Tensor:
    """Fixed positional encoding of Vaswani et al. (2017).

    PE[t, 2i]   = sin(t / 10000^(2i/C))
    PE[t, 2i+1] = cos(t / 10000^(2i/C))
    Each pair of dimensions is a rotating 2-D vector whose angular speed
    decreases geometrically with i, so relative offsets are linear maps.
    """
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    i = torch.arange(0, C, 2, dtype=torch.float32)[None, :]
    angle = pos / (10000.0 ** (i / C))
    pe = torch.zeros(T, C)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (lower-triangular) mask.

    For each head h:   A_h = softmax( Q_h K_h^T / sqrt(D) + M ),  out_h = A_h V_h
    where M[t, s] = -inf if s > t, else 0, so position t can only read
    positions <= t.  This mask is what makes the model *autoregressive*: the
    prediction for token t+1 cannot peek at token t+1 itself.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)  # Q, K, V projections fused
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)  # output projection W_O
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.dropout = cfg.dropout
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False, fast: bool = False):
        B, T, C = x.shape
        H, D = self.n_head, C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B,T,C) -> (B,H,T,D): each head gets its own D-dim slice
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        if fast and not return_attn:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
            att = None
        else:
            # explicit version: the reference implementation
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,T,T)
            att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
            att = torch.softmax(att, dim=-1)
            y = self.attn_drop(att) @ v  # (B,H,T,D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concatenate heads
        y = self.resid_drop(self.proj(y))
        return y, (att if return_attn else None)


class MLP(nn.Module):
    """Position-wise feed-forward network: C -> 4C -> C with GELU."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm residual block (GPT-2 style).

    Pre-norm (LayerNorm *inside* the residual branch) keeps an identity path
    from input to output, so gradients reach early layers undiminished and
    training is stable without learning-rate warm-up tricks being critical.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, return_attn=False, fast=False):
        a, att = self.attn(self.ln1(x), return_attn=return_attn, fast=fast)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(cfg.block_size, cfg.d_model))
        elif cfg.pos_encoding == "sinusoidal":
            self.register_buffer("pos_emb", sinusoidal_table(cfg.block_size, cfg.d_model), persistent=False)
        elif cfg.pos_encoding == "none":
            self.pos_emb = None  # "NoPE": position must be inferred from the causal mask alone
        else:
            raise ValueError(cfg.pos_encoding)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.d_model)
        # unembedding is tied to tok_emb (Press & Wolf 2017): logits = h @ E^T
        self.apply(self._init)
        # GPT-2: scale residual-output projections by 1/sqrt(2 * n_layer) so the
        # variance of the residual stream does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))
        if cfg.pos_encoding == "learned":
            nn.init.normal_(self.pos_emb, 0.0, 0.01)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attn: bool = False,
        return_hidden: int | None = None,
        fast: bool = False,
    ):
        """Return logits (B,T,V); optionally the mean loss, attention maps, and
        the residual stream after block ``return_hidden`` (for the SAE)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb[:T]
        x = self.drop(x)
        attns, hidden = [], None
        for i, blk in enumerate(self.blocks):
            x, att = blk(x, return_attn=return_attn, fast=fast)
            if return_attn:
                attns.append(att)
            if return_hidden is not None and i == return_hidden:
                hidden = x
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        if return_attn:
            out["attn"] = torch.stack(attns, dim=1)  # (B, L, H, T, T)
        if hidden is not None:
            out["hidden"] = hidden
        return out

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
        stop_token: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling: predict, sample, append, repeat."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size :]
            logits = self(ctx)["logits"][:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None and (nxt == stop_token).all():
                break
        return idx


def load_model(path: str, map_location: str = "cpu") -> GPT:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
