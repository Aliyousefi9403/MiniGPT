"""AdamW and the learning-rate schedule, written out explicitly.

AdamW (Loshchilov & Hutter, 2019) keeps two running averages per parameter:
    m <- b1 m + (1 - b1) g          (first moment: a smoothed gradient)
    v <- b2 v + (1 - b2) g^2        (second moment: a smoothed squared gradient)
and steps along  m_hat / (sqrt(v_hat) + eps),  where the hats undo the bias
towards zero that the averages have early on.  Each coordinate therefore gets
its own step size ~ lr / (typical gradient magnitude).

The "W" is *decoupled* weight decay:  p <- p - lr * wd * p,  applied directly
to the weights instead of being added to the gradient (which, inside Adam,
would get rescaled by 1/sqrt(v) and stop acting like L2 regularisation).
tests/test_optim.py checks this matches ``torch.optim.AdamW`` to 1e-6.
"""

from __future__ import annotations

import math

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v, g = state["t"], state["m"], state["v"], p.grad
                p.mul_(1 - lr * wd)  # decoupled weight decay
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Decay matrices (Linear/Embedding weights) only; never biases, LayerNorm
    gains or positional tables, where shrinking towards 0 has no regularising
    meaning and just distorts the function."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() == 2 and "pos_emb" not in name else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, max_steps: int, peak: float, warmup: int, floor_frac: float = 0.1) -> float:
    """Linear warm-up to ``peak`` then cosine decay to ``floor_frac * peak``."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * progress)))
