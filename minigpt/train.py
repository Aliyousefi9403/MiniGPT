"""The training loop.

One step:  sample a batch of windows -> forward -> cross-entropy loss ->
backward -> clip the global gradient norm -> AdamW step with the scheduled
learning rate.  Every ``eval_every`` steps the loss on a fixed set of training
and validation windows is measured and the best-validation checkpoint is kept
("early stopping by checkpoint selection").
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import Batcher, eval_windows
from .model import GPT, GPTConfig
from .optim import AdamW, lr_at, param_groups


@dataclass
class TrainConfig:
    steps: int = 6000
    batch_size: int = 32
    lr: float = 1e-3
    warmup: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_windows: int = 96  # windows per split per evaluation (fixed set)
    seed: int = 0
    threads: int = 4
    snapshots: tuple = ()  # extra steps at which to save the weights (training dynamics)
    extra: dict = field(default_factory=dict)


@torch.no_grad()
def mean_loss(model: GPT, windows: torch.Tensor, batch: int = 64) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(windows), batch):
        w = windows[i : i + batch]
        out = model(w[:, :-1], targets=w[:, 1:], fast=True)
        tot += out["loss"].item() * len(w)
        n += len(w)
    model.train()
    return tot / max(n, 1)


def train_model(
    mcfg: GPTConfig,
    tcfg: TrainConfig,
    train_stream: np.ndarray,
    val_stream: np.ndarray,
    out_path: str | Path,
    log=print,
) -> dict:
    torch.manual_seed(tcfg.seed)
    torch.set_num_threads(tcfg.threads)
    model = GPT(mcfg)
    opt = AdamW(param_groups(model, tcfg.weight_decay), lr=tcfg.lr, betas=(0.9, 0.95))
    batches = Batcher(train_stream, mcfg.block_size, tcfg.batch_size, seed=tcfg.seed)

    g = torch.Generator().manual_seed(1234)
    tr_w = eval_windows(train_stream, mcfg.block_size)
    tr_w = tr_w[torch.randperm(len(tr_w), generator=g)[: tcfg.eval_windows]]
    va_w = eval_windows(val_stream, mcfg.block_size)
    va_w = va_w[torch.randperm(len(va_w), generator=g)[: tcfg.eval_windows]]

    log(
        f"params {model.n_params() / 1e6:.2f}M | train tokens {len(train_stream):,} | "
        f"{tcfg.steps} steps x {tcfg.batch_size} x {mcfg.block_size} = "
        f"{tcfg.steps * tcfg.batch_size * mcfg.block_size / len(train_stream):.1f} epochs"
    )
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "elapsed_s": []}
    best_val, best_step, start = math.inf, -1, 0
    resume_path = Path(str(out_path) + ".resume")
    t0 = time.time()
    if resume_path.exists():  # continue an interrupted run exactly where it stopped
        r = torch.load(resume_path, weights_only=False)
        model.load_state_dict(r["model"])
        opt.load_state_dict(r["opt"])
        batches.g.set_state(r["batch_rng"])
        torch.set_rng_state(r["torch_rng"])
        history, best_val, best_step, start = r["history"], r["best_val"], r["best_step"], r["step"]
        t0 -= history["elapsed_s"][-1]
        log(f"resumed from step {start}")
    model.train()
    for step in range(start, tcfg.steps + 1):
        if (step % tcfg.eval_every == 0 and step != start) or step == tcfg.steps or (step == 0 and start == 0):
            trl, val = mean_loss(model, tr_w), mean_loss(model, va_w)
            el = time.time() - t0
            for k, v in zip(history, [step, trl, val, opt.param_groups[0]["lr"], el]):
                history[k].append(v)
            flag = ""
            if val < best_val:
                best_val, best_step, flag = val, step, " *"
                torch.save(
                    {"config": mcfg.to_dict(), "model": model.state_dict(), "step": step, "val_loss": val},
                    out_path,
                )
            log(f"step {step:5d} | train {trl:.4f} | val {val:.4f} | {el / 60:5.1f} min{flag}")
            if 0 < step < tcfg.steps:  # full state, so a killed run can resume here
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "batch_rng": batches.g.get_state(), "torch_rng": torch.get_rng_state(),
                            "history": history, "best_val": best_val, "best_step": best_step, "step": step},
                           resume_path)
        if step in tcfg.snapshots:
            snap = Path(out_path).parent / "snapshots"
            snap.mkdir(exist_ok=True)
            torch.save({"config": mcfg.to_dict(), "model": model.state_dict(), "step": step},
                       snap / f"{Path(out_path).stem}_step{step}.pt")
        if step == tcfg.steps:
            break
        lr = lr_at(step, tcfg.steps, tcfg.lr, tcfg.warmup)
        for grp in opt.param_groups:
            grp["lr"] = lr
        x, y = batches()
        loss = model(x, targets=y, fast=True)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

    resume_path.unlink(missing_ok=True)
    summary = {
        "model_config": mcfg.to_dict(),
        "train_config": tcfg.__dict__,
        "n_params": model.n_params(),
        "train_tokens": int(len(train_stream)),
        "best_val_loss": best_val,
        "best_step": best_step,
        "minutes": (time.time() - t0) / 60,
        "history": history,
    }
    Path(str(out_path) + ".json").write_text(json.dumps(summary, indent=1))
    return summary
