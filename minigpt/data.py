"""Corpus loading, the section-level split, and training batches.

The split is the backbone of the privacy experiments, so it is made once,
deterministically, at the level of whole Ganjoor sections (a "bakhsh", ~50
couplets on average), never individual lines:

    A  (45%)  target model's training data      -> "members"
    B  (45%)  reference model's training data   (disjoint from A, V and H)
    V   (5%)  validation: model selection, reported val loss
    H   (5%)  held out from every model          -> "non-members"

Sections are assigned at random (seeded), not by position in the epic, so A and
H are drawn from the same distribution.  That matters: if non-members came
from, say, the last chapters, a membership "attack" could score well just by
detecting the change of story (the distribution-shift trap documented by Duan
et al. 2024 and Das et al. 2024).  tests/test_data.py checks disjointness and
RESULTS.md reports a model-free "blind" baseline to confirm there is no shift.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SPLIT_FRACTIONS = {"A": 0.45, "B": 0.45, "V": 0.05, "H": 0.05}
SPLIT_SEED = 1010  # the Shahnameh was completed c. 1010 CE


@dataclass
class Section:
    idx: int
    story: str
    title: str
    couplets: list[str]

    @property
    def text(self) -> str:
        return section_text(self.couplets)


def section_text(couplets: list[str]) -> str:
    """A document is its couplets, one per line, followed by a blank line."""
    return "\n".join(couplets) + "\n\n"


def load_sections(data_dir: Path = DATA) -> list[Section]:
    lines = (data_dir / "shahnameh.txt").read_text(encoding="utf-8").split("\n")
    meta = json.loads((data_dir / "sections.json").read_text(encoding="utf-8"))
    out = []
    for i, m in enumerate(meta):
        cps = lines[m["first_line"] : m["first_line"] + m["n_couplets"]]
        assert all(" / " in c for c in cps), f"malformed section {i}"
        out.append(Section(i, m["story"], m["title"], cps))
    return out


def split_sections(n: int, seed: int = SPLIT_SEED) -> dict[str, list[int]]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    out, start = {}, 0
    names = list(SPLIT_FRACTIONS)
    for j, name in enumerate(names):
        k = n - start if j == len(names) - 1 else round(SPLIT_FRACTIONS[name] * n)
        out[name] = sorted(order[start : start + k])
        start += k
    return out


def build_stream(docs: list[str], tokenizer) -> np.ndarray:
    ids: list[int] = []
    for d in docs:
        ids.extend(tokenizer.encode(d))
    return np.asarray(ids, dtype=np.int32)


class Batcher:
    """Random contiguous windows from a token stream (nanoGPT-style).

    Each example is ``block_size + 1`` consecutive tokens: inputs are the first
    ``block_size``, targets are the same window shifted by one.  Sampling
    windows uniformly means every token is seen ~``tokens_seen / len(stream)``
    times in expectation: that ratio is what this repo calls an "epoch".
    """

    def __init__(self, stream: np.ndarray, block_size: int, batch_size: int, seed: int = 0):
        self.data = torch.from_numpy(stream.astype(np.int64))
        self.block_size, self.batch_size = block_size, batch_size
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.data) - self.block_size - 1
        ix = torch.randint(0, n, (self.batch_size,), generator=self.g)
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x, y


def eval_windows(stream: np.ndarray, block_size: int) -> torch.Tensor:
    """Non-overlapping windows covering a stream, for deterministic eval loss."""
    data = torch.from_numpy(stream.astype(np.int64))
    n = (len(data) - 1) // block_size
    return torch.stack([data[i * block_size : i * block_size + block_size + 1] for i in range(n)])
