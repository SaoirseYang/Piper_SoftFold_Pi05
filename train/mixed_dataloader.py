#!/usr/bin/env python3
"""7:3 virtual/local mixed sampling utilities.

Provides:
  - MixedIndexSampler: yields (source, index) pairs with configured ratios
  - Weighted loss helper for amplifying local samples

Used by train_stage*.py; can also be imported into custom training loops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class MixSource:
    name: str
    root: Path
    sample_ratio: float
    loss_weight: float


def load_mix_manifest(path: Path) -> list[MixSource]:
    data = json.loads(Path(path).read_text())
    sources = []
    for name in ("virtual", "local"):
        block = data[name]
        sources.append(
            MixSource(
                name=name,
                root=Path(block["root"]),
                sample_ratio=float(block["sample_ratio"]),
                loss_weight=float(block["loss_weight"]),
            )
        )
    return sources


class MixedIndexSampler:
    """Infinite / epoch sampler over multiple dataset lengths."""

    def __init__(
        self,
        lengths: dict[str, int],
        ratios: dict[str, float],
        *,
        seed: int = 0,
        epoch_size: int | None = None,
    ):
        self.names = list(lengths.keys())
        self.lengths = lengths
        self.ratios = ratios
        self.rng = np.random.default_rng(seed)
        total = sum(ratios[n] for n in self.names)
        self.probs = np.array([ratios[n] / total for n in self.names], dtype=np.float64)
        if epoch_size is None:
            epoch_size = max(lengths.values())
        self.epoch_size = int(epoch_size)

    def __iter__(self) -> Iterator[tuple[str, int]]:
        for _ in range(self.epoch_size):
            name = str(self.rng.choice(self.names, p=self.probs))
            idx = int(self.rng.integers(0, self.lengths[name]))
            yield name, idx

    def __len__(self) -> int:
        return self.epoch_size


def loss_weight_for(name: str, sources: list[MixSource]) -> float:
    for s in sources:
        if s.name == name:
            return s.loss_weight
    return 1.0
