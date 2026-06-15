"""Episodic sampler for metric-learning (Phase 2).

Draws N-way (K-shot + Q-query) episodes from a per-class index of samples. A
``load(label, item)`` callback materializes one fixed-length audio clip, so the
same sampler serves the real Speech Commands index and the synthetic test
fixtures. Episode audio is returned class-major as ``[N*(K+Q), L]``: the first
``N*K`` rows are support (K per class, class order), the rest are query.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor


class EpisodicSampler:
    """Sample few-shot episodes from ``{label: [item, ...]}`` via a ``load`` fn."""

    def __init__(
        self,
        items_by_class: dict[str, list[Any]],
        load: Callable[[str, Any], np.ndarray],
        *,
        seed: int = 0,
    ) -> None:
        self.items_by_class = items_by_class
        self.labels = sorted(items_by_class)
        self.load = load
        self.rng = np.random.default_rng(seed)

    def episode(self, n_way: int, k_shot: int, q_query: int) -> Tensor:
        if n_way > len(self.labels):
            raise ValueError(f"n_way={n_way} exceeds available classes ({len(self.labels)})")
        chosen = self.rng.choice(len(self.labels), size=n_way, replace=False)
        per = k_shot + q_query
        support: list[np.ndarray] = []
        query: list[np.ndarray] = []
        for ci in chosen:
            label = self.labels[int(ci)]
            items = self.items_by_class[label]
            sel = self.rng.choice(len(items), size=per, replace=len(items) < per)
            clips = [self.load(label, items[int(i)]) for i in sel]
            support.extend(clips[:k_shot])
            query.extend(clips[k_shot:])
        audio = np.stack(support + query).astype(np.float32)
        return torch.from_numpy(audio)
