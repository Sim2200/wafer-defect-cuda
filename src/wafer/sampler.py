"""Class-balanced sampling that also partitions work across DDP ranks.

WM-811K is 85% "none". A plain DistributedSampler would hand every worker the same skew and the
model would learn to say "none". This sampler draws indices with probability inverse to class
frequency (so each class is drawn about equally often per epoch), then gives each rank a disjoint,
equal-length slice of that draw, so the per-rank class mix is balanced and the ranks never see
the same wafer in the same epoch. `set_epoch` reseeds the draw so epochs differ.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler


class DistributedBalancedSampler(Sampler[int]):
    def __init__(self, labels: np.ndarray, num_replicas: int = 1, rank: int = 0, samples_per_epoch: int | None = None,
                 seed: int = 0, power: float = 1.0):
        self.labels = np.asarray(labels)
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        counts = np.bincount(self.labels, minlength=int(self.labels.max()) + 1).astype(np.float64)
        weights = np.where(counts > 0, 1.0 / np.maximum(counts, 1) ** power, 0.0)[self.labels]
        self.probs = weights / weights.sum()
        total = samples_per_epoch or len(self.labels)
        self.per_rank = total // num_replicas
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        draw = rng.choice(len(self.labels), size=self.per_rank * self.num_replicas, replace=True, p=self.probs)
        mine = draw[self.rank::self.num_replicas]
        return iter(torch.from_numpy(mine).tolist())

    def __len__(self) -> int:
        return self.per_rank
