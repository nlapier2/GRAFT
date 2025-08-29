from __future__ import annotations
from typing import Dict, Iterable, Iterator, List, Optional
import numpy as np


def _as_int_array(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype != np.int64 and arr.dtype != np.int32:
        arr = arr.astype(np.int64, copy=False)
    return arr


class DatasetBalancedSampler:
    """
    Uniformly picks a dataset (environment), then samples a mini-batch of rows
    from that dataset's index pool.

    Use this when you want equal influence per dataset in the objective—even if
    some datasets have many more cells.

    Parameters
    ----------
    dataset_to_idx : Dict[str, np.ndarray]
        Mapping from dataset_id -> 1D array of row indices (into your GraftDataset).
        You can get this from `GraftDataset.split_by_dataset()`.
    batch_size : int
        Number of rows per batch.
    with_replacement_small : bool
        If a dataset has < batch_size rows, sample with replacement from that pool.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        dataset_to_idx: Dict[str, np.ndarray],
        batch_size: int,
        with_replacement_small: bool = True,
        seed: int = 13,
    ):
        pools = {k: _as_int_array(v) for k, v in dataset_to_idx.items() if len(v) > 0}
        if not pools:
            raise ValueError("No non-empty dataset pools provided.")
        self.pools: Dict[str, np.ndarray] = pools
        self.datasets: List[str] = list(pools.keys())
        self.batch_size = int(batch_size)
        self.with_replacement_small = bool(with_replacement_small)
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Infinite generator of index batches (np.ndarray[int] of length batch_size).
        """
        while True:
            ds = self.rng.choice(self.datasets)
            pool = self.pools[ds]
            if (len(pool) >= self.batch_size) or (not self.with_replacement_small):
                replace = len(pool) < self.batch_size
            else:
                replace = True
            yield self.rng.choice(pool, size=self.batch_size, replace=replace)

    def set_seed(self, seed: int) -> None:
        """Reset RNG seed (useful for epoch-to-epoch shuffling)."""
        self.rng = np.random.default_rng(int(seed))


class DatasetWeightedSampler:
    """
    Like DatasetBalancedSampler, but lets you specify a **probability weight**
    per dataset when picking which dataset supplies the next batch.

    Use cases
    ---------
    - Slightly downweight massive datasets without making them rare.
    - Curriculum schedules (e.g., start balanced, then anneal toward empirical).

    Parameters
    ----------
    dataset_to_idx : Dict[str, np.ndarray]
        Mapping dataset_id -> index pool.
    weights : Dict[str, float]
        Non-negative weights per dataset. They will be normalized internally.
        Datasets missing from this dict default to weight=0 (excluded).
    batch_size : int
        Rows per batch.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        dataset_to_idx: Dict[str, np.ndarray],
        weights: Dict[str, float],
        batch_size: int,
        seed: int = 13,
    ):
        pools = {k: _as_int_array(v) for k, v in dataset_to_idx.items() if len(v) > 0}
        if not pools:
            raise ValueError("No non-empty dataset pools provided.")
        # Keep only datasets with positive weight
        w = {k: float(weights.get(k, 0.0)) for k in pools.keys()}
        w = {k: v for k, v in w.items() if v > 0}
        if not w:
            raise ValueError("All dataset weights are zero or missing.")
        total = sum(w.values())
        self.probs = np.array([w[k] / total for k in w.keys()], dtype=np.float64)
        self.datasets: List[str] = list(w.keys())
        self.pools = {k: pools[k] for k in self.datasets}
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            i = self.rng.choice(len(self.datasets), p=self.probs)
            ds = self.datasets[i]
            pool = self.pools[ds]
            replace = len(pool) < self.batch_size
            yield self.rng.choice(pool, size=self.batch_size, replace=replace)

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))


class InterleavedGlobalSampler:
    """
    Interleave **dataset-balanced** batches with occasional **global** batches.

    Motivation
    ----------
    - Balanced batches keep the objective from being dominated by big datasets.
    - Global batches let the optimizer see more cross-dataset variability and
      speed up convergence (especially when some datasets are tiny).

    Parameters
    ----------
    dataset_to_idx : Dict[str, np.ndarray]
        Mapping dataset_id -> index pool.
    batch_size : int
        Rows per batch.
    p_global : float in [0,1]
        Probability that a drawn batch is a global batch (from all rows).
        e.g., p_global=0.2 means ~1 in 5 batches uses the full pool.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        dataset_to_idx: Dict[str, np.ndarray],
        batch_size: int,
        p_global: float = 0.2,
        seed: int = 13,
    ):
        if not (0.0 <= p_global <= 1.0):
            raise ValueError("p_global must be in [0, 1].")
        pools = {k: _as_int_array(v) for k, v in dataset_to_idx.items() if len(v) > 0}
        if not pools:
            raise ValueError("No non-empty dataset pools provided.")
        self.pools = pools
        self.datasets = list(pools.keys())
        self.all_idx = np.concatenate(list(pools.values()), axis=0)
        self.batch_size = int(batch_size)
        self.p_global = float(p_global)
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            if self.rng.random() < self.p_global:
                # Global batch
                replace = len(self.all_idx) < self.batch_size
                yield self.rng.choice(self.all_idx, size=self.batch_size, replace=replace)
            else:
                # Dataset-balanced batch
                ds = self.rng.choice(self.datasets)
                pool = self.pools[ds]
                replace = len(pool) < self.batch_size
                yield self.rng.choice(pool, size=self.batch_size, replace=replace)

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
