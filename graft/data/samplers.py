# graft/data/samplers.py
# Lightweight "dataset chooser" utilities for the streamed pipeline.
# These yield dataset_ids (environments) in the desired order/policy.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import random


# ------------------------------ helper: weights ------------------------------ #

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, v) for v in w.values()))
    if total <= 0.0:
        # fall back to uniform
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: max(0.0, v) / total for k, v in w.items()}


def derive_weights_from_sizes(
    sizes: Dict[str, int],
    mode: str = "sqrt",
    floor: float = 1e-6,
) -> Dict[str, float]:
    """
    Turn per-dataset sizes into sampling weights.

    mode:
      - "uniform" -> equal probability for all datasets
      - "count"   -> proportional to size
      - "sqrt"    -> proportional to sqrt(size) (less dominance by huge datasets)
    """
    if not sizes:
        return {}
    if mode == "uniform":
        return {k: 1.0 for k in sizes}
    if mode == "count":
        return {k: float(max(0, int(v))) for k, v in sizes.items()}
    if mode == "sqrt":
        return {k: float(np.sqrt(max(0, int(v))) + floor) for k, v in sizes.items()}
    raise ValueError(f"Unknown mode: {mode}")


# -------------------------- policies: dataset chooser ------------------------ #

@dataclass
class BalancedRoundRobin:
    """
    Cycles through dataset_ids in a stable, balanced order.
    Optionally shuffles order per epoch.
    """
    dataset_ids: Sequence[str]
    steps: Optional[int] = None      # total number of ids to emit; None -> infinite
    shuffle_each_epoch: bool = False
    seed: int = 123

    def __iter__(self) -> Iterator[str]:
        rng = random.Random(self.seed)
        ids = list(self.dataset_ids)
        if not ids:
            return
        produced = 0
        while self.steps is None or produced < self.steps:
            if self.shuffle_each_epoch:
                rng.shuffle(ids)
            for dsid in ids:
                yield dsid
                produced += 1
                if self.steps is not None and produced >= self.steps:
                    break


@dataclass
class WeightedDatasetSampler:
    """
    Samples dataset_ids i.i.d. from a probability distribution.
    Useful when you want to dampen dominance of large datasets (e.g., sqrt-size).
    """
    dataset_ids: Sequence[str]
    weights: Optional[Dict[str, float]] = None  # dict[dsid] -> weight
    steps: Optional[int] = None                 # total number of ids to emit; None -> infinite
    seed: int = 123

    def __iter__(self) -> Iterator[str]:
        if not self.dataset_ids:
            return
        rng = np.random.default_rng(self.seed)
        ids = np.array(self.dataset_ids, dtype=object)
        if self.weights is None:
            p = np.ones(len(ids), dtype=np.float64) / len(ids)
        else:
            w = np.array([float(self.weights.get(d, 0.0)) for d in ids], dtype=np.float64)
            if (w <= 0).all():
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
            p = w
        produced = 0
        while self.steps is None or produced < self.steps:
            # draw in vectorized chunks for speed
            chunk = min(1024, (self.steps - produced) if self.steps is not None else 1024)
            idx = rng.choice(len(ids), size=chunk, replace=True, p=p)
            for j in idx:
                yield str(ids[j])
                produced += 1
                if self.steps is not None and produced >= self.steps:
                    break


@dataclass
class InterleavedGlobalSampler:
    """
    Wrap another chooser and interleave periodic 'global' steps (e.g., for global invariance penalties).
    Emits strings that are either a dataset_id or the sentinel '__GLOBAL__'.
    """
    base: Iterable[str]
    every: int = 8  # insert a global step after every `every` dataset-specific steps

    def __iter__(self) -> Iterator[str]:
        c = 0
        for dsid in self.base:
            yield dsid
            c += 1
            if self.every > 0 and (c % self.every == 0):
                yield "__GLOBAL__"


# ------------------------------- factory helpers ----------------------------- #

def make_dataset_chooser(
    dataset_ids: Sequence[str],
    sizes: Optional[Dict[str, int]] = None,
    policy: str = "balanced",            # "balanced" | "weighted"
    weight_mode: str = "sqrt",           # for weighted: "uniform" | "count" | "sqrt"
    steps: Optional[int] = None,
    shuffle_each_epoch: bool = False,
    seed: int = 123,
):
    """
    Build a chooser iterator over dataset_ids.

    Typical usage in train loop:
        chooser = make_dataset_chooser(ds.get_dataset_ids(), sizes=ds_sizes, policy="weighted", weight_mode="sqrt", steps=total_steps)
        for key in chooser:
            if key == "__GLOBAL__":
                # run any global step
                continue
            for batch in ds.iter_batches([key]):
                train_step(batch)   # your trainer consumes one mini-batch
                break               # consume exactly one batch per dataset step
    """
    if policy == "balanced":
        return BalancedRoundRobin(dataset_ids=dataset_ids, steps=steps, shuffle_each_epoch=shuffle_each_epoch, seed=seed)
    elif policy == "weighted":
        if sizes is None:
            weights = {dsid: 1.0 for dsid in dataset_ids}
        else:
            weights = derive_weights_from_sizes(sizes, mode=weight_mode)
        weights = normalize_weights(weights)
        return WeightedDatasetSampler(dataset_ids=dataset_ids, weights=weights, steps=steps, seed=seed)
    else:
        raise ValueError(f"Unknown policy: {policy}")


def estimate_dataset_sizes(by_ds: Dict[str, "pd.DataFrame"]) -> Dict[str, int]:
    """
    Convenience helper to compute per-dataset sizes from the streaming dataset's internal tables.
    (We keep the import optional to avoid a hard pandas dependency here.)
    """
    sizes: Dict[str, int] = {}
    for dsid, df in by_ds.items():
        try:
            sizes[dsid] = int(getattr(df, "shape", [0])[0])
        except Exception:
            sizes[dsid] = 0
    return sizes
