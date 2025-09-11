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


@dataclass
class PriorityMixtureChooser:
    """
    Emit the priority dataset with probability p; otherwise emit from 'others'.
    This guarantees ~p fraction of steps from the priority dataset.
    """
    priority_id: str
    p: float
    others: Iterable[str]           # an iterator over the non-priority datasets
    steps: Optional[int] = None
    seed: int = 123

    def __iter__(self) -> Iterator[str]:
        if self.steps is not None and self.steps <= 0:
            return
        rng = random.Random(self.seed)
        produced = 0
        other_iter = iter(self.others)
        while self.steps is None or produced < self.steps:
            if rng.random() < self.p:
                yield self.priority_id
                produced += 1
            else:
                try:
                    v = next(other_iter)
                except StopIteration:
                    other_iter = iter(self.others)  # recycle
                    v = next(other_iter)
                yield v
                produced += 1


# ------------------------------- factory helpers ----------------------------- #

def make_dataset_chooser(
    dataset_ids: Sequence[str],
    sizes: Optional[Dict[str, int]] = None,
    policy: str = "balanced",
    weight_mode: str = "sqrt",
    steps: Optional[int] = None,
    shuffle_each_epoch: bool = False,
    seed: int = 123,
    priority: Optional[Dict[str, object]] = None,
):
    """
    Build a chooser iterator over dataset_ids.

    priority (optional): dict with keys:
      - dataset_id: str
      - frac: float in (0,1) (default 0.5)
      - others_policy: "balanced" | "weighted" (default = policy)
      - others_weight_mode: "uniform" | "count" | "sqrt" (default = weight_mode)
    """
    # ---- fast path: base chooser (no priority) ----
    def _base(ids, sz, pol, wm, st, shuffle, sd):
        if pol == "balanced":
            return BalancedRoundRobin(dataset_ids=ids, steps=st, shuffle_each_epoch=shuffle, seed=sd)
        elif pol == "weighted":
            if sz is None:
                weights = {dsid: 1.0 for dsid in ids}
            else:
                weights = derive_weights_from_sizes(sz, mode=wm)
            weights = normalize_weights(weights)
            return WeightedDatasetSampler(dataset_ids=ids, weights=weights, steps=st, seed=sd)
        else:
            raise ValueError(f"Unknown policy: {pol}")

    if not priority:
        return _base(dataset_ids, sizes, policy, weight_mode, steps, shuffle_each_epoch, seed)

    # ---- priority mixture path ----
    pid = str(priority["dataset_id"])
    if pid not in dataset_ids:
        raise ValueError(f"priority dataset_id '{pid}' not in dataset_ids")

    frac = float(priority.get("frac", 0.5))
    if not (0.0 < frac < 1.0):
        raise ValueError("priority.frac must be in (0,1)")

    other_ids = [d for d in dataset_ids if d != pid]
    if not other_ids:
        # degenerate case: only priority id exists
        return BalancedRoundRobin(dataset_ids=[pid], steps=steps, shuffle_each_epoch=False, seed=seed)

    other_sizes = {d: (sizes[d] if sizes is not None and d in sizes else 1) for d in other_ids}
    others_chooser = _base(other_ids, other_sizes, policy, weight_mode, steps, shuffle_each_epoch, seed + 1)

    return PriorityMixtureChooser(priority_id=pid, p=frac, others=others_chooser, steps=steps, seed=seed + 2)


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
