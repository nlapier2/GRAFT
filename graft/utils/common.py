"""
Common utilities shared across data prep, training, and evaluation.

This module intentionally sticks to **lightweight helpers** that avoid importing
heavy ML frameworks. The goal is to centralize the little things we do in many
places so scripts stay clean and consistent.

What’s inside (high level)
--------------------------
- I/O helpers:
    * read_parquet_indexed: read parquet and set index from a column if present
    * resolve_index: make 'cell_id' the index when available
    * ensure_columns: add missing columns with default values
- ID / categorical helpers:
    * encode_categories: map strings → int codes + provide mappings
    * build_id_map: fast {id → position} dict for lookups
- Gene mapping:
    * build_gene_to_idx: gene symbol → column index in model order
    * map_targets_to_idx: target gene Series → int indices (−1 for controls/missing)
- Reproducibility:
    * seed_everything: seed numpy/python torch (when available)
- Small batching utilities:
    * chunks: yield contiguous index slices of a given size
"""

from __future__ import annotations
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import os
import random

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# -------------------------
# I/O and index utilities
# -------------------------
def resolve_index(df: pd.DataFrame, index_col: str = "cell_id", drop: bool = True) -> pd.DataFrame:
    """
    If `index_col` exists as a column, set it as the index. Otherwise return df unchanged.
    """
    if index_col in df.columns:
        return df.set_index(index_col, drop=drop)
    return df


def read_parquet_indexed(path: str, index_col: str = "cell_id") -> pd.DataFrame:
    """
    Read a parquet and, if `index_col` exists as a column, make it the index.
    """
    df = pd.read_parquet(path)
    return resolve_index(df, index_col=index_col, drop=True)


def ensure_columns(df: pd.DataFrame, defaults: Dict[str, object]) -> pd.DataFrame:
    """
    Ensure the given columns exist; if missing, create with provided default value.
    Operates in-place and also returns df for chaining.
    """
    for k, v in (defaults or {}).items():
        if k not in df.columns:
            df[k] = v
    return df


# -------------------------
# ID / categorical helpers
# -------------------------
def encode_categories(values: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Encode string-like labels into contiguous int codes [0..E-1].
    Returns (codes, str->int, int->str).
    """
    arr = np.asarray(values).astype(str)
    uniq = np.unique(arr)
    mapping = {s: i for i, s in enumerate(uniq)}
    inv = {i: s for s, i in mapping.items()}
    codes = np.array([mapping[s] for s in arr], dtype=np.int64)
    return codes, mapping, inv


def build_id_map(ids: Iterable[str]) -> Dict[str, int]:
    """
    Build a dictionary mapping id → position (0-based). Useful to map cell_ids to row indices.
    """
    return {k: i for i, k in enumerate(ids)}


# -------------------------
# Gene mapping utilities
# -------------------------
def build_gene_to_idx(genes: Sequence[str]) -> Dict[str, int]:
    """
    Gene symbol -> column index in model order.
    """
    return {g: i for i, g in enumerate(list(genes))}


def map_targets_to_idx(
    target_series: pd.Series,
    gene_to_idx: Dict[str, int],
    is_control: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """
    Map target gene symbols to integer indices; controls/missing set to -1.

    Parameters
    ----------
    target_series : pd.Series of strings (gene symbols) or None
    gene_to_idx   : mapping from symbol -> column index
    is_control    : optional boolean mask; when True, output -1 regardless of symbol
    """
    n = len(target_series)
    out = np.full(n, -1, dtype=np.int64)
    ctrl = np.asarray(is_control, dtype=bool) if is_control is not None else np.zeros(n, dtype=bool)
    vals = target_series.values if isinstance(target_series, pd.Series) else np.asarray(target_series)
    for i in range(n):
        if ctrl[i]:
            continue
        tg = vals[i]
        if tg is None:
            continue
        if isinstance(tg, float) and np.isnan(tg):
            continue
        idx = gene_to_idx.get(str(tg), -1)
        out[i] = idx
    return out


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 13) -> None:
    """
    Seed numpy, python, and torch (if installed) for repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------
# Batching helpers
# -------------------------
def chunks(n_rows: int, batch_size: int) -> Iterator[slice]:
    """
    Yield contiguous slices that cover range(n_rows) in steps of batch_size.
    """
    bs = int(batch_size)
    for s in range(0, n_rows, bs):
        yield slice(s, min(s + bs, n_rows))
