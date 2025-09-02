# graft/utils/chunk_preprocess.py
"""
Minimal chunk preprocessor:
- Load a backed AnnData slice into memory.
- Filter cells using the canonical index parquet (if provided).
- Subset/reorder genes by a TSV gene list (from build_gene_maps.py).
- Return a small in-memory AnnData chunk (X aligned to gene_list order).

Assumptions:
- Each raw .h5ad corresponds to a single dataset_id.
- Global cell IDs follow "dataset_id::local_cell_id" convention (see build_index.py).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


def load_gene_list(path: str) -> list[str]:
    """Read one-gene-per-line TSV → list[str]."""
    with open(path, "r") as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    return genes


def load_allowed_cells(index_parquet: str,
                       dataset_id: Optional[str] = None,
                       controls_only: bool = False,
                       cell_type: Optional[str] = None) -> set[str]:
    """
    Read the canonical cell index parquet and return an allowed set of *global* cell_ids.
    Filters by dataset_id / controls_only / cell_type if provided.
    """
    idx = pd.read_parquet(index_parquet)
    if dataset_id is not None:
        idx = idx[idx["dataset_id"] == dataset_id]
    if controls_only and "is_control" in idx.columns:
        idx = idx[idx["is_control"].astype(bool)]
    if cell_type is not None and "cell_type" in idx.columns:
        idx = idx[idx["cell_type"] == cell_type]
    return set(idx["cell_id"].astype(str).tolist())


def _materialize_rows(A_backed: ad.AnnData, row_index: Sequence[int] | slice) -> ad.AnnData:
    """Materialize just the requested rows to memory."""
    Aview = A_backed[row_index, :]
    if hasattr(Aview, "to_memory"):
        return Aview.to_memory()
    # Fallback copy
    X = A_backed.X[row_index, :]
    if sparse.issparse(X):
        X = X.tocsr().copy()
    else:
        X = np.array(X, copy=True)
    obs = A_backed.obs.iloc[range(*row_index.indices(A_backed.n_obs))].copy() if isinstance(row_index, slice) else A_backed.obs.iloc[list(row_index)].copy()
    var = A_backed.var.copy()
    out = ad.AnnData(X=X, obs=obs, var=var)
    out.obs_names = Aview.obs_names.copy()
    out.var_names = Aview.var_names.copy()
    return out


def _build_projection(var_names_src: Iterable[str], gene_list_dst: Sequence[str]) -> sparse.csr_matrix:
    """
    Build a sparse projection P (G_src x G_dst) so that X_aligned = X_src @ P
    reorders columns by name and pads zeros for missing genes.
    """
    src = list(map(str, var_names_src))
    dst = list(map(str, gene_list_dst))
    dst_pos = {g: j for j, g in enumerate(dst)}

    rows, cols = [], []
    for i, g in enumerate(src):
        j = dst_pos.get(g)
        if j is not None:
            rows.append(i); cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    Gs, Gd = len(src), len(dst)
    if len(rows) == 0:
        raise ValueError("No overlapping genes between dataset var_names and target gene_list.")
    return sparse.csr_matrix((data, (rows, cols)), shape=(Gs, Gd))


def preprocess_chunk(
    A_backed: ad.AnnData,
    row_index: Sequence[int] | slice,
    gene_list: Sequence[str],
    dataset_id: Optional[str] = None,
    allowed_cell_ids: Optional[set[str]] = None,
    counts_layer: Optional[str] = None,
) -> ad.AnnData:
    """
    Core routine:
      1) materialize A_backed[row_index, :] into memory
      2) (optional) swap X <- counts_layer
      3) (optional) filter rows by allowed_cell_ids (global)
      4) align columns to gene_list order (by name) via sparse projection
      5) reindex obs_names to global "dataset_id::local_id" if dataset_id provided

    Returns an in-memory AnnData with X aligned to gene_list order.
    """
    # 1) to memory
    A = _materialize_rows(A_backed, row_index)

    # 2) use raw counts layer if requested
    if counts_layer is not None and counts_layer in (A.layers.keys() if A.layers is not None else {}):
        X = A.layers[counts_layer]
        A = ad.AnnData(X=X, obs=A.obs.copy(), var=A.var.copy())
        A.obs_names = A_backed[row_index, :].obs_names.copy()
        A.var_names = A_backed.var_names.copy()

    # 3) filter by allowed global cell_ids
    if allowed_cell_ids is not None:
        if dataset_id is None:
            # Assume obs_names were already global
            global_ids = A.obs_names.astype(str)
        else:
            # Compose "dataset_id::local_id"
            global_ids = pd.Index([f"{dataset_id}::{cid}" for cid in A.obs_names.astype(str)], name="cell_id")
        mask = pd.Index(global_ids).isin(allowed_cell_ids).to_numpy()
        if mask.any():
            A = A[mask, :].copy()
            global_ids = pd.Index(np.array(global_ids)[mask], name="cell_id")
        else:
            # No allowed cells in this slice
            return A[:0, :].copy()
    else:
        # still standardize global_ids if dataset_id provided
        global_ids = pd.Index([f"{dataset_id}::{cid}" for cid in A.obs_names.astype(str)], name="cell_id") if dataset_id is not None else A.obs_names.astype(str)

    # 4) gene alignment by name using sparse projection
    P = _build_projection(A.var_names.astype(str), gene_list)
    X = A.X
    if sparse.issparse(X):
        X_aligned = X @ P
    else:
        X_aligned = sparse.csr_matrix(X) @ P  # cast to sparse for big G

    # 5) assemble output AnnData
    var = pd.DataFrame(index=pd.Index(gene_list, name=A.var_names.name))
    obs = pd.DataFrame(index=global_ids)  # keep minimal; downstream can join metadata as needed
    out = ad.AnnData(X=X_aligned, obs=obs, var=var)
    return out
