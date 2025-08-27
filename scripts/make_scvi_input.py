# scripts/make_scvi_input.py  (v1-simplified + controls-only + chunked reads)
# Minimal: per-dataset contiguous selection (cap) + chunked H5AD scanning to extract selected cells.
# New:
#   --controls-only     : filter to controls only (uses 'is_control' in the built index)
#   --chunk-size INT    : read H5AD in row chunks (default 100k) and extract desired rows from each chunk.
#
# Notes:
# - We keep .to_memory() on each small chunk (safe).
# - We only carry minimal, HDF5-safe obs columns.
# - We assemble with scipy.sparse.vstack for speed.

import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import anndata as ad
import yaml
from scipy import sparse

from utils.normalize import normalize_hgnc


# ------------------------------ helpers ------------------------------

def read_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("datasets", cfg)

def load_map(dataset_id):
    with open(f"artifacts/gene_map/{dataset_id}.json") as f:
        mp = json.load(f)
    return {int(k): int(v) for k, v in mp["to_common_idx"].items()}

def subset_rows_contiguous(idx: pd.DataFrame,
                           cell_type: str,
                           datasets=None,
                           max_cells_per_ds=None,
                           seed=13,
                           controls_only: bool = False):
    """
    Pick ONE contiguous block per dataset up to cap (max_cells_per_ds).
    Optionally restrict to controls only using 'is_control' column in the index.
    """
    rng = np.random.default_rng(seed)
    sub = idx[(idx["cell_type"] == cell_type) & (idx["common_gene_set"])].copy()
    if controls_only and "is_control" in sub.columns:
        sub = sub[sub["is_control"].astype(bool)]
    if datasets:
        sub = sub[sub["dataset_id"].isin(datasets)]

    parts = []
    for ds, g in sub.groupby("dataset_id", sort=False):
        n = len(g)
        if n == 0:
            continue
        cap = min(max_cells_per_ds or n, n)
        if cap == n:
            sel = g  # take all
        else:
            # choose a contiguous window in the observed file order
            start = int(rng.integers(0, n - cap + 1))
            sel = g.iloc[start:start+cap]
        parts.append(sel)

    if not parts:
        return sub.iloc[0:0]
    return pd.concat(parts, axis=0)

def _build_keep_and_order(ds_nvars: int, gene_map: dict):
    """Return (keep_bool, order_cols) for dataset->common gene mapping."""
    to_common = np.full(ds_nvars, -1, dtype=int)
    for k, v in gene_map.items():
        if 0 <= k < ds_nvars:
            to_common[k] = v
    keep_bool = to_common >= 0
    order_cols = np.argsort(to_common[keep_bool])
    return keep_bool, order_cols

def _make_obs_frame(rows_meta: pd.DataFrame, index_order: np.ndarray) -> pd.DataFrame:
    """
    Build a minimal, HDF5-safe obs DataFrame in the order of `index_order`
    where `index_order` are positions into rows_meta.
    """
    rm = rows_meta.iloc[index_order]
    obs = pd.DataFrame(index=rm["cell_id"].values)  # global ids as obs_names
    # Keep only minimal scalar/categorical columns
    obs["dataset_id"]  = rm["dataset_id"].astype("category").values
    obs["lab_id"]      = rm["lab_id"].astype("category").values
    obs["batch_id"]    = rm["batch_id"].astype("category").values
    obs["cell_type"]   = rm["cell_type"].astype("category").values
    obs["is_control"]  = rm["is_control"].astype(bool).values
    obs["target_gene"] = rm["target_gene"].astype("category").values
    return obs

def fetch_dataset_by_chunks(raw_path: str,
                            rows_meta: pd.DataFrame,
                            gene_map: dict,
                            gene_list,
                            chunk_size: int = 100_000):
    """
    Scan H5AD in contiguous chunks of rows and extract only the desired cells.
    Returns (X: csr_matrix [n_selected x G], obs: DataFrame with global cell_id index).
    """
    A_b = ad.read_h5ad(raw_path, backed="r")
    G = A_b.n_vars

    # Map requested cell_ids to *dataset-local* ids
    cid_global = rows_meta["cell_id"].tolist()
    # If global ids look like "<dataset>::<local>", strip the prefix
    if not all(c in A_b.obs_names for c in cid_global):
        cid_local = [c.split("::", 1)[1] for c in cid_global]
    else:
        cid_local = cid_global

    # Desired order = rows_meta order
    id_to_pos = {cid_local[i]: i for i in range(len(cid_local))}
    want_set = set(id_to_pos.keys())

    # Column mapping once
    keep_bool, order_cols = _build_keep_and_order(G, gene_map)

    X_blocks = []
    obs_blocks = []
    ord_blocks = []

    n = A_b.n_obs
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Materialize this chunk
        A_chunk = A_b[start:end, :].to_memory()  # (C, G)
        ids_chunk = np.asarray(A_chunk.obs_names)

        # Find which rows we want from this chunk
        mask = np.isin(ids_chunk, list(want_set))
        if not mask.any():
            continue
        sel_idx = np.nonzero(mask)[0]

        # Subset rows in-memory, then columns in-memory
        A_sel = A_chunk[sel_idx, :]
        A_sel = A_sel[:, keep_bool]
        A_sel = A_sel[:, order_cols]

        # Determine desired global order positions for these rows
        ord_vals = np.array([id_to_pos[i] for i in ids_chunk[sel_idx]], dtype=np.int64)
        # Reorder rows inside this block to match rows_meta order (ascending ord)
        order = np.argsort(ord_vals)
        A_sel = A_sel[order, :]
        ord_vals = ord_vals[order]

        # Build obs for this block (in the correct order)
        obs_block = _make_obs_frame(rows_meta, ord_vals)

        # Collect
        X_blocks.append(A_sel.X.tocsr() if hasattr(A_sel.X, "tocsr") else sparse.csr_matrix(A_sel.X))
        obs_blocks.append(obs_block)
        ord_blocks.append(ord_vals)

        # free chunk
        del A_chunk, A_sel

    if not X_blocks:
        raise ValueError(f"No requested cells found in {raw_path}")

    # Concatenate blocks (already in rows_meta order within each block)
    X = sparse.vstack(X_blocks, format="csr")
    obs = pd.concat(obs_blocks, axis=0)

    # Final global reorder to ensure exact rows_meta order across blocks
    all_ord = np.concatenate(ord_blocks, axis=0)
    perm = np.argsort(all_ord)
    X = X[perm, :]
    obs = obs.iloc[perm]

    # Ensure var names match the common gene list
    var = pd.DataFrame(index=gene_list)
    return X, obs


# ------------------------------ main ------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="configs/datasets.yaml", help="Datasets YAML")
    ap.add_argument("--index", default="artifacts/cell_index.parquet", help="Cell index parquet path")
    ap.add_argument("--gene-list", default="artifacts/gene_list.tsv", help="Common gene list (one per line)")
    ap.add_argument("--cell-type", required=True, help="Cell type to build (e.g., K562)")
    ap.add_argument("--datasets", nargs="*", default=None, help="Optional list of dataset_ids to include")
    ap.add_argument("--max-cells-per-ds", type=int, default=200000, help="Cap per dataset to limit memory")
    ap.add_argument("--controls-only", action="store_true", help="If set, include only control cells")
    ap.add_argument("--chunk-size", type=int, default=100000, help="Row chunk size for backed reads")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=None, help="Output h5ad path")
    args = ap.parse_args()

    gene_list = [g.strip() for g in open(args.gene_list) if g.strip()]
    datasets_cfg = read_yaml(args.yaml)
    idx = pd.read_parquet(args.index)

    rows = subset_rows_contiguous(
        idx, args.cell_type, args.datasets, args.max_cells_per_ds,
        seed=args.seed, controls_only=args.controls_only
    )
    if rows.empty:
        scope = "controls" if args.controls_only else "cells"
        raise SystemExit(f"No {scope} for cell_type={args.cell_type} with common_gene_set=True")

    X_list, obs_list = [], []
    for ds, g in rows.groupby("dataset_id", sort=False):
        raw_path = datasets_cfg[ds].get("raw_path") or datasets_cfg[ds].get("path")
        if not raw_path or not os.path.exists(raw_path):
            print(f"[WARN] {ds}: raw_path not found, skip")
            continue
        gmap = load_map(ds)
        X_ds, obs_ds = fetch_dataset_by_chunks(
            raw_path=raw_path,
            rows_meta=g,
            gene_map=gmap,
            gene_list=gene_list,
            chunk_size=max(1, int(args.chunk_size))
        )
        X_list.append(X_ds)
        obs_list.append(obs_ds)
        print(f"[INFO] added {ds}: ({X_ds.shape[0]}, {X_ds.shape[1]}) via chunked scan")

    if not X_list:
        raise SystemExit("No datasets could be loaded.")

    # Assemble final AnnData
    X = sparse.vstack(X_list, format="csr")
    obs = pd.concat(obs_list, axis=0)
    var = pd.DataFrame(index=gene_list)
    Aall = ad.AnnData(X=X, obs=obs, var=var)

    # Write
    out = args.out or f"artifacts/scvi_input_{args.cell_type}{'_controls' if args.controls_only else ''}.h5ad"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Aall.write_h5ad(out, compression="lzf")
    print(f"[OK] wrote {out}: {Aall.shape}")

if __name__ == "__main__":
    main()
