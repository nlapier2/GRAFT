#!/usr/bin/env python3
"""
make_scvi_input_v2.py
---------------------
Build a controls-only (or filtered) scVI input H5AD by streaming raw dataset H5ADs
in backed mode and using a shared, minimal chunk preprocessor.

Key ideas:
- Use the global TSV gene list (from build_gene_maps.py) to define var order.
- Use the index parquet (from build_index.py) to define which cells to include
  (and to build the final obs table).
- For each dataset: open raw .h5ad (backed), iterate row-chunks, call
  graft.utils.chunk_preprocess.preprocess_chunk to materialize + align genes +
  filter to allowed cells. Collect blocks, then concatenate once.
- Impose a single stable row order at the very end (sorted by cell_id).

Outputs:
- H5AD with X aligned to gene list, obs taken from the index parquet (for included cells),
  and var = gene_list.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse

from graft.utils.chunk_preprocess import preprocess_chunk, load_gene_list


def load_datasets_yaml(datasets_yaml: str) -> pd.DataFrame:
    """
    Expected YAML structure:
      datasets:
        - dataset_id: NAME
          raw_path: /path/to/file.h5ad
          # (optional) counts_layer: counts

    Returns a DataFrame with at least columns [dataset_id, raw_path, counts_layer?].
    """
    with open(datasets_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    rows = []
    for item in cfg.get("datasets", []):
        rows.append({
            "dataset_id": str(item["dataset_id"]),
            "raw_path": str(item["raw_path"]),
            "counts_layer": str(item.get("counts_layer")) if item.get("counts_layer") is not None else None,
        })
    if not rows:
        raise ValueError("No datasets found in datasets.yaml (expected key 'datasets').")
    return pd.DataFrame(rows)


def filter_index(index_parquet: str,
                 dataset_ids: Optional[List[str]] = None,
                 controls_only: bool = True,
                 cell_type: Optional[str] = None) -> pd.DataFrame:
    """
    Load the global index parquet and return a filtered table of rows to include.
    We keep index metadata to build the final obs cleanly.

    We expect at least columns: cell_id, dataset_id, (optional) batch_id, is_control, cell_type, ...
    """
    idx = pd.read_parquet(index_parquet)
    if dataset_ids is not None:
        idx = idx[idx["dataset_id"].astype(str).isin(list(map(str, dataset_ids)))]
    if controls_only and "is_control" in idx.columns:
        idx = idx[idx["is_control"].astype(bool)]
    if cell_type is not None and "cell_type" in idx.columns:
        idx = idx[idx["cell_type"] == cell_type]
    # enforce string ids
    idx["cell_id"] = idx["cell_id"].astype(str)
    idx["dataset_id"] = idx["dataset_id"].astype(str)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-yaml", required=True, help="YAML listing datasets with raw_path, dataset_id")
    ap.add_argument("--index-parquet", required=True, help="Global cell index parquet from build_index.py")
    ap.add_argument("--gene-list-tsv", required=True, help="One-gene-per-line TSV (global gene universe/order)")
    ap.add_argument("--out-h5ad", required=True, help="Path to write the scVI input H5AD")
    ap.add_argument("--controls-only", action="store_true", help="Keep only control cells")
    ap.add_argument("--cell-type", default=None, help="Optional cell_type filter (exact match)")
    ap.add_argument("--chunk-size", type=int, default=100_000, help="Rows per chunk to materialize from each dataset")
    ap.add_argument("--max-cells", type=int, default=None, help="Optional cap on total cells (for smoke tests)")
    args = ap.parse_args()

    # 1) Load datasets config (unchanged logic)
    ds_tbl = load_datasets_yaml(args.datasets_yaml)
    gene_list = load_gene_list(args.gene_list_tsv)

    # 2) Read index and filter rows we want to include
    idx = filter_index(
        index_parquet=args.index_parquet,
        dataset_ids=ds_tbl["dataset_id"].tolist(),
        controls_only=args.controls_only,
        cell_type=args.cell_type,
    )
    if idx.empty:
        raise ValueError("No cells pass the index filtering conditions.")

    # Pre-allocate collectors
    X_blocks = []
    cell_ids_collected: List[str] = []

    # 3) For each dataset, stream its H5AD and extract the needed cells via preprocess_chunk
    want_by_ds = {dsid: g for dsid, g in idx.groupby("dataset_id")}
    n_total = 0

    for _, row in ds_tbl.iterrows():
        dsid = row["dataset_id"]
        raw_path = row["raw_path"]
        counts_layer = row.get("counts_layer", None)

        if dsid not in want_by_ds:
            continue  # nothing to do for this dataset

        rows_meta = want_by_ds[dsid].copy()
        allowed = set(rows_meta["cell_id"].astype(str).tolist())

        # Stream the backed file in big contiguous slices
        A_b = ad.read_h5ad(raw_path, backed="r")
        n_obs = A_b.n_obs
        cs = int(args.chunk_size)

        for start in range(0, n_obs, cs):
            end = min(start + cs, n_obs)
            A_chunk = preprocess_chunk(
                A_backed=A_b,
                row_index=slice(start, end),
                gene_list=gene_list,
                dataset_id=dsid,
                allowed_cell_ids=allowed,
                counts_layer=counts_layer,
            )
            if A_chunk.n_obs == 0:
                continue

            # Keep in the order returned; we will impose global order once at the end
            X_blocks.append(A_chunk.X.tocsr() if hasattr(A_chunk.X, "tocsr") else sparse.csr_matrix(A_chunk.X))
            cell_ids_collected.extend(list(map(str, A_chunk.obs_names)))

            n_total += A_chunk.n_obs
            if args.max_cells is not None and n_total >= args.max_cells:
                break

        if args.max_cells is not None and n_total >= args.max_cells:
            break

    if not X_blocks:
        raise ValueError("No cells materialized. Check dataset paths and index filters.")

    # 4) Concatenate and impose a single stable row order
    X = sparse.vstack(X_blocks, format="csr")
    collected = pd.Index(cell_ids_collected, name="cell_id")
    # Build final order by sorting cell_id (or, if you prefer, join back to idx and impose idx order)
    order = np.argsort(collected.values)
    X = X[order, :]
    cell_ids_final = collected.values[order]

    # 5) Build obs from the index parquet (authoritative metadata)
    obs = idx.set_index("cell_id").loc[cell_ids_final].copy()
    # Construct tech_batch_id if not present and batch info exists
    if "tech_batch_id" not in obs.columns and {"dataset_id", "batch_id"}.issubset(obs.columns):
        obs["tech_batch_id"] = (obs["dataset_id"].astype(str) + "_" + obs["batch_id"].astype(str)).astype("category")

    # 6) Assemble AnnData and write
    var = pd.DataFrame(index=pd.Index(gene_list, name="gene"))
    A_out = ad.AnnData(X=X, obs=obs, var=var)
    # Preserve global cell ids as obs_names
    A_out.obs_names = pd.Index(cell_ids_final, name="cell_id")

    os.makedirs(os.path.dirname(args.out_h5ad), exist_ok=True)
    A_out.write_h5ad(args.out_h5ad)
    print(f"[ok] wrote {args.out_h5ad} with shape {A_out.n_obs} x {A_out.n_vars}")
    print(f"[note] controls_only={args.controls_only} cell_type={args.cell_type} chunk_size={args.chunk_size} max_cells={args.max_cells}")
    print("[done]")


if __name__ == "__main__":
    main()
