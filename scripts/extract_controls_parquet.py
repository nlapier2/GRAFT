#!/usr/bin/env python3
"""
Streamed control cell extractor for a single .h5ad file.

- Reads cells in chunks, aligned to a target gene_list and canonical index parquet.
- Filters for cells identified as controls. 
  Control definition matches original pseudobulk logic:
    1. is_control=True
    2. OR target_label matches --control_label
    3. OR target_label is missing (fallback)
- Handles missing genes: genes not present in the source .h5ad are marked as missing and filled with NaN or -1 at the END.
- Output: an AnnData with rows = control cells, columns = gene_list.

Assumes:
- You have already run build_index.py to create the canonical index parquet for this dataset.
- We use chunk_preprocess.preprocess_chunk to materialize aligned slices.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

# Ensure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graft.utils.chunk_preprocess import preprocess_chunk, load_gene_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, help="Path to backed AnnData file (.h5ad) for ONE dataset")
    ap.add_argument("--dataset_id", required=True, help="Dataset ID matching canonical index")
    ap.add_argument("--index_parquet", required=True, help="Canonical cell index parquet (from build_index.py)")
    ap.add_argument("--gene_list", required=False, help="TSV with one gene per line, in target dataset order")
    ap.add_argument("--target_h5ad", required=False, help="If provided, use this .h5ad's var_names as gene_list instead")
    ap.add_argument("--out_h5ad", required=True, help="Output extracted controls .h5ad")
    ap.add_argument("--counts_layer", default=None, help="Optional counts layer name to use instead of X")
    ap.add_argument("--target_label", default="target_gene", help="Obs column name holding perturbation label")
    ap.add_argument("--control_label", default="non-targeting", help="String label used to identify control rows in target_label")
    ap.add_argument("--chunk_rows", type=int, default=100_000, help="Number of rows to read per chunk")
    ap.add_argument("--missing_fill", default="nan", choices=["nan", "-1"], help="Placeholder for genes missing in this dataset")
    ap.add_argument("--filter_by_index", action="store_true", help="Filter to cell_ids present in index (recommended)")

    args = ap.parse_args()

    # Load canonical index and filter to this dataset
    index_df = pd.read_parquet(args.index_parquet)
    index_df = index_df[index_df["dataset_id"] == args.dataset_id].copy()
    if index_df.empty:
        raise SystemExit(f"No rows in index for dataset_id={args.dataset_id}")

    # Load gene list and determine which genes are present in the source .h5ad
    if args.target_h5ad is not None:
        if args.gene_list is not None:
            raise SystemExit("Error: --gene_list and --target_h5ad are mutually exclusive")
        A_target = ad.read_h5ad(args.target_h5ad, backed="r")
        gene_list = list(map(str, A_target.var_names))
        print(f"[info] Using {len(gene_list)} genes from target .h5ad {args.target_h5ad}", file=sys.stderr)
    elif args.gene_list is not None:
        print(f"[info] Using gene list from {args.gene_list}", file=sys.stderr)
        gene_list = load_gene_list(args.gene_list)  # ordering to enforce
    else:
        raise SystemExit("Error: one of --gene_list or --target_h5ad must be provided")

    A_backed = ad.read_h5ad(args.h5ad, backed="r")
    src_genes = A_backed.var_names.astype(str).tolist()
    src_set_lower = {g.lower() for g in src_genes}

    # Boolean mask: which target genes are present in source? (case-insensitive)
    present_mask = np.array([(g in src_genes) or (g.lower() in src_set_lower) for g in gene_list], dtype=bool)

    # Accumulators for extracted cells
    all_X: List[np.ndarray] = []
    all_obs: List[pd.DataFrame] = []
    total_extracted = 0

    # Stream in row chunks
    n_obs = A_backed.n_obs
    for start in range(0, n_obs, args.chunk_rows):
        stop = min(start + args.chunk_rows, n_obs)
        sl = slice(start, stop)
        print('Reading rows', start, 'to', stop, 'of', n_obs, file=sys.stderr)

        # Materialize and align this chunk to gene_list
        chunk = preprocess_chunk(
            A_backed=A_backed,
            row_index=sl,
            gene_list=gene_list,
            dataset_id=args.dataset_id,
            index_df=index_df,
            counts_layer=args.counts_layer,
            keep_cols=["dataset_id", "cell_id", "lab_id", "batch_id", "cell_type",
                       "is_control", "perturbation", "tech_batch_id", args.target_label],
            filter_by_index=args.filter_by_index,
        )
        if chunk.n_obs == 0:
            continue

        # --- Identify Control Cells ---
        # We replicate the logic from the original pseudobulk script:
        # 1. is_control is explicitly True
        mask_is_control = chunk.obs["is_control"].fillna(False).astype(bool)

        # Get target labels as strings, handling NaNs
        target_vals = chunk.obs[args.target_label].fillna("").astype(str)

        # 2. target_label matches control_label explicitly
        mask_match_label = (target_vals == args.control_label)

        # 3. Fallback: if target_label is empty or NaN, treat as control (conservative)
        mask_missing = (target_vals == "") | (target_vals.str.lower() == "nan")

        # Combine
        is_ctrl = mask_is_control | mask_match_label | mask_missing
        
        if not is_ctrl.any():
            continue

        # Extract subsets
        chunk_X = chunk.X[is_ctrl]
        chunk_obs = chunk.obs[is_ctrl].copy()

        # Convert to dense to ensure consistent handling of "missing_fill" later
        if sparse.issparse(chunk_X):
            chunk_X = chunk_X.toarray()

        all_X.append(chunk_X)
        all_obs.append(chunk_obs)
        total_extracted += chunk_X.shape[0]

    # Build output AnnData
    if total_extracted == 0:
        raise SystemExit("No control cells found (after filtering).")

    print(f"[info] Concatenating {total_extracted} control cells...", file=sys.stderr)
    X_final = np.vstack(all_X)
    obs_final = pd.concat(all_obs)
    var_final = pd.DataFrame(index=pd.Index(gene_list, name="gene"))

    # Fill genes that are globally missing in this dataset with placeholder
    if args.missing_fill == "nan":
        fill_val = np.nan
    else:
        fill_val = -1.0
        
    if not present_mask.all():
        X_final[:, ~present_mask] = fill_val

    out = ad.AnnData(X=X_final, obs=obs_final, var=var_final)
    # Minimal hygiene: ensure float32 to cut disk size
    out.X = np.asarray(out.X, dtype=np.float32)

    # Save
    os.makedirs(os.path.dirname(args.out_h5ad), exist_ok=True)
    out.write_h5ad(args.out_h5ad, compression="gzip")
    print(f"[OK] wrote extracted controls to {args.out_h5ad} with shape {out.shape}")


if __name__ == "__main__":
    main()