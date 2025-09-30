#!/usr/bin/env python3
"""
Streamed pseudobulk (and semi-pseudobulk) generator for a single .h5ad file.

- Reads cells in chunks, aligned to a target gene_list and canonical index parquet.
- Aggregates running sums/counts per group without loading entire matrix.
- Groups:
    * pseudobulk: per perturbation (target label), with controls grouped under --control_label
    * semi-pseudobulk (optional): per (pert, shard) where shard = hash(cell_id) % K, or per (pert, stratify_by)
- Handles missing genes: genes not present in the source .h5ad are marked as missing and filled with NaN or -1 at the END.
- Output: an AnnData with rows per group, columns = gene_list (same order as target dataset), X = mean expressions.

Assumes:
- You have already run build_index.py to create the canonical index parquet for this dataset.  
- You have a gene_list.tsv and consistent obs naming via that index parquet.
- We use chunk_preprocess.preprocess_chunk to materialize aligned slices.      
"""

from __future__ import annotations

import argparse
import os
import sys
import hashlib
from typing import Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

# Ensure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graft.utils.chunk_preprocess import preprocess_chunk, load_gene_list  


def stable_hash_to_int(s: str) -> int:
    # Deterministic 64-bit hash → Python int
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "little", signed=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, help="Path to backed AnnData file (.h5ad) for ONE dataset")
    ap.add_argument("--dataset_id", required=True, help="Dataset ID matching canonical index")
    ap.add_argument("--index_parquet", required=True, help="Canonical cell index parquet (from build_index.py)")  
    ap.add_argument("--gene_list", required=False, help="TSV with one gene per line, in target dataset order")
    ap.add_argument("--target_h5ad", required=False, help="If provided, use this .h5ad's var_names as gene_list instead")
    ap.add_argument("--out_h5ad", required=True, help="Output pseudobulk .h5ad")
    ap.add_argument("--counts_layer", default=None, help="Optional counts layer name to use instead of X")
    ap.add_argument("--target_label", default="target_gene", help="Obs column name holding perturbation label (in canonical index)")
    ap.add_argument("--control_label", default="non-targeting", help="Name to assign to control group row(s)")
    ap.add_argument("--chunk_rows", type=int, default=100_000, help="Number of rows to read per chunk")
    ap.add_argument("--semi_k", type=int, default=0, help="If >0, create K hash shards per perturbation (semi-pseudobulk)")
    ap.add_argument("--stratify_by", default=None, choices=[None, "tech_batch_id", "batch_id", "cell_type"], help="Alternative to semi_k: group by (pert, stratify_by)")
    ap.add_argument("--missing_fill", default="nan", choices=["nan","-1"], help="Placeholder for genes missing in this dataset")
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
    
    gene_to_pos = {g: i for i, g in enumerate(gene_list)}

    A_backed = ad.read_h5ad(args.h5ad, backed="r")
    src_genes = A_backed.var_names.astype(str).tolist()
    src_set_lower = {g.lower() for g in src_genes}

    # Boolean mask: which target genes are present in source? (case-insensitive)
    present_mask = np.array([ (g in src_genes) or (g.lower() in src_set_lower) for g in gene_list ], dtype=bool)
    # map lowercased gene -> canonical index for quick membership check
    gene_to_pos_lower = {g.lower(): i for i, g in enumerate(gene_list)}

    G = len(gene_list)

    # Accumulators: dict[group_key] = (sum_vec, count)
    sums: Dict[Tuple[str, Optional[int], Optional[str]], np.ndarray] = {}
    counts: Dict[Tuple[str, Optional[int], Optional[str]], int] = {}
    # Track homogeneity for tech_batch_id and cell_type per group (minimal memory).
    tech_val: Dict[Tuple[str, Optional[int], Optional[str]], Optional[str]] = {}
    tech_mixed: Dict[Tuple[str, Optional[int], Optional[str]], bool] = {}
    cell_val: Dict[Tuple[str, Optional[int], Optional[str]], Optional[str]] = {}
    cell_mixed: Dict[Tuple[str, Optional[int], Optional[str]], bool] = {}

    # round-robin shard pointer per perturbation (optionally per-strata if you ever add that)
    rr_next = defaultdict(int)

    # Helper: decide group key for each row in a chunk's obs
    def row_to_group(row: pd.Series) -> Tuple[str, Optional[int], Optional[str]]:
        # Determine perturbation label name
        if bool(row.get("is_control", False)):
            pert = args.control_label
        else:
            # Use the standardized target/pert label from index
            pert = str(row.get(args.target_label, ""))
            if pert == "" or pd.isna(pert):
                # If not set, fallback to control (conservative)
                pert = args.control_label

        shard: Optional[int] = None
        strata: Optional[str] = None

        if args.stratify_by is not None:
            # e.g., tech_batch_id; require the column to exist in obs/index
            val = row.get(args.stratify_by, None)
            strata = str(val) if val is not None else "NA"
        elif args.semi_k and args.semi_k > 0:
            # simple per-perturbation round-robin sharding
            k = int(args.semi_k)
            base = (pert, strata)
            idx = rr_next[base] % k
            rr_next[base] = (rr_next[base] + 1) % k
            shard = idx

        return (pert, shard, strata)

    # Stream in row chunks
    n_obs = A_backed.n_obs
    for start in range(0, n_obs, args.chunk_rows):
        stop = min(start + args.chunk_rows, n_obs)
        sl = slice(start, stop)
        print('Reading rows', start, 'to', stop, 'of', n_obs, file=sys.stderr)

        # Materialize and align this chunk to gene_list using your helper (handles obs from index)
        chunk = preprocess_chunk(
            A_backed=A_backed,
            row_index=sl,
            gene_list=gene_list,
            dataset_id=args.dataset_id,
            index_df=index_df,
            counts_layer=args.counts_layer,
            keep_cols=["dataset_id","cell_id","lab_id","batch_id","cell_type",
                       "is_control","perturbation", "tech_batch_id", args.target_label],
            filter_by_index=args.filter_by_index,
        )
        if chunk.n_obs == 0:
            continue

        X = chunk.X
        if sparse.issparse(X):
            X = X.toarray()

        # Accumulate per group
        for i in range(chunk.n_obs):
            row = chunk.obs.iloc[i]
            key = row_to_group(row)
            if key not in sums:
                sums[key] = np.zeros(G, dtype=np.float64)
                counts[key] = 0
            sums[key] += X[i, :]
            counts[key] += 1
            # Update homogeneity for tech_batch_id
            tv = row.get("tech_batch_id", None)
            tv = None if pd.isna(tv) else str(tv)
            prev_tv = tech_val.get(key)
            if prev_tv is None:
                tech_val[key] = tv
            elif tv is not None and prev_tv != tv:
                tech_mixed[key] = True
            # Update homogeneity for cell_type
            cv = row.get("cell_type", None)
            cv = None if pd.isna(cv) else str(cv)
            prev_cv = cell_val.get(key)
            if prev_cv is None:
                cell_val[key] = cv
            elif cv is not None and prev_cv != cv:
                cell_mixed[key] = True

    # Build output AnnData
    if not sums:
        raise SystemExit("No data accumulated (after filtering).")

    keys = list(sums.keys())
    X_mean = np.vstack([ (sums[k] / max(counts[k], 1)) for k in keys ])

    # Fill genes that are globally missing in this dataset with placeholder
    if args.missing_fill == "nan":
        fill_val = np.nan
    else:
        fill_val = -1.0
    if not present_mask.all():
        X_mean[:, ~present_mask] = fill_val

    # Construct obs
    obs_rows = []
    for k in keys:
        pert, shard, strata = k
        is_control = (pert == args.control_label)
        
        # infer pert_type/target_idx/target_present
        if is_control:
            pert_type = "control"
            target_idx = -1
            target_present = False
        else:
            pl = str(pert)
            idx = gene_to_pos_lower.get(pl.lower(), None)
            if idx is not None:
                pert_type = "gene"
                target_idx = int(idx)
                target_present = True
            else:
                pert_type = "non_gene"
                target_idx = -1
                target_present = False

        # Decide tech_batch_id / cell_type values: prefer stratifier value; else homogeneous value; else "mixed"
        tech_v = tech_val.get(k)
        tech_value = (strata if args.stratify_by=="tech_batch_id" else
                      (tech_v if (tech_v is not None and not tech_mixed.get(k, False)) else "mixed"))
        cell_v = cell_val.get(k)
        cell_value = (strata if args.stratify_by=="cell_type" else
                      (cell_v if (cell_v is not None and not cell_mixed.get(k, False)) else "mixed"))


        row = {
            "dataset_id": args.dataset_id,
            args.target_label: pert,
            "is_control": bool(is_control),
            "n_cells": counts[k],
            "pert_type": pert_type,
            "target_idx": target_idx,
            "target_present": target_present,
            # carry common covariates; if grouping mixes them, set "mixed"
            "tech_batch_id": tech_value,
            "cell_type": cell_value,
        }
        if shard is not None:
            row["shard"] = int(shard)
        if strata is not None:
            row[args.stratify_by] = strata
        obs_rows.append(row)
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"{args.dataset_id}::{r[args.target_label]}" + (f"::sh{r['shard']}" if 'shard' in r else "") +
                 (f"::{args.stratify_by}={r[args.stratify_by]}" if args.stratify_by in r else "")
                 for _, r in obs.iterrows()]

    var = pd.DataFrame(index=pd.Index(gene_list, name="gene"))

    out = ad.AnnData(X=X_mean, obs=obs, var=var)
    # Minimal hygiene: ensure float32 to cut disk size
    out.X = np.asarray(out.X, dtype=np.float32)

    # Save
    os.makedirs(os.path.dirname(args.out_h5ad), exist_ok=True)
    out.write_h5ad(args.out_h5ad, compression="gzip")
    print(f"[OK] wrote pseudobulk to {args.out_h5ad} with shape {out.shape} and {obs['n_cells'].sum():,} total cells")

if __name__ == "__main__":
    main()
