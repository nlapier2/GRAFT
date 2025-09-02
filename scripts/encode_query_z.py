#!/usr/bin/env python3
"""
encode_query_z.py
-----------------
Encode a single dataset's AnnData (.h5ad) into a control-trained scVI model to produce
per-cell latents `z` (and optionally normalized means x̄) in the same space as controls.

Key features in this version:
- Opens the query AnnData in **backed='r'** mode and materializes **only the current chunk**.
- Optional `--max-chunks` to process just the first N chunks (useful for quick tests).
- Keeps memory usage low and avoids loading huge H5ADs at once.

Usage:
  python scripts/encode_query_z.py     --scvi-model-dir artifacts_v2/scvi_k562_controls/scvi_K562     --query-h5ad /data/dataset.h5ad     --out-parquet artifacts_v2/MyDataset/z.parquet     --cell-id-key cell_id     --forward-batch-size 4096     --chunk-size 20000     --ensure-obs-cols tech_batch_id=ref     --save-xbar false     --transform-batch None     --max-chunks 1
"""
from __future__ import annotations

import argparse
import gc
import os
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scvi


def parse_kv_list(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise argparse.ArgumentError(None, f"--ensure-obs-cols expects key=value, got '{it}'")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def add_missing_obs_cols(adata: ad.AnnData, required: Dict[str, str]) -> None:
    for k, v in (required or {}).items():
        if k not in adata.obs.columns:
            adata.obs[k] = v


def to_memory_chunk(A_backed: ad.AnnData, idx_slice: slice) -> ad.AnnData:
    """Materialize a backed AnnData slice into memory.

    Uses .to_memory() when available; otherwise constructs a fresh AnnData.
    """
    Aview = A_backed[idx_slice]
    if hasattr(Aview, "to_memory"):
        return Aview.to_memory()

    X = A_backed.X[idx_slice, :]
    if sp.issparse(X):
        X = X.tocsr().copy()
    else:
        X = np.array(X, copy=True)
    obs = A_backed.obs.iloc[range(*idx_slice.indices(A_backed.n_obs))].copy()
    var = A_backed.var.copy()
    out = ad.AnnData(X=X, obs=obs, var=var)
    out.obs_names = Aview.obs_names.copy()
    out.var_names = Aview.var_names.copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scvi-model-dir", required=True, help="Path to saved scVI model directory")
    ap.add_argument("--query-h5ad", required=True, help="Path to *one* dataset AnnData (.h5ad)")
    ap.add_argument("--out-parquet", required=True, help="Where to write the z parquet (index=cell_id)")
    ap.add_argument("--cell-id-key", default=None, help="obs key with unique cell IDs; uses obs_names if None")
    ap.add_argument("--filter-obs-query", default=None, help="Optional pandas query string to subset obs before encoding")
    ap.add_argument("--forward-batch-size", type=int, default=4096, help="scVI forward batch size")
    ap.add_argument("--chunk-size", type=int, default=20000, help="Process query AnnData in chunks of this many rows")
    ap.add_argument("--max-chunks", type=int, default=None, help="If set, process only the first N chunks (for quick tests)")
    ap.add_argument("--ensure-obs-cols", nargs="*", default=[], help="List of key=value obs columns to inject if missing (e.g., tech_batch_id=ref)")
    ap.add_argument("--save-xbar", type=str, default="false", help="Also save normalized means x̄ alongside z (true/false)")
    ap.add_argument("--transform-batch", default=None, help="If saving x̄, decode under this reference batch/category")
    args = ap.parse_args()

    save_xbar = str(args.save_xbar).lower() in ("1", "true", "yes", "y")

    # Load base model (trained on controls). We'll attach per-chunk query data.

    # Open query AnnData in backed mode
    A = ad.read_h5ad(args.query_h5ad, backed="r")

    # Build integer positions to process (optionally filtered by obs query)
    if args.filter_obs_query:
        sel_idx = A.obs.query(args.filter_obs_query).index
        name_to_pos = {name: i for i, name in enumerate(A.obs_names)}
        pos = np.array([name_to_pos[n] for n in sel_idx], dtype=np.int64)
    else:
        pos = np.arange(A.n_obs, dtype=np.int64)

    ensure = parse_kv_list(args.ensure_obs_cols)

    # Prepare accumulators
    z_frames: List[pd.DataFrame] = []
    xbar_frames: List[pd.DataFrame] = []

    # Process in chunks (materialize only the current slice)
    bs = int(args.chunk_size)
    chunk_count = 0
    for start in range(0, len(pos), bs):
        stop = min(start + bs, len(pos))
        rows = pos[start:stop]

        # Build runs of consecutive indices to reduce slicing calls
        groups = []
        run_start = rows[0]
        prev = rows[0]
        for r in rows[1:]:
            if r == prev + 1:
                prev = r
            else:
                groups.append(slice(run_start, prev + 1))
                run_start = r
                prev = r
        groups.append(slice(run_start, prev + 1))

        # Concatenate in-memory chunks
        chunks = [to_memory_chunk(A, g) for g in groups]
        Aq = chunks[0].concatenate(*chunks[1:], join="outer", index_unique=None) if len(chunks) > 1 else chunks[0]

        # Inject missing obs cols if required by the model registry
        add_missing_obs_cols(Aq, ensure)

        # Attach query data and run scVI
        qmodel = scvi.model.SCVI.load_query_data(Aq, args.scvi_model_dir, inplace_subset_query_vars=True)

        # Latents
        z = qmodel.get_latent_representation(batch_size=args.forward_batch_size)
        ids = (Aq.obs[args.cell_id_key].astype(str).values if args.cell_id_key else Aq.obs_names.astype(str).values)
        df = pd.DataFrame(z, index=ids)
        z_frames.append(df)

        # Optional normalized means (decoded under a reference batch)
        if save_xbar:
            xbar = qmodel.get_normalized_expression(transform_batch=(None if args.transform_batch in [None, "None"] else args.transform_batch),
                                                    batch_size=args.forward_batch_size)
            if isinstance(xbar, np.ndarray):
                xbar_df = pd.DataFrame(xbar, index=ids, columns=Aq.var_names)
            else:
                xbar_df = xbar
                xbar_df.index = ids
            xbar_frames.append(xbar_df.astype(np.float32))

        # Cleanup
        del qmodel, Aq, z
        gc.collect()

        chunk_count += 1
        if args.max_chunks is not None and chunk_count >= int(args.max_chunks):
            print(f"[note] Reached --max-chunks={args.max_chunks}; stopping early for quick test.")
            break

    # Concatenate and write z
    Z = pd.concat(z_frames, axis=0)
    Z = Z.loc[~Z.index.duplicated(keep="first")]
    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    Z.to_parquet(args.out_parquet, index=True)
    print(f"[ok] wrote z parquet: {args.out_parquet}  (rows={len(Z)}, dim={Z.shape[1]})")

    # Optional: write x̄ in a sibling file
    if save_xbar and xbar_frames:
        X = pd.concat(xbar_frames, axis=0)
        X = X.loc[Z.index]  # align to kept z
        out_x = os.path.splitext(args.out_parquet)[0] + ".xbar.parquet"
        X.to_parquet(out_x, index=True)
        print(f"[ok] wrote xbar parquet: {out_x}  (rows={len(X)}, genes={X.shape[1]})")


if __name__ == "__main__":
    main()
