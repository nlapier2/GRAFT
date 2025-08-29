#!/usr/bin/env python3
"""
encode_query_z.py
-----------------
Map a *single* dataset's AnnData (.h5ad) into a control-trained scVI model to produce
per-cell latents `z` (and optionally normalized means x̄) in the *same* space as controls.

You run this once per dataset shard. The outputs (parquet files) can then be
consumed by the GRAFT trainer, without ever concatenating all cells into one AnnData.

Typical usage:
  python scripts/encode_query_z.py     --scvi-model-dir artifacts_v2/scvi_k562_controls/scvi_K562     --query-h5ad /data/K562_ReplogleWeissman_perturb.h5ad     --out-parquet artifacts_v2/z_by_dataset/K562_ReplogleWeissman.parquet     --cell-id-key cell_id     --forward-batch-size 4096     --chunk-size 20000     --ensure-obs-cols tech_batch_id=ref     --save-xbar false

Notes
-----
- We do not require that the query AnnData contains the *same* batch covariates as
  the training controls. If the scVI registry expects a key (e.g., 'tech_batch_id'),
  add it here via --ensure-obs-cols to avoid KeyErrors. A single default category is fine.
- For x̄ (normalized means), set --save-xbar true and optionally --transform-batch
  to decode under a reference batch (recommended for invariance).
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import os, sys, gc
from typing import Dict, Optional, List

import scvi


def parse_kv_list(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items or []:
        if '=' not in it:
            raise argparse.ArgumentError(None, f"--ensure-obs-cols expects key=value, got '{it}'")
        k, v = it.split('=', 1)
        out[k] = v
    return out


def add_missing_obs_cols(adata: ad.AnnData, required: Dict[str, str]) -> None:
    for k, v in (required or {}).items():
        if k not in adata.obs.columns:
            adata.obs[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scvi-model-dir", required=True, help="Path to saved scVI model directory")
    ap.add_argument("--query-h5ad", required=True, help="Path to *one* dataset AnnData (.h5ad)")
    ap.add_argument("--out-parquet", required=True, help="Where to write the z parquet (index=cell_id)")
    ap.add_argument("--cell-id-key", default=None, help="obs key with unique cell IDs; uses obs_names if None")
    ap.add_argument("--filter-obs-query", default=None, help="Optional pandas query string to subset obs before encoding")
    ap.add_argument("--forward-batch-size", type=int, default=4096, help="scVI forward batch size")
    ap.add_argument("--chunk-size", type=int, default=20000, help="Process query AnnData in chunks of this many rows")
    ap.add_argument("--ensure-obs-cols", nargs='*', default=[], help="List of key=value obs columns to inject if missing (e.g., tech_batch_id=ref)")
    ap.add_argument("--save-xbar", type=str, default="false", help="Also save normalized means x̄ alongside z (true/false)")
    ap.add_argument("--transform-batch", default=None, help="If saving x̄, decode under this reference batch/category")
    args = ap.parse_args()

    save_xbar = str(args.save_xbar).lower() in ("1","true","yes","y")

    # Load base model (trained on controls)
    model = scvi.model.SCVI.load(args.scvi_model_dir, adata=None)  # we'll attach query adata per-chunk

    # Open query AnnData (can be big; we slice into chunks)
    A = ad.read_h5ad(args.query_h5ad)
    if args.filter_obs_query:
        A = A[A.obs.query(args.filter_obs_query).index].copy()

    # Prepare output accumulators
    z_frames: List[pd.DataFrame] = []
    xbar_frames: List[pd.DataFrame] = []

    ensure = parse_kv_list(args.ensure_obs_cols)

    # Process in chunks to keep memory in check
    N = A.shape[0]
    bs = int(args.chunk_size)
    for start in range(0, N, bs):
        stop = min(start + bs, N)
        sl = slice(start, stop)
        Aq = A[sl].copy()
        # Inject missing obs cols if needed
        add_missing_obs_cols(Aq, ensure)

        # Attach query data to model
        qmodel = model.load_query_data(Aq, inplace=False)

        # Latents
        z = qmodel.get_latent_representation(batch_size=args.forward_batch_size)
        ids = (Aq.obs[args.cell_id_key].astype(str).values if args.cell_id_key else Aq.obs_names.astype(str).values)
        df = pd.DataFrame(z, index=ids)
        z_frames.append(df)

        # Optional normalized means (decoded under a reference batch)
        if save_xbar:
            xbar = qmodel.get_normalized_expression(transform_batch=args.transform_batch, batch_size=args.forward_batch_size)
            # scvi may return ndarray or DataFrame
            if isinstance(xbar, np.ndarray):
                xbar_df = pd.DataFrame(xbar, index=ids, columns=Aq.var_names)
            else:
                xbar_df = xbar
                xbar_df.index = ids
            xbar_frames.append(xbar_df.astype(np.float32))

        # Cleanup
        del qmodel, Aq, z
        gc.collect()

    # Concatenate and write z
    Z = pd.concat(z_frames, axis=0)
    Z = Z.loc[~Z.index.duplicated(keep="first")]
    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    Z.to_parquet(args.out_parquet, index=True)
    print(f"[ok] wrote z parquet: {args.out_parquet}  (rows={len(Z)}, dim={Z.shape[1]})")

    # Optional: write x̄ in a sibling file
    if save_xbar:
        X = pd.concat(xbar_frames, axis=0)
        X = X.loc[Z.index]  # align to kept z
        out_x = os.path.splitext(args.out_parquet)[0] + ".xbar.parquet"
        X.to_parquet(out_x, index=True)
        print(f"[ok] wrote xbar parquet: {out_x}  (rows={len(X)}, genes={X.shape[1]})")


if __name__ == "__main__":
    main()
