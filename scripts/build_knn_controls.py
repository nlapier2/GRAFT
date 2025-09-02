#!/usr/bin/env python3
"""
build_knn_controls.py
---------------------
Precompute top-K **control** neighbors for **perturbed** cells *within a single dataset*
using scVI latents `z` produced by encode_query_z.py.

You run this **once per dataset**. The outputs (a long-form parquet and a small JSON summary)
can be consumed at training time to quickly sample matched controls (e.g., stochastic top-K).

Inputs
------
- index parquet (global): rows for all cells with at least:
    * cell_id (index or column)
    * dataset_id (str)
    * is_control (bool)
- z parquet (per-dataset): scVI latents for this dataset only (index = cell_id)
  produced by encode_query_z.py

Outputs
-------
- neighbors parquet (long): columns [cell_id, rank, control_id, dist, dataset_id]
- summary json: counts and params

Usage
-----
python scripts/build_knn_controls.py \
  --index-parquet artifacts_v2/cell_index.parquet \
  --z-parquet artifacts_v2/z_by_dataset/K562_ReplogleWeissman.parquet \
  --dataset-id K562_ReplogleWeissman \
  --out-parquet artifacts_v2/knn_by_dataset/K562_ReplogleWeissman.knn.parquet \
  --k 32 --metric euclidean --chunk 20000 --normalize false
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import os, sys, json, math, gc
from typing import Tuple


def _resolve_index(df: pd.DataFrame) -> pd.DataFrame:
    if "cell_id" in df.columns:
        df = df.set_index("cell_id", drop=True)
    return df


def _cosine_prep(Zq: np.ndarray, Zc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Zqn = Zq / (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-8)
    Zcn = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-8)
    return Zqn.astype(np.float32), Zcn.astype(np.float32)


def _topk_euclidean(Zq: np.ndarray, Zc: np.ndarray, K: int, chunk: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, dists) where indices has shape (Q, K) with entries in [0, C),
    and dists has shape (Q, K). Computed in blocks for memory safety.
    """
    Q, d = Zq.shape
    C = Zc.shape[0]
    idx_out = np.empty((Q, K), dtype=np.int64)
    dst_out = np.empty((Q, K), dtype=np.float32)
    for s in range(0, Q, chunk):
        t = min(s + chunk, Q)
        Z = Zq[s:t]  # (B, d)
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
        x2 = np.sum(Z * Z, axis=1, keepdims=True)          # (B, 1)
        y2 = np.sum(Zc * Zc, axis=1, keepdims=True).T      # (1, C)
        d2 = x2 + y2 - 2.0 * (Z @ Zc.T)                    # (B, C)
        # Protect against tiny negatives due to precision
        d2 = np.maximum(d2, 0.0)
        # Argpartition for top-K smallest distances
        part = np.argpartition(d2, K-1, axis=1)[:, :K]     # (B, K) unsorted indices
        # Gather and sort top-K
        topd = np.take_along_axis(d2, part, axis=1)
        ord = np.argsort(topd, axis=1)
        part_sorted = np.take_along_axis(part, ord, axis=1)
        topd_sorted = np.take_along_axis(topd, ord, axis=1)
        idx_out[s:t, :] = part_sorted
        dst_out[s:t, :] = np.sqrt(topd_sorted, dtype=np.float32)
        del Z, x2, y2, d2, part, topd, ord, part_sorted, topd_sorted
        gc.collect()
    return idx_out, dst_out


def _topk_cosine(Zq: np.ndarray, Zc: np.ndarray, K: int, chunk: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine distance = 1 - cosine similarity. We find top-K **nearest** (smallest distance).
    """
    Zq, Zc = _cosine_prep(Zq, Zc)
    Q, d = Zq.shape
    C = Zc.shape[0]
    idx_out = np.empty((Q, K), dtype=np.int64)
    dst_out = np.empty((Q, K), dtype=np.float32)
    for s in range(0, Q, chunk):
        t = min(s + chunk, Q)
        Z = Zq[s:t]  # (B, d)
        sim = Z @ Zc.T                               # (B, C)
        dist = 1.0 - sim
        part = np.argpartition(dist, K-1, axis=1)[:, :K]
        topd = np.take_along_axis(dist, part, axis=1)
        ord = np.argsort(topd, axis=1)
        part_sorted = np.take_along_axis(part, ord, axis=1)
        topd_sorted = np.take_along_axis(topd, ord, axis=1)
        idx_out[s:t, :] = part_sorted
        dst_out[s:t, :] = topd_sorted.astype(np.float32)
        del Z, sim, dist, part, topd, ord, part_sorted, topd_sorted
        gc.collect()
    return idx_out, dst_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-parquet", required=True, help="Global index parquet with dataset_id and is_control")
    ap.add_argument("--z-parquet", required=True, help="Per-dataset z parquet (index=cell_id)")
    ap.add_argument("--dataset-id", required=True, help="Dataset identifier as in index parquet (dataset_id)")
    ap.add_argument("--out-parquet", required=True, help="Where to write long-form neighbors parquet")
    ap.add_argument("--out-summary", default=None, help="Optional JSON summary path (defaults next to out-parquet)")
    ap.add_argument("--k", type=int, default=32, help="Top-K neighbors to save")
    ap.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    ap.add_argument("--chunk", type=int, default=20000, help="Query chunk size for distance blocks")
    ap.add_argument("--max-perturbed", type=int, default=None, help="If set, limit to first N perturbed cells (fast path)")
    args = ap.parse_args()

    # Load index and filter to this dataset
    idx_df = pd.read_parquet(args.index_parquet)
    idx_df = _resolve_index(idx_df)
    if "dataset_id" not in idx_df.columns or "is_control" not in idx_df.columns:
        raise ValueError("index parquet must contain 'dataset_id' and 'is_control' columns")
    idx_ds = idx_df[idx_df["dataset_id"].astype(str) == str(args.dataset_id)].copy()
    if idx_ds.empty:
        raise ValueError(f"No rows for dataset_id='{args.dataset_id}' in index parquet.")

    # Load z for this dataset and align
    Z = pd.read_parquet(args.z_parquet)
    Z = _resolve_index(Z)
    common = Z.index.intersection(idx_ds.index)
    if len(common) == 0:
        raise ValueError("No overlapping cell_ids between z parquet and index parquet for this dataset.")
    Z = Z.loc[common].astype(np.float32)
    idx_ds = idx_ds.loc[common]

    # Split controls vs perturbeds
    is_ctrl = idx_ds["is_control"].astype(bool).values
    cell_ids = idx_ds.index.to_numpy()
    ctrl_ids = cell_ids[is_ctrl]
    pert_ids = cell_ids[~is_ctrl]
    # Fast-path: optionally limit the number of perturbed cells processed
    if args.max_perturbed is not None and len(pert_ids) > args.max_perturbed:
        pert_ids = pert_ids[: args.max_perturbed]
    if len(ctrl_ids) == 0:
        raise ValueError("This dataset has zero controls; cannot build control neighbor index.")
    if len(pert_ids) == 0:
        print("[warn] This dataset has no perturbed cells; nothing to do.")
        # Still write empty outputs for consistency
        out_dir = os.path.dirname(args.out_parquet)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(columns=["cell_id","rank","control_id","dist","dataset_id"]).to_parquet(args.out_parquet, index=False)
        if args.out_summary is None:
            args.out_summary = os.path.splitext(args.out_parquet)[0] + ".summary.json"
        with open(args.out_summary, "w") as f:
            json.dump({"dataset_id": args.dataset_id, "k": args.k, "metric": args.metric, "n_controls": 0, "n_perturbed": 0}, f, indent=2)
        return

    Zc = Z.loc[ctrl_ids].values  # (C, d)
    Zq = Z.loc[pert_ids].values  # (Q, d)

    # Compute top-K control neighbors per perturbed
    K = min(int(args.k), len(ctrl_ids))
    if args.metric == "euclidean":
        nn_idx, nn_dist = _topk_euclidean(Zq, Zc, K=K, chunk=args.chunk)
    else:
        nn_idx, nn_dist = _topk_cosine(Zq, Zc, K=K, chunk=args.chunk)

    # Map neighbor indices to control cell_ids
    ctrl_ids_array = np.asarray(ctrl_ids)
    # Build long-form dataframe
    rows = []
    for qi, qcell in enumerate(pert_ids):
        for r in range(K):
            rows.append((qcell, r, ctrl_ids_array[nn_idx[qi, r]], float(nn_dist[qi, r])))
    out_df = pd.DataFrame(rows, columns=["cell_id","rank","control_id","dist"])
    out_df["dataset_id"] = args.dataset_id

    # Write outputs
    out_dir = os.path.dirname(args.out_parquet)
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)
    print(f"[ok] wrote neighbors: {args.out_parquet}  (rows={len(out_df)}, Q={len(pert_ids)}, K={K})")

    # Summary
    if args.out_summary is None:
        args.out_summary = os.path.splitext(args.out_parquet)[0] + ".summary.json"
    summary = {
        "dataset_id": args.dataset_id,
        "k": K,
        "metric": args.metric,
        "n_controls": int(len(ctrl_ids)),
        "n_perturbed": int(len(pert_ids)),
        "neighbors_rows": int(len(out_df)),
    }
    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] wrote summary:   {args.out_summary}")
    

if __name__ == "__main__":
    main()
