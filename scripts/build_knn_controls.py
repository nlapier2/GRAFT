#!/usr/bin/env python3
"""
build_knn_controls.py
===============================
Build a fast kNN index over control latents directly from the NPZ produced by train_scvi.py.

Inputs:
  --z-npz         Path to NPZ with arrays: z (N x d, float32) and cell_ids (N, str)
  --out-dir       Output directory for ANN artifacts

Optional:
  --metric        l2 | cosine  (default: cosine)
  --index-type    auto | faiss_flat | faiss_ivf_hnsw | hnswlib
                  default: auto → faiss_flat if faiss available, else hnswlib
  --faiss-nlist   IVF list count (default 4096)
  --faiss-nprobe  IVF search probes (default 32)
  --hnsw-M        HNSW graph M (default 32)
  --hnsw-efC      HNSW efConstruction (default 200)
  --hnsw-efS      HNSW efSearch at query time (default 100)

Outputs (under out-dir):
  ctrl_ids.parquet   (single column 'cell_id' in the exact row order used by the index)
  knn.index          (binary ANN index file; FAISS or hnswlib)
  knn_meta.json      (index metadata: backend, metric, dim, counts, params, filenames)

Notes:
- If metric=cosine, vectors are L2-normalized before indexing and querying.
- Order of ctrl_ids matches the order vectors were added to the index.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd

# Optional ANN backends
_HAS_FAISS = False
_HAS_HNSW = False
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    pass

try:
    import hnswlib  # type: ignore
    _HAS_HNSW = True
except Exception:
    pass


def _load_z_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        d = np.load(path, allow_pickle=False)
    except ValueError:
        # tolerate old dumps that might need allow_pickle
        d = np.load(path, allow_pickle=True)
    if "z" not in d or "cell_ids" not in d:
        raise ValueError(f"{path} must contain arrays 'z' and 'cell_ids'")
    Z = np.asarray(d["z"], dtype=np.float32, order="C")
    cell_ids = np.asarray(d["cell_ids"]).astype("U")
    if Z.ndim != 2:
        raise ValueError(f"z must be 2D (N x d); got shape {Z.shape}")
    if len(cell_ids) != Z.shape[0]:
        raise ValueError(f"cell_ids length {len(cell_ids)} != z rows {Z.shape[0]}")
    return Z, cell_ids


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _build_faiss_index(
    vectors: np.ndarray,
    metric: str,
    index_type: str,
    nlist: int = 4096,
    nprobe: int = 32,
):
    d = vectors.shape[1]
    if metric == "cosine":
        xb = _l2_normalize_rows(vectors.astype(np.float32, copy=False))
        metric_f = faiss.METRIC_INNER_PRODUCT
    else:
        xb = vectors.astype(np.float32, copy=False)
        metric_f = faiss.METRIC_L2

    if index_type == "faiss_flat":
        index = faiss.IndexFlatIP(d) if metric_f == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    elif index_type == "faiss_ivf_hnsw":
        quantizer = faiss.IndexHNSWFlat(d, 32, metric_f == faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, metric_f)
    else:
        raise ValueError(f"Unknown FAISS index_type: {index_type}")

    if hasattr(index, "is_trained") and not index.is_trained:
        index.train(xb)
    index.add(xb)
    try:
        index.nprobe = nprobe
    except Exception:
        pass
    return index


def _build_hnsw_index(
    vectors: np.ndarray,
    metric: str,
    M: int = 32,
    efC: int = 200,
    efS: int = 100,
):
    d = vectors.shape[1]
    space = "cosine" if metric == "cosine" else "l2"
    p = hnswlib.Index(space=space, dim=d)  # type: ignore
    p.init_index(max_elements=vectors.shape[0], ef_construction=efC, M=M)
    xb = _l2_normalize_rows(vectors.astype(np.float32, copy=False)) if metric == "cosine" else vectors.astype(np.float32, copy=False)
    labels = np.arange(xb.shape[0], dtype=np.int64)
    p.add_items(xb, labels)
    p.set_ef(efS)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-npz", required=True, help="NPZ with arrays 'z' (N x d) and 'cell_ids' (N), from train_scvi.py")
    ap.add_argument("--out-dir", required=True, help="Where to write index artifacts")
    ap.add_argument("--metric", choices=["l2", "cosine"], default="cosine")
    ap.add_argument("--index-type", choices=["auto", "faiss_flat", "faiss_ivf_hnsw", "hnswlib"], default="auto")
    ap.add_argument("--faiss-nlist", type=int, default=4096)
    ap.add_argument("--faiss-nprobe", type=int, default=32)
    ap.add_argument("--hnsw-M", type=int, default=32)
    ap.add_argument("--hnsw-efC", type=int, default=200)
    ap.add_argument("--hnsw-efS", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load Z and cell_ids directly from NPZ (as written by train_scvi.py)
    Z, cell_ids = _load_z_npz(args.z_npz)
    N, d = Z.shape
    print(f"[info] loaded z: N={N}, d={d}")

    # Pick index backend
    index_type = args.index_type
    if index_type == "auto":
        if _HAS_FAISS:
            index_type = "faiss_flat"
        elif _HAS_HNSW:
            index_type = "hnswlib"
        else:
            raise RuntimeError("No ANN backend available. Install faiss or hnswlib, or choose a specific index-type that is available.")

    # Build index
    print(f"[info] building index: type={index_type}, metric={args.metric}, dim={d}, N={N}")
    meta = {
        "metric": args.metric,
        "dim": int(d),
        "n_items": int(N),
        "backend": None,
        "type": index_type,
        "z_source": os.path.basename(args.z_npz),
        "ctrl_ids": "ctrl_ids.parquet",
        "index_file": "knn.index",
    }

    if index_type.startswith("faiss"):
        if not _HAS_FAISS:
            raise RuntimeError("FAISS not available but index-type requires it.")
        index = _build_faiss_index(
            Z, metric=args.metric, index_type=index_type,
            nlist=args.faiss_nlist, nprobe=args.faiss_nprobe
        )
        index_path = os.path.join(args.out_dir, "knn.index")
        faiss.write_index(index, index_path)  # type: ignore
        meta.update({
            "backend": "faiss",
            "faiss_nlist": args.faiss_nlist,
            "faiss_nprobe": args.faiss_nprobe,
        })
    else:
        if not _HAS_HNSW:
            raise RuntimeError("hnswlib not available but index-type requires it.")
        p = _build_hnsw_index(
            Z, metric=args.metric,
            M=args.hnsw_M, efC=args.hnsw_efC, efS=args.hnsw_efS
        )
        index_path = os.path.join(args.out_dir, "knn.index")
        p.save_index(index_path)
        meta.update({
            "backend": "hnswlib",
            "hnsw_M": args.hnsw_M,
            "hnsw_efC": args.hnsw_efC,
            "hnsw_efS": args.hnsw_efS,
        })

    # Persist the control cell_ids in the same added order
    ids_path = os.path.join(args.out_dir, "ctrl_ids.parquet")
    pd.DataFrame({"cell_id": cell_ids}).to_parquet(ids_path, index=False)

    # Meta
    meta_path = os.path.join(args.out_dir, "knn_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] wrote index: {index_path}")
    print(f"[ok] wrote ctrl_ids: {ids_path}")
    print(f"[ok] wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
