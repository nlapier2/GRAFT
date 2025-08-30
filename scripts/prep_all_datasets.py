#!/usr/bin/env python3
"""
prep_all_datasets.py
====================
Reads a datasets.yaml and a full index parquet, and for each dataset:
  1) runs encode_query_z.py to produce artifacts_v2/<DATASET_ID>/z.parquet
  2) runs knn_matcher.py to produce artifacts_v2/<DATASET_ID>/knn_controls.parquet

Usage:
  python scripts/prep_all_datasets.py \
    --datasets-yaml datasets.yaml \
    --scvi-model-dir artifacts_v2/scvi_k562_mak200k_control_only/scvi_K562/ \
    --index-parquet artifacts_v2/full_index.parquet \
    [--transform-batch None] [--overwrite]
"""

from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import yaml

def sh(args):
    print(">>", " ".join(map(str,args)))
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-yaml", required=True, help="Path to datasets.yaml")
    ap.add_argument("--scvi-model-dir", required=True, help="Directory of the trained scVI model")
    ap.add_argument("--index-parquet", required=True, help="Full index parquet (not controls-only)")
    ap.add_argument("--transform-batch", default=None, help="Optional scVI transform_batch")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds_cfg = yaml.safe_load(Path(args.datasets_yaml).read_text())

    # Resolve script paths robustly (works if train_gnn.py is top-level)
    repo_root = Path.cwd()
    encode_script = repo_root / "scripts" / "encode_query_z.py"
    knn_script    = repo_root / "scripts" / "knn_matcher.py"
    assert encode_script.exists(), f"encode_query_z.py not found at {encode_script}"
    assert knn_script.exists(), f"knn_matcher.py not found at {knn_script}"

    for ds in ds_cfg.get("datasets", []):
        ds_id   = str(ds["id"])
        h5ad    = str(ds["h5ad"])
        out_dir = Path("artifacts_v2") / ds_id
        out_dir.mkdir(parents=True, exist_ok=True)

        z_out   = out_dir / "z.parquet"
        knn_out = out_dir / "knn_controls.parquet"

        # 1) Encode z
        if args.overwrite or not z_out.exists():
            sh([sys.executable, str(encode_script),
                "--model-dir", args.scvi_model_dir,
                "--h5ad", h5ad,
                "--out", str(z_out),
                "--transform-batch", str(args.transform_batch) if args.transform_batch is not None else "None",
            ])
        else:
            print(f"✓ z exists: {z_out}")

        # 2) kNN controls (optional but recommended)
        # The matcher will use the index parquet to find controls and restrict to same dataset.
        if args.overwrite or not knn_out.exists():
            sh([sys.executable, str(knn_script),
                "--z-parquet", str(z_out),
                "--index-parquet", args.index_parquet,
                "--out", str(knn_out),
                "--k", "1",
            ])
        else:
            print(f"✓ knn exists: {knn_out}")

    print("All datasets processed. Outputs are under artifacts_v2/<DATASET_ID>/")

if __name__ == "__main__":
    main()
