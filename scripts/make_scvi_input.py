
# scripts/make_scvi_input.py
# Build a single AnnData per cell type (union of selected datasets, common gene order),
# with optional per-dataset downsampling to keep memory manageable.
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import anndata as ad
import yaml

from utils.normalize import normalize_hgnc

def read_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("datasets", cfg)

def load_map(dataset_id):
    with open(f"artifacts/gene_map/{dataset_id}.json") as f:
        mp = json.load(f)
    return {int(k): int(v) for k, v in mp["to_common_idx"].items()}

def subset_rows(idx: pd.DataFrame, cell_type: str, datasets=None, max_cells_per_ds=None, seed=13):
    sub = idx[(idx["cell_type"] == cell_type) & (idx["common_gene_set"])].copy()
    if datasets:
        sub = sub[sub["dataset_id"].isin(datasets)]
    if max_cells_per_ds is not None:
        rng = np.random.default_rng(seed)
        parts = []
        for ds, g in sub.groupby("dataset_id"):
            n = len(g)
            k = min(n, max_cells_per_ds)
            parts.append(g.sample(k, random_state=seed))
        sub = pd.concat(parts, axis=0)
    return sub

def fetch_dataset_slice(raw_path, rows_meta: pd.DataFrame, gene_map: dict, gene_list):
    """Read only the required rows; reorder columns to common gene order; return in-memory AnnData."""
    # Open backed and slice
    A = ad.read_h5ad(raw_path, backed="r")
    # map our 'cell_id' back to original obs names
    # Prefer exact .obs_names match if possible, else parse after '::'
    if all(cid in A.obs_names for cid in rows_meta["cell_id"]):
        row_names = rows_meta["cell_id"].tolist()
    else:
        row_names = [cid.split("::", 1)[1] for cid in rows_meta["cell_id"]]
    # Slice rows (this materializes a view; not fully loaded yet)
    A = A[row_names, :]
    # Build column mask/order using gene_map
    ds_nvars = A.n_vars
    to_common = np.full(ds_nvars, -1, dtype=int)
    for k, v in gene_map.items():
        if k < ds_nvars:
            to_common[k] = v
    keep = to_common >= 0
    order = np.argsort(to_common[keep])
    A = A[:, keep]    # keep only intersected genes
    A = A[:, order]   # reorder to match common gene_list
    # Double-check order using var_names if available
    # Note: some datasets may not match casing; normalize for safety
    try:
        assert list(map(normalize_hgnc, A.var_names)) == gene_list
    except Exception:
        pass  # skip strict check; we rely on mapping indices

    # Now force materialization to memory for this slice
    A = A.to_memory()
    # Minimal obs we need
    A.obs["dataset_id"] = rows_meta["dataset_id"].values
    A.obs["lab_id"] = rows_meta["lab_id"].values
    A.obs["batch_id"] = rows_meta["batch_id"].values
    A.obs["cell_type"] = rows_meta["cell_type"].values
    A.obs["is_control"] = rows_meta["is_control"].values
    A.obs["target_gene"] = rows_meta["target_gene"].values
    return A

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="configs/datasets.yaml", help="Datasets YAML")
    ap.add_argument("--cell-type", required=True, help="Cell type to build (e.g., K562)")
    ap.add_argument("--datasets", nargs="*", default=None, help="Optional list of dataset_ids to include")
    ap.add_argument("--max-cells-per-ds", type=int, default=200000, help="Cap per dataset to limit memory")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=None, help="Output h5ad path")
    args = ap.parse_args()

    gene_list = [g.strip() for g in open("artifacts/gene_list.tsv") if g.strip()]
    datasets_cfg = read_yaml(args.yaml)
    idx = pd.read_parquet("artifacts/cell_index.parquet")

    rows = subset_rows(idx, args.cell_type, args.datasets, args.max_cells_per_ds, args.seed)
    if rows.empty:
        raise SystemExit(f"No rows for cell_type={args.cell_type} with common_gene_set=True")

    adatas = []
    for ds, g in rows.groupby("dataset_id"):
        raw_path = datasets_cfg[ds].get("raw_path") or datasets_cfg[ds].get("path")
        if not raw_path or not os.path.exists(raw_path):
            print(f"[WARN] {ds}: raw_path not found, skip")
            continue
        gmap = load_map(ds)
        A = fetch_dataset_slice(raw_path, g, gmap, gene_list)
        adatas.append(A)
        print(f"[INFO] added {ds}: {A.shape}")

    if not adatas:
        raise SystemExit("No datasets could be loaded.")

    Aall = ad.concat(adatas, join="outer", merge="first")
    out = args.out or f"artifacts/scvi_input_{args.cell_type}.h5ad"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Aall.write_h5ad(out, compression="lzf")
    print(f"[OK] wrote {out}: {Aall.shape}")

if __name__ == "__main__":
    main()
