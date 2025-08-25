# scripts/make_scvi_input.py  (v1-simplified)
# Minimal: per-dataset contiguous block sampling + one backed block read.
# Changes from original:
#   (1) Does .to_memory() after a SINGLE backed slice (no chained views).
#   (2) Loads a single contiguous block per dataset (as large as needed for the cap).
# No strata, no row-block args, no extra knobs.

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

def subset_rows_contiguous(idx: pd.DataFrame, cell_type: str, datasets=None, max_cells_per_ds=None, seed=13):
    """Pick ONE contiguous block per dataset up to cap (max_cells_per_ds)."""
    rng = np.random.default_rng(seed)
    sub = idx[(idx["cell_type"] == cell_type) & (idx["common_gene_set"])].copy()
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

def fetch_dataset_slice_block(raw_path, rows_meta: pd.DataFrame, gene_map: dict, gene_list):
    """Fast path: read a SINGLE contiguous [start:end) block from backed H5AD, then filter in-memory.
       Steps:
         - Map requested rows to integer positions in file.
         - Take start=min(pos), end=max(pos)+1 → one backed slice → .to_memory().
         - In-memory: pick requested rows from that block; then column keep+reorder.
    """
    A_b = ad.read_h5ad(raw_path, backed="r")

    # Map our global cell_id to dataset obs_names, then to positions
    cid = rows_meta["cell_id"].tolist()
    if not all(c in A_b.obs_names for c in cid):
        cid = [c.split("::", 1)[1] for c in cid]
    pos = A_b.obs_names.get_indexer(cid)
    if (pos < 0).any():
        missing = int((pos < 0).sum())
        raise ValueError(f"{missing} requested cell_ids not found in {raw_path}")

    start = int(pos.min())
    end   = int(pos.max()) + 1

    # SINGLE backed slice, then materialize
    A_blk = A_b[start:end, :].to_memory()

    # Filter ONLY the requested rows (relative to start), and restore original order
    rel = (pos - start).astype(int)
    mask = np.zeros(A_blk.n_obs, dtype=bool)
    mask[rel] = True
    A_sel = A_blk[mask, :]

    # Reorder rows to match rows_meta's original order
    # rows_meta order corresponds to our selection order; A_sel currently in file order among selected
    order_map = {r: i for i, r in enumerate(rel)}
    rel_in_sel = np.where(mask)[0]
    desired_idx = np.array([order_map[r] for r in rel_in_sel])
    inv = np.argsort(desired_idx)
    A_sel = A_sel[inv, :]

    # Column keep+reorder in-memory
    keep_bool, order_cols = _build_keep_and_order(A_b.n_vars, gene_map)
    A_sel = A_sel[:, keep_bool]
    A_sel = A_sel[:, order_cols]

    # Attach *only* the minimal obs we need and ensure safe dtypes
    A_sel.obs_names = rows_meta["cell_id"].values  # make global IDs the index
    obs_keep = pd.DataFrame(index=A_sel.obs_names)
    obs_keep["dataset_id"]  = rows_meta["dataset_id"].astype("category").values
    obs_keep["lab_id"]      = rows_meta["lab_id"].astype("category").values
    obs_keep["batch_id"]    = rows_meta["batch_id"].astype("category").values
    obs_keep["cell_type"]   = rows_meta["cell_type"].astype("category").values
    obs_keep["is_control"]  = rows_meta["is_control"].astype(bool).values
    # target_gene may be missing/NA for controls; categories handle NA cleanly
    obs_keep["target_gene"] = rows_meta["target_gene"].astype("category").values
    A_sel.obs = obs_keep

    # Ensure common var names
    try:
        assert list(map(normalize_hgnc, A_sel.var_names)) == gene_list
    except Exception:
        A_sel.var_names = gene_list

    return A_sel

# ------------------------------ main ------------------------------

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

    rows = subset_rows_contiguous(idx, args.cell_type, args.datasets, args.max_cells_per_ds, args.seed)
    if rows.empty:
        raise SystemExit(f"No rows for cell_type={args.cell_type} with common_gene_set=True")

    X_list, obs_list = [], []
    for ds, g in rows.groupby("dataset_id", sort=False):
        raw_path = datasets_cfg[ds].get("raw_path") or datasets_cfg[ds].get("path")
        if not raw_path or not os.path.exists(raw_path):
            print(f"[WARN] {ds}: raw_path not found, skip")
            continue
        gmap = load_map(ds)
        A = fetch_dataset_slice_block(raw_path, g, gmap, gene_list)
        # Collect
        X_list.append(A.X.tocsr() if hasattr(A.X, "tocsr") else sparse.csr_matrix(A.X))
        obs_list.append(A.obs)

        print(f"[INFO] added {ds}: {A.shape} (read [{A.n_obs} rows] from ONE contiguous block)")

    if not X_list:
        raise SystemExit("No datasets could be loaded.")

    # Assemble final AnnData via vstack (fast & simple)
    X = sparse.vstack(X_list, format="csr")
    obs = pd.concat(obs_list, axis=0)
    var = pd.DataFrame(index=gene_list)
    Aall = ad.AnnData(X=X, obs=obs, var=var)

    out = args.out or f"artifacts/scvi_input_{args.cell_type}.h5ad"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Aall.write_h5ad(out, compression="gzip")
    print(f"[OK] wrote {out}: {Aall.shape}")

if __name__ == "__main__":
    main()
