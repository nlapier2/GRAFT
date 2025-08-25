
# scripts/make_scvi_input.py
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


def _choose_blocks(n, block_size, n_blocks, rng):
    """Return a list of (start, end) half-open intervals within [0, n)."""
    if block_size <= 0:
        return [(0, n)]
    if n <= block_size or n_blocks <= 1:
        return [(0, min(n, block_size))]
    max_start = max(1, n - block_size)
    starts = rng.integers(low=0, high=max_start, size=n_blocks)
    starts = sorted(set(int(s) for s in starts))
    return [(s, min(n, s + block_size)) for s in starts]

def subset_rows_blocked(idx: pd.DataFrame,
                        cell_type: str,
                        datasets=None,
                        max_cells_per_ds=None,
                        seed=13,
                        block_size=20000,
                        blocks_per_stratum=3,
                        strata=("batch_id","is_control")):
    """Sample contiguous blocks per dataset (optionally stratified) to minimize random HDF5 I/O.
       Returns a concatenated DataFrame of selected rows preserving within-dataset file order.
    """
    rng = np.random.default_rng(seed)
    sub = idx[(idx["cell_type"] == cell_type) & (idx["common_gene_set"])].copy()
    if datasets:
        sub = sub[sub["dataset_id"].isin(datasets)]
    out_parts = []
    strata = tuple([s for s in strata if s in sub.columns])
    for ds, g in sub.groupby("dataset_id", sort=False):
        n = len(g)
        if n == 0:
            continue
        cap = max_cells_per_ds or n
        if block_size <= 0:
            sel = g.iloc[:cap]
        elif len(strata) == 0:
            n_blocks = max(1, int(np.ceil(cap / block_size)))
            blocks = _choose_blocks(n, block_size, n_blocks, rng)
            sel = pd.concat([g.iloc[s:e] for (s,e) in blocks], axis=0)
        else:
            sel_parts = []
            tot_blocks = max(1, int(np.ceil(cap / block_size)))
            sizes = g.groupby(list(strata), sort=False).size()
            weights = (sizes / sizes.sum()).clip(lower=0)
            alloc = (weights * tot_blocks).round().astype(int).replace(0, 1)
            diff = int(alloc.sum() - tot_blocks)
            if diff != 0:
                order = alloc.sort_values(ascending=(diff>0))
                for key in order.index:
                    if diff == 0: break
                    if diff > 0 and alloc[key] > 1:
                        alloc[key] -= 1; diff -= 1
                    elif diff < 0:
                        alloc[key] += 1; diff += 1
            for key, n_blocks in alloc.items():
                if isinstance(key, tuple):
                    mask = (g[list(strata)].apply(tuple, axis=1) == key)
                    grp = g.loc[mask]
                else:
                    grp = g[g[strata[0]] == key]
                m = len(grp)
                if m == 0: continue
                blocks = _choose_blocks(m, block_size, n_blocks, rng)
                sel_parts.extend([grp.iloc[s:e] for (s,e) in blocks])
            sel = pd.concat(sel_parts, axis=0) if sel_parts else g.head(min(cap, n))
        if len(sel) > cap:
            sel = sel.iloc[:cap]
        out_parts.append(sel)
    if not out_parts:
        return sub.iloc[0:0]
    return pd.concat(out_parts, axis=0)


def subset_rows(idx: pd.DataFrame, cell_type: str, datasets=None, max_cells_per_ds=None, seed=13):
    sub = idx[(idx["cell_type"] == cell_type) & (idx["common_gene_set"])].copy()
    if datasets:
        sub = sub[sub["dataset_id"].isin(datasets)]
    if max_cells_per_ds is not None:
        parts = []
        for ds, g in sub.groupby("dataset_id"):
            k = min(len(g), max_cells_per_ds)
            parts.append(g.sample(k, random_state=seed))
        sub = pd.concat(parts, axis=0)
    return sub

def _build_keep_and_order(ds_nvars: int, gene_map: dict):
    to_common = np.full(ds_nvars, -1, dtype=int)
    for k, v in gene_map.items():
        if 0 <= k < ds_nvars:
            to_common[k] = v
    keep_bool = to_common >= 0
    order_cols = np.argsort(to_common[keep_bool])
    return keep_bool, order_cols

def fetch_dataset_slice_fast(raw_path, rows_meta: pd.DataFrame, gene_map: dict, gene_list, row_block=None):
    A_b = ad.read_h5ad(raw_path, backed="r")
    cid = rows_meta["cell_id"].tolist()
    if not all(c in A_b.obs_names for c in cid):
        cid = [c.split("::", 1)[1] for c in cid]
    row_pos = A_b.obs_names.get_indexer(cid)
    if (row_pos < 0).any():
        missing = sum(row_pos < 0)
        raise ValueError(f"{missing} requested cell_ids not found in {raw_path}")
    order = np.argsort(row_pos)
    row_pos_sorted = row_pos[order]
    rows_meta_sorted = rows_meta.iloc[order].reset_index(drop=True)
    keep_bool, order_cols = _build_keep_and_order(A_b.n_vars, gene_map)
    blocks = []
    if row_block and row_block > 0:
        for s in range(0, len(row_pos_sorted), row_block):
            block_pos = row_pos_sorted[s:s+row_block]
            A_blk = A_b[block_pos, :].to_memory()
            A_blk = A_blk[:, keep_bool]
            A_blk = A_blk[:, order_cols]
            rm = rows_meta_sorted.iloc[s:s+row_block]
            A_blk.obs["dataset_id"]  = rm["dataset_id"].values
            A_blk.obs["lab_id"]      = rm["lab_id"].values
            A_blk.obs["batch_id"]    = rm["batch_id"].values
            A_blk.obs["cell_type"]   = rm["cell_type"].values
            A_blk.obs["is_control"]  = rm["is_control"].values
            A_blk.obs["target_gene"] = rm["target_gene"].values
            blocks.append(A_blk)
        A_sorted = ad.concat(blocks, join="outer", merge="first")
    else:
        A_rows = A_b[row_pos_sorted, :].to_memory()
        A_rows = A_rows[:, keep_bool]
        A_rows = A_rows[:, order_cols]
        A_rows.obs["dataset_id"]  = rows_meta_sorted["dataset_id"].values
        A_rows.obs["lab_id"]      = rows_meta_sorted["lab_id"].values
        A_rows.obs["batch_id"]    = rows_meta_sorted["batch_id"].values
        A_rows.obs["cell_type"]   = rows_meta_sorted["cell_type"].values
        A_rows.obs["is_control"]  = rows_meta_sorted["is_control"].values
        A_rows.obs["target_gene"] = rows_meta_sorted["target_gene"].values
        A_sorted = A_rows
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    A = A_sorted[inv, :]
    try:
        assert list(map(normalize_hgnc, A.var_names)) == gene_list
    except Exception:
        A.var_names = gene_list
    return A

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="configs/datasets.yaml", help="Datasets YAML")
    ap.add_argument("--cell-type", required=True, help="Cell type to build (e.g., K562)")
    ap.add_argument("--datasets", nargs="*", default=None, help="Optional list of dataset_ids to include")
    ap.add_argument("--max-cells-per-ds", type=int, default=200000, help="Cap per dataset to limit memory")
    ap.add_argument("--block-size", type=int, default=20000, help="If >0, sample contiguous blocks (faster I/O)")
    ap.add_argument("--blocks-per-stratum", type=int, default=3, help="Blocks per stratum when block sampling")
    ap.add_argument("--strata", nargs="*", default=[], help="Strata columns for block sampling (if present)")
    ap.add_argument("--row-block", type=int, default=0, help="Optional row block size for backed reads (0=off)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=None, help="Output h5ad path")
    ap.add_argument("--assemble", choices=["vstack","concat"], default="vstack", help="How to assemble per-dataset chunks into one AnnData (default: vstack)")
    args = ap.parse_args()
    gene_list = [g.strip() for g in open("artifacts/gene_list.tsv") if g.strip()]
    datasets_cfg = read_yaml(args.yaml)
    idx = pd.read_parquet("artifacts/cell_index.parquet")
    rows = subset_rows_blocked(idx, args.cell_type, args.datasets, args.max_cells_per_ds, args.seed,
                           block_size=args.block_size if args.block_size and args.block_size>0 else 0,
                           blocks_per_stratum=args.blocks_per_stratum,
                           strata=tuple(args.strata)) if (args.block_size and args.block_size>0) else \
                           subset_rows(idx, args.cell_type, args.datasets, args.max_cells_per_ds, args.seed)
    if rows.empty:
        raise SystemExit(f"No rows for cell_type={args.cell_type} with common_gene_set=True")
    adatas = []
    for ds, g in rows.groupby("dataset_id"):
        raw_path = datasets_cfg[ds].get("raw_path") or datasets_cfg[ds].get("path")
        if not raw_path or not os.path.exists(raw_path):
            print(f"[WARN] {ds}: raw_path not found, skip")
            continue
        gmap = load_map(ds)
        A = fetch_dataset_slice_fast(raw_path, g, gmap, gene_list, row_block=args.row_block)
        adatas.append(A)
        print(f"[INFO] added {ds}: {A.shape}")
        if not adatas:
            raise SystemExit("No datasets could be loaded.")

    out = args.out or f"artifacts/scvi_input_{args.cell_type}.h5ad"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.assemble == "vstack":
        from scipy import sparse
        X_list = [A.X.tocsr() if hasattr(A.X, "tocsr") else sparse.csr_matrix(A.X) for A in adatas]
        obs = pd.concat([A.obs for A in adatas], axis=0)
        var = pd.DataFrame(index=gene_list)
        X = sparse.vstack(X_list, format="csr")
        Aall = ad.AnnData(X=X, obs=obs, var=var)
    else:
        Aall = ad.concat(adatas, join="outer", merge="first")
    Aall.write_h5ad(out, compression="lzf")
    print(f"[OK] wrote {out}: {Aall.shape}")

if __name__ == "__main__":
    main()
