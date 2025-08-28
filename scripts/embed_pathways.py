#!/usr/bin/env python3
"""
Embed many pathway sources into a compact meta-anchor matrix for the factor encoder.

Inputs
------
- YAML pathway config (same format you already use), e.g.:

KEGG_Pathways:
  file: /data/misc_data/gene_list_kegg_pathways_enhanced.tsv
  gene_col: Gene_Symbol
  pathway_col: MSigDB_URL
  format: tsv

PRESAGE_GO_CellComp:
  file: /data/PRESAGE/cache/pathway_embeddings/c5.go.cc.v2023.2.Hs.symbols.pkl
  gene_col: NA
  pathway_col: NA
  format: presage

- A gene list to align to (either: --genes-from-h5ad or --genes-list)

Outputs
-------
- {outdir}/M_meta.npy           # (R x G) float32, nonnegative, column-normalized meta-anchors
- {outdir}/genes.txt            # gene order (one per line; matches M_meta columns)
- {outdir}/path2meta.parquet    # mapping: raw pathway → meta-anchor weights (K x R, sparse-ish)
- {outdir}/meta_config.json     # parameters used

Notes
-----
This produces the "anchored block" you feed to the factor encoder: rows=factors, cols=genes.
"""

import os, sys, json, argparse
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy import sparse

# Make local package imports work when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reuse your existing pathway IO helpers
from utils.load_pathways import load_pathway_sources, make_pathway_matrix  # noqa: E402
# ^ These are the same functions used by run_delta_methods.py to read your YAML & matrices
#   (keeps input format identical).  :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

try:
    import anndata as ad
except Exception:
    ad = None


# -------------------------- helpers --------------------------

def _read_genes_from_h5ad(path: str) -> List[str]:
    if ad is None:
        raise RuntimeError("anndata is required for --genes-from-h5ad")
    A = ad.read_h5ad(path, backed="r")
    genes = list(A.var_names)
    A.file.close()
    return genes

def _read_genes_list(path: str) -> List[str]:
    # accepts txt (one per line) or csv with headerless single column
    if path.lower().endswith(".csv"):
        return list(pd.read_csv(path, header=None)[0].astype(str))
    else:
        with open(path, "r") as fh:
            return [ln.strip() for ln in fh if ln.strip()]

def _nnz_per_column(X: pd.DataFrame) -> np.ndarray:
    arr = X.values
    return np.count_nonzero(arr, axis=0)

def _tfidf_weight(X: pd.DataFrame) -> pd.DataFrame:
    """Simple TF–IDF on gene×pathway matrix X (genes as rows, pathways as cols).
    tf = column-normalized per pathway; idf = log((K+1)/(df+1)).
    """
    GxK = X.values.astype(np.float64, copy=False)
    # term frequency per pathway (normalize each column)
    col_sum = GxK.sum(axis=0)
    col_sum[col_sum == 0] = 1.0
    tf = GxK / col_sum[None, :]
    # document frequency per gene (across pathways)
    df = (GxK > 0).sum(axis=1).astype(np.float64)
    idf = np.log((GxK.shape[1] + 1.0) / (df + 1.0))
    W = tf * idf[:, None]
    return pd.DataFrame(W, index=X.index, columns=X.columns)

def _column_normalize_nonneg(W_fg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    W_fg = np.maximum(W_fg, 0.0)
    colsum = W_fg.sum(axis=0, keepdims=True)
    colsum = np.where(colsum < eps, 1.0, colsum)
    return (W_fg / colsum).astype(np.float32, copy=False)

def _save_sparse_path2meta(H_kr: np.ndarray, path_names: List[str], out_path: str, topk: int = 10):
    """Save a sparse-ish mapping: for each raw pathway, top-k meta-anchors and weights."""
    # H is (K x R): weight of each meta-anchor per raw pathway
    K, R = H_kr.shape
    rows = []
    for i in range(K):
        w = H_kr[i]
        if not np.any(w):
            continue
        top_idx = np.argsort(w)[::-1][:topk]
        rows.append(
            pd.DataFrame({
                "pathway": path_names[i],
                "meta_idx": top_idx,
                "weight": w[top_idx].astype(np.float32)
            })
        )
    if rows:
        df = pd.concat(rows, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["pathway", "meta_idx", "weight"])
    df.to_parquet(out_path, index=False)


# -------------------------- main work --------------------------

def build_gene_by_pathway_matrix(
    pathway_cfg_yaml: str,
    genes: List[str],
    min_size: int,
    max_size: int,
    prefix_source: bool = True,
) -> pd.DataFrame:
    """
    Read all pathway sources (your YAML format), align to 'genes',
    and horizontally concatenate into a single DataFrame (genes × total_pathways).
    """
    sources = load_pathway_sources(pathway_cfg_yaml)  # same reader used by run_delta_methods.py  :contentReference[oaicite:5]{index=5}
    mats = []
    for src_name, meta in sources.items():
        M = make_pathway_matrix(
            file_name=meta["file"],
            gene_col=meta["gene_col"],
            pathway_col=meta["pathway_col"],
            format=meta["format"],
            var_names=genes,  # align to our gene list
        )  # returns genes×pathways membership as a DataFrame
        if not isinstance(M, pd.DataFrame):
            M = pd.DataFrame(M, index=genes)
            # If columns are unknown, synthesize names
            M.columns = [f"{src_name}__p{i}" for i in range(M.shape[1])]
        else:
            # ensure correct index order & cast
            M = M.reindex(index=genes).fillna(0.0)
            # add source prefix to avoid collisions
            if prefix_source:
                M = M.rename(columns=lambda c: f"{src_name}__{c}")
        # size filter by nonzero counts (robust to continuous weights)
        nnz = _nnz_per_column(M)
        keep = (nnz >= min_size) & (nnz <= max_size)
        M = M.loc[:, keep]
        if M.shape[1] == 0:
            print(f"[warn] source {src_name} yielded 0 pathways after size filtering")
            continue
        mats.append(M.astype(np.float32, copy=False))
    if not mats:
        raise RuntimeError("No pathways survived filters; relax min/max size or check YAML.")
    X = pd.concat(mats, axis=1)
    # drop any all-zero columns
    X = X.loc[:, (X.values != 0).any(axis=0)]
    return X


def reduce_to_meta_anchors(
    X_gene_by_path: pd.DataFrame,
    n_components: int = 256,
    method: str = "nmf",
    tfidf: bool = True,
    nmf_max_iter: int = 500,
    nmf_init: str = "nndsvd",
    svd_random_state: int = 0,
):
    """
    Convert gene×pathway to meta-anchors × gene (R×G) and pathway→meta map (K×R).
    - For NMF: fit on (K×G) & return components_ (R×G), H = transform(K×G) (K×R)
    - For SVD: fit on (K×G), take components_ (R×G), clip +, and H = (K×R) via transform
    """
    # Optionally TF–IDF weight to de-emphasize massive pathways
    Xw = _tfidf_weight(X_gene_by_path) if tfidf else X_gene_by_path
    # We fit on (K×G): pathways as samples, genes as features
    X_k_by_g = Xw.T.values  # (K, G)
    genes = list(Xw.index)
    path_names = list(Xw.columns)

    if method == "nmf":
        from sklearn.decomposition import NMF
        model = NMF(
            n_components=n_components,
            init=nmf_init,
            max_iter=nmf_max_iter,
            random_state=0,
            solver="cd",
            beta_loss="frobenius",
        )
        H_kr = model.fit_transform(X_k_by_g)        # (K, R)
        W_rg = model.components_                    # (R, G) nonnegative
        W_rg = _column_normalize_nonneg(W_rg)       # col-normalize over factors
        return W_rg, H_kr, genes, path_names

    elif method == "svd":
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=n_components, random_state=svd_random_state)
        Z_kr = svd.fit_transform(X_k_by_g)          # (K, R): low-D coords of pathways
        C_rg = svd.components_                      # (R, G): can have negatives
        W_rg = _column_normalize_nonneg(np.maximum(C_rg, 0.0))  # clip negatives, normalize
        return W_rg, Z_kr, genes, path_names

    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathway-config", required=True, help="YAML listing pathway sources (same format you already use)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--genes-from-h5ad", help="h5ad file whose var_names define gene order")
    g.add_argument("--genes-list", help="Text/CSV with one gene symbol per line")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-components", type=int, default=256)
    ap.add_argument("--method", choices=["nmf", "svd"], default="nmf")
    ap.add_argument("--no-tfidf", action="store_true", help="Disable TF–IDF weighting")
    ap.add_argument("--min-size", type=int, default=10, help="Min nonzero genes per raw pathway to keep")
    ap.add_argument("--max-size", type=int, default=999999999, help="Max nonzero genes per raw pathway to keep")
    ap.add_argument("--nmf-max-iter", type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Resolve genes
    if args.genes_from_h5ad:
        genes = _read_genes_from_h5ad(args.genes_from_h5ad)
    else:
        genes = _read_genes_list(args.genes_list)

    # Build a single gene×pathway matrix from all sources
    X = build_gene_by_pathway_matrix(
        pathway_cfg_yaml=args.pathway_config,
        genes=genes,
        min_size=args.min_size,
        max_size=args.max_size,
        prefix_source=True,
    )

    # Reduce to meta-anchors
    W_rg, H_kr, genes, path_names = reduce_to_meta_anchors(
        X_gene_by_path=X,
        n_components=args.n_components,
        method=args.method,
        tfidf=(not args.no_tfidf),
        nmf_max_iter=args.nmf_max_iter,
    )

    # Save artifacts
    np.save(os.path.join(args.outdir, "M_meta.npy"), W_rg)                 # (R, G)
    with open(os.path.join(args.outdir, "genes.txt"), "w") as fh:
        fh.write("\n".join(genes))
    _save_sparse_path2meta(H_kr, path_names, os.path.join(args.outdir, "path2meta.parquet"), topk=10)

    meta = {
        "n_components": int(args.n_components),
        "method": args.method,
        "tfidf": not args.no_tfidf,
        "min_size": int(args.min_size),
        "max_size": int(args.max_size),
        "nmf_max_iter": int(args.nmf_max_iter),
        "genes_from": args.genes_from_h5ad or args.genes_list,
        "pathway_config": args.pathway_config,
        "W_shape": [int(x) for x in W_rg.shape],
        "X_shape": [int(x) for x in X.shape],
    }
    with open(os.path.join(args.outdir, "meta_config.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[OK] wrote M_meta.npy (R×G = {W_rg.shape}), genes.txt, and path2meta.parquet to {args.outdir}")


if __name__ == "__main__":
    main()
