#!/usr/bin/env python3
"""
QC for meta-anchors produced by embed_pathways.py.

Inputs
------
- --embed-outdir : directory containing:
    - M_meta.npy      (R x G) meta-anchor matrix (rows=factors, cols=genes)
    - genes.txt       gene order used for columns of M_meta
    - path2meta.parquet (optional) pathway → meta weights (top-k per pathway)
- --pathway-config : YAML listing pathway sources (same format you already use)
  (We re-load the raw pathways to compute enrichment against the meta-anchors.)

Outputs
-------
- {out}/meta_top_genes.csv
- {out}/meta_enrichment.parquet
- {out}/meta_top_paths_by_weight.csv  (if path2meta available)
- {out}/qc_meta_report.md

Notes
-----
- Enrichment uses hypergeometric tail (SciPy), vectorized over all raw pathways.
- We test each meta against all pathways (after size filtering); FDR is applied per meta.
- Top-genes per meta are taken directly from M_meta row weights.
"""

import os, sys, argparse, json
from typing import List, Tuple
import numpy as np
import pandas as pd

# make local package imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.load_pathways import load_pathway_sources, make_pathway_matrix  # reuse your loaders

try:
    from scipy.stats import hypergeom
except Exception as e:
    raise SystemExit("scipy is required for enrichment tests. pip install scipy") from e


# -------------------------- helpers --------------------------

def _load_meta(embed_outdir: str) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    W = np.load(os.path.join(embed_outdir, "M_meta.npy"))
    with open(os.path.join(embed_outdir, "genes.txt"), "r") as fh:
        genes = [ln.strip() for ln in fh if ln.strip()]
    p2m_path = os.path.join(embed_outdir, "path2meta.parquet")
    P2M = pd.read_parquet(p2m_path) if os.path.exists(p2m_path) else None
    return W.astype(np.float32, copy=False), genes, P2M


def _build_gene_by_pathway_matrix(pathway_cfg_yaml: str, genes: List[str],
                                  min_size: int, max_size: int,
                                  prefix_source: bool = True) -> pd.DataFrame:
    """
    Recreate the raw gene×pathway membership (genes as index, pathways as columns),
    using the same YAML + readers as your GP pipeline.
    """
    sources = load_pathway_sources(pathway_cfg_yaml)
    mats = []
    for src_name, meta in sources.items():
        M = make_pathway_matrix(
            file_name=meta["file"],
            gene_col=meta["gene_col"],
            pathway_col=meta["pathway_col"],
            format=meta["format"],
            var_names=genes,
        )
        if not isinstance(M, pd.DataFrame):
            M = pd.DataFrame(M, index=genes)
            M.columns = [f"{src_name}__p{i}" for i in range(M.shape[1])]
        else:
            M = M.reindex(index=genes).fillna(0.0)
            if prefix_source:
                M = M.rename(columns=lambda c: f"{src_name}__{c}")
        # filter by size (nonzero genes per pathway)
        nnz = (M.values != 0).sum(axis=0)
        keep = (nnz >= min_size) & (nnz <= max_size)
        M = M.loc[:, keep]
        if M.shape[1] == 0:
            continue
        mats.append(M.astype(np.float32, copy=False))
    if not mats:
        raise RuntimeError("No pathways survived filters; relax min/max size or check YAML.")
    X = pd.concat(mats, axis=1)
    # drop any all-zero columns
    X = X.loc[:, (X.values != 0).any(axis=0)]
    return X


def _topk_genes_per_meta(W_rg: np.ndarray, genes: List[str], topk: int) -> pd.DataFrame:
    rows = []
    R, G = W_rg.shape
    g = np.array(genes)
    for r in range(R):
        w = W_rg[r]
        if topk >= len(w):
            idx = np.argsort(w)[::-1]
        else:
            idx = np.argpartition(w, -topk)[-topk:]
            idx = idx[np.argsort(w[idx])[::-1]]
        rows.append(pd.DataFrame({
            "meta_idx": r,
            "rank": np.arange(1, len(idx)+1),
            "gene": g[idx],
            "weight": w[idx].astype(np.float32)
        }))
    return pd.concat(rows, ignore_index=True)


def _enrichment_for_meta_set(X_bool, set_mask: np.ndarray) -> pd.DataFrame:
    """
    Vectorized hypergeometric enrichment of a single gene set against all pathways.
    X_bool: scipy.sparse (preferred) or numpy bool array with shape (G, K)
    set_mask: boolean mask length G
    Returns DataFrame with: 'overlap','path_size','meta_size','pval'
    """
    G, K = X_bool.shape
    m = int(set_mask.sum())
    if m == 0:
        return pd.DataFrame({"overlap": [], "path_size": [], "meta_size": [], "pval": []})

    try:
        # sparse path: index selected rows and sum across rows
        from scipy import sparse as sp
        if sp.issparse(X_bool):
            # ensure efficient column ops
            Xc = X_bool.tocsc(copy=False)
            # overlaps = sum over selected rows for each column
            overlaps = np.asarray(Xc[set_mask, :].sum(axis=0)).ravel()
            # path_sizes = column sums
            path_sizes = np.asarray(Xc.sum(axis=0)).ravel()
        else:
            # dense fallback
            Xd = X_bool.astype(bool, copy=False)
            overlaps = (Xd[set_mask, :]).sum(axis=0)
            path_sizes = Xd.sum(axis=0)
    except Exception:
        # very conservative dense fallback
        Xd = np.asarray(X_bool, dtype=bool)
        overlaps = (Xd[set_mask, :]).sum(axis=0)
        path_sizes = Xd.sum(axis=0)

    from scipy.stats import hypergeom
    pvals = hypergeom.sf(overlaps - 1, G, path_sizes, m)
    return pd.DataFrame({
        "overlap": overlaps.astype(np.int32),
        "path_size": path_sizes.astype(np.int32),
        "meta_size": m,
        "pval": pvals.astype(np.float64),
    })


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR (vectorized)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q[ranks.argsort()] = q_sorted
    return np.clip(q, 0, 1)


# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-outdir", required=True, help="Dir with M_meta.npy, genes.txt, path2meta.parquet")
    ap.add_argument("--pathway-config", required=True, help="YAML listing pathway sources (same as GP pipeline)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topk-genes", type=int, default=50, help="Top-k genes per meta for QC and enrichment")
    ap.add_argument("--min-path-size", type=int, default=10)
    ap.add_argument("--max-path-size", type=int, default=999999999)
    ap.add_argument("--enrich-topn", type=int, default=20, help="Top-N enriched pathways to report per meta")
    ap.add_argument("--report-first", type=int, default=20, help="How many metas to summarize in markdown")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load meta-anchors + genes + (optional) path2meta
    W_rg, genes, P2M = _load_meta(args.embed_outdir)
    R, G = W_rg.shape
    if G != len(genes):
        raise SystemExit(f"Gene length mismatch: M_meta has {G} columns, genes.txt has {len(genes)}")

    # 2) Rebuild raw gene×pathway membership aligned to the same genes
    X = _build_gene_by_pathway_matrix(args.pathway_config, genes,
                                      min_size=args.min_path_size, max_size=args.max_path_size,
                                      prefix_source=True)
    # Ensure gene order matches
    X = X.reindex(index=genes).fillna(0.0)
    path_names = np.array(list(X.columns))
    # Boolean (sparse) for fast overlaps
    try:
        from scipy.sparse import csc_matrix
        X_bool = csc_matrix((X.values != 0).astype(np.int8))  # G x K
    except Exception:
        X_bool = (X.values != 0)

    # 3) Top genes per meta
    top_genes_df = _topk_genes_per_meta(W_rg, genes, args.topk_genes)
    top_genes_csv = os.path.join(args.outdir, "meta_top_genes.csv")
    top_genes_df.to_csv(top_genes_csv, index=False)
    print(f"[save] {top_genes_csv}")

    # 4) Enrichment per meta (vectorized)
    enrich_rows = []
    for r in range(R):
        w = W_rg[r]
        if args.topk_genes >= len(w):
            idx = np.argsort(w)[::-1]
        else:
            idx = np.argpartition(w, -args.topk_genes)[-args.topk_genes:]
            idx = idx[np.argsort(w[idx])[::-1]]
        mask = np.zeros(G, dtype=bool)
        mask[idx] = True

        df = _enrichment_for_meta_set(X_bool, mask)
        df["meta_idx"] = r
        # attach pathway names
        df["pathway"] = path_names
        # FDR per meta
        df = df.sort_values("pval").reset_index(drop=True)
        df["qval"] = _fdr_bh(df["pval"].values)
        # keep top-N by qval (or pval if tied)
        keep = df.nsmallest(args.enrich_topn, ["qval", "pval"]).copy()
        enrich_rows.append(keep)

    enrich_df = pd.concat(enrich_rows, ignore_index=True)
    enrich_path = os.path.join(args.outdir, "meta_enrichment.parquet")
    enrich_df.to_parquet(enrich_path, index=False)
    print(f"[save] {enrich_path}")

    # 5) Top raw pathways by weight per meta (from path2meta if present)
    top_paths_csv = None
    if P2M is not None and not P2M.empty and {"meta_idx","pathway","weight"}.issubset(P2M.columns):
        # aggregate weights per (meta_idx, pathway) and take top per meta
        grp = P2M.groupby(["meta_idx","pathway"], as_index=False)["weight"].sum()
        rows = []
        for r, g in grp.groupby("meta_idx"):
            gg = g.sort_values("weight", ascending=False).head(args.enrich_topn)
            gg = gg.assign(meta_idx=int(r))
            rows.append(gg[["meta_idx","pathway","weight"]])
        top_paths_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["meta_idx","pathway","weight"])
        top_paths_csv = os.path.join(args.outdir, "meta_top_paths_by_weight.csv")
        top_paths_df.to_csv(top_paths_csv, index=False)
        print(f"[save] {top_paths_csv}")
    else:
        print("[info] path2meta.parquet not found or missing columns; skipping top-by-weight table.")

    # 6) Compact markdown report for the first N metas
    report_path = os.path.join(args.outdir, "qc_meta_report.md")
    with open(report_path, "w") as fh:
        fh.write(f"# Meta-anchor QC Report\n\n")
        fh.write(f"- R (meta-anchors): **{R}**\n")
        fh.write(f"- G (genes): **{G}**\n")
        fh.write(f"- Raw pathways after filters: **{X.shape[1]}**\n")
        fh.write(f"- Top-k genes per meta: **{args.topk_genes}**\n")
        fh.write(f"- Enrichment top-N per meta: **{args.enrich_topn}**\n\n")

        # pre-index tables for quick lookup
        tg = top_genes_df.set_index("meta_idx")
        if top_paths_csv:
            tp = pd.read_csv(top_paths_csv).groupby("meta_idx")
        else:
            tp = None
        enr = enrich_df.groupby("meta_idx")

        for r in range(min(args.report_first, R)):
            fh.write(f"## Meta {r}\n\n")
            # genes
            gtab = tg.loc[r][["rank","gene","weight"]]
            if isinstance(gtab, pd.Series):  # when only one row
                gtab = gtab.to_frame().T
            fh.write("**Top genes**\n\n")
            fh.write(gtab.head(10).to_markdown(index=False))
            fh.write("\n\n")
            # enrichment
            if r in enr.groups:
                etab = enr.get_group(r).sort_values(["qval","pval"]).head(10)
                etab = etab[["pathway","overlap","path_size","meta_size","pval","qval"]]
                fh.write("**Top enriched raw pathways**\n\n")
                fh.write(etab.to_markdown(index=False))
                fh.write("\n\n")
            # top-by-weight from path2meta
            if tp is not None and r in tp.groups.groups:
                wtab = tp.get_group(r).sort_values("weight", ascending=False).head(10)
                fh.write("**Top raw pathways by meta weight (from path2meta)**\n\n")
                fh.write(wtab.to_markdown(index=False))
                fh.write("\n\n")

    print(f"[save] {report_path}")
    print("[OK] QC complete.")

if __name__ == "__main__":
    main()
