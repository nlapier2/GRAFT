#!/usr/bin/env python3
"""
Build a gene-gene similarity CSR (cosine Top-K) from pathway memberships.

Usage:
  python build_gene_similarity.py \
    --pathway-cfg configs/pathways.yaml \
    --genes-csv data/genes.csv \
    --out-npz artifacts/gene_similarity_topk128.npz \
    --topk 128 --no-tfidf

Inputs:
  --pathway-cfg  YAML consumed by your load_pathways.py helpers.
  --genes-csv    CSV/TXT with one gene per line (or column 'gene'); OR use --genes-from-h5ad
  --genes-from-h5ad  Path to .h5ad to take var_names as genes (mutually exclusive with --genes-csv)

Output (.npz):
  indptr:int64[G+1], indices:int64[E], data:float32[E], shape:int64[2], genes:object[G]
  plus metadata: topk:int64, tfidf:uint8
"""
import argparse, math, os, sys
from typing import List
import numpy as np
import pandas as pd
import torch
import anndata as ad

# ---- import your existing loader utilities ----
# Expecting these in your repo; adjust import path if needed.
from load_pathways import load_pathway_sources, make_pathway_matrix  # noqa


def _read_genes(genes_csv: str) -> List[str]:
    # Flexible: one gene per line, or a CSV with a 'gene' column
    try:
        import pandas as pd
        df = pd.read_csv(genes_csv)
        if 'gene' in df.columns:
            genes = df['gene'].astype(str).tolist()
        else:
            # Treat as single-column with or without header
            if df.shape[1] == 1:
                genes = df.iloc[:, 0].astype(str).tolist()
            else:
                raise ValueError("CSV must have a single column or a 'gene' column.")
    except Exception:
        # Fallback: plain text file, one per line
        with open(genes_csv, 'r') as f:
            genes = [ln.strip() for ln in f if ln.strip()]
    return genes


@torch.no_grad()
def _build_csr_from_pathways(
    genes: List[str],
    pathway_cfg: str,
    device: str = "cuda",
    tfidf: bool = True,
    topk: int = 128,
    row_chunk: int = 1024,
    add_dummy_for_empty: bool = True,
):
    """
    Returns (indptr:int64[G+1], indices:int64[E], data:float32[E])
    """
    # Load pathway sources via your helper
    srcs = load_pathway_sources(pathway_cfg)
    # Choose one or merge: for now, simple union via make_pathway_matrix on the first entry
    # You can iterate & sum/stack if you want a union-of-priors; minimal version keeps it simple.
    name, meta = next(iter(srcs.items()))
    pm = make_pathway_matrix(
        file_name=meta["file"],
        gene_col=meta["gene_col"],
        pathway_col=meta["pathway_col"],
        format=meta["format"],
        var_names=genes,  # ensures rows follow 'genes' order
    )
    pm = pm.reindex(index=pd.Index(genes, name=pm.index.name), fill_value=0.0)
    assert pm.shape[0] == len(genes), "Gene reindexing failed"
    # pm is (genes x pathways) (per your helper); convert to torch
    X = torch.tensor(pm.values, dtype=torch.float32, device=device)  # (G,P)
    G, P = X.shape

    # Add dummy column for genes with no pathways
    if add_dummy_for_empty:
        has_any = (X != 0).any(dim=1)
        if (~has_any).any():
            dummy = torch.zeros(G, 1, device=device, dtype=X.dtype)
            dummy[~has_any, 0] = 1.0
            X = torch.cat([X, dummy], dim=1)
            P += 1

    # TF-IDF reweight columns (pathways)
    if tfidf:
        df = (X != 0).sum(dim=0).clamp_min(1).to(torch.float32)  # (P,)
        idf = torch.log1p(G / df)
        X = X * idf  # broadcast

    # Row L2 normalize → cosine becomes dot
    X = torch.nn.functional.normalize(X, p=2, dim=1, eps=1e-8)

    # Prepare outputs
    k = min(topk, max(1, G - 1))
    indptr = torch.empty(G + 1, dtype=torch.int64, device=device)
    indptr[0] = 0
    indices = torch.empty(G * k, dtype=torch.int64, device=device)
    data = torch.empty(G * k, dtype=torch.float32, device=device)

    XT = X.transpose(0, 1).contiguous()  # (P,G)

    write_ptr = 0
    for start in range(0, G, row_chunk):
        end = min(G, start + row_chunk)
        X_blk = X[start:end, :]                    # (B,P)
        S = X_blk @ XT                             # (B,G) cosine sims
        # exclude self
        row_idx = torch.arange(start, end, device=device)
        S[torch.arange(end - start, device=device), row_idx] = -math.inf
        v, i = torch.topk(S, k=k, dim=1, largest=True, sorted=False)  # (B,k)
        blk = end - start
        indices[write_ptr:write_ptr + blk * k] = i.reshape(-1)
        data[write_ptr:write_ptr + blk * k] = v.reshape(-1).to(torch.float32)
        # indptr increases by fixed k per row
        # fill for rows [start+1 ... end]
        base = (start + 1) * k
        indptr[start + 1:end + 1] = torch.arange(base, base + blk * k, step=k, device=device)
        write_ptr += blk * k

    return indptr, indices, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathway-cfg", required=True, help="YAML for pathway sources (load_pathways.py schema)")
    ap.add_argument("--genes-from-h5ad", required=True, help=".h5ad to read var_names as genes")
    ap.add_argument("--out-npz", required=True, help="Output .npz file (CSR)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--row-chunk", type=int, default=999999)
    ap.add_argument("--no-tfidf", action="store_true", help="Disable TF-IDF reweighting")
    ap.add_argument("--no-dummy", action="store_true", help="Disable dummy pathway for empty genes")
    args = ap.parse_args()

    # Resolve gene list
    adata = ad.read_h5ad(args.genes_from_h5ad, backed='r')
    genes = list(map(str, adata.var_names))

    indptr, indices, data = _build_csr_from_pathways(
        genes=genes,
        pathway_cfg=args.pathway_cfg,
        device=args.device,
        tfidf=(not args.no_tfidf),
        topk=args.topk,
        row_chunk=args.row_chunk,
        add_dummy_for_empty=(not args.no_dummy),
    )

    # After constructing indptr/indices/data:
    assert indptr.shape[0] == len(genes) + 1
    assert int(indptr[-1]) == int(indices.shape[0]) == int(data.shape[0])

    # Save as a single npz with metadata
    np.savez_compressed(
        args.out_npz,
        indptr=indptr.cpu().numpy(),
        indices=indices.cpu().numpy(),
        data=data.cpu().numpy(),
        shape=np.array([len(genes), len(genes)], dtype=np.int64),
        genes=np.array(genes, dtype=object),
        topk=np.array([args.topk], dtype=np.int64),
        tfidf=np.array([0 if args.no_tfidf else 1], dtype=np.uint8),
    )
    print(f"[save] {args.out_npz}  (G={len(genes)}, topk={args.topk})")


if __name__ == "__main__":
    main()
