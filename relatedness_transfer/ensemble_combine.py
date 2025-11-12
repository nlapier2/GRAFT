#!/usr/bin/env python3
import argparse
from types import SimpleNamespace
import numpy as np
import scanpy as sc
from multi_dataset_krr import evaluate_model, write_cell_level_predictions  # same call sites throughout your code
import scipy.sparse as sp
import pandas as pd

def _pseudobulk_and_ctrl_mean(adata: sc.AnnData, target_label: str, control_label: str):
    """
    Returns:
      perts: list[str] of non-control perts (unique, stable order)
      genes: list[str] var_names order
      ctrl_mean: (G,) float32 array
      bulk_by_pert: dict[pert] -> (G,) mean expression
    """
    labels = adata.obs[target_label].astype(str).values
    genes = list(map(str, adata.var_names))

    # Control mean (over all control cells)
    ctrl_mask = (labels == str(control_label))
    if ctrl_mask.any():
        ctrl_mean = np.asarray(adata.X[ctrl_mask].mean(axis=0)).ravel().astype(np.float32)
    else:
        raise ValueError(f"No control cells found for control_label={control_label!r}")

    # Mean per perturbation (exclude control)
    perts_all = list(map(str, labels))
    seen = set()
    perts = []
    for p in perts_all:
        if p != control_label and p not in seen:
            perts.append(p); seen.add(p)

    bulk_by_pert = {}
    for p in perts:
        m = (labels == p)
        if not m.any():
            continue
        bulk_by_pert[p] = np.asarray(adata.X[m].mean(axis=0)).ravel().astype(np.float32)

    return perts, genes, ctrl_mean.astype(np.float32), bulk_by_pert

def _deltas_from_bulk(perts, genes, ctrl_mean, bulk_by_pert):
    """Stack deltas (bulk - ctrl_mean) in the provided ordering."""
    G = len(genes)
    N = len(perts)
    D = np.zeros((N, G), dtype=np.float32)
    for i, p in enumerate(perts):
        v = bulk_by_pert.get(p, None)
        if v is None:
            raise ValueError(f"Missing pseudobulk for perturbation {p}")
        D[i, :] = (v - ctrl_mean)
    return D  # (N,G)

def _align_three(perts1, genes1, perts2, genes2, pertsT, genesT):
    """Intersection alignment across the three inputs (order = sorted by name)."""
    Gs = sorted(set(genes1) & set(genes2) & set(genesT))
    Ps = sorted(set(perts1) & set(perts2) & set(pertsT))
    if len(Ps) == 0 or len(Gs) == 0:
        raise ValueError("Empty intersection across inputs (perts or genes).")
    return Ps, Gs

def _matrix_from_dict(D_dict, perts, genes, ctrl_mean):
    """Helper to build expression and delta matrices aligned to (perts, genes)."""
    # Reconstruct expression = delta + ctrl_mean
    D = np.stack([D_dict[p] for p in perts], axis=0).astype(np.float32)
    E = D + ctrl_mean[None, :]
    return D, E

def _combine(D1, D2, mode: str):
    """
    Combine two delta matrices (N,G) by:
      - mean:    (D1 + D2)/2
      - sum:     D1 + D2
      - maxmag:  per row, pick row (from D1 or D2) with larger L1 norm
      - maxrank: per row, pick row from matrix where that row's L1 norm
                 ranks higher among that matrix's rows (1 = highest)
    """
    mode = mode.lower()
    if mode == "mean":
        return 0.5 * (D1 + D2)
    if mode == "sum":
        return D1 + D2

    l1_1 = np.abs(D1).sum(axis=1)
    l1_2 = np.abs(D2).sum(axis=1)

    if mode == "maxmag":
        take1 = (l1_1 >= l1_2)
        D = D2.copy()
        D[take1, :] = D1[take1, :]
        return D

    if mode == "maxrank":
        # ranks: 1 = largest
        r1 = 1 + np.argsort(np.argsort(-l1_1))  # [1..N]
        r2 = 1 + np.argsort(np.argsort(-l1_2))
        take1 = (r1 < r2)  # smaller rank number = higher magnitude rank
        D = D2.copy()
        D[take1, :] = D1[take1, :]
        return D

    raise ValueError(f"Unknown combine mode: {mode}")

def main():
    ap = argparse.ArgumentParser(description="Simple two-model ensemble for perturbation predictions.")
    ap.add_argument("--pred1_h5ad", required=True, help="Path to first predicted AnnData (normalized/log1p).")
    ap.add_argument("--pred2_h5ad", required=True, help="Path to second predicted AnnData (normalized/log1p).")
    ap.add_argument("--true_h5ad",  required=True, help="Path to TRUE target AnnData (normalized/log1p).")
    ap.add_argument("--target_label", required=True, help="obs column with perturbation labels.")
    ap.add_argument("--control_label", required=True, help="Name used for control rows in target_label.")
    ap.add_argument("--mode", choices=["mean", "sum", "maxmag", "maxrank"], default="mean",
                    help="Rule for combining the two predicted delta matrices.")
    ap.add_argument("--out_h5ad", default=None,
                    help="If set, write a synthetic predicted AnnData here.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for control-cell sampling.")
    args = ap.parse_args()

    # Load inputs
    A1 = sc.read_h5ad(args.pred1_h5ad)
    A2 = sc.read_h5ad(args.pred2_h5ad)
    AT = sc.read_h5ad(args.true_h5ad)
    if np.max(A1.X) > 100:
        print("[ensemble] normalizing/log1p first predicted AnnData")
        sc.pp.normalize_total(A1, inplace=True)
        sc.pp.log1p(A1)
    if np.max(A2.X) > 100:
        print("[ensemble] normalizing/log1p second predicted AnnData")
        sc.pp.normalize_total(A2, inplace=True)
        sc.pp.log1p(A2)
    if np.max(AT.X) > 100:
        print("[ensemble] normalizing/log1p true target AnnData")
        sc.pp.normalize_total(AT, inplace=True)
        sc.pp.log1p(AT)

    # Pseudobulk + control mean for each
    p1, g1, c1, bulk1 = _pseudobulk_and_ctrl_mean(A1, args.target_label, args.control_label)
    p2, g2, c2, bulk2 = _pseudobulk_and_ctrl_mean(A2, args.target_label, args.control_label)
    pT, gT, cT, bulkT = _pseudobulk_and_ctrl_mean(AT, args.target_label, args.control_label)

    # Align across the three
    perts, genes = pT, gT  # _align_three(p1, g1, p2, g2, pT, gT)

    # Build DELTAS in aligned order
    D1 = _deltas_from_bulk(perts, genes, c1, bulk1)       # (N,G)
    D2 = _deltas_from_bulk(perts, genes, c2, bulk2)       # (N,G)
    DT = _deltas_from_bulk(perts, genes, cT, bulkT)       # (N,G)

    # Combine predicted deltas
    D_ens = _combine(D1, D2, args.mode)                   # (N,G)

    # Reconstruct EXPRESSION using the TRUE control mean (to match your eval)
    E_pred = D_ens + cT[None, :]                           # (N,G)
    E_true = DT   + cT[None, :]                            # (N,G)

    # Minimal args for your evaluate_model (it only needs target_label)
    eval_args = SimpleNamespace(target_label=args.target_label)
    # Use the TRUE AnnData (same gene space) for evaluation context
    print("\n=== Evaluating ensemble against truth ===")
    evaluate_model(
        adata=AT,  # evaluation context (var_names, target label mapping, etc.)
        args=eval_args,
        pred_bundle=(E_pred, E_true, perts, cT.astype(np.float32)),
    )

    # === Optional: write synthetic predicted AnnData (use proven helper) ===
    if args.out_h5ad:
        write_cell_level_predictions(
            adata_test_orig=AT,              # ORIGINAL single-cell target AnnData
            eval_gene_names=genes,           # column order of E_pred / D_ens
            pred_mat_eval=E_pred.astype(np.float32),  # (N,G) expression, genes in `genes`
            names_eval=perts,                # row order
            ctrl_mean_eval=cT.astype(np.float32),     # control mean in `genes` order
            target_label=args.target_label,
            control_label=args.control_label,
            out_path=args.out_h5ad,
            random_state=args.seed,
        )
        print(f"[ensemble] wrote synthetic predicted AnnData → {args.out_h5ad}")

if __name__ == "__main__":
    main()
