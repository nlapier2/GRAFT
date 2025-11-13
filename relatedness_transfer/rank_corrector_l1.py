#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L1 pathway-based rank-only corrector (one-list).

- Builds pathway offsets from descriptive rank errors (computed on-the-fly).
- Forms per-pert, per-gene correction scores using perturbed-pathway × responder-pathway structure.
- Produces a unique total order 1..G by sorting priority q = r - eta * c with deterministic tie-breakers.
- Reassigns the entire vector of predicted deltas to the new order (allows sign flips).
- Evaluates MAE and PDS using evaluate_model() from multi_dataset_krr.py.

Filtering (as requested):
  - Drop responder pathways with < 20 genes.
  - Drop responder pathways with < 3 perturbed genes among your perturbation set.
  - Drop perturbations whose targeted gene has ≤ 1 pathway annotation (after pathway filtering).

Prints how many pathways/perts were filtered and remaining.
"""

import os
import argparse
from typing import Tuple, List
import numpy as np
import pandas as pd
import anndata as ad

# Local imports
from load_pathways import load_pathway_sources, make_pathway_matrix

# Import evaluator from user's code
import sys, pathlib
USER_DIR = str(pathlib.Path(__file__).resolve().parent)
if USER_DIR not in sys.path:
    sys.path.insert(0, USER_DIR)
from multi_dataset_krr import evaluate_model  # type: ignore


def compute_control_mean(X: np.ndarray, labels: pd.Series, control_label: str) -> np.ndarray:
    m = labels.astype(str) == str(control_label)
    if not np.any(m):
        raise ValueError(f"No control rows found for control_label='{control_label}'.")
    return X[m].mean(axis=0)


def deltas_from_pseudobulk(adata: ad.AnnData, target_label: str, control_label: str
                           ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    labels = adata.obs[target_label].astype(str)
    X = adata.X.A if hasattr(adata.X, "A") else np.asarray(adata.X)
    ctrl = compute_control_mean(X, labels, control_label)
    mask = labels != str(control_label)
    perts = list(labels[mask])
    Xp = X[mask]
    # effect = pred - ctrl  (so positive means up vs control)
    deltas = Xp - ctrl[None, :]
    return deltas, perts, ctrl


def intersect_genes(pred: ad.AnnData, true: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData]:
    common = pred.var_names.intersection(true.var_names)
    if common.size == 0:
        raise ValueError("No overlapping genes between predicted and true AnnData.")
    if common.size < pred.n_vars or common.size < true.n_vars:
        print(f"[info] Restricting to {common.size} common genes.")
    pred2 = pred[:, common].copy()
    true2 = true[:, common].copy()
    return pred2, true2


def ranks_desc(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with 1 = largest value; deterministic tiebreakers via stable argsort order."""
    order = np.lexsort((np.arange(x.size), -x))  # sort by value desc, then by index asc
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    return ranks


def signed_rank_error_one_list(true_delta: np.ndarray, pred_delta: np.ndarray) -> np.ndarray:
    """e = r_pred - r_true with one full descending list (1=most positive)."""
    r_true = ranks_desc(true_delta)
    r_pred = ranks_desc(pred_delta)
    return r_pred - r_true  # positive = under-ranked (too low)


def responder_pathway_profile(e_mat: np.ndarray, resp_mat: pd.DataFrame) -> pd.Series:
    """Median over perts of median error within each responder pathway (one-list)."""
    K = resp_mat.shape[1]
    vals = []
    for k in range(K):
        mask = resp_mat.iloc[:, k].values > 0
        if not np.any(mask):
            vals.append(np.nan)
            continue
        per_pert = np.nanmedian(e_mat[:, mask], axis=1)
        vals.append(np.nanmedian(per_pert))
    return pd.Series(vals, index=resp_mat.columns, name="median_rank_error")


def pert_to_resp_matrix(e_mat: np.ndarray, pert_mat: pd.DataFrame, resp_mat: pd.DataFrame) -> pd.DataFrame:
    """M[alpha,beta] = median_{perts in alpha} median_{genes in beta} e_p(g)."""
    P, G = e_mat.shape
    Ka = pert_mat.shape[1]; Kb = resp_mat.shape[1]
    M = np.full((Ka, Kb), np.nan, dtype=float)
    # precompute per-pert per-beta medians
    per_pert_beta = np.full((P, Kb), np.nan, dtype=float)
    resp_masks = [(resp_mat.iloc[:, j].values > 0) for j in range(Kb)]
    for j in range(Kb):
        mg = resp_masks[j]
        if np.any(mg):
            per_pert_beta[:, j] = np.nanmedian(e_mat[:, mg], axis=1)
    # aggregate over perts that have alpha
    pert_masks = [(pert_mat.iloc[:, i].values > 0) for i in range(Ka)]
    for i in range(Ka):
        mp = pert_masks[i]
        if np.any(mp):
            M[i, :] = np.nanmedian(per_pert_beta[mp, :], axis=0)
    return pd.DataFrame(M, index=pert_mat.columns, columns=resp_mat.columns)


def build_corrector_terms(one_list_errors: np.ndarray,
                          resp_gp: pd.DataFrame,
                          pert_gp: pd.DataFrame,
                          clip_negatives: bool = True):
    """Return a (Kb,) responder offset vector 'a', and a (Ka,Kb) pair matrix 'C'."""
    # Responder offsets
    prof = responder_pathway_profile(one_list_errors, resp_gp)
    # a = prof.values.copy()
    # if clip_negatives:
    #     a[a < 0] = 0.0  # only push up pathways that are under-ranked on average
    # Drop global responder offset to avoid homogenizing perts
    a = np.zeros_like(prof.values, dtype=float)
    # Pair matrix
    M = pert_to_resp_matrix(one_list_errors, pert_gp, resp_gp).values
    if clip_negatives:
        M[M < 0] = 0.0
    return a, M, prof.index.tolist(), list(pert_gp.columns), list(resp_gp.columns)


def apply_rank_only_reassignment(pred_delta: np.ndarray,
                                 a_vec: np.ndarray,
                                 C_mat: np.ndarray,
                                 P_row: np.ndarray,
                                 R_mat: np.ndarray,
                                 eta: float,
                                 pair_topL: int = 100) -> np.ndarray:
    """
    pred_delta: (G,) predicted deltas for a single perturbation.
    a_vec: (Kb,) responder pathway offsets.
    C_mat: (Ka,Kb) pair effects (alpha->beta).
    P_row: (Ka,) binary vector pathways of the perturbed gene.
    R_mat: (G,Kb) responder pathway membership (per gene).
    """
    G, Kb = R_mat.shape
    # Current ranks (descending; 1=most positive)
    r = ranks_desc(pred_delta)
    # Correction: c_p(g) = R @ (P @ C)  (drop global responder offset)
    v = (P_row @ C_mat)               # (Kb,)
    # Sparsify responder pathways per-perturbation (keep top-|v| entries)
    if pair_topL is not None and pair_topL > 0 and pair_topL < v.shape[0]:
        keep = np.argpartition(np.abs(v), -(pair_topL))[-pair_topL:]
        mask = np.zeros_like(v, dtype=bool); mask[keep] = True
        v = v * mask
    c = R_mat @ v                     # (G,)
    # Center (median-zero) to avoid uniform shifts
    c = c - np.median(c)
    # Orthogonalize against current ranks to remove global mode
    var_r = np.var(r) + 1e-12
    beta = np.cov(c, r, ddof=0)[0, 1] / var_r
    c = c - beta * r                   # (G,)
    # Priority and unique order
    q = r - eta * c
    # unique order via lexsort: primary q asc, secondary original rank asc, tertiary gene index asc
    order = np.lexsort((np.arange(G), r, q))
    # Reassignment: sort values desc, then stamp onto new order
    values_desc = np.sort(pred_delta)[::-1]
    new = np.empty_like(pred_delta)
    new[order] = values_desc
    return new


def main():
    ap = argparse.ArgumentParser(description="L1 pathway-based rank-only corrector (one-list).")
    ap.add_argument("--pred_h5ad", required=True)
    ap.add_argument("--true_h5ad", required=True)
    ap.add_argument("--pathways_yaml", required=True)
    ap.add_argument("--target_label", required=True)
    ap.add_argument("--control_label", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eta", type=float, default=0.5, help="Scale for corrections in priority q = r - eta*c.")
    ap.add_argument("--clip_negatives", action='store_true', help="If true, set negative pathway effects to 0.")
    ap.add_argument("--pair_topL", type=int, default=100,
                    help="Per-perturbation: keep only top-|effect| responder pathways from (P @ C). Set 0 to disable.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data and intersect genes
    pred = ad.read_h5ad(args.pred_h5ad)
    true = ad.read_h5ad(args.true_h5ad)
    pred, true = intersect_genes(pred, true)
    genes = list(pred.var_names)

    # Deltas and alignment of perts
    pred_delta, pred_perts, ctrl = deltas_from_pseudobulk(pred, args.target_label, args.control_label)
    true_delta, true_perts, _ = deltas_from_pseudobulk(true, args.target_label, args.control_label)

    perts_common = sorted(set(pred_perts).intersection(true_perts))
    if not perts_common:
        raise ValueError("No overlapping perturbations between predicted and true.")
    def reindex(rows, names, keep):
        idx = pd.Index(names).get_indexer(keep)
        return rows[idx, :]
    pred_delta = reindex(pred_delta, pred_perts, perts_common)
    true_delta = reindex(true_delta, true_perts, perts_common)
    Pn, G = pred_delta.shape

    # Load pathways (first source) and build gene×pathway
    srcs = load_pathway_sources(args.pathways_yaml)
    if not srcs:
        raise ValueError("No pathway sources found in YAML.")
    first_name = list(srcs.keys())[0]
    meta = srcs[first_name]
    gp = make_pathway_matrix(
        file_name=meta["file"],
        gene_col=meta["gene_col"],
        pathway_col=meta["pathway_col"],
        format=meta["format"],
        var_names=genes,
    )
    # Binarize
    gp = (gp > 0).astype(float)  # (G, Kraw)

    # Determine perturbed gene pathway memberships
    idx_genes = pd.Index(genes)
    pert_gene_rows = idx_genes.get_indexer(perts_common)
    pert_gp = np.zeros((Pn, gp.shape[1]), dtype=float)
    for i, ridx in enumerate(pert_gene_rows):
        if ridx >= 0:
            pert_gp[i, :] = gp.iloc[ridx].values

    # === Filtering as requested ===
    # Pathways with at least 20 genes
    mask_size = (gp.sum(axis=0).values >= 20)
    # Pathways with at least 3 perturbed genes
    mask_perts = (pert_gp.sum(axis=0) >= 3)
    keep_pw = mask_size & mask_perts
    n_pw_init = gp.shape[1]
    gp_f = gp.loc[:, keep_pw]
    pert_gp_f = pert_gp[:, keep_pw]
    n_pw_kept = gp_f.shape[1]

    # Drop perturbations with <= 1 pathway annotation (after filtering)
    pert_annot_counts = pert_gp_f.sum(axis=1)
    keep_perts = pert_annot_counts > 1.0
    n_pert_init = Pn
    pred_delta = pred_delta[keep_perts, :]
    true_delta = true_delta[keep_perts, :]
    perts_common = [p for p, k in zip(perts_common, keep_perts) if k]
    pert_gp_f = pert_gp_f[keep_perts, :]
    Pn_kept = pred_delta.shape[0]

    print(f"[filter] pathways: {n_pw_init} -> {n_pw_kept} kept; removed {n_pw_init - n_pw_kept}")
    print(f"[filter] perts:     {n_pert_init} -> {Pn_kept} kept; removed {n_pert_init - Pn_kept}")

    # Build one-list rank error matrix (Pn x G)
    e_mat = np.vstack([signed_rank_error_one_list(true_delta[i], pred_delta[i]) for i in range(Pn_kept)])

    # Build corrector terms (a_vec, C_mat). Clip negatives if requested.
    a_vec, C_mat, resp_names, pert_pw_names, resp_pw_names = build_corrector_terms(
        one_list_errors=e_mat,
        resp_gp=gp_f,
        pert_gp=pd.DataFrame(pert_gp_f, index=perts_common, columns=gp_f.columns),
        clip_negatives=bool(args.clip_negatives),
    )
    # R_mat for all genes
    R_mat = gp_f.values  # (G, Kb)
    Ka, Kb = C_mat.shape

    # Apply rank-only reassignment for each perturbation
    pred_delta_corr = np.empty_like(pred_delta)
    for i in range(Pn_kept):
        P_row = pert_gp_f[i, :]  # (Ka,)
        pred_delta_corr[i, :] = apply_rank_only_reassignment(
            pred_delta=pred_delta[i, :],
            a_vec=a_vec,
            C_mat=C_mat,
            P_row=P_row,
            R_mat=R_mat,
            eta=args.eta,
            pair_topL=args.pair_topL
        )

    # Map back to predicted expressions by adding control mean
    pred_expr_corr = pred_delta_corr + ctrl[None, :]

    # Evaluate with user's evaluate_model()
    # Build bundle (pred_mat, true_mat, pert_names, ctrl_mean)
    pred_bundle = (pred_expr_corr, true_delta + ctrl[None, :], perts_common, ctrl)
    # Dummy args namespace with required fields used by evaluate_model; reuse target_label/control_label
    class _Args:
        pass
    ev_args = _Args()
    ev_args.target_label = args.target_label
    ev_args.control_label = args.control_label

    # Evaluate
    metrics = evaluate_model(adata=true, args=ev_args, pred_bundle=pred_bundle)
    # Save corrected matrix and metrics
    out_pred = os.path.join(args.out_dir, "pred_corrected_pseudobulk.h5ad")
    ad.AnnData(pred_expr_corr, obs=pd.DataFrame({args.target_label: perts_common}), var=pred.var).write(out_pred)

    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(args.out_dir, "metrics_corrected.csv"), index=False)
    print("[done] Wrote:", out_pred)
    # print("[done] Metrics:", metrics)


if __name__ == "__main__":
    main()
