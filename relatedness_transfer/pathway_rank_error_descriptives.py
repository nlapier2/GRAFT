#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptive analysis of signed rank errors vs pathways.
(See header in the file for details.)
"""
import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from typing import Tuple, Dict, List

from load_pathways import load_pathway_sources, make_pathway_matrix


def compute_control_mean(X: np.ndarray, labels: pd.Series, control_label: str) -> np.ndarray:
    mask_ctrl = labels.astype(str) == str(control_label)
    if not np.any(mask_ctrl):
        raise ValueError(f"No control rows found for control_label='{control_label}'.")
    return X[mask_ctrl].mean(axis=0)


def deltas_from_pseudobulk(adata: ad.AnnData, target_label: str, control_label: str
                           ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    genes = list(adata.var_names)
    labels = adata.obs[target_label].astype(str)
    X = adata.X.A if hasattr(adata.X, "A") else np.asarray(adata.X)
    ctrl_mean = compute_control_mean(X, labels, control_label)
    mask_pert = labels != str(control_label)
    pert_names = list(labels[mask_pert])
    Xp = X[mask_pert]
    deltas = ctrl_mean[None, :] - Xp
    return deltas, pert_names, ctrl_mean


def intersect_genes(pred: ad.AnnData, true: ad.AnnData) -> Tuple[ad.AnnData, ad.AnnData]:
    common = pred.var_names.intersection(true.var_names)
    if common.size == 0:
        raise ValueError("No overlapping genes between predicted and true AnnData.")
    if common.size < pred.n_vars or common.size < true.n_vars:
        print(f"[info] Restricting to {common.size} common genes.")
    pred2 = pred[:, common].copy()
    true2 = true[:, common].copy()
    return pred2, true2


def rank_positions(values: np.ndarray, descending: bool) -> np.ndarray:
    order = np.argsort(values)[::-1] if descending else np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def signed_rank_error_per_pert(true_delta: np.ndarray, pred_delta: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    G = true_delta.shape[0]
    e_up = np.full(G, np.nan, dtype=float)
    e_dn = np.full(G, np.nan, dtype=float)
    up_idx = np.where(true_delta > 0)[0]
    dn_idx = np.where(true_delta < 0)[0]
    if up_idx.size > 0:
        r_true = rank_positions(true_delta[up_idx], descending=True)
        r_pred = rank_positions(pred_delta[up_idx], descending=True)
        e_up_vals = r_pred - r_true
        e_up[up_idx] = e_up_vals
    if dn_idx.size > 0:
        r_true = rank_positions(true_delta[dn_idx], descending=False)
        r_pred = rank_positions(pred_delta[dn_idx], descending=False)
        e_dn_vals = r_true - r_pred
        e_dn[dn_idx] = e_dn_vals
    return e_up, e_dn


def aggregate_responder_pathway_profiles(e_mat: np.ndarray, pathway_mat: pd.DataFrame
                                        ) -> pd.DataFrame:
    K = pathway_mat.shape[1]
    res = []
    for k in range(K):
        mask_g = pathway_mat.iloc[:, k].values > 0
        if not np.any(mask_g):
            res.append(np.full(e_mat.shape[0], np.nan))
            continue
        vals = np.nanmedian(e_mat[:, mask_g], axis=1)
        res.append(vals)
    per_pert = np.vstack(res).T
    out = np.nanmedian(per_pert, axis=0)
    return pd.DataFrame({"median_rank_error": out}, index=pathway_mat.columns)


def aggregate_pert_to_resp_matrix(e_mat: np.ndarray,
                                  pert_pathways: pd.DataFrame,
                                  resp_pathways: pd.DataFrame
                                  ) -> pd.DataFrame:
    P, G = e_mat.shape
    Ka = pert_pathways.shape[1]
    Kb = resp_pathways.shape[1]
    M = np.full((Ka, Kb), np.nan, dtype=float)
    per_pert_per_beta = np.full((P, Kb), np.nan, dtype=float)
    resp_masks = [(resp_pathways.iloc[:, j].values > 0) for j in range(Kb)]
    for j in range(Kb):
        mask_g = resp_masks[j]
        if np.any(mask_g):
            per_pert_per_beta[:, j] = np.nanmedian(e_mat[:, mask_g], axis=1)
    pert_masks = [(pert_pathways.iloc[:, i].values > 0) for i in range(Ka)]
    for i in range(Ka):
        mask_p = pert_masks[i]
        if np.any(mask_p):
            M[i, :] = np.nanmedian(per_pert_per_beta[mask_p, :], axis=0)
    return pd.DataFrame(M, index=pert_pathways.columns, columns=resp_pathways.columns)


def catastrophic_enrichment(e_mat: np.ndarray,
                            true_delta: np.ndarray,
                            top_k: int,
                            far_k: int,
                            resp_pathways: pd.DataFrame,
                            mode: str) -> pd.DataFrame:
    P, G = e_mat.shape
    pathway_names = resp_pathways.columns
    hits = np.zeros(len(pathway_names), dtype=float)
    totals = np.zeros(len(pathway_names), dtype=float)
    nonhits = np.zeros(len(pathway_names), dtype=float)
    nontotals = np.zeros(len(pathway_names), dtype=float)
    for p in range(P):
        t = true_delta[p]
        if mode == "up":
            idx = np.where(t > 0)[0]
            if idx.size == 0:
                continue
            r_true = rank_positions(t[idx], descending=True)
            e = e_mat[p, idx]
            catastrophic = (r_true <= top_k) & ((r_true + e) > far_k)
            mask_cat = np.zeros(G, dtype=bool); mask_cat[idx] = catastrophic
        else:
            idx = np.where(t < 0)[0]
            if idx.size == 0:
                continue
            r_true = rank_positions(t[idx], descending=False)
            e = e_mat[p, idx]
            catastrophic = (r_true <= top_k) & ((r_true - e) > far_k)
            mask_cat = np.zeros(G, dtype=bool); mask_cat[idx] = catastrophic
        for j, pname in enumerate(pathway_names):
            in_path = resp_pathways.iloc[:, j].values > 0
            hits[j] += np.sum(mask_cat & in_path)
            totals[j] += np.sum(in_path)
            nonhits[j] += np.sum(mask_cat & (~in_path))
            nontotals[j] += np.sum(~in_path)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_in = hits / np.maximum(totals, 1e-9)
        rate_out = nonhits / np.maximum(nontotals, 1e-9)
        odds = rate_in / np.maximum(rate_out, 1e-12)
    df = pd.DataFrame({
        "hits": hits,
        "in_total": totals,
        "out_hits": nonhits,
        "out_total": nontotals,
        "odds_like": odds
    }, index=pathway_names).sort_values("odds_like", ascending=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Descriptive analysis of signed rank errors vs pathways.")
    ap.add_argument("--pred_h5ad", required=True)
    ap.add_argument("--true_h5ad", required=True)
    ap.add_argument("--pathways_yaml", required=True)
    ap.add_argument("--target_label", required=True)
    ap.add_argument("--control_label", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--far_k", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pred = ad.read_h5ad(args.pred_h5ad)
    true = ad.read_h5ad(args.true_h5ad)
    pred, true = intersect_genes(pred, true)

    pred_delta, pred_perts, _ = deltas_from_pseudobulk(pred, args.target_label, args.control_label)
    true_delta, true_perts, _ = deltas_from_pseudobulk(true, args.target_label, args.control_label)

    perts_common = sorted(set(pred_perts).intersection(true_perts))
    if not perts_common:
        raise ValueError("No overlapping perturbations between predicted and true.")
    def reindex(rows, names, keep):
        idx = pd.Index(names).get_indexer(keep)
        return rows[idx, :]
    pred_delta = reindex(pred_delta, pred_perts, perts_common)
    true_delta = reindex(true_delta, true_perts, perts_common)
    P, G = true_delta.shape
    genes = list(pred.var_names)

    # Load pathway sources and take the first
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
    gp = (gp > 0).astype(float)  # ensure binary

    # Build perturbed-gene pathway matrix (P x Kα): look up each pert gene in gp rows
    pert_gene_names = perts_common  # assume pert label is the targeted gene name
    idx = pd.Index(genes)
    rows = idx.get_indexer([g if g in idx else None for g in pert_gene_names])
    pert_gp = np.zeros((P, gp.shape[1]), dtype=float)
    for i, ridx in enumerate(rows):
        if ridx >= 0:
            pert_gp[i, :] = gp.iloc[ridx].values

    # === NEW: Filtering rules ===
    # Keep responder pathways with >=20 genes AND present in >=3 perturbed genes
    mask_size = (gp.sum(axis=0).values >= 20)
    mask_perts = (pert_gp.sum(axis=0) >= 3)
    keep_pw = mask_size & mask_perts
    n_pw_init = gp.shape[1]
    gp_f = gp.loc[:, keep_pw]
    pert_gp_f = pert_gp[:, keep_pw]
    n_pw_kept = gp_f.shape[1]

    # Drop perturbations with <=1 pathway annotation (after filtering)
    pert_annot_counts = pert_gp_f.sum(axis=1)
    keep_perts = pert_annot_counts > 1.0
    n_pert_init = P
    pred_delta = pred_delta[keep_perts, :]
    true_delta = true_delta[keep_perts, :]
    perts_common = [p for p, k in zip(perts_common, keep_perts) if k]
    pert_gp_f = pert_gp_f[keep_perts, :]
    P = pred_delta.shape[0]  # update after filtering

    # Report filtering stats
    print(f"[filter] pathways: {n_pw_init} -> {n_pw_kept} kept; removed {n_pw_init - n_pw_kept}")
    print(f"[filter] perts:     {n_pert_init} -> {P} kept; removed {n_pert_init - P}")

    # Final DataFrames aligned to filtered sets
    pert_gp_df = pd.DataFrame(pert_gp_f, index=perts_common, columns=[f"PW_{c}" for c in gp_f.columns])
    resp_gp_df = pd.DataFrame(gp_f.values, index=genes, columns=[f"PW_{c}" for c in gp_f.columns])

    # Compute signed rank errors per perturbation (after filtering)
    e_up = np.full((P, G), np.nan, dtype=float)
    e_dn = np.full((P, G), np.nan, dtype=float)
    for i in range(P):
        eu, ed = signed_rank_error_per_pert(true_delta[i], pred_delta[i])
        e_up[i] = eu
        e_dn[i] = ed

    # A) Responder pathway profiles
    prof_up = aggregate_responder_pathway_profiles(e_up, resp_gp_df)
    prof_dn = aggregate_responder_pathway_profiles(e_dn, resp_gp_df)
    prof_up.to_csv(os.path.join(args.out_dir, "responder_pathway_profiles_up.csv"))
    prof_dn.to_csv(os.path.join(args.out_dir, "responder_pathway_profiles_down.csv"))

    # B) Perturbed-pathway -> responder-pathway matrices
    M_up = aggregate_pert_to_resp_matrix(e_up, pert_gp_df, resp_gp_df)
    M_dn = aggregate_pert_to_resp_matrix(e_dn, pert_gp_df, resp_gp_df)
    M_up.to_csv(os.path.join(args.out_dir, "pert_to_resp_matrix_up.csv"))
    M_dn.to_csv(os.path.join(args.out_dir, "pert_to_resp_matrix_down.csv"))

    # C) Catastrophic under-rank enrichment per responder pathway
    enr_up = catastrophic_enrichment(e_up, true_delta, args.top_k, args.far_k, resp_gp_df, mode="up")
    enr_dn = catastrophic_enrichment(e_dn, true_delta, args.top_k, args.far_k, resp_gp_df, mode="down")
    enr_up.to_csv(os.path.join(args.out_dir, "catastrophic_enrichment_by_resp_up.csv"))
    enr_dn.to_csv(os.path.join(args.out_dir, "catastrophic_enrichment_by_resp_down.csv"))

    with open(os.path.join(args.out_dir, "summaries.txt"), "w") as fh:
        fh.write("Top responder pathways by median under-rank (UP):\n")
        fh.write(prof_up.sort_values("median_rank_error", ascending=False).head(20).to_string())
        fh.write("\n\nTop responder pathways by median under-rank (DOWN):\n")
        fh.write(prof_dn.sort_values("median_rank_error", ascending=False).head(20).to_string())
        fh.write("\n")

    print("[done] Wrote outputs to", args.out_dir)


if __name__ == "__main__":
    main()
