#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tail-boost pipeline with clean train/test separation and modular selectors.

Methods:
  - random: pick middle genes uniformly at random (baseline).
  - pair_score: compute a perturbation-specific pathway score from TRAIN ONLY:
      * Build gene×pathway matrix (from --pathways_yaml, first source), filter pathways:
          - ≥20 genes
          - present in ≥3 perturbed genes (train perts only)
          - drop train perts with ≤1 annotation (after filtering)
      * Compute one-list rank errors on TRAIN (r_pred - r_true; descending 1..G).
      * Build C[alpha,beta] = median_{perts in alpha}( median_{genes in beta}(error) ).
      * For TEST pert p:
          v_p = P_p · C                         (responder pathway effects; P_p from target gene membership)
          v_p  <- sparsify top-|v| L entries    (--topL)
          score(g) = (R_norm @ v_p)[g]          (R_norm = gp_f / size^gamma, gamma default 0.5)
          Positive scores → positive tail; negative → negative tail.
      * Select K middle genes with largest score for + tail and K with smallest score for − tail.
      * Apply tail means to those selected genes; NEVER use TEST truth.
"""

import os
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from typing import Tuple, List, Optional, Dict
from load_pathways import load_pathway_sources, make_pathway_matrix
from sklearn.decomposition import PCA


# Optional evaluation import
_EVAL_AVAILABLE = False
try:
    from multi_dataset_krr import evaluate_model  # type: ignore
    _EVAL_AVAILABLE = True
except Exception:
    _EVAL_AVAILABLE = False

# ---- Utilities ----

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
    deltas = Xp - ctrl[None, :]  # effect space
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


def topk_indices_desc(x: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, x.size)
    if k_eff <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-x, k_eff-1)[:k_eff]
    idx = idx[np.argsort(-x[idx], kind="mergesort")]
    return idx


def bottomk_indices_asc(x: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, x.size)
    if k_eff <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(x, k_eff-1)[:k_eff]
    idx = idx[np.argsort(x[idx], kind="mergesort")]
    return idx


def ranks_desc(values: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(values.size), -values))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    return ranks


def one_list_rank_error(true_delta: np.ndarray, pred_delta: np.ndarray) -> np.ndarray:
    return ranks_desc(pred_delta) - ranks_desc(true_delta)


# ---- Pathways loader ----
def load_gene_pathway_matrix(pathways_yaml: str, genes: List[str]) -> pd.DataFrame:
    from load_pathways import load_pathway_sources, make_pathway_matrix  # user's helper
    srcs = load_pathway_sources(pathways_yaml)
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
    return (gp > 0).astype(float)

# ---- Pathways: read ALL sources and build PCA gene embeddings ----
def load_all_pathway_matrices(pathways_yaml: str, genes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Returns dict {source_name: gene x pathways DataFrame} for ALL sources in YAML.
    Keeps numeric values if present (no forced binarization).
    """
    srcs = load_pathway_sources(pathways_yaml)
    if not srcs:
        raise ValueError("No pathway sources found in YAML.")
    out = {}
    for name, meta in srcs.items():
        m = make_pathway_matrix(
            file_name=meta["file"],
            gene_col=meta["gene_col"],
            pathway_col=meta["pathway_col"],
            format=meta["format"],
            var_names=genes,
        )
        out[name] = m  # keep as-is (numeric or binary)
    return out

def build_gene_embeddings_all_sources(
    genes: List[str],
    perts_train: List[str],
    perts_test: List[str],
    pathways_yaml: str,
    n_pcs: int = 256,
    gamma_size_norm: float = 0.0,
) -> Tuple[np.ndarray, float, pd.Index, List[str]]:
    """
    Concatenate all sources (column-wise) after filtering:
      - keep only pathway columns that (i) appear in >=1 gene overall,
        (ii) are present in at least one TRAIN perturbed gene and at least one TEST perturbed gene.
    Optionally size-normalize columns by (#genes)^gamma_size_norm (default 0.0 = off).
    Returns:
      - PCs (G x n_pcs), cumulative variance explained (float 0..1),
      - gene index (same order as 'genes'),
      - list of kept pathway column names with source prefixes.
    """
    src2mtx = load_all_pathway_matrices(pathways_yaml, genes)
    gene_index = pd.Index(genes)
    # concat with source prefix to avoid column name collisions
    mats = []
    colnames = []
    for src, df in src2mtx.items():
        df = df.copy()
        df.columns = [f"{src}::{c}" for c in df.columns]
        mats.append(df.values)
        colnames.extend(df.columns.tolist())
    if not mats:
        raise ValueError("No pathway matrices constructed.")
    M = np.concatenate(mats, axis=1)           # shape (G, K_total)
    colnames = list(colnames)                  # length K_total

    # Presence-by-gene
    gene_presence = np.any(M != 0, axis=0)     # K_total

    # Determine presence in TRAIN and TEST perturbed genes
    def rows_for(perts: List[str]) -> np.ndarray:
        idx = gene_index.get_indexer(perts)
        idx = idx[idx >= 0]
        return idx
    tr_rows = rows_for(perts_train)
    te_rows = rows_for(perts_test)
    present_train = np.any(M[tr_rows, :] != 0, axis=0) if tr_rows.size else np.zeros(M.shape[1], dtype=bool)
    present_test  = np.any(M[te_rows, :] != 0, axis=0) if te_rows.size else np.zeros(M.shape[1], dtype=bool)

    keep = gene_presence & present_train & present_test
    if not np.any(keep):
        raise ValueError("After filtering, no pathway columns remain that are present in both TRAIN and TEST.")
    M = M[:, keep]
    kept_cols = [c for c, k in zip(colnames, keep) if k]

    # Optional size normalization: divide each column by (#genes with nonzero)^gamma
    if gamma_size_norm and gamma_size_norm != 0.0:
        sizes = np.sum(M != 0, axis=0).astype(float)
        sizes = np.maximum(sizes, 1.0)
        M = M / (sizes ** gamma_size_norm)[None, :]

    # Center columns, do PCA
    M_centered = M - np.mean(M, axis=0, keepdims=True)
    n_pcs_eff = min(n_pcs, M_centered.shape[1])
    pca = PCA(n_components=n_pcs_eff, svd_solver="auto", random_state=0)
    PCs = pca.fit_transform(M_centered)        # shape (G, n_pcs_eff)
    var_explained = float(np.sum(pca.explained_variance_ratio_))
    return PCs, var_explained, gene_index, kept_cols


# ---- Modular selectors ----
class BaseTailSelector:
    def __init__(self, K: int):
        self.K = K
    def fit(self, **kwargs):
        return self
    def select(self, **kwargs):
        raise NotImplementedError


class RandomTailSelector(BaseTailSelector):
    def __init__(self, K: int, rng: np.random.Generator):
        super().__init__(K)
        self.rng = rng
    def select(self, pred_delta_vec: np.ndarray, pos_frozen: np.ndarray, neg_frozen: np.ndarray):
        G = pred_delta_vec.size
        middle_mask = np.ones(G, dtype=bool)
        middle_mask[pos_frozen] = False
        middle_mask[neg_frozen] = False
        middle = np.where(middle_mask)[0]
        k_eff = min(self.K, middle.size)
        if k_eff <= 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        pos_boost = self.rng.choice(middle, size=k_eff, replace=(middle.size < k_eff))
        neg_boost = self.rng.choice(middle, size=k_eff, replace=(middle.size < k_eff))
        return np.array(pos_boost, dtype=int), np.array(neg_boost, dtype=int)


class PairScoreTailSelector(BaseTailSelector):
    """
    TRAIN-only pathway pair scoring, applied to TEST without seeing TEST truth.
    """
    def __init__(self, K: int, rng: np.random.Generator, topL: int = 100, gamma: float = 0.5):
        super().__init__(K)
        self.rng = rng
        self.topL = topL
        self.gamma = gamma
        self.C = None                # (Ka,Kb)
        self.gp_f = None             # (G,Kb) binary membership after filtering
        self.R_norm = None           # (G,Kb) normalized membership
        self.pathway_cols = None     # list of pathway names
        self.genes_index = None      # Index of genes

    def fit(self,
            pred_delta_train: np.ndarray,
            true_delta_train: np.ndarray,
            perts_train: List[str],
            genes: List[str],
            pathways_yaml: str):
        gp = load_gene_pathway_matrix(pathways_yaml, genes)  # (G,Kraw)
        G, Kraw = gp.shape
        self.genes_index = pd.Index(genes)

        # TRAIN perturbed-gene pathway membership
        idx = self.genes_index.get_indexer(perts_train)
        P_train = np.zeros((len(perts_train), Kraw), dtype=float)
        for i, ridx in enumerate(idx):
            if ridx >= 0:
                P_train[i, :] = gp.iloc[ridx].values

        # FILTERING on TRAIN
        mask_size = (gp.sum(axis=0).values >= 20)
        mask_perts = (P_train.sum(axis=0) >= 3)
        keep_pw = mask_size & mask_perts
        gp_f = gp.loc[:, keep_pw]
        P_train_f = P_train[:, keep_pw]
        Kb = gp_f.shape[1]

        # Drop TRAIN perts with ≤1 annotation after filtering
        keep_perts = (P_train_f.sum(axis=1) > 1.0)
        pred_delta_train = pred_delta_train[keep_perts, :]
        true_delta_train = true_delta_train[keep_perts, :]
        P_train_f = P_train_f[keep_perts, :]
        perts_train = [p for p, k in zip(perts_train, keep_perts) if k]

        # Rank error on TRAIN
        Pn = pred_delta_train.shape[0]
        e_mat = np.vstack([one_list_rank_error(true_delta_train[i], pred_delta_train[i]) for i in range(Pn)])

        # Build C: Ka==Kb since we use same gp universe for pert/responder
        Ka = Kb
        C = np.full((Ka, Kb), np.nan, dtype=float)
        resp_masks = [(gp_f.iloc[:, j].values > 0) for j in range(Kb)]
        per_pert_beta = np.full((Pn, Kb), np.nan, dtype=float)
        for j in range(Kb):
            mg = resp_masks[j]
            if np.any(mg):
                per_pert_beta[:, j] = np.nanmedian(e_mat[:, mg], axis=1)
        for a in range(Ka):
            mp = (P_train_f[:, a] > 0)
            if np.any(mp):
                C[a, :] = np.nanmedian(per_pert_beta[mp, :], axis=0)

        # Normalize responder pathways by size^gamma
        sizes = gp_f.sum(axis=0).values.astype(float)
        size_norm = np.power(np.maximum(sizes, 1.0), self.gamma)
        R_norm = gp_f.values / size_norm[None, :]

        # Save
        self.C = C
        self.gp_f = gp_f.values.astype(float)
        self.R_norm = R_norm
        self.pathway_cols = list(gp_f.columns)
        return self

    def _P_row_for_label(self, pert_label: str) -> np.ndarray:
        # Map pert label (assumed gene symbol) to membership over filtered pathways
        P_row = np.zeros((self.gp_f.shape[1],), dtype=float)
        if pert_label in self.genes_index:
            ridx = self.genes_index.get_loc(pert_label)
            # gene membership across filtered pathways is gp_f[ridx, :]
            # But gp_f is (G,Kb) stored; ensure row index aligns with gene order used in fit
            # genes_index is the order of genes used to build gp_f
            # The row in gp_f for gene ridx:
            P_row = self.gp_f[ridx, :]
        return P_row

    def select(self, pred_delta_vec: np.ndarray, pos_frozen: np.ndarray, neg_frozen: np.ndarray, pert_label: str):
        G, Kb = self.R_norm.shape
        # Build v_p from pert label
        P_row = self._P_row_for_label(pert_label)               # (Kb,)
        v = P_row @ self.C                                      # (Kb,)
        # sparsify v
        if self.topL is not None and 0 < self.topL < v.size:
            keep = np.argpartition(np.abs(v), -(self.topL))[-self.topL:]
            mask = np.zeros_like(v, dtype=bool); mask[keep] = True
            v = v * mask
        score = self.R_norm @ v                                  # (G,)

        middle_mask = np.ones(G, dtype=bool)
        middle_mask[pos_frozen] = False
        middle_mask[neg_frozen] = False
        middle = np.where(middle_mask)[0]
        k_eff = min(self.K, middle.size)
        if k_eff <= 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        pos_order = middle[np.argsort(-score[middle], kind="mergesort")]
        neg_order = middle[np.argsort(score[middle],  kind="mergesort")]
        return pos_order[:k_eff], neg_order[:k_eff]
    

class GOTermCountTailSelector(BaseTailSelector):
    """
    Simple GO-term counting baseline (first pathway file only).

    TRAIN:
      - Load first pathway source (assumed GO) and take rows only for perturbed genes (TRAIN+TEST perts).
      - Keep GO terms present in ≥1 TRAIN pert and ≥1 TEST pert.
      - Drop TRAIN perts that have ≤1 GO term after filtering.
      - For each remaining TRAIN pert i:
          * Freeze ±K by predicted deltas.
          * Consider middle genes; define oracle top-K (+tail) and bottom-K (−tail) by TRUE deltas within the middle.
          * For each GO term t present in this pert, increment:
                pos_counts[t, g] for g ∈ oracle_plus
                neg_counts[t, g] for g ∈ oracle_minus
            and term_den[t] += 1
      - Fractions:
            frac_pos[t, g] = pos_counts[t, g] / term_den[t]
            frac_neg[t, g] = neg_counts[t, g] / term_den[t]

    TEST:
      - For a test pert with GO terms T_p:
            score_pos[g] = sum_{t ∈ T_p} frac_pos[t, g]
            score_neg[g] = sum_{t ∈ T_p} frac_neg[t, g]
        Select top-K from middle by score_pos (for +tail) and by score_neg (for −tail).

    Notes:
      - Only perturbed-gene GO membership matrices are kept in memory (TRAIN+TEST).
      - If a TEST pert has 0 kept GO terms (rare after your filtering), it will select nothing for that pert.
    """
    def __init__(self, K: int, rng: np.random.Generator):
        super().__init__(K)
        self.rng = rng
        # learned artifacts
        self.frac_pos: np.ndarray | None = None   # (Kkeep, G)
        self.frac_neg: np.ndarray | None = None   # (Kkeep, G)
        self.term_cols: list[str] | None = None   # kept GO term names
        self.gene_index: pd.Index | None = None   # index over all genes (var_names order)
        # perturbed-gene GO membership (kept terms only)
        self.P_train_kept: np.ndarray | None = None  # (P_train_kept, Kkeep)
        self.P_test_kept:  np.ndarray | None = None  # (P_test, Kkeep)
        self.perts_train_kept: list[str] | None = None
        self.perts_test: list[str] | None = None
        # map pert label -> row of kept-term membership (as bool vector) for TEST perts
        self._test_term_rows: dict[str, np.ndarray] = {}

    def fit(self,
            pred_delta_train: np.ndarray,
            true_delta_train: np.ndarray,
            perts_train: list[str],
            perts_test: list[str],
            genes: list[str],
            pathways_yaml: str):
        # --- Load first pathway source and get gene x GO matrix (binary) ---
        from load_pathways import load_pathway_sources, make_pathway_matrix
        srcs = load_pathway_sources(pathways_yaml)
        if not srcs:
            raise ValueError("[go_score] No pathway sources found in YAML.")
        first_name = list(srcs.keys())[0]
        meta = srcs[first_name]
        gp_full = make_pathway_matrix(
            file_name=meta["file"],
            gene_col=meta["gene_col"],
            pathway_col=meta["pathway_col"],
            format=meta["format"],
            var_names=genes,
        )
        gp_full = (gp_full > 0).astype(np.float32)  # ensure binary
        G, Kraw = gp_full.shape
        self.gene_index = pd.Index(genes)

        # --- Build perturbed-gene membership rows for TRAIN and TEST ---
        def rows_for(perts: list[str]) -> np.ndarray:
            idx = self.gene_index.get_indexer(perts)
            return idx[idx >= 0]

        tr_rows_all = rows_for(perts_train)
        te_rows_all = rows_for(perts_test)

        # GO terms present in ≥1 TRAIN pert and ≥1 TEST pert
        present_train = np.any(gp_full.iloc[tr_rows_all, :].values > 0, axis=0) if tr_rows_all.size else np.zeros(Kraw, bool)
        present_test  = np.any(gp_full.iloc[te_rows_all, :].values > 0, axis=0) if te_rows_all.size else np.zeros(Kraw, bool)
        keep_terms_mask = present_train & present_test
        if not np.any(keep_terms_mask):
            raise ValueError("[go_score] After filtering, no GO terms remain that are present in both TRAIN and TEST perts.")

        gp_kept_terms = gp_full.loc[:, keep_terms_mask].copy()  # (G, Kkeep)
        self.term_cols = list(gp_kept_terms.columns)
        Kkeep = gp_kept_terms.shape[1]

        # Slice perturbed-gene rows (TRAIN and TEST) on kept terms
        P_train = np.zeros((len(perts_train), Kkeep), dtype=np.float32)
        for i, p in enumerate(perts_train):
            ridx = self.gene_index.get_loc(p) if p in self.gene_index else -1
            if ridx >= 0:
                P_train[i, :] = gp_kept_terms.iloc[ridx, :].values

        P_test  = np.zeros((len(perts_test), Kkeep), dtype=np.float32)
        for i, p in enumerate(perts_test):
            ridx = self.gene_index.get_loc(p) if p in self.gene_index else -1
            if ridx >= 0:
                P_test[i, :] = gp_kept_terms.iloc[ridx, :].values

        # --- Drop TRAIN perts with ≤1 kept GO term ---
        keep_train_mask = (P_train.sum(axis=1) > 1.0)
        n_before = P_train.shape[0]
        P_train = P_train[keep_train_mask, :]
        pred_delta_train = pred_delta_train[keep_train_mask, :]
        true_delta_train = true_delta_train[keep_train_mask, :]
        perts_train_kept = [p for p, k in zip(perts_train, keep_train_mask) if k]
        n_after = P_train.shape[0]
        print(f"[go_score] Dropped {n_before - n_after} TRAIN perts with ≤1 kept GO term; kept {n_after}.")

        # (We do not drop TEST perts; if a TEST pert has 0–1 terms post-filter, it will naturally select nothing.)

        # --- Count tables over TRAIN perts (middle-restricted oracle) ---
        pos_counts = np.zeros((Kkeep, G), dtype=np.float32)
        neg_counts = np.zeros((Kkeep, G), dtype=np.float32)
        term_den   = np.zeros((Kkeep,), dtype=np.float32)  # number of TRAIN perts where term is present

        Pn = pred_delta_train.shape[0]
        for i in range(Pn):
            terms_i = np.where(P_train[i, :] > 0)[0]
            if terms_i.size == 0:
                continue
            xd = pred_delta_train[i]; td = true_delta_train[i]
            # build middle (freeze ±K)
            pos_frozen = topk_indices_desc(xd, self.K)
            neg_frozen = bottomk_indices_asc(xd, self.K)
            middle = np.ones(G, dtype=bool); middle[pos_frozen] = False; middle[neg_frozen] = False
            mid_idx = np.where(middle)[0]
            if mid_idx.size == 0:
                continue
            k_eff = min(self.K, mid_idx.size)
            pos_oracle = mid_idx[np.argsort(-td[mid_idx], kind="mergesort")][:k_eff]
            neg_oracle = mid_idx[np.argsort(td[mid_idx],  kind="mergesort")][:k_eff]

            # update counts for each term present in this TRAIN pert
            for t in terms_i:
                term_den[t] += 1.0
                pos_counts[t, pos_oracle] += 1.0
                neg_counts[t, neg_oracle] += 1.0

        # Fractions per term per gene
        den = np.maximum(term_den, 1.0).astype(np.float32)  # safe divide
        self.frac_pos = (pos_counts / den[:, None]).astype(np.float32)  # (Kkeep, G)
        self.frac_neg = (neg_counts / den[:, None]).astype(np.float32)

        # --- Cache TEST perturbed-gene term rows for fast selection ---
        self.P_test_kept = P_test
        self.P_train_kept = P_train
        self.perts_train_kept = perts_train_kept
        self.perts_test = perts_test
        self._test_term_rows = {}
        for i, p in enumerate(perts_test):
            self._test_term_rows[p] = P_test[i, :].astype(bool)

        kept_terms = int(Kkeep)
        used_terms = int((term_den > 0).sum())
        print(f"[go_score] kept_terms={kept_terms}, terms_with_train_support={used_terms}, "
              f"G={G}, train_kept={len(perts_train_kept)}, test={len(perts_test)}")
        return self

    def select(self,
               pred_delta_vec: np.ndarray,
               pos_frozen: np.ndarray,
               neg_frozen: np.ndarray,
               pert_label: str) -> tuple[np.ndarray, np.ndarray]:
        assert self.frac_pos is not None and self.frac_neg is not None
        assert self.gene_index is not None
        assert self.P_test_kept is not None and self._test_term_rows is not None

        G = pred_delta_vec.size
        # compute middle indices
        middle = np.ones(G, dtype=bool); middle[pos_frozen] = False; middle[neg_frozen] = False
        mid_idx = np.where(middle)[0]
        if mid_idx.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        # test pert's GO-term membership over kept terms
        terms_mask = self._test_term_rows.get(pert_label, None)
        if terms_mask is None:
            # unseen label (not in TEST list): select nothing (no fallback/random)
            return np.array([], dtype=int), np.array([], dtype=int)

        if not terms_mask.any():
            # no kept GO terms: select nothing
            return np.array([], dtype=int), np.array([], dtype=int)

        # scores: sum rows of frac_* over terms this pert has
        score_pos = self.frac_pos[terms_mask, :].sum(axis=0)  # (G,)
        score_neg = self.frac_neg[terms_mask, :].sum(axis=0)  # (G,)

        k_eff = min(self.K, mid_idx.size)
        pos_order = mid_idx[np.argsort(-score_pos[mid_idx], kind="mergesort")][:k_eff]
        neg_order = mid_idx[np.argsort(-score_neg[mid_idx], kind="mergesort")][:k_eff]
        return pos_order, neg_order
    

# ---- NEW: Supervised selector with XGBoost over PCA gene embeddings ----
class XGBTailSelector(BaseTailSelector):
    """
    Learn to pick middle genes for + and - tails with two XGBoost classifiers:
      - Train labels from TRAIN perts only, using middle-restricted oracle top-K/bottom-K.
      - Features: [x_p || x_g || x_p * x_g] where x_* are PCA embeddings of genes across ALL pathway sources.
    Inference on TEST uses ONLY predicted deltas (for freezing) + learned model (no test truth).
    """
    def __init__(self,
                 K: int,
                 rng: np.random.Generator,
                 PCs: np.ndarray,                 # (G, d)
                 gene_index: pd.Index,
                 xgb_params: dict):
        super().__init__(K)
        self.rng = rng
        self.PCs = PCs
        self.gene_index = gene_index
        self.xgb_params = xgb_params
        self.model_pos = None
        self.model_neg = None
        self.d = PCs.shape[1]

    def _feat_pair(self, p_idx: int, g_idx: int) -> np.ndarray:
        xp = self.PCs[p_idx]
        xg = self.PCs[g_idx]
        return np.concatenate([xp, xg, xp * xg], axis=0)

    def _oracle_labels_for_train(self, pred_delta: np.ndarray, true_delta: np.ndarray) -> tuple:
        """
        Build training rows only from TRAIN perts, middle genes only.
        Returns (X_pos, y_pos), (X_neg, y_neg), where y_* are {0,1}, with strong negative subsampling for balance.
        """
        P_train, G = pred_delta.shape
        d = self.d
        Xp_pos, yp_pos = [], []
        Xp_neg, yp_neg = [], []
        # subsample rate for negatives to keep ~K positives vs ~4K negatives manageable
        neg_downsample = max(1, int(G / (8 * self.K)))

        for i in range(P_train):
            xd = pred_delta[i]; td = true_delta[i]
            pos_frozen = topk_indices_desc(xd, self.K)
            neg_frozen = bottomk_indices_asc(xd, self.K)
            middle = np.ones(G, dtype=bool); middle[pos_frozen] = False; middle[neg_frozen] = False
            mid_idx = np.where(middle)[0]
            if mid_idx.size == 0:
                continue
            # oracle-in-middle
            pos_oracle = mid_idx[np.argsort(-td[mid_idx], kind="mergesort")][:min(self.K, mid_idx.size)]
            neg_oracle = mid_idx[np.argsort(td[mid_idx],  kind="mergesort")][:min(self.K, mid_idx.size)]
            pos_set = set(map(int, pos_oracle))
            neg_set = set(map(int, neg_oracle))
            # indices: perturbed gene row (by name equals label)
            # if pert name isn't found in gene list, skip
            # (caller ensures perts are gene symbols)
            # we infer p_idx here:
            # NOTE: the i-th row corresponds to perts_train[i] externally; we pass p_idx at select-time by name.
            # At train time we can recover p_idx by name list we pass into fit.
            # We'll store perts_train_gene_indices during fit.
            pass
        # We'll assemble using perts_train_gene_indices we stored in fit().
        return (None, None), (None, None)

    def fit(self,
            pred_delta_train: np.ndarray,
            true_delta_train: np.ndarray,
            perts_train: List[str]):
        # map pert labels to gene indices (skip perts not in var_names)
        self.perts_train_gene_indices = []
        for p in perts_train:
            if p in self.gene_index:
                self.perts_train_gene_indices.append(self.gene_index.get_loc(p))
            else:
                self.perts_train_gene_indices.append(-1)

        # Build TRAIN datasets
        P_train, G = pred_delta_train.shape
        X_pos, y_pos = [], []
        X_neg, y_neg = [], []

        for i in range(P_train):
            p_idx = self.perts_train_gene_indices[i]
            if p_idx < 0:
                continue
            xd = pred_delta_train[i]; td = true_delta_train[i]
            pos_frozen = topk_indices_desc(xd, self.K)
            neg_frozen = bottomk_indices_asc(xd, self.K)
            middle = np.ones(G, dtype=bool); middle[pos_frozen] = False; middle[neg_frozen] = False
            mid_idx = np.where(middle)[0]
            if mid_idx.size == 0:
                continue
            # oracle-in-middle
            k_eff = min(self.K, mid_idx.size)
            pos_oracle = mid_idx[np.argsort(-td[mid_idx], kind="mergesort")][:k_eff]
            neg_oracle = mid_idx[np.argsort(td[mid_idx],  kind="mergesort")][:k_eff]
            pos_set = set(map(int, pos_oracle))
            neg_set = set(map(int, neg_oracle))

            # build examples; subsample negatives to control size
            # target prevalence ~ K / (G-2K); set ratio ~ 1:4 (pos:neg) per pert for stability
            # pick at most 4*k_eff negatives from the middle
            self.rng.shuffle(mid_idx)
            neg_pool = [g for g in mid_idx if (g not in pos_set and g not in neg_set)]
            neg_pool = np.array(neg_pool, dtype=int)
            n_neg_cap = 4 * k_eff
            if neg_pool.size > n_neg_cap:
                neg_pool = neg_pool[:n_neg_cap]

            for g in pos_oracle:
                X_pos.append(self._feat_pair(p_idx, int(g))); y_pos.append(1)
            # draw equal number of hard negatives for pos-head: sample from middle not in pos_oracle
            # (we already truncated neg_pool)
            for g in neg_pool:
                X_pos.append(self._feat_pair(p_idx, int(g))); y_pos.append(0)

            for g in neg_oracle:
                X_neg.append(self._feat_pair(p_idx, int(g))); y_neg.append(1)
            # negatives for neg-head: reuse the same pool
            for g in neg_pool:
                X_neg.append(self._feat_pair(p_idx, int(g))); y_neg.append(0)

        if not X_pos or not X_neg:
            raise ValueError("[xgb] No training examples built. Check K and TRAIN size.")
        X_pos = np.vstack(X_pos); y_pos = np.asarray(y_pos, dtype=int)
        X_neg = np.vstack(X_neg); y_neg = np.asarray(y_neg, dtype=int)

        # Fit two XGB classifiers
        try:
            import xgboost as xgb
        except Exception as e:
            raise ImportError("xgboost not available. Please `pip install xgboost`.") from e

        self.model_pos = xgb.XGBClassifier(**self.xgb_params)
        self.model_neg = xgb.XGBClassifier(**self.xgb_params)

        self.model_pos.fit(X_pos, y_pos)
        self.model_neg.fit(X_neg, y_neg)
        return self

    def select(self,
               pred_delta_vec: np.ndarray,
               pos_frozen: np.ndarray,
               neg_frozen: np.ndarray,
               pert_label: str):
        """
        Score only middle genes for this test perturbation; pick top-K for + and bottom-K for - by probability.
        """
        assert self.model_pos is not None and self.model_neg is not None
        G = pred_delta_vec.size
        # middle mask
        middle = np.ones(G, dtype=bool); middle[pos_frozen] = False; middle[neg_frozen] = False
        mid_idx = np.where(middle)[0]
        if mid_idx.size == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        # map pert label to gene index (if not found, fall back to random middle)
        if pert_label in self.gene_index:
            p_idx = self.gene_index.get_loc(pert_label)
        else:
            # fallback
            k_eff = min(self.K, mid_idx.size)
            pos_boost = self.rng.choice(mid_idx, size=k_eff, replace=(mid_idx.size < k_eff))
            neg_boost = self.rng.choice(mid_idx, size=k_eff, replace=(mid_idx.size < k_eff))
            return np.array(pos_boost, dtype=int), np.array(neg_boost, dtype=int)

        # build features for all mid candidates
        feats = np.vstack([self._feat_pair(p_idx, int(g)) for g in mid_idx])
        pos_prob = self.model_pos.predict_proba(feats)[:, 1]
        neg_prob = self.model_neg.predict_proba(feats)[:, 1]

        k_eff = min(self.K, mid_idx.size)
        pos_order = mid_idx[np.argsort(-pos_prob, kind="mergesort")][:k_eff]
        neg_order = mid_idx[np.argsort(-neg_prob, kind="mergesort")][:k_eff]
        return pos_order, neg_order


# ---- Boost application ----
def apply_boost_once(pred_delta: np.ndarray,
                     pos_frozen: np.ndarray, neg_frozen: np.ndarray,
                     pos_boost: np.ndarray, neg_boost: np.ndarray) -> np.ndarray:
    out = pred_delta.copy()
    pos_mean = float(np.mean(pred_delta[pos_frozen])) if pos_frozen.size > 0 else 0.0
    neg_mean = float(np.mean(pred_delta[neg_frozen])) if neg_frozen.size > 0 else 0.0
    if pos_boost.size > 0:
        out[pos_boost] = pos_mean
    if neg_boost.size > 0:
        out[neg_boost] = neg_mean
    return out


def selection_accuracy(true_delta: np.ndarray,
                       pred_delta: np.ndarray,
                       pos_selected: list[np.ndarray],
                       neg_selected: list[np.ndarray],
                       K_pos: int,
                       K_neg: int,
                       pert_names: list[str] | None = None) -> pd.DataFrame:
    """
    Compute selection accuracy **against a middle-restricted oracle**:
    - Freeze top-K_pos and bottom-K_neg by *predicted* deltas (same as the method).
    - Define middle = all other genes.
    - Oracle positive set = top-K_pos by *true* delta within the middle.
      Oracle negative set = bottom-K_neg by *true* delta within the middle.
    Report the fraction of your selected genes that match these oracle sets.
    """
    P, G = true_delta.shape
    assert len(pos_selected) == P and len(neg_selected) == P, "Selections must align with P."
    rows = []
    for i in range(P):
        td = true_delta[i]
        xd = pred_delta[i]
        # Freeze by predicted deltas (same as algorithm)
        pos_frozen = topk_indices_desc(xd, K_pos)
        neg_frozen = bottomk_indices_asc(xd, K_neg)
        middle_mask = np.ones(G, dtype=bool)
        middle_mask[pos_frozen] = False
        middle_mask[neg_frozen] = False
        middle = np.where(middle_mask)[0]
        # Oracle sets: top-K_pos and bottom-K_neg within the middle by *true* delta
        if K_pos > 0 and middle.size > 0:
            mid_pos = middle[np.argsort(-td[middle], kind="mergesort")]
            oracle_pos = mid_pos[:min(K_pos, mid_pos.size)]
        else:
            oracle_pos = np.array([], dtype=int)
        if K_neg > 0 and middle.size > 0:
            mid_neg = middle[np.argsort(td[middle], kind="mergesort")]
            oracle_neg = mid_neg[:min(K_neg, mid_neg.size)]
        else:
            oracle_neg = np.array([], dtype=int)
        pos_set = set(map(int, oracle_pos)); neg_set = set(map(int, oracle_neg))
        sel_pos = np.asarray(pos_selected[i], dtype=int)
        sel_neg = np.asarray(neg_selected[i], dtype=int)
        pos_acc = np.nan if sel_pos.size == 0 else (np.isin(sel_pos, list(pos_set)).sum() / sel_pos.size)
        neg_acc = np.nan if sel_neg.size == 0 else (np.isin(sel_neg, list(neg_set)).sum() / sel_neg.size)
        rows.append({
            "pert_idx": i,
            "pert": pert_names[i] if pert_names is not None else i,
            "pos_sel": int(sel_pos.size),
            "neg_sel": int(sel_neg.size),
            "pos_correct": np.nan if np.isnan(pos_acc) else float(pos_acc),
            "neg_correct": np.nan if np.isnan(neg_acc) else float(neg_acc),
            "pos_oracle_size": int(len(pos_set)),
            "neg_oracle_size": int(len(neg_set)),
        })
    df = pd.DataFrame(rows)
    overall = {
        "pert_idx": "overall",
        "pert": "overall",
        "pos_sel": int(np.nansum(df["pos_sel"])),
        "neg_sel": int(np.nansum(df["neg_sel"])),
        "pos_correct": float(np.nanmean(df["pos_correct"])),
        "neg_correct": float(np.nanmean(df["neg_correct"])),
    }
    return pd.concat([df, pd.DataFrame([overall])], ignore_index=True)

def main():
    ap = argparse.ArgumentParser(description="Tail-boost train/test (modular).")
    ap.add_argument("--pred_h5ad", required=True)
    ap.add_argument("--true_h5ad", required=True)
    ap.add_argument("--target_label", required=True)
    ap.add_argument("--control_label", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--K", type=int, default=1500, help="Top-K size for each tail to freeze and to boost.")
    ap.add_argument("--test_pct_pert", type=float, default=0.2, help="Fraction of perturbations in TEST (0..1).")
    ap.add_argument("--method", choices=["random", "pair_score", "xgb", "go_score"], default="random")
    ap.add_argument("--pathways_yaml", type=str, default="", help="Required for method=pair_score.")
    ap.add_argument("--topL", type=int, default=100, help="Sparsity for v_p: keep top-|v| entries per test pert.")
    ap.add_argument("--gamma", type=float, default=0.5, help="Pathway size normalization exponent (0..1).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--evaluate", type=int, default=1, help="If 1 and evaluator importable, compute TEST metrics.")
    ap.add_argument("--n_pcs", type=int, default=256, help="Number of PCA components for gene embeddings (all sources).")
    ap.add_argument("--emb_gamma", type=float, default=0.0, help="Size normalization exponent for embedding columns (0..1).")
    # XGBoost knobs (safe, light defaults)
    ap.add_argument("--xgb_max_depth", type=int, default=4)
    ap.add_argument("--xgb_estimators", type=int, default=400)
    ap.add_argument("--xgb_lr", type=float, default=0.05)
    ap.add_argument("--xgb_subsample", type=float, default=0.9)
    ap.add_argument("--xgb_colsample", type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load & intersect
    pred = ad.read_h5ad(args.pred_h5ad)
    true = ad.read_h5ad(args.true_h5ad)
    pred, true = intersect_genes(pred, true)

    # Deltas
    pred_delta_all, pred_perts_all, ctrl = deltas_from_pseudobulk(pred, args.target_label, args.control_label)
    true_delta_all, true_perts_all, _ = deltas_from_pseudobulk(true, args.target_label, args.control_label)
    perts_common = np.array(sorted(set(pred_perts_all).intersection(true_perts_all)))
    if perts_common.size == 0:
        raise ValueError("No overlapping perturbations between predicted and true.")
    def reindex(rows, names, keep):
        idx = pd.Index(names).get_indexer(keep)
        return rows[idx, :]
    pred_delta_all = reindex(pred_delta_all, pred_perts_all, perts_common)
    true_delta_all = reindex(true_delta_all, true_perts_all, perts_common)

    P, G = pred_delta_all.shape
    perm = rng.permutation(P)
    n_test = int(round(args.test_pct_pert * P))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    perts_train = list(perts_common[train_idx])
    perts_test = list(perts_common[test_idx])
    print(f"[split] P={P} -> TRAIN={len(train_idx)} TEST={len(test_idx)} (test_pct_pert={args.test_pct_pert})")

    # Build selector
    if args.method == "random":
        selector = RandomTailSelector(K=args.K, rng=rng)
    elif args.method == "pair_score":
        if not args.pathways_yaml:
            raise ValueError("--pathways_yaml is required for method=pair_score.")
        selector = PairScoreTailSelector(K=args.K, rng=rng, topL=args.topL, gamma=args.gamma)
        selector.fit(pred_delta_train=pred_delta_all[train_idx, :],
                     true_delta_train=true_delta_all[train_idx, :],
                     perts_train=perts_train,
                     genes=list(pred.var_names),
                     pathways_yaml=args.pathways_yaml)
    elif args.method == "go_score":
        if not args.pathways_yaml:
            raise ValueError("--pathways_yaml is required for method=go_score.")
        selector = GOTermCountTailSelector(K=args.K, rng=rng)
        selector.fit(
            pred_delta_train=pred_delta_all[train_idx, :],
            true_delta_train=true_delta_all[train_idx, :],
            perts_train=perts_train,
            perts_test=perts_test,
            genes=list(pred.var_names),
            pathways_yaml=args.pathways_yaml,
        )
    else:  # xgb
        if not args.pathways_yaml:
            raise ValueError("--pathways_yaml is required for method=xgb.")
        # Build gene embeddings (ALL sources) once
        PCs, var_expl, gene_index, kept_cols = build_gene_embeddings_all_sources(
            genes=list(pred.var_names),
            perts_train=perts_train,
            perts_test=perts_test,
            pathways_yaml=args.pathways_yaml,
            n_pcs=args.n_pcs,
            gamma_size_norm=args.emb_gamma,
        )
        print(f"[embeddings] Using {PCs.shape[1]} PCs; variance explained = {100.0*var_expl:.2f}% "
              f"(columns kept: {len(kept_cols)})")
        # Import and construct the XGBTailSelector (defined below)
        selector = XGBTailSelector(
            K=args.K,
            rng=rng,
            PCs=PCs,
            gene_index=gene_index,
            xgb_params=dict(
                max_depth=args.xgb_max_depth,
                n_estimators=args.xgb_estimators,
                learning_rate=args.xgb_lr,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample,
                n_jobs=0,
                random_state=args.seed,
                tree_method="hist",
            ),
        )
        # Fit on TRAIN only (labels come from oracle-in-middle)
        selector.fit(
            pred_delta_train=pred_delta_all[train_idx, :],
            true_delta_train=true_delta_all[train_idx, :],
            perts_train=perts_train,
        )

    # Apply to TEST
    pred_delta_test = pred_delta_all[test_idx, :].copy()
    pred_delta_test_boosted = np.empty_like(pred_delta_test)
    # record which indices we boosted (for accuracy calc)
    pos_boost_list: list[np.ndarray] = []
    neg_boost_list: list[np.ndarray] = []
    for i, pert_label in enumerate(perts_test):
        x = pred_delta_test[i, :]
        pos_frozen = topk_indices_desc(x, args.K)
        neg_frozen = bottomk_indices_asc(x, args.K)
        if args.method == "random":
            pos_boost, neg_boost = selector.select(x, pos_frozen=pos_frozen, neg_frozen=neg_frozen)
        elif args.method == "pair_score":
            pos_boost, neg_boost = selector.select(x, pos_frozen=pos_frozen, neg_frozen=neg_frozen, pert_label=pert_label)
        elif args.method == "go_score":
            pos_boost, neg_boost = selector.select(
                pred_delta_vec=x,
                pos_frozen=pos_frozen,
                neg_frozen=neg_frozen,
                pert_label=pert_label,
            )
        else:  # xgb
            pos_boost, neg_boost = selector.select(
                pred_delta_vec=x,
                pos_frozen=pos_frozen,
                neg_frozen=neg_frozen,
                pert_label=pert_label,
            )
        pred_delta_test_boosted[i, :] = apply_boost_once(x, pos_frozen, neg_frozen, pos_boost, neg_boost)
        pos_boost_list.append(np.asarray(pos_boost, dtype=int))
        neg_boost_list.append(np.asarray(neg_boost, dtype=int))

    # Map back to expression, save & evaluate
    pred_expr_test_boosted = pred_delta_test_boosted + ctrl[None, :]
    out_h5ad = os.path.join(args.out_dir, f"tail_boost_TEST_only_{args.method}.h5ad")
    ad.AnnData(pred_expr_test_boosted, obs=pd.DataFrame({args.target_label: perts_test}), var=pred.var).write(out_h5ad)
    print("[done] Wrote:", out_h5ad)

    if args.evaluate:
        true_delta_test = true_delta_all[test_idx, :]
        # Save selection accuracy regardless of evaluator availability
        acc_df = selection_accuracy(true_delta_test, pred_delta_test, pos_boost_list, neg_boost_list,
                                    K_pos=args.K, K_neg=args.K, pert_names=perts_test)
        acc_path = os.path.join(args.out_dir, f"selection_accuracy_TEST_{args.method}.csv")
        acc_df.to_csv(acc_path, index=False)
        try:
            overall_row = acc_df.iloc[-1]
            print(f"[selection-accuracy][TEST] pos_correct={overall_row['pos_correct']:.4f} "
                  f"neg_correct={overall_row['neg_correct']:.4f} (K={args.K})")
        except Exception:
            pass
        if _EVAL_AVAILABLE:
            # Build bundle: (pred, true, perts, ctrl)
            metrics = evaluate_model(
                adata=true,
                args=type("A", (), {"target_label": args.target_label, "control_label": args.control_label})(),
                pred_bundle=(pred_expr_test_boosted, true_delta_test + ctrl[None, :], perts_test, ctrl)
            )
            pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(args.out_dir, f"metrics_TEST_only_{args.method}.csv"), index=False)
            # print("[metrics][TEST]", metrics)
    elif args.evaluate and not _EVAL_AVAILABLE:
        print("[warn] evaluate_model not found; skipping TEST metrics.")

    pd.DataFrame({"pert": perts_train, "split": "train"}).to_csv(os.path.join(args.out_dir, "perts_train.csv"), index=False)
    pd.DataFrame({"pert": perts_test,  "split": "test" }).to_csv(os.path.join(args.out_dir, "perts_test.csv"),  index=False)


if __name__ == "__main__":
    main()
