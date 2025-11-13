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
from typing import Tuple, List

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
                       pos_selected: list[np.ndarray],
                       neg_selected: list[np.ndarray],
                       K_pos: int,
                       K_neg: int,
                       pert_names: list[str] | None = None) -> pd.DataFrame:
    """
    Compute fraction of selected middle genes that truly belong to the top/bottom K for each perturbation.
    Returns per-pert rows plus an 'overall' row with means (NaNs ignored).
    """
    P, G = true_delta.shape
    assert len(pos_selected) == P and len(neg_selected) == P, "Selections must align with P."
    rows = []
    for i in range(P):
        td = true_delta[i]
        top_idx = np.argpartition(-td, K_pos-1)[:K_pos] if K_pos > 0 else np.array([], dtype=int)
        bot_idx = np.argpartition(td, K_neg-1)[:K_neg] if K_neg > 0 else np.array([], dtype=int)
        top_set = set(map(int, top_idx)); bot_set = set(map(int, bot_idx))
        sel_pos = np.asarray(pos_selected[i], dtype=int)
        sel_neg = np.asarray(neg_selected[i], dtype=int)
        pos_acc = np.nan if sel_pos.size == 0 else (np.isin(sel_pos, list(top_set)).sum() / sel_pos.size)
        neg_acc = np.nan if sel_neg.size == 0 else (np.isin(sel_neg, list(bot_set)).sum() / sel_neg.size)
        rows.append({
            "pert_idx": i,
            "pert": pert_names[i] if pert_names is not None else i,
            "pos_sel": int(sel_pos.size),
            "neg_sel": int(sel_neg.size),
            "pos_correct": np.nan if np.isnan(pos_acc) else float(pos_acc),
            "neg_correct": np.nan if np.isnan(neg_acc) else float(neg_acc),
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
    ap.add_argument("--method", choices=["random", "pair_score"], default="random")
    ap.add_argument("--pathways_yaml", type=str, default="", help="Required for method=pair_score.")
    ap.add_argument("--topL", type=int, default=100, help="Sparsity for v_p: keep top-|v| entries per test pert.")
    ap.add_argument("--gamma", type=float, default=0.5, help="Pathway size normalization exponent (0..1).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--evaluate", type=int, default=1, help="If 1 and evaluator importable, compute TEST metrics.")
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
    else:
        if not args.pathways_yaml:
            raise ValueError("--pathways_yaml is required for method=pair_score.")
        selector = PairScoreTailSelector(K=args.K, rng=rng, topL=args.topL, gamma=args.gamma)
        selector.fit(pred_delta_train=pred_delta_all[train_idx, :],
                     true_delta_train=true_delta_all[train_idx, :],
                     perts_train=perts_train,
                     genes=list(pred.var_names),
                     pathways_yaml=args.pathways_yaml)

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
        else:
            pos_boost, neg_boost = selector.select(x, pos_frozen=pos_frozen, neg_frozen=neg_frozen, pert_label=pert_label)
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
        acc_df = selection_accuracy(true_delta_test, pos_boost_list, neg_boost_list,
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
