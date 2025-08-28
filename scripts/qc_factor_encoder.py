#!/usr/bin/env python3
"""
qc_factor_encoder.py

Quality checks for a learned pathway-anchored dictionary W (factors -> genes).
Optionally, stream denoised gene means x̄ from a trained scVI model to compute
activation quality and lab invariance summaries.

Outputs (written to --outdir):
- qc_summary.txt: human-readable summary of key stats
- per_factor_metrics.parquet: per-factor table (norms, outside mass, etc.)
- top_genes_per_factor.csv: top-k genes per factor (by loading)
- precision_at_k.csv: precision vs membership sets (if --membership-npy provided)
- activation_correlations.csv: corr(a_lin, simple set score) if scVI provided
- lab_effects.csv: per-factor ANOVA/KW p-values across labs (if lab_id exists)

Usage (dictionary-only checks):
  python qc_factor_encoder.py --W artifacts_v2/learned_factor_encoders/factor_W_K562.npy \
                              --outdir artifacts_v2/qc_W_K562

With scVI streaming (activation checks):
  python qc_factor_encoder.py --W artifacts_v2/learned_factor_encoders/factor_W_K562.npy \
                              --scvi-input artifacts_v2/scvi_input_K562_max200k_controls.h5ad \
                              --scvi-model-dir artifacts_v2/scvi_k562_mak200k_control_only/scvi_K562 \
                              --sample-cells 20000 \
                              --outdir artifacts_v2/qc_W_K562

Optional membership prior (improves interpretability metrics):
  --membership-npy artifacts_v2/pathways_onlypresage_K562_svd256/M.npy --n-anchors 256
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception:
    ad = None

# Optional: use your project's streaming util if available
try:
    from utils.scvi_stream import ScviOnTheFly
    _HAS_STREAM = True
except Exception:
    _HAS_STREAM = False


def load_W(W_path: str) -> np.ndarray:
    W = np.load(W_path)
    if W.ndim != 2:
        raise ValueError(f"W must be 2D; got shape {W.shape}")
    # Expect shape (F, G) as in your training code; if transposed, try to fix
    if W.shape[0] < W.shape[1]:
        pass  # (F, G) already
    else:
        # If someone saved (G, F) by mistake and G >> F, flip
        if W.shape[0] > 10000 and W.shape[1] < 2000:
            W = W.T
    return W.astype(np.float32, copy=False)


def basic_dictionary_checks(W: np.ndarray,
                            genes: Optional[List[str]] = None,
                            membership: Optional[np.ndarray] = None,
                            n_anchors: Optional[int] = None,
                            topk: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Returns:
      per_factor_metrics, top_genes_df, precision_at_k (or None)
    """
    F, G = W.shape
    # norms (factor rows over genes)
    row_l2 = np.linalg.norm(W, axis=1)
    row_l1 = np.sum(np.abs(W), axis=1)
    neg_count = np.sum(W < 0, axis=1)
    neg_frac = neg_count / G

    # outside mass vs membership sets (if provided)
    prec_df = None
    out_mass = np.full(F, np.nan, dtype=np.float32)
    in_mass = np.full(F, np.nan, dtype=np.float32)

    if membership is not None:
        if membership.shape[1] != G:
            raise ValueError(f"Membership gene dim {membership.shape[1]} != W gene dim {G}")
        if n_anchors is None:
            n_anchors = min(membership.shape[0], F)
        M = membership[:n_anchors, :].astype(bool)
        Wpos = np.maximum(W[:n_anchors, :], 0.0)
        denom = (Wpos.sum(axis=1) + 1e-8)
        in_mass[:n_anchors] = (Wpos * M).sum(axis=1) / denom
        out_mass[:n_anchors] = (Wpos * (~M)).sum(axis=1) / denom

        # precision@k using membership sets (for anchored factors)
        rows = []
        for f in range(n_anchors):
            w = W[f, :]
            idx = np.argsort(-w)[:topk]
            set_mask = M[f, :]
            prec = np.mean(set_mask[idx]) if set_mask.any() else np.nan
            rows.append({"factor": f, "precision_at_k": prec, "k": topk, "anchor_size": int(set_mask.sum())})
        prec_df = pd.DataFrame(rows)

    # top-k genes per factor
    rows2 = []
    for f in range(F):
        idx = np.argsort(-W[f, :])[:topk]
        picks = [(i, float(W[f, i])) for i in idx]
        if genes is not None:
            tg = [(genes[i], v) for i, v in picks]
        else:
            tg = picks
        rows2.append({"factor": f, "top_genes": tg})
    top_df = pd.DataFrame(rows2)

    metrics = pd.DataFrame({
        "factor": np.arange(F, dtype=int),
        "l2_norm": row_l2,
        "l1_norm": row_l1,
        "neg_frac": neg_frac,
        "in_mass": in_mass,
        "out_mass": out_mass,
    })
    return metrics, top_df, prec_df


def ridge_project_batch(W: np.ndarray, X: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """
    Compute a_lin = argmin_A ||X - A W||^2 + lam ||A||^2
    W: (F, G), X: (B, G)  ->  A: (B, F)
    Uses A = X (W W^T + lam I)^-1 W  (implemented as X @ K.T where K solves the linear system)
    """
    F, G = W.shape
    WWt = W @ W.T  # F x F
    K = np.linalg.solve(WWt + lam * np.eye(F, dtype=W.dtype), W)  # F x G
    A = X @ K.T
    A[A < 0] = 0.0
    return A


def simple_set_scores(X: np.ndarray, sets_bool: np.ndarray) -> np.ndarray:
    """
    Mean expression per set (rows = cells, cols = sets).
    X: (N, G), sets_bool: (S, G)
    Returns S-score matrix (N, S).
    """
    denom = np.maximum(1, sets_bool.sum(axis=1)).astype(np.float32)
    return (X @ sets_bool.T) / denom[None, :]


def activation_qc_with_scvi(W: np.ndarray,
                            scvi_input: str,
                            model_dir: str,
                            sample_cells: int = 20000,
                            lam: float = 0.1,
                            membership: Optional[np.ndarray] = None,
                            n_anchors: Optional[int] = None):
    if not _HAS_STREAM:
        raise RuntimeError("utils.scvi_stream.ScviOnTheFly not importable. Run within the repo environment.")
    # stream xbar
    stream = ScviOnTheFly(model_dir=model_dir, scvi_input_h5ad=scvi_input, library_size=1e4)
    adata = stream.adata
    N = adata.n_obs
    G = adata.n_vars
    if W.shape[1] != G:
        raise ValueError(f"W gene dim {W.shape[1]} != adata var dim {G}")

    rng = np.random.default_rng(13)
    n_samp = min(sample_cells, N)
    idx = np.sort(rng.choice(N, size=n_samp, replace=False))
    # fetch xbar in chunks to avoid memory spikes
    chunk = 8192
    xs = []
    for start in range(0, n_samp, chunk):
        end = min(n_samp, start + chunk)
        xbar = stream.get_xbar(indices=idx[start:end])
        xs.append(xbar.astype(np.float32, copy=False))
    X = np.vstack(xs)  # (n_samp, G)

    # ridge project to get a_lin
    A = ridge_project_batch(W, X, lam=lam)  # (n_samp, F)

    labs = None
    if "lab_id" in adata.obs.columns:
        labs = adata.obs["lab_id"].astype(str).values[idx]

    out = {"A": A, "X": X, "idx": idx, "labs": labs, "genes": adata.var_names.to_list()}
    if membership is not None:
        if n_anchors is None:
            n_anchors = min(membership.shape[0], W.shape[0])
        sets = membership[:n_anchors, :].astype(bool)
        S = simple_set_scores(X, sets)  # (n_samp, n_anchors)
        # corr across cells between A and S per anchored factor
        cors = []
        for f in range(n_anchors):
            a = A[:, f]
            s = S[:, f]
            a_mean = a.mean(); s_mean = s.mean()
            a_std = a.std() + 1e-8; s_std = s.std() + 1e-8
            r = float(((a - a_mean) * (s - s_mean)).mean() / (a_std * s_std))
            cors.append(r)
        out["anchor_corr"] = np.array(cors, dtype=np.float32)
    return out


def anova_by_lab(A: np.ndarray, labs: np.ndarray) -> pd.DataFrame:
    """
    Simple one-way ANOVA-like F-stat proxy via between/within variance ratio per factor.
    (No scipy dependency; report as effect-size style statistic.)
    """
    F = A.shape[1]
    df = []
    uniq = np.unique(labs)
    for f in range(F):
        y = A[:, f]
        mu = y.mean()
        ssb = 0.0
        ssw = 0.0
        n = len(y)
        for g in uniq:
            m = y[labs == g]
            if m.size == 0:
                continue
            ssb += m.size * (m.mean() - mu) ** 2
            ssw += ((m - m.mean()) ** 2).sum()
        stat = (ssb / (len(uniq) - 1 + 1e-8)) / (ssw / (n - len(uniq) + 1e-8) + 1e-8)
        df.append({"factor": f, "lab_effect_F": float(stat)})
    return pd.DataFrame(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--W", required=True, help="Path to learned W (npy). Shape (F, G).")
    p.add_argument("--membership-npy", default=None, help="Optional prior membership M (F_anchor, G).")
    p.add_argument("--n-anchors", type=int, default=None, help="If provided, number of anchored factors (assumed first).")
    p.add_argument("--scvi-input", default=None, help="Optional scVI input .h5ad (controls).")
    p.add_argument("--scvi-model-dir", default=None, help="Optional scVI saved model dir.")
    p.add_argument("--sample-cells", type=int, default=20000, help="Cells to sample for activation checks.")
    p.add_argument("--lambda-ridge", type=float, default=0.1, help="Ridge lambda for a_lin projection.")
    p.add_argument("--topk", type=int, default=50, help="Top-k genes to report per factor.")
    p.add_argument("--outdir", required=True, help="Where to write QC artifacts.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    W = load_W(args.W)  # (F, G)
    F, G = W.shape

    # Try to get gene names from scVI input if available
    genes = None
    if args.scvi_input is not None and ad is not None:
        try:
            A = ad.read_h5ad(args.scvi_input)
            genes = A.var_names.to_list()
        except Exception:
            genes = None

    M = None
    if args.membership_npy:
        mp = args.membership_npy
        if os.path.isdir(mp):
            candidate = os.path.join(mp, "M.npy")
            if os.path.exists(candidate):
                mp = candidate
        M = np.load(mp).astype(bool)
        if M.shape[1] != G:
            raise ValueError(f"Membership gene dim {M.shape[1]} does not match W gene dim {G}.")

    # Dictionary-only checks
    metrics, top_df, prec_df = basic_dictionary_checks(W, genes=genes, membership=M,
                                                       n_anchors=args.n_anchors,
                                                       topk=args.topk)
    metrics.to_parquet(os.path.join(args.outdir, "per_factor_metrics.parquet"))
    # Expand top genes into a digestible CSV
    rows = []
    for _, r in top_df.iterrows():
        f = int(r["factor"])
        tg = r["top_genes"]
        for rank, (g, val) in enumerate(tg, 1):
            rows.append({"factor": f, "rank": rank, "gene": g if isinstance(g, str) else str(g), "weight": float(val)})
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "top_genes_per_factor.csv"), index=False)
    if prec_df is not None:
        prec_df.to_csv(os.path.join(args.outdir, "precision_at_k.csv"), index=False)

    # Activation checks if scVI provided
    activation_stats = None
    if args.scvi_input and args.scvi_model_dir:
        activation_stats = activation_qc_with_scvi(W, args.scvi_input, args.scvi_model_dir,
                                                   sample_cells=args.sample_cells,
                                                   lam=args.lambda_ridge,
                                                   membership=M, n_anchors=args.n_anchors)
        if "anchor_corr" in activation_stats:
            cors = activation_stats["anchor_corr"]
            pd.DataFrame({"factor": np.arange(len(cors)), "corr_activation_vs_setscore": cors}).to_csv(
                os.path.join(args.outdir, "activation_correlations.csv"), index=False
            )
        if activation_stats["labs"] is not None:
            lab_df = anova_by_lab(activation_stats["A"], activation_stats["labs"])
            lab_df.to_csv(os.path.join(args.outdir, "lab_effects.csv"), index=False)

    # Summarize
    lines = []
    lines.append(f"W shape: {W.shape} (F factors x G genes)")
    lines.append(f"Negativity violations (mean frac): {metrics['neg_frac'].mean():.6f}")
    if "in_mass" in metrics.columns:
        valid = np.isfinite(metrics["in_mass"].values)
        if valid.any():
            lines.append(f"Anchored factors in-mass (median over anchored): {np.nanmedian(metrics['in_mass']):.3f}")
            lines.append(f"Anchored factors out-mass (median over anchored): {np.nanmedian(metrics['out_mass']):.3f}")
    if prec_df is not None:
        lines.append(f"Median precision@{args.topk} (anchored): {prec_df['precision_at_k'].median():.3f}")
    if activation_stats is not None and "anchor_corr" in activation_stats:
        cors = activation_stats["anchor_corr"]
        lines.append(f"Activation vs set-score corr (median over anchored): {np.nanmedian(cors):.3f}")
    if activation_stats is not None and activation_stats["labs"] is not None:
        lines.append("Lab effects: see lab_effects.csv (higher F means stronger lab dependence).")

    with open(os.path.join(args.outdir, "qc_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"[OK] W QC written to {args.outdir}")

if __name__ == "__main__":
    main()
