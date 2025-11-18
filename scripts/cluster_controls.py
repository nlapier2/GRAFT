#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


# --- Simple gene signatures (customize if you like) ---

CELL_CYCLE_S_GENES = [
    "MCM5", "PCNA", "MCM2", "MCM4", "RRM1",
    "UNG", "GINS2", "MCM6", "CDCA7", "DTL",
]

CELL_CYCLE_G2M_GENES = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5",
    "TPX2", "TOP2A", "NDC80", "CKS2", "MKI67",
]

STRESS_GENES = [
    "DDIT3", "ATF4", "HSPA1A", "HSPA1B", "HSPB1",
    "HMOX1", "FOS", "JUN", "JUNB", "EGR1",
]


# --- Utilities ---

def get_control_mask(adata, control_col=None, control_value=None):
    """
    Returns a boolean mask for control cells.

    - If control_col is None or not in adata.obs, treat ALL cells as 'controls'.
    - If control_col is set but control_value is None, treat all non-missing
      entries in that column as controls.
    - If both are set, treat rows where obs[control_col] == control_value as controls.
    """
    n = adata.n_obs
    if control_col is None or control_col not in adata.obs.columns:
        return np.ones(n, dtype=bool)

    col = adata.obs[control_col].astype(str)
    if control_value is None:
        return col.notna().values

    return (col.values == str(control_value))


def _is_sparse(x):
    return sp.issparse(x)


def load_gene_dataframe(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if ext in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataframe extension: {ext}")


# --- Core steps ---

def preprocess_and_cluster(
    adata,
    n_hvgs=2000,
    n_pcs=50,
    n_neighbors=15,
    resolution=1.0,
    cluster_key="leiden",
    min_genes=200,
    min_cells=3,
    random_state=0,
):
    # Basic filtering
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells)

    # Normalization & log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Highly variable genes
    if n_hvgs is not None:
        # IMPORTANT: do NOT subset, just flag HVGs
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, subset=False)

    # Scaling & PCA (use only HVGs for PCA if we have them,
    # but keep all genes in adata.var_names)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(
        adata,
        n_comps=n_pcs,
        svd_solver="arpack",
        use_highly_variable=(n_hvgs is not None),
    )

    # Neighbors, UMAP, clustering
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)


def score_signatures_and_annotate(adata, cluster_key, outdir: Path):
    """
    Scores cell cycle and stress signatures and assigns a simple heuristic label to each cluster.
    """
    # Restrict signatures to observed genes
    var_names = set(adata.var_names)
    s_genes = [g for g in CELL_CYCLE_S_GENES if g in var_names]
    g2m_genes = [g for g in CELL_CYCLE_G2M_GENES if g in var_names]
    stress_genes = [g for g in STRESS_GENES if g in var_names]

    if len(s_genes) >= 5 and len(g2m_genes) >= 5:
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    else:
        print("Skipping cell cycle scoring: too few S/G2M genes present.")

    if len(stress_genes) >= 3:
        sc.tl.score_genes(adata, stress_genes, score_name="stress_score")
    else:
        print("Skipping stress scoring: too few stress genes present.")

    clusters = sorted(adata.obs[cluster_key].astype(str).unique())
    records = []

    for cl in clusters:
        mask = adata.obs[cluster_key].astype(str) == cl
        rec = {
            "cluster": cl,
            "n_cells": int(mask.sum()),
        }

        # Median scores per cluster
        for score_col in ["S_score", "G2M_score", "stress_score"]:
            if score_col in adata.obs:
                rec[f"median_{score_col}"] = float(
                    np.nanmedian(adata.obs.loc[mask, score_col])
                )
            else:
                rec[f"median_{score_col}"] = np.nan

        s_val = rec.get("median_S_score", np.nan)
        g2m_val = rec.get("median_G2M_score", np.nan)
        stress_val = rec.get("median_stress_score", np.nan)

        label = "unannotated"
        if not np.isnan(s_val) and not np.isnan(g2m_val) and not np.isnan(stress_val):
            cc_val = max(s_val, g2m_val)
            if cc_val < 0 and stress_val < 0:
                label = "quiescent/other"
            elif cc_val >= stress_val:
                label = "cell_cycle_S" if s_val >= g2m_val else "cell_cycle_G2M"
            else:
                label = "stress_response"

        rec["annotation"] = label
        records.append(rec)

    annot_df = pd.DataFrame(records)
    annot_df.to_csv(outdir / "cluster_annotations.csv", index=False)

    # Add annotation to obs
    annot_map = dict(zip(annot_df["cluster"].astype(str), annot_df["annotation"]))
    adata.obs["cluster_annotation"] = (
        adata.obs[cluster_key].astype(str).map(annot_map)
    )


def find_cluster_marker_genes(adata, cluster_key, outdir: Path, n_genes_plot=10):
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon")

    markers_df = sc.get.rank_genes_groups_df(adata, group=None)
    markers_df.to_csv(outdir / "cluster_marker_genes.csv", index=False)

    # Plotting top markers
    sc.pl.rank_genes_groups(
        adata,
        n_genes=n_genes_plot,
        sharey=False,
        save="_rank_genes_groups.png",
    )
    


def summarize_genes_of_interest(adata, cluster_key, gene_df_path: Path, outdir: Path):
    df = load_gene_dataframe(gene_df_path)
    genes = list(df.columns)

    present_genes = [g for g in genes if g in adata.var_names]
    missing_genes = sorted(set(genes) - set(present_genes))

    if missing_genes:
        print(f"Warning: {len(missing_genes)} genes not found in adata.var_names.")
        if len(missing_genes) <= 20:
            print("Missing genes:", ", ".join(missing_genes))
        else:
            print("Example missing genes:", ", ".join(missing_genes[:20]), "...")

    if not present_genes:
        print("No genes of interest found in the AnnData object; skipping summary.")
        return None, []

    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    gene_indices = [var_idx[g] for g in present_genes]

    clusters = sorted(adata.obs[cluster_key].astype(str).unique())
    rows = []

    for cl in clusters:
        mask = adata.obs[cluster_key].astype(str) == cl
        sub = adata[mask, :]

        if _is_sparse(sub.X):
            sub_X = sub.X[:, gene_indices]
            means = np.asarray(sub_X.mean(axis=0)).ravel()
        else:
            means = sub.X[:, gene_indices].mean(axis=0)

        row = {"cluster": cl, "n_cells": int(mask.sum())}
        row.update({g: float(m) for g, m in zip(present_genes, means)})
        rows.append(row)

    expr_df = pd.DataFrame(rows)
    expr_df.to_csv(outdir / "genes_of_interest_expression_by_cluster.csv", index=False)
    return expr_df, present_genes


def compute_genes_of_interest_correlations(
    expr_df: pd.DataFrame,
    genes: list[str],
    outdir: Path,
):
    """
    Compute a gene x gene correlation matrix based on mean expression
    across clusters (rows in expr_df).
    """
    if expr_df is None or not genes:
        print("No expression dataframe / genes provided; skipping correlation.")
        return

    # Use clusters as observations, genes as variables
    mat = expr_df.set_index("cluster")[genes]

    corr = mat.corr(method="pearson")
    corr.to_csv(outdir / "genes_of_interest_cluster_correlation.csv")
    print(
        f"Saved correlation matrix for {len(genes)} genes to "
        f"{outdir / 'genes_of_interest_cluster_correlation.csv'}"
    )


def compute_genes_of_interest_abs_magnitude_correlation(
    expr_df: pd.DataFrame,
    genes: list[str],
    outdir: Path,
):
    """
    Absolute magnitude correlation across clusters.

    For each gene g:
      - Take its mean expression per cluster: mu_g(c)
      - Center across clusters: mu_g(c) - mean_c mu_g(c)
      - Take absolute value: |mu_g(c) - mean_c mu_g(c)|

    This captures how strongly (not in which direction) a gene deviates from
    its average in each cluster. We then compute Pearson correlation between
    these |centered| profiles for all pairs of genes.
    """
    if expr_df is None or not genes:
        print("No expression dataframe / genes provided; skipping abs magnitude correlation.")
        return

    # clusters x genes matrix
    mat = expr_df.set_index("cluster")[genes]

    # center within each gene across clusters
    gene_means = mat.mean(axis=0)              # Series, no keepdims
    mat_centered = mat.sub(gene_means, axis=1) # subtract per-column means

    # take absolute deviations
    mat_abs = mat_centered.abs()

    # correlate |deviation| profiles across clusters
    corr_abs = mat_abs.corr(method="pearson")

    out_path = outdir / "genes_of_interest_abs_magnitude_correlation_by_cluster.csv"
    corr_abs.to_csv(out_path)
    print(f"Saved absolute magnitude correlation (cluster means) to {out_path}")


def make_plots(adata, cluster_key, outdir: Path):
    # Cluster UMAP
    sc.pl.umap(
        adata,
        color=[cluster_key],
        legend_loc="on data",
        save="_clusters.png",
    )

    # Cell cycle / stress scores if present
    score_cols_for_plot = [
        c
        for c in ["S_score", "G2M_score", "stress_score", "cluster_annotation"]
        if c in adata.obs
    ]
    if score_cols_for_plot:
        sc.pl.umap(
            adata,
            color=score_cols_for_plot,
            save="_scores_and_annotations.png",
        )

def compute_local_residual_correlations_for_genes(
    adata,
    genes: list[str],
    outdir: Path,
    control_col: str | None = None,
    control_value: str | None = None,
    n_neighbors: int = 15,
):
    """
    Local / manifold-aware co-expression on *control cells*.

    Idea:
    - Work only on control cells.
    - For each cell, compute a neighborhood on the PCA manifold.
    - Use the neighbor graph to compute a local mean expression for each gene.
    - Define 'local residuals' as X - local_mean(X).
    - Correlate genes based on these local residuals across cells.
      => Genes are similar if they co-vary in their *local deviations*
         from nearby cells on the manifold.
    """
    if not genes:
        print("No genes of interest provided; skipping local residual correlation.")
        return

    control_mask = get_control_mask(adata, control_col, control_value)
    if control_mask.sum() == 0:
        print("No control cells selected; skipping local residual correlation.")
        return

    adata_ctrl = adata[control_mask].copy()
    n_cells = adata_ctrl.n_obs

    if n_cells <= 1:
        print("Not enough control cells for local correlation; skipping.")
        return

    if "X_pca" not in adata_ctrl.obsm_keys():
        raise ValueError(
            "X_pca embedding not found in adata.obsm. "
            "Make sure PCA is computed before calling this."
        )

    # Build neighbors on control cells using PCA representation
    k = min(n_neighbors, max(1, n_cells - 1))
    print(f"Building neighbor graph for local residuals on {n_cells} control cells (k={k})...")
    sc.pp.neighbors(adata_ctrl, n_neighbors=k, use_rep="X_pca")

    # Neighbor graph: use connectivities; add self-loops and row-normalize
    from scipy.sparse import eye as sp_eye

    W = adata_ctrl.obsp["connectivities"].tocsr()
    W = W + sp_eye(W.shape[0], format="csr")  # self-loops

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv_row = 1.0 / row_sums
    W_norm = sp.diags(inv_row) @ W  # row-normalized

    # Extract expression for genes of interest
    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    gene_indices = [var_idx[g] for g in genes]

    X = adata_ctrl[:, gene_indices].X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    # Local means via neighbor-averaging
    local_means = W_norm @ X  # shape: (n_cells, G)

    # Local residuals
    R = X - local_means

    # Center each gene's residuals across cells
    R = R - R.mean(axis=0, keepdims=True)

    # Compute Pearson correlation between genes using residuals
    std = R.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    R_norm = R / std
    # correlation matrix G x G
    corr = (R_norm.T @ R_norm) / (R_norm.shape[0] - 1)

    corr_df = pd.DataFrame(corr, index=genes, columns=genes)
    out_path = outdir / "genes_of_interest_local_residual_correlation_controls.csv"
    corr_df.to_csv(out_path)
    print(f"Saved local residual correlation (control cells) to {out_path}")


def compute_linear_state_similarity_for_genes(
    adata,
    genes: list[str],
    outdir: Path,
    control_col: str | None = None,
    control_value: str | None = None,
    n_pcs: int | None = None,
):
    """
    For each gene, fit a linear model of expression vs PCA latent state on *control cells*:
        expression_g ≈ β_g^T z

    - z is the PCA embedding (X_pca) for control cells.
    - β_g is a vector of slopes for gene g.
    - Similarity between genes is defined as cosine similarity between their β vectors.
    """
    if not genes:
        print("No genes of interest provided; skipping linear state similarity.")
        return

    control_mask = get_control_mask(adata, control_col, control_value)
    if control_mask.sum() == 0:
        print("No control cells selected; skipping linear state similarity.")
        return

    adata_ctrl = adata[control_mask].copy()
    n_cells = adata_ctrl.n_obs

    if "X_pca" not in adata_ctrl.obsm_keys():
        raise ValueError(
            "X_pca embedding not found in adata.obsm. "
            "Make sure PCA is computed before calling this."
        )

    Z = adata_ctrl.obsm["X_pca"]
    Z = np.asarray(Z, dtype=float)
    if n_pcs is not None and n_pcs < Z.shape[1]:
        Z = Z[:, :n_pcs]

    # Center and scale Z (so PCs have comparable influence)
    Z = Z - Z.mean(axis=0, keepdims=True)
    Z_std = Z.std(axis=0, ddof=1)
    Z_std[Z_std == 0] = 1.0
    Z = Z / Z_std

    # Expression matrix for genes of interest
    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    gene_indices = [var_idx[g] for g in genes]

    X = adata_ctrl[:, gene_indices].X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    # Center genes (intercept gets absorbed into centered expression)
    X = X - X.mean(axis=0, keepdims=True)

    # Solve Z * B = X in least squares sense (B: d x G)
    # each column of B is β_g for a gene
    B, _, _, _ = np.linalg.lstsq(Z, X, rcond=None)  # shape: (n_pcs_used, G)

    # Save coefficients
    coef_df = pd.DataFrame(
        B.T,
        index=genes,
        columns=[f"PC{i+1}" for i in range(B.shape[0])],
    )
    coef_path = outdir / "genes_of_interest_pca_coefficients_controls.csv"
    coef_df.to_csv(coef_path)
    print(f"Saved PCA linear coefficients (control cells) to {coef_path}")

    # Cosine similarity between β vectors
    norms = np.linalg.norm(B, axis=0)
    norms[norms == 0] = 1.0
    B_norm = B / norms
    cos_sim = B_norm.T @ B_norm  # G x G

    cos_df = pd.DataFrame(cos_sim, index=genes, columns=genes)
    cos_path = outdir / "genes_of_interest_gradient_cosine_similarity_controls.csv"
    cos_df.to_csv(cos_path)
    print(f"Saved gradient cosine similarity (control cells) to {cos_path}")


def check_precomputed_adata(adata, cluster_key: str):
    """
    Sanity checks for using a precomputed AnnData:
    - cluster_key present in adata.obs
    - PCA and UMAP embeddings exist
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"cluster_key '{cluster_key}' not found in adata.obs. "
            "When using --precomputed-h5ad, it must contain this column."
        )

    if "X_pca" not in adata.obsm_keys():
        raise ValueError(
            "X_pca embedding not found in precomputed AnnData. "
            "When using --precomputed-h5ad, PCA must already exist."
        )

    if "X_umap" not in adata.obsm_keys():
        print(
            "Warning: X_umap not found in precomputed AnnData. "
            "UMAP plots may fail unless you recompute UMAP."
        )


def main(args):
    input_path = Path(args.input_h5ad)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    sc.settings.verbosity = 3
    sc.settings.autoshow = False
    sc.settings.figdir = str(outdir)

    # --- choose raw vs precomputed AnnData ---

    if args.precomputed_h5ad is not None:
        pre_path = Path(args.precomputed_h5ad)
        print(f"Reading PRECOMPUTED AnnData from {pre_path}...")
        adata = sc.read_h5ad(pre_path)
        check_precomputed_adata(adata, cluster_key=args.cluster_key)
        print(
            "Using precomputed PCA/UMAP/clusters; "
            "skipping preprocessing and clustering steps."
        )
    else:
        print(f"Reading AnnData from {input_path}...")
        adata = sc.read_h5ad(input_path)

        print("Preprocessing and clustering...")
        preprocess_and_cluster(
            adata,
            n_hvgs=args.n_hvgs,
            n_pcs=args.n_pcs,
            n_neighbors=args.n_neighbors,
            resolution=args.resolution,
            cluster_key=args.cluster_key,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            random_state=args.random_state,
        )

    print("Scoring signatures and annotating clusters...")
    score_signatures_and_annotate(adata, args.cluster_key, outdir)

    print("Finding cluster marker genes...")
    #find_cluster_marker_genes(
    #    adata,
    #    cluster_key=args.cluster_key,
    #    outdir=outdir,
    #    n_genes_plot=args.n_marker_genes_plot,
    #)

    if args.genes_of_interest is not None:
        print("Summarizing genes of interest by cluster...")
        expr_df, present_genes = summarize_genes_of_interest(
            adata,
            cluster_key=args.cluster_key,
            gene_df_path=Path(args.genes_of_interest),
            outdir=outdir,
        )

        print("Computing gene-gene correlations across clusters for genes of interest...")
        compute_genes_of_interest_correlations(
            expr_df=expr_df,
            genes=present_genes,
            outdir=outdir,
        )

        print("Computing absolute magnitude correlation across clusters for genes of interest...")
        compute_genes_of_interest_abs_magnitude_correlation(
            expr_df=expr_df,
            genes=present_genes,
            outdir=outdir,
        )

        print("Computing local residual correlation in control cells...")
        compute_local_residual_correlations_for_genes(
            adata=adata,
            genes=present_genes,
            outdir=outdir,
            control_col=args.control_col,
            control_value=args.control_value,
            n_neighbors=args.local_k,
        )

        print("Computing linear state-model similarity in control cells...")
        compute_linear_state_similarity_for_genes(
            adata=adata,
            genes=present_genes,
            outdir=outdir,
            control_col=args.control_col,
            control_value=args.control_value,
            n_pcs=args.n_pcs,
        )

    print("Making plots...")
    make_plots(adata, cluster_key=args.cluster_key, outdir=outdir)

    # Save processed AnnData
    out_adata_path = outdir / "adata_with_clusters_and_scores.h5ad"
    adata.write(out_adata_path)
    print(f"Finished. Results written to {outdir}")
    print(f"Processed AnnData saved to {out_adata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Cluster cells in an AnnData .h5ad file, generate UMAP plots, "
            "annotate clusters (cell cycle / stress), find marker genes, and "
            "summarize genes of interest by cluster."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        required=True,
        help="Path to input .h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write plots and reports.",
    )
    parser.add_argument(
        "--genes-of-interest",
        help=(
            "Optional: path to a dataframe (csv/tsv/pkl/parquet) whose COLUMNS "
            "are gene names of interest; expression by cluster will be reported."
        ),
        default=None,
    )
    parser.add_argument(
        "--cluster-key",
        default="leiden",
        help="Name of the clustering column to create in adata.obs (default: leiden).",
    )
    parser.add_argument(
        "--n-hvgs",
        type=int,
        default=2000,
        help="Number of highly variable genes to select (default: 2000).",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=50,
        help="Number of principal components to use (default: 50).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for the graph (default: 15).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden clustering resolution (default: 1.0).",
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        default=200,
        help="Filter cells with fewer genes than this (default: 200).",
    )
    parser.add_argument(
        "--min-cells",
        type=int,
        default=3,
        help="Filter genes expressed in fewer cells than this (default: 3).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for neighbors/UMAP/leiden (default: 0).",
    )
    parser.add_argument(
        "--n-marker-genes-plot",
        type=int,
        default=10,
        help="Number of top marker genes per cluster to show in plots (default: 10).",
    )
    parser.add_argument(
        "--control-col",
        help=(
            "Optional: adata.obs column indicating perturbation/control status; "
            "used to select control cells for control-based analyses."
        ),
        default=None,
    )
    parser.add_argument(
        "--control-value",
        help=(
            "If --control-col is set, this value (as string) marks control cells "
            "(e.g. 'CTRL', 'NT', 'non-targeting'). If omitted, all non-missing "
            "values in that column are treated as controls."
        ),
        default=None,
    )
    parser.add_argument(
        "--local-k",
        type=int,
        default=15,
        help=(
            "Number of neighbors for local residual correlation in control cells "
            "(default: 15)."
        ),
    )
    parser.add_argument(
        "--precomputed-h5ad",
        help=(
            "Optional: path to a .h5ad that already has PCA/UMAP and the "
            "clustering column (--cluster-key). If provided, this file is "
            "used instead of --input-h5ad and preprocessing+clustering are skipped."
        ),
        default=None,
    )

    args = parser.parse_args()
    main(args)
