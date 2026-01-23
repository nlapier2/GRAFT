import pandas as pd
import numpy as np
from scipy import stats
import math
import os
import argparse
import matplotlib.pyplot as plt
import anndata as ad
import scipy.sparse as sp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Active Learning Baselines: Random vs. Magnitude Sampling")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the TSV file containing gene_name, LoF_gamma, and HBA1_beta."
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=100, 
        help="Number of genes to reveal in each batch (default: 100)."
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print progress every N batches (default: 10)."
    )
    
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.05,
        help="P-value threshold for statistical significance (default: 0.05)."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save the correlation plots."
    )

    # New Arguments for Control Strategy
    parser.add_argument(
        "--control_h5ad",
        type=str,
        default=None,
        help="Path to H5AD file containing control cells for covariance calculation."
    )
    
    parser.add_argument(
        "--target_label",
        type=str,
        default="target_gene",
        help="Obs column name identifying perturbation/control status (default: target_gene)."
    )
    
    parser.add_argument(
        "--control_label",
        type=str,
        default="",
        help="Value in target_label that identifies control cells. If blank, uses all cells (default: '')."
    )

    return parser.parse_args()


def compute_control_covariance(h5ad_path, target_gene, obs_label, control_val):
    """
    Loads an AnnData file, filters for control cells (case-insensitive), 
    and calculates the covariance between every gene and the 'target_gene'.
    Returns a pandas Series mapping gene_name -> absolute_covariance.
    """
    if not os.path.exists(h5ad_path):
        print(f"Warning: Control H5AD not found at {h5ad_path}")
        return None

    print(f"Loading control data from {h5ad_path}...")
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"Error loading H5AD: {e}")
        return None

    # 1. Filter for control cells (if label is provided)
    if control_val is not None and str(control_val).strip() != "":
        if obs_label in adata.obs.columns:
            # Case-insensitive comparison
            c_val_lower = str(control_val).lower()
            mask = adata.obs[obs_label].astype(str).str.lower() == c_val_lower
            
            n_total = adata.n_obs
            adata = adata[mask].copy()
            print(f"Filtered control cells: {adata.n_obs} / {n_total} cells (label '{obs_label}' ~= '{control_val}')")
        else:
            print(f"Warning: obs column '{obs_label}' not found. Using all {adata.n_obs} cells.")
    else:
        # Blank control label -> Trust that all cells are controls
        print(f"Control label is blank. Using all {adata.n_obs} cells as controls.")

    if adata.n_obs < 5:
        print("Error: Too few control cells to compute covariance.")
        return None

    # 2. Check for target gene
    if target_gene not in adata.var_names:
        print(f"Error: Target gene '{target_gene}' not found in H5AD var_names.")
        return None

    # 3. Compute Covariance
    # Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
    
    # Extract Target Vector
    target_idx = adata.var_names.get_loc(target_gene)
    X = adata.X
    
    # Handle Sparse vs Dense (Convert to dense for simple vectorization)
    if sp.issparse(X):
        try:
            X = X.toarray() 
        except MemoryError:
            print("Error: Control matrix too large to densify for covariance calc.")
            return None
        
    # Get target column and center it
    y_vec = X[:, target_idx]
    y_centered = y_vec - np.mean(y_vec)
    
    # Center all genes
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean[None, :]
    
    # Calculate Covariance: (X_c . y_c) / (N - 1)
    N = adata.n_obs
    covariances = np.dot(X_centered.T, y_centered) / (N - 1)
    
    # Return as Series
    return pd.Series(covariances, index=adata.var_names)


def run_simulation_strategy(name, df_sorted, total_genes, batch_size, p_threshold, print_every=10):
    """
    Runs the batch simulation for a specific ordering of genes (df_sorted).
    """
    print(f"\nRunning Strategy: {name}")
    print(f"{'Batch':<8} | {'Revealed':<8} | {'Corr (ObsGenes)':<15} | {'P (ObsGenes)':<15} | {'Corr (ImpGenes)':<15} | {'P (ImpGenes)':<15}")
    print("-" * 75)

    # These are the full "Ground Truth" vectors
    all_lof_true = df_sorted['LoF_gamma'].values
    all_hba1_true = df_sorted['HBA1_beta'].values
    
    total_batches = math.ceil(total_genes / batch_size)
    
    sig_batch_obs = None
    sig_batch_imp = None

    history_obs = []
    history_imp = []

    for batch_idx in range(1, total_batches + 1):
        n_revealed = min(batch_idx * batch_size, total_genes)
        
        # --- Slice Data (Simulate "Revealing" the batch) ---
        # The genes 0..n_revealed are "Known".
        # The genes n_revealed..end are "Unknown".
        
        lof_known = all_lof_true[:n_revealed]
        hba1_known = all_hba1_true[:n_revealed]
        
        # ==========================================
        # Metric 1: Observed Correlation
        # ==========================================
        # Correlation computed ONLY on the subset we have sampled.
        # We use the real LoF values here just for the calculation.
        if n_revealed >= 2:
            # Handle constant input case to avoid warnings
            if np.std(lof_known) == 0 or np.std(hba1_known) == 0:
                corr_obs, p_obs = 0.0, 1.0
            else:
                corr_obs, p_obs = stats.pearsonr(lof_known, hba1_known)
        else:
            corr_obs, p_obs = 0.0, 1.0
            
        if sig_batch_obs is None and p_obs < p_threshold:
            sig_batch_obs = batch_idx

        history_obs.append(p_obs)

        # ==========================================
        # Metric 2: Imputed Correlation (Average Known)
        # ==========================================
        # LOF: We use the FULL REAL LoF vector (no imputation, we just use the truth).
        # HBA1: We use the HYBRID vector (Real for knowns, Mean of knowns for unknowns).
        
        n_missing = total_genes - n_revealed
        
        if n_missing > 0:
            # Impute HBA1 only
            if len(hba1_known) > 0:
                mean_hba1 = np.mean(hba1_known)
            else:
                mean_hba1 = 0.0 # Fallback if nothing revealed yet
                
            hba1_imputed_tail = np.full(n_missing, mean_hba1)
            hba1_full_hybrid = np.concatenate([hba1_known, hba1_imputed_tail])
        else:
            hba1_full_hybrid = hba1_known
            
        # Calculate correlation: Full Real LoF vs Full Hybrid HBA1
        if np.std(all_lof_true) == 0 or np.std(hba1_full_hybrid) == 0:
             corr_imp, p_imp = 0.0, 1.0
        else:
            corr_imp, p_imp = stats.pearsonr(all_lof_true, hba1_full_hybrid)

        if sig_batch_imp is None and p_imp < p_threshold:
            sig_batch_imp = batch_idx

        history_imp.append(p_imp)

        # Print row (Every N, plus first and last)
        if batch_idx == 1 or batch_idx % print_every == 0 or batch_idx == total_batches:
            print(f"{batch_idx:<8} | {n_revealed:<8} | {corr_obs:+.4f}     | {p_obs:.2e}    | {corr_imp:+.4f}     | {p_imp:.2e}")

    return sig_batch_obs, sig_batch_imp, history_obs, history_imp


def plot_pvalue_history(p_values, method_name, output_dir):
    """
    Plots the -log10(p-value) over batches for a single approach.
    """
    if not p_values:
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batches = range(1, len(p_values) + 1)
    
    # Cap p-values at 1e-20
    min_p = 1e-20
    capped_p_values = [max(p, min_p) for p in p_values]
    
    # Convert to -log10
    # Handle case where p might be 0 (though max(p, 1e-20) handles that)
    nlog10_p = [-np.log10(p) for p in capped_p_values]
    
    # Reference values (also capped/converted)
    final_p = capped_p_values[-1]
    final_nlog10 = -np.log10(final_p)
    
    thresh_p = 0.05
    thresh_nlog10 = -np.log10(thresh_p)
    
    plt.figure(figsize=(10, 6))
    plt.plot(batches, nlog10_p, label='-log10(p-value)', linewidth=2)
    
    # Horizontal lines
    plt.axhline(y=thresh_nlog10, color='r', linestyle='--', alpha=0.7, label=f'Marginal Sig (0.05)')
    plt.axhline(y=final_nlog10, color='g', linestyle='--', alpha=0.7, label=f'Final P-value')
    
    plt.title(f"Significance Trajectory: {method_name}")
    plt.xlabel("Batches")
    plt.ylabel("-log10(p-value)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
    out_path = os.path.join(output_dir, f"{safe_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")


def main():
    args = parse_arguments()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found at {args.input_file}")
        return
        
    print(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file, sep='\t')
    
    # Preprocessing: Drop NAs
    # We remove rows that don't have ground truth, as we can't simulate checking them.
    df = df.dropna(subset=['LoF_gamma', 'HBA1_beta'])
    total_genes = len(df)
    
    print(f"Valid genes for simulation: {total_genes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Significance Threshold: {args.p_threshold}")
    
    if total_genes < 3:
        print("Not enough genes to run simulation.")
        return

    # ---------------------------------------------------------
    # Strategy 1: GammaMagnitude Sampling (LoF Known)
    # ---------------------------------------------------------
    # Sort by absolute LoF_gamma
    df_mag = df.copy()
    df_mag['abs_lof'] = df_mag['LoF_gamma'].abs()
    df_mag = df_mag.sort_values(by='abs_lof', ascending=False)
    
    mag_obs, mag_imp, mag_hist_obs, mag_hist_imp = run_simulation_strategy(
        "GammaMagnitude Sorting (LoF Known)", df_mag, total_genes, args.batch_size, args.p_threshold, args.print_every
    )

    # Plot GammaMagnitude results
    plot_pvalue_history(mag_hist_obs, "GammaMagnitude_ObservedGenes", args.output_dir)
    plot_pvalue_history(mag_hist_imp, "GammaMagnitude_ImputedGenes", args.output_dir)

    # ---------------------------------------------------------
    # Strategy 2: Random Sampling (LoF Unknown)
    # ---------------------------------------------------------
    # Shuffle randomly
    df_rnd = df.copy()
    df_rnd = df_rnd.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    rnd_obs, rnd_imp, rnd_hist_obs, rnd_hist_imp = run_simulation_strategy(
        "Random Sampling (LoF Unknown)", df_rnd, total_genes, args.batch_size, args.p_threshold, args.print_every
    )

    # Plot Random results
    plot_pvalue_history(rnd_hist_obs, "Random_ObservedGenes", args.output_dir)
    plot_pvalue_history(rnd_hist_imp, "Random_ImputedGenes", args.output_dir)

    # ---------------------------------------------------------
    # Strategy 3: Control Covariance (Optional)
    # ---------------------------------------------------------
    cov_obs, cov_imp = None, None

    if args.control_h5ad:
        # 1. Compute Covariance
        # We assume the target gene name is "HBA1" based on the TSV context (HBA1_beta)
        cov_series = compute_control_covariance(
            args.control_h5ad, "HBA1", args.target_label, args.control_label
        )

        if cov_series is not None:
            # 2. Merge Covariance into DF
            df_cov = df.copy()
            # Map covariance values to genes. Fill missing with 0.
            df_cov['covariance'] = df_cov['gene_name'].map(cov_series).fillna(0.0)

        if cov_series is not None:
            # 2. Merge Covariance into DF
            df_cov = df.copy()
            # Map covariance values to genes. Fill missing with 0.
            df_cov['covariance'] = df_cov['gene_name'].map(cov_series).fillna(0.0)

            # 3. Sort by Absolute Covariance (Descending)
            # Active learning assumption: genes strongly correlated (pos or neg) are most informative.
            df_cov['abs_cov'] = df_cov['covariance'].abs()
            df_cov = df_cov.sort_values(by='abs_cov', ascending=False)

            cov_obs, cov_imp, cov_hist_obs, cov_hist_imp = run_simulation_strategy(
                "Control Covariance Sorting", df_cov, total_genes, args.batch_size, args.p_threshold, args.print_every
            )

            plot_pvalue_history(cov_hist_obs, "ControlCovariance_ObservedGenes", args.output_dir)
            plot_pvalue_history(cov_hist_imp, "ControlCovariance_ImputedGenes", args.output_dir)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("FINAL SUMMARY: Batches needed for Significance")
    print("="*50)
    print(f"{'Strategy':<30} | {'ObservedGenes':<18} | {'ImputedGenes':<18}")
    print("-" * 72)

    def fmt(val): return str(val) if val else "> Max Batches"

    print(f"{'GammaMagnitude Sorting':<30} | {fmt(mag_obs):<18} | {fmt(mag_imp):<18}")
    print(f"{'Random Sampling':<30} | {fmt(rnd_obs):<18} | {fmt(rnd_imp):<18}")
    if args.control_h5ad and cov_obs is not None:
        print(f"{'Control Covariance Sorting':<30} | {fmt(cov_obs):<18} | {fmt(cov_imp):<18}")
    print("-" * 72)

if __name__ == "__main__":
    main()