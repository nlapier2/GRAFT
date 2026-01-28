import pandas as pd
import numpy as np
from scipy import stats
import math
import os
import argparse
import matplotlib.pyplot as plt
import anndata as ad
import scipy.sparse as sp

from active_gp import ActiveGPLearner


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

    # New Argument for External Perturbation Strategy
    parser.add_argument(
        "--external_h5ad",
        type=str,
        default=None,
        help="Path to 'external' pseudobulked H5AD for perturbation effect calculation."
    )

    parser.add_argument(
        "--center_data",
        action="store_true",
        help="Center LoF_gamma and HBA1_beta at 0 by subtracting their means."
    )

    # --- GP Imputation Arguments ---
    parser.add_argument("--external_list", type=str, default="", help="List of external h5ad files for GP kernel.")
    parser.add_argument("--embeddings_yaml", type=str, default="", help="Single YAML file defining pathway/embedding sources.")
    parser.add_argument("--kernel_agg", type=str, default="mean", choices=["mean", "wmean"], help="GP Kernel aggregation method.")
    parser.add_argument("--gp_noise_var", type=float, default=0.01, help="GP noise variance (lambda).")
    parser.add_argument("--gp_recompute_freq", type=int, default=5, help="How often to re-weight GP kernels (batches).")

    return parser.parse_args()


def run_simulation_strategy(name, df_sorted, total_genes, batch_size, p_threshold, print_every=10, gp_learner=None):
    """
    Runs the batch simulation for a specific ordering of genes (df_sorted).
    If gp_learner is provided, uses GP prediction for the 'Imputed' metric instead of Mean Imputation.
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
    history_mse = []

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
        # Metric 2: Imputed Correlation (Mean OR GP)
        # ==========================================
        # LOF: We use the FULL REAL LoF vector (no imputation, we just use the truth).
        # HBA1: We use the HYBRID vector (Real for knowns, Mean of knowns for unknowns).
        
        n_missing = total_genes - n_revealed
        
        if gp_learner is not None:
            # --- GP IMPUTATION ---
            # 1. Update weights periodically
            if batch_idx % gp_learner.args.gp_recompute_freq == 0:
                # Create mask for CURRENTLY known genes in the ORIGINAL order
                # We need to map the sorted df back to the original indices the GP knows
                # BUT ActiveGPLearner was init with df['gene_name'].values from the MAIN df.
                # So we just pass the names of currently revealed genes.
                pass # Optimization: ActiveGP.update takes indices.
                
            # Create boolean mask for the learner
            # The learner stores genes in the original order. We need to match.
            # Get names of revealed genes
            rev_names = df_sorted['gene_name'].iloc[:n_revealed].values
            
            # Create mask aligned to learner.genes
            # Use np.isin for speed
            mask = np.isin(gp_learner.genes, rev_names)
            
            # Get observed values aligned to that mask
            # We need to ensure Y is aligned to learner.genes[mask]
            # Create a lookup series
            y_series = pd.Series(hba1_known, index=rev_names)
            y_aligned_obs = y_series.reindex(gp_learner.genes[mask]).values
            
            # Update weights (optional frequency check inside update or here)
            if batch_idx % gp_learner.args.gp_recompute_freq == 0:
                gp_learner.update(mask, y_aligned_obs)
                
            # Predict
            hba1_full_hybrid = gp_learner.predict(mask, y_aligned_obs)
            
            # Note: hba1_full_hybrid is aligned to gp_learner.genes (Original Order)
            # We need to compare it to all_lof_true (Sorted Order)
            # So we map it back
            pred_series = pd.Series(hba1_full_hybrid, index=gp_learner.genes)
            hba1_full_hybrid = pred_series.reindex(df_sorted['gene_name']).values
            
        else:
            # --- MEAN IMPUTATION (Baseline) ---
            if n_missing > 0:
                if len(hba1_known) > 0:
                    mean_hba1 = np.mean(hba1_known)
                else:
                    mean_hba1 = 0.0 
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

        # ==========================================
        # Metric 3: Mean Squared Error (Full Vector)
        # ==========================================
        # MSE between Truth and Hybrid (Imputed) Vector
        mse = np.mean((all_hba1_true - hba1_full_hybrid) ** 2)
        history_mse.append(mse)

        # Print row (Every N, plus first and last)
        if batch_idx == 1 or batch_idx % print_every == 0 or batch_idx == total_batches:
            print(f"{batch_idx:<8} | {n_revealed:<8} | {corr_obs:+.4f}     | {p_obs:.2e}    | {corr_imp:+.4f}     | {p_imp:.2e}")

    return sig_batch_obs, sig_batch_imp, history_obs, history_imp, history_mse


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


def plot_mse_comparison(mse_histories, output_dir):
    """
    Plots MSE trajectories. Automatically splits the y-axis (broken axis)
    if one method's initial error is significantly (>5x) higher than the median max error.
    """
    if not mse_histories:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Analyze Data to decide on Broken Axis
    # Get the maximum MSE for each strategy
    max_values = [max(hist) for hist in mse_histories.values() if hist]
    if not max_values: 
        return
        
    global_max = max(max_values)
    median_max = np.median(max_values)
    
    # Threshold: If the worst method is >5x higher than the median method, break the axis.
    use_broken_axis = global_max > (5.0 * median_max)

    if use_broken_axis:
        # --- BROKEN AXIS PLOT ---
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.1)  # adjust space between axes

        # Plot data on both axes
        for name, history in mse_histories.items():
            batches = range(1, len(history) + 1)
            ax1.plot(batches, history, label=name, linewidth=2, alpha=0.8)
            ax2.plot(batches, history, label=name, linewidth=2, alpha=0.8)

        # zoom-in / limit the view to different portions of the data
        # Ax1 (Top): Shows the outliers. Y-lim from (median_max*2) to (global_max * 1.05)
        ax1.set_ylim(median_max * 1.5, global_max * 1.05)
        
        # Ax2 (Bottom): Shows the details. Y-lim from 0 to (median_max * 1.2)
        ax2.set_ylim(0, median_max * 1.2)

        # Hide the spines between ax and ax2
        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        # Add diagonal lines to indicate the break
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        ax1.set_title("Imputation Error Trajectory (MSE) - Split Axis")
        ax2.set_ylabel("Mean Squared Error")
        ax2.set_xlabel("Batches")
        
        # Legend only on top to avoid clutter
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
    else:
        # --- STANDARD PLOT ---
        plt.figure(figsize=(10, 6))
        for name, history in mse_histories.items():
            batches = range(1, len(history) + 1)
            plt.plot(batches, history, label=name, linewidth=2, alpha=0.8)
        
        plt.title("Imputation Error Trajectory (MSE)")
        plt.xlabel("Batches")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "MSE_Comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved MSE comparison plot to {out_path}")


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


def compute_external_perturbation_effect(h5ad_path, target_gene, obs_label, control_val):
    """
    Loads an external H5AD (pseudobulk or single-cell), finds the control population,
    and calculates the absolute difference in `target_gene` expression between 
    each perturbation and the control.
    Returns: pd.Series mapping perturbation_name -> absolute_effect_size
    """
    if not os.path.exists(h5ad_path):
        print(f"Warning: External H5AD not found at {h5ad_path}")
        return None

    print(f"Loading external perturbation data from {h5ad_path}...")
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"Error loading H5AD: {e}")
        return None

    # 1. Verify Columns
    if obs_label not in adata.obs.columns:
        print(f"Error: Target label column '{obs_label}' not found in external H5AD.")
        return None
    
    if target_gene not in adata.var_names:
        print(f"Error: Target gene '{target_gene}' not found in external H5AD.")
        return None

    # 2. Extract Data for Target Gene
    # We only need the column corresponding to HBA1
    gene_idx = adata.var_names.get_loc(target_gene)
    X_vec = adata.X[:, gene_idx]
    
    # Densify if sparse
    if sp.issparse(X_vec):
        X_vec = X_vec.toarray().flatten()
    else:
        X_vec = np.asarray(X_vec).flatten()

    # 3. Identify Control Mean
    # We use case-insensitive matching for robustness, or exact if preferred.
    # Given the previous instruction, we'll try exact first, then case-insensitive.
    obs_vals = adata.obs[obs_label].astype(str)
    
    is_ctrl = obs_vals == str(control_val)
    if not is_ctrl.any():
        # Try case-insensitive
        is_ctrl = obs_vals.str.lower() == str(control_val).lower()
    
    if not is_ctrl.any():
        print(f"Error: No control cells found with label '{control_val}' in column '{obs_label}'.")
        return None
    
    ctrl_mean = np.mean(X_vec[is_ctrl])
    print(f"External Dataset: Found {is_ctrl.sum()} control observations. Mean {target_gene}: {ctrl_mean:.4f}")

    # 4. Compute Means per Perturbation
    # We group by the perturbation label
    # Create a DataFrame for easy groupby
    df_temp = pd.DataFrame({
        'pert': obs_vals,
        'expr': X_vec
    })
    
    # Filter out controls from the perturbation list (optional, but keeps the series clean)
    df_pert = df_temp[~is_ctrl]
    
    # Group by perturbation and calculate mean
    pert_means = df_pert.groupby('pert')['expr'].mean()
    
    # 5. Calculate Absolute Delta
    abs_deltas = (pert_means - ctrl_mean).abs()
    
    return abs_deltas


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

    # Optional Centering
    if args.center_data:
        print("Centering LoF_gamma and HBA1_beta at 0...")
        df['LoF_gamma'] = df['LoF_gamma'] - df['LoF_gamma'].mean()
        df['HBA1_beta'] = df['HBA1_beta'] - df['HBA1_beta'].mean()

    print(f"Valid genes for simulation: {total_genes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Significance Threshold: {args.p_threshold}")
    
    if total_genes < 3:
        print("Not enough genes to run simulation.")
        return

    # Dictionary to store MSE histories for comparison plot
    all_mse_histories = {}

    # ---------------------------------------------------------
    # Strategy 1: GammaMagnitude Sampling (LoF Known)
    # ---------------------------------------------------------
    # Sort by absolute LoF_gamma
    df_mag = df.copy()
    df_mag['abs_lof'] = df_mag['LoF_gamma'].abs()
    df_mag = df_mag.sort_values(by='abs_lof', ascending=False)
    
    mag_obs, mag_imp, mag_hist_obs, mag_hist_imp, mag_mse = run_simulation_strategy(
        "GammaMagnitude Sorting (LoF Known)", df_mag, total_genes, args.batch_size, args.p_threshold, args.print_every
    )
    all_mse_histories["GammaMagnitude"] = mag_mse

    # Plot GammaMagnitude results
    plot_pvalue_history(mag_hist_obs, "GammaMagnitude_ObservedGenes", args.output_dir)
    plot_pvalue_history(mag_hist_imp, "GammaMagnitude_ImputedGenes", args.output_dir)

    # ---------------------------------------------------------
    # Strategy 2: Random Sampling (LoF Unknown)
    # ---------------------------------------------------------
    # Shuffle randomly
    df_rnd = df.copy()
    df_rnd = df_rnd.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    rnd_obs, rnd_imp, rnd_hist_obs, rnd_hist_imp, rnd_mse = run_simulation_strategy(
        "Random Sampling (LoF Unknown)", df_rnd, total_genes, args.batch_size, args.p_threshold, args.print_every
    )
    all_mse_histories["Random"] = rnd_mse

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

            cov_obs, cov_imp, cov_hist_obs, cov_hist_imp, cov_mse = run_simulation_strategy(
                "Control Covariance Sorting", df_cov, total_genes, args.batch_size, args.p_threshold, args.print_every
            )
            all_mse_histories["ControlCovariance"] = cov_mse

            plot_pvalue_history(cov_hist_obs, "ControlCovariance_ObservedGenes", args.output_dir)
            plot_pvalue_history(cov_hist_imp, "ControlCovariance_ImputedGenes", args.output_dir)


    # ---------------------------------------------------------
    # Strategy 4: External Perturbation Effect (Optional)
    # ---------------------------------------------------------
    ext_obs, ext_imp = None, None

    if args.external_h5ad:
        # 1. Compute Deltas in External Dataset
        ext_series = compute_external_perturbation_effect(
            args.external_h5ad, "HBA1", args.target_label, args.control_label
        )

        if ext_series is not None:
            # 2. Merge into DF
            df_ext = df.copy()
            # Map effects. Fill missing with 0 (assumption: unmeasured = no info = low priority)
            df_ext['ext_effect'] = df_ext['gene_name'].map(ext_series).fillna(0.0)

            # 3. Sort by Absolute Effect (Descending)
            df_ext = df_ext.sort_values(by='ext_effect', ascending=False)

            ext_obs, ext_imp, ext_hist_obs, ext_hist_imp, ext_mse = run_simulation_strategy(
                "External Perturbation Sorting", df_ext, total_genes, args.batch_size, args.p_threshold, args.print_every
            )
            all_mse_histories["ExternalPerturbation"] = ext_mse

            plot_pvalue_history(ext_hist_obs, "ExternalPerturbation_ObservedGenes", args.output_dir)
            plot_pvalue_history(ext_hist_imp, "ExternalPerturbation_ImputedGenes", args.output_dir)

    # ---------------------------------------------------------
    # Final Comparative Plots
    # ---------------------------------------------------------
    if all_mse_histories:
        plot_mse_comparison(all_mse_histories, args.output_dir)

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
    if args.external_h5ad and ext_obs is not None:
        print(f"{'External Perturbation Sorting':<30} | {fmt(ext_obs):<18} | {fmt(ext_imp):<18}")
    print("-" * 72)

if __name__ == "__main__":
    main()