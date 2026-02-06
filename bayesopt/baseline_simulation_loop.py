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
from active_strategies import StaticStrategy, StaticGPStrategy, HighLeverageStrategy, UncertaintyStrategy, DiversityStrategy, PCUncertaintyStrategy, VarianceReductionStrategy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Active Learning Baselines: Random vs. Magnitude Sampling")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the TSV file containing gene_name, LoF_gamma, and HBA1_beta.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of genes to reveal in each batch (default: 100).")
    parser.add_argument("--print_every", type=int, default=10, help="Print progress every N batches (default: 10).")
    parser.add_argument("--p_threshold", type=float, default=0.05, help="P-value threshold for statistical significance (default: 0.05).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--center_data", action="store_true", help="Center LoF_gamma and HBA1_beta at 0 by subtracting their means.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the correlation plots.")
    parser.add_argument("--plot_target", type=str, default="HBA1", help="The specific target gene to generate plots and detailed logs for.")

    # Arguments for Control Strategy
    parser.add_argument("--control_h5ad", type=str, default=None, help="Path to H5AD file containing control cells for covariance calculation.")
    parser.add_argument("--target_label", type=str, default="target_gene", help="Obs column name identifying perturbation/control status (default: target_gene).")
    parser.add_argument("--control_label", type=str, default="", help="Value in target_label that identifies control cells. If blank, uses all cells (default: '').")

    # Argument for External Perturbation Strategy
    parser.add_argument("--external_h5ad", type=str, default=None, help="Path to 'external' pseudobulked H5AD for perturbation effect calculation.")

    # --- GP Imputation Arguments ---
    parser.add_argument("--external_list", type=str, default="", help="List of external h5ad files for GP kernel.")
    parser.add_argument("--embeddings_yaml", type=str, default="", help="Single YAML file defining pathway/embedding sources.")
    parser.add_argument("--emb_metric", type=str, default="cosine", help="Metric for embedding kernels (cosine/rbf).")
    parser.add_argument("--kernel_weight_gamma", type=float, default=1.0, help="Gamma parameter for kernel alignment weights.")
    parser.add_argument("--kernel_agg", type=str, default="mean", choices=["mean", "wmean"], help="GP Kernel aggregation method.")
    parser.add_argument("--gp_noise_var", type=float, default=0.01, help="GP noise variance (lambda).")
    parser.add_argument("--gp_recompute_freq", type=int, default=5, help="How often to re-weight GP kernels (batches).")
    
    # --- Active Learning Arguments ---
    parser.add_argument("--run_active_leverage", action="store_true", help="Run the Active High Leverage strategy.")
    parser.add_argument("--run_active_uncertainty", action="store_true", help="Run the Active Uncertainty strategy.")
    parser.add_argument("--run_active_diversity", action="store_true", help="Run the Active Diversity strategy.")
    parser.add_argument("--run_active_pca", action="store_true", help="Run the Active PC-Uncertainty strategy.")
    parser.add_argument("--run_active_var_reduction", action="store_true", help="Run the Active Stepwise Variance Reduction (Kriging Believer) strategy.")

    parser.add_argument("--acq_beta", type=float, default=1.0, help="Beta parameter for acquisition (mean vs std trade-off).")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to run (optional limit).")
    parser.add_argument("--pca_recompute_freq", type=int, default=1, help="Batch frequency to recompute PCA for PC-Uncertainty.")
    parser.add_argument("--pca_top_k", type=int, default=50, help="Number of Principal Components to use for uncertainty.")
    parser.add_argument("--stepwise_subset_size", type=int, default=400, help="Size of the 'Working Set' for stepwise variance reduction (speed optimization).")

    # --- Static Strategy Options ---
    parser.add_argument("--imputation_method", type=str, default="mean", choices=["mean", "zero"], help="Imputation method for static strategies: 'mean' (AverageKnown) or 'zero'.")
    parser.add_argument("--sampling_strategy", type=str, default="strongest", choices=["strongest", "uniform"], help="Order to pick genes for static strategies: 'strongest' (Magnitude Descending) or 'uniform' (Stratified across range).")
    parser.add_argument("--random_samp_pct", type=float, default=0.0, help="Percentage of batch (0.0-1.0) to select randomly for static strategies.")
    parser.add_argument("--static_only", action="store_true", help="Skip all active learning strategies (GP/Active).")

    return parser.parse_args()


def load_adata_once(h5ad_path, obs_label=None, control_val=None):
    """
    Loads an H5AD file once and optionally filters it for control cells.
    Returns the AnnData object or None if loading fails.
    """
    if not h5ad_path or not os.path.exists(h5ad_path):
        return None

    print(f"Loading H5AD: {h5ad_path} ...")
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"Error loading H5AD: {e}")
        return None

    # Filter only if specific control criteria are provided (e.g. for Control Covariance)
    # For External data, we usually keep everything (controls + perturbations) so we skip this block if control_val is None
    if obs_label and control_val is not None and str(control_val).strip() != "":
        if obs_label in adata.obs.columns:
            c_val_lower = str(control_val).lower()
            mask = adata.obs[obs_label].astype(str).str.lower() == c_val_lower
            adata = adata[mask].copy()
            print(f"  -> Filtered to {adata.n_obs} cells (label '{obs_label}' ~= '{control_val}').")
        else:
            print(f"  -> Warning: '{obs_label}' not found. Using all cells.")

    return adata


def print_final_summary(results_list):
    """
    Prints the final summary tables for Correlated vs Uncorrelated genes.
    Expects a list of dictionaries containing keys:
    ['strategy', 'true_p', 'success_obs', 'error_nlog10_obs', 'final_mse', ...]
    """
    if not results_list:
        print("\nNo results to summarize.")
        return

    df_res = pd.DataFrame(results_list)

    # Split into Correlated (True P < 0.05) vs Null (True P >= 0.05)
    df_corr = df_res[df_res['true_p'] < 0.05]
    df_null = df_res[df_res['true_p'] >= 0.05]

    def _print_group(name, sub_df):
        print(f"\n\n>>> SUMMARY: {name} Genes (Count: {len(sub_df['gene'].unique())})")
        if sub_df.empty:
            print("No genes in this category.")
            return

        # Group by Strategy
        grp = sub_df.groupby('strategy')
        
        # Headers (Added MSE column at the end)
        header = (f"{'Strategy':<35} | {'Succ% (Obs)':<11} | {'Bias (Obs)':<10} | "
                  f"{'MAE (Obs)':<10} | {'Succ% (Imp)':<11} | {'MSE (Mean +/- SE)':<20}")
        print(header)
        print("-" * len(header))
        
        for strat, g in grp:
            # Stats: Success Rate, Bias (Mean Error), MAE (Mean Absolute Error)
            succ_obs = (g['success_obs'].sum() / len(g)) * 100
            bias_obs = g['error_nlog10_obs'].mean()
            mae_obs = g['abs_error_nlog10_obs'].mean()
            
            succ_imp = (g['success_imp'].sum() / len(g)) * 100
            
            # MSE Stats (Mean and Standard Error)
            mse_mean = g['final_mse'].mean()
            mse_se = g['final_mse'].sem()
            
            # Format string
            row = (f"{strat:<35} | {succ_obs:9.1f}%  | {bias_obs:9.2f}  | "
                   f"{mae_obs:9.2f}  | {succ_imp:9.1f}%  | "
                   f"{mse_mean:.2e} +/- {mse_se:.2e}")
            print(row)

    _print_group("TRUE CORRELATED", df_corr)
    _print_group("UNCORRELATED (NULL)", df_null)
    print("\n" + "="*60)


def run_simulation_strategy(strategy, df_master, total_genes, batch_size, p_threshold, 
                          target_gene_name, y_true_values, ground_truth_p,
                          max_batches=None, print_every=10, silent=False):
    """
    Runs the simulation for a specific Strategy on a specific Target Gene.
    Returns: A dictionary of summary statistics.
    """
    if not silent:
        print(f"\nRunning Strategy: {strategy.name}")
        print(f"{'Batch':<8} | {'Revealed':<8} | {'Corr (Obs)':<12} | {'P (Obs)':<10} | {'Corr (Imp)':<12} | {'P (Imp)':<10}")
        print("-" * 75)

    all_lof_true = df_master['LoF_gamma'].values
    # y_true_values is passed explicitly (the beta column for this target)
    
    revealed_mask = np.zeros(total_genes, dtype=bool)
    n_revealed = 0
    
    sig_batch_obs = None
    sig_batch_imp = None
    
    final_p_obs = 1.0
    final_p_imp = 1.0

    history_obs = []
    history_imp = []
    history_mse = []
    
    # Diagnostics storage
    diag_lof = []
    diag_hba1 = []
    diag_conn = []
    
    batch_idx = 0
    
    while n_revealed < total_genes:
        # Check max batches
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Reached max_batches ({max_batches}). Stopping.")
            break

        batch_idx += 1
        
        # 1. Select
        new_indices = strategy.select_next_batch(batch_size, revealed_mask, y_true_values[revealed_mask])
        if len(new_indices) == 0:
            break
            
        # 2. Reveal
        revealed_mask[new_indices] = True
        n_revealed = np.sum(revealed_mask)
        
        lof_known = all_lof_true[revealed_mask]
        y_known = y_true_values[revealed_mask]
        
        # 3. Update
        strategy.update(new_indices, y_true_values[new_indices])

        # ==========================================
        # Diagnostic: Analyze Selected Batch Quality
        # ==========================================
        # 1. Magnitude of Effects
        batch_lof_abs = np.mean(np.abs(all_lof_true[new_indices]))
        batch_hba1_abs = np.mean(np.abs(y_true_values[new_indices]))
        
        diag_lof.append(batch_lof_abs)
        diag_hba1.append(batch_hba1_abs)
        
        # 2. Connectivity (Are we picking central hubs or outliers?)
        # We check if the strategy has a 'learner' with a fused kernel
        learner = getattr(strategy, 'learner', None)
        if learner is not None and getattr(learner, 'K_fused', None) is not None:
            # Get the rows of the kernel for the selected genes
            # Calculate mean absolute similarity to ALL other genes (Connectivity)
            # K_fused is (N, N). We take rows [new_indices]. 
            K_sub = learner.K_fused[new_indices, :]
            batch_conn = np.mean(np.abs(K_sub)) # Average correlation to the universe
        else:
            batch_conn = np.nan
        diag_conn.append(batch_conn)

        # Metric 1: Observed
        if n_revealed >= 2 and np.std(lof_known) > 0 and np.std(y_known) > 0:
            corr_obs, p_obs = stats.pearsonr(lof_known, y_known)
        else:
            corr_obs, p_obs = 0.0, 1.0
            
        if sig_batch_obs is None and p_obs < p_threshold:
            sig_batch_obs = batch_idx
        
        final_p_obs = p_obs
        history_obs.append(p_obs)

        # Metric 2: Imputed
        prediction = strategy.predict(revealed_mask, y_known)
        
        if prediction is not None:
            y_full_hybrid = prediction
        else:
            # Fallback
            imp_method = getattr(strategy.args, 'imputation_method', 'mean')
            if imp_method == 'zero':
                mean_val = 0.0
            else:
                mean_val = np.mean(y_known) if len(y_known) > 0 else 0.0
            
            y_full_hybrid = np.copy(y_true_values)
            y_full_hybrid[~revealed_mask] = mean_val
            
        if np.std(all_lof_true) > 0 and np.std(y_full_hybrid) > 0:
            corr_imp, p_imp = stats.pearsonr(all_lof_true, y_full_hybrid)
        else:
            corr_imp, p_imp = 0.0, 1.0

        if sig_batch_imp is None and p_imp < p_threshold:
            sig_batch_imp = batch_idx
            
        final_p_imp = p_imp
        history_imp.append(p_imp)

        # Metric 3: MSE
        mse = np.mean((y_true_values - y_full_hybrid) ** 2)
        history_mse.append(mse)

        if not silent:
            if batch_idx == 1 or batch_idx % print_every == 0 or n_revealed == total_genes:
                print(f"{batch_idx:<8} | {n_revealed:<8} | {corr_obs:+.4f}     | {p_obs:.2e}  | {corr_imp:+.4f}     | {p_imp:.2e}")

# Aggregate Diagnostics
    avg_diag = {
        'lof': np.mean(diag_lof) if diag_lof else 0.0,
        'hba1': np.mean(diag_hba1) if diag_hba1 else 0.0,
        'conn': np.nanmean(diag_conn) if not np.all(np.isnan(diag_conn)) else 0.0
    }

    # --- Compile Stats ---
    # Log10 P-value stats
    nlog10_true = -np.log10(ground_truth_p) if ground_truth_p > 1e-300 else 300
    nlog10_pred_obs = -np.log10(final_p_obs) if final_p_obs > 1e-300 else 300
    nlog10_pred_imp = -np.log10(final_p_imp) if final_p_imp > 1e-300 else 300
    
    # Determine the limit for reporting failure (if max_batches is None, use actual batches run)
    limit = max_batches if max_batches is not None else batch_idx

    return {
        "gene": target_gene_name,
        "strategy": strategy.name,
        "true_p": ground_truth_p,
        "final_p_obs": final_p_obs,
        "final_p_imp": final_p_imp,
        "final_mse": history_mse[-1] if history_mse else 0.0,
        "batches_obs": sig_batch_obs if sig_batch_obs else (limit + 1),
        "batches_imp": sig_batch_imp if sig_batch_imp else (limit + 1),
        "success_obs": (sig_batch_obs is not None),
        "success_imp": (sig_batch_imp is not None),
        "error_nlog10_obs": (nlog10_pred_obs - nlog10_true), # Bias
        "error_nlog10_imp": (nlog10_pred_imp - nlog10_true),
        "abs_error_nlog10_obs": abs(nlog10_pred_obs - nlog10_true),
        "abs_error_nlog10_imp": abs(nlog10_pred_imp - nlog10_true),
        "history_obs": history_obs,
        "history_imp": history_imp,
        "history_mse": history_mse,
        "avg_diag": avg_diag
    }


def plot_pvalue_history(p_values, method_name, output_dir, target_name, true_p_val=None):
    """
    Plots the -log10(p-value) over batches for a single approach.
    If true_p_val is provided, it is plotted as the 'Ground Truth' reference line.
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
    nlog10_p = [-np.log10(p) for p in capped_p_values]
    
    # Determine Reference Line (Ground Truth or Final Batch)
    if true_p_val is not None:
        ref_p = max(true_p_val, min_p)
        label_text = "Ground Truth"
    else:
        ref_p = capped_p_values[-1]
        label_text = "Final Batch"

    ref_nlog10 = -np.log10(ref_p)
    
    thresh_p = 0.05
    thresh_nlog10 = -np.log10(thresh_p)
    
    plt.figure(figsize=(10, 6))
    plt.plot(batches, nlog10_p, label='-log10(p-value)', linewidth=2)
    
    # Horizontal lines
    plt.axhline(y=thresh_nlog10, color='r', linestyle='--', alpha=0.7, label=f'Marginal Sig (0.05)')
    plt.axhline(y=ref_nlog10, color='g', linestyle='--', alpha=0.7, label=f'{label_text} P-value')
    
    # Update Title to include Target Name
    plt.title(f"{target_name}: Significance Trajectory - {method_name}")
    plt.xlabel("Batches")
    plt.ylabel("-log10(p-value)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save with Target Name in filename to avoid overwrites
    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
    safe_target = target_name.replace("/", "_")
    out_path = os.path.join(output_dir, f"{safe_target}_{safe_name}.png")
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


def compute_control_covariance(adata, target_gene):
    """
    Calculates the covariance between every gene and the 'target_gene'
    using the provided control AnnData object.
    Returns a pandas Series mapping gene_name -> absolute_covariance.
    """
    if adata is None:
        return None
        
    if adata.n_obs < 5:
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


def compute_external_perturbation_effect(adata, target_gene, obs_label, control_val):
    """
    Calculates the absolute difference in `target_gene` expression between 
    each perturbation and the control using the provided AnnData.
    Returns: pd.Series mapping perturbation_name -> absolute_effect_size
    """
    if adata is None:
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


def get_stratified_indices(scores, n_bins=10, seed=42):
    """
    Returns indices sorted such that traversing them samples uniformly 
    across the POPULATION PERCENTILES of 'scores'.
    
    1. Bins scores into equal-frequency intervals (pd.qcut).
       - Top Bin = Top 10% of genes (not just top 10% of value range).
    2. Shuffles indices within each bin.
    3. Interleaves the bins (Round Robin) to create the final queue.
    """
    try:
        # Use qcut for Quantile binning (Equal Frequency)
        # duplicates='drop' merges bins if many genes have identical scores (e.g. 0.0)
        bins = pd.qcut(scores, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Fallback to cut (Equal Width) if qcut fails (e.g. all scores identical)
        try:
            bins = pd.cut(scores, bins=n_bins, labels=False, duplicates='drop')
        except ValueError:
            return scores.index.values
    
    # Determine number of actual bins created (might be < n_bins due to duplicates)
    if hasattr(bins, 'categories'):
        n_actual = len(bins.categories)
    else:
        # If labels=False, bins are integers. max() + 1 gives count.
        n_actual = int(bins.max()) + 1
    
    # Group indices by bin
    bin_indices = [[] for _ in range(n_actual)]
    rng = np.random.default_rng(seed)
    
    # This iteration might be slow for huge DFs, but fast for 20k.
    # A vectorized way:
    df_temp = pd.DataFrame({'score': scores, 'bin': bins})
    
    # Group and shuffle
    # We explicitly iterate 0..n_actual-1 to ensure order (lowest bin to highest bin)
    # Note: qcut assigns 0 to the lowest scores and N to highest.
    # If we want to sample uniformly, order doesn't matter much, 
    # but usually we want to cycle Low -> Med -> High -> Low...
    
    rng = np.random.default_rng(seed)
    queues = []
    max_len = 0
    
    for b in range(n_actual):
        # Extract indices belonging to this bin
        indices = df_temp[df_temp['bin'] == b].index.values
        if len(indices) > 0:
            rng.shuffle(indices)
            queues.append(list(indices))
            if len(indices) > max_len:
                max_len = len(indices)
    
    # Interleave (Round Robin)
    stratified_order = []
    for i in range(max_len):
        for q in queues:
            if i < len(q):
                stratified_order.append(q[i])
                
    return np.array(stratified_order, dtype=int)


def main():
    args = parse_arguments()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found at {args.input_file}")
        return
        
    print(f"Loading Master TSV from {args.input_file}...")
    df = pd.read_csv(args.input_file, sep='\t')
    
    # Identify Beta Columns
    beta_cols = [c for c in df.columns if c.endswith("_beta")]
    if not beta_cols:
        print("Error: No columns ending in '_beta' found in input file.")
        return
    
    print(f"Found {len(beta_cols)} target genes: {[c.replace('_beta','') for c in beta_cols]}")
    
    # Load Ancillary Data Once
    adata_ctrl = load_adata_once(args.control_h5ad, args.target_label, args.control_label)
    # Don't filter external data! We need perturbations + controls.
    adata_ext = load_adata_once(args.external_h5ad, None, None)
    
    # Initialize Shared GP Learner Once
    shared_learner = None
    has_external_data = (args.external_list or args.external_h5ad or args.embeddings_yaml)
    
    # Clean DF for Learner (Must have valid LoF for alignment)
    df = df.dropna(subset=['LoF_gamma']).reset_index(drop=True)
    total_genes = len(df)
    
    if not args.static_only and ActiveGPLearner is not None and has_external_data:
        print("\nInitializing Shared Active GP Learner (Kernel Computation)...")
        shared_learner = ActiveGPLearner(df['gene_name'].values, args)

    results_list = []
    
    # ==========================================
    # LOOP OVER TARGET GENES
    # ==========================================
    for target_col in beta_cols:
        target_name = target_col.replace("_beta", "")
        
        # Check if this is the "Plot Target"
        is_plot_target = (target_name == args.plot_target)
        silent = not is_plot_target
        
        if not silent:
            print(f"\n{'='*60}")
            print(f"PROCESSING TARGET: {target_name}")
            print(f"{'='*60}")
        else:
            print(f"... Processing {target_name} ...")

        # Extract vectors (Handle Missing Data by Filling 0 or Skipping?)
        # Simulation requires full vector. We assume clean input or fill 0.
        y_true = df[target_col].fillna(0.0).values
        
        # Optional Centering (Per Target)
        if args.center_data:
            y_true = y_true - np.mean(y_true)
        
        # Ground Truth
        lof_vals = df['LoF_gamma'].values
        if args.center_data: lof_vals = lof_vals - np.mean(lof_vals)
            
        if np.std(lof_vals) > 0 and np.std(y_true) > 0:
            _, ground_truth_p = stats.pearsonr(lof_vals, y_true)
        else:
            ground_truth_p = 1.0
            
        if not silent:
            print(f"Ground Truth P-value: {ground_truth_p:.2e}")

        # --- PREPARE STRATEGIES ---
        strategies = []
        
        # 1. GammaMagnitude
        # (Re-instantiate to be safe, though indices are static)
        mag_scores = df['LoF_gamma'].abs()
        if args.sampling_strategy == "uniform":
            mag_idxs = get_stratified_indices(mag_scores, n_bins=10, seed=args.seed)
            mag_name = "GammaMagnitude (Uniform)"
        else:
            mag_idxs = mag_scores.sort_values(ascending=False).index.values
            mag_name = "GammaMagnitude (Strongest)"
        strategies.append(StaticStrategy(total_genes, args, mag_idxs, name=mag_name))
        
        # 2. Random
        rnd_idxs = df.sample(frac=1, random_state=args.seed).index.values
        strategies.append(StaticStrategy(total_genes, args, rnd_idxs, name="Random Sampling"))
        
        # 3. Control Covariance (Target Specific)
        cov_indices = None
        if adata_ctrl:
            cov_series = compute_control_covariance(adata_ctrl, target_name)
            if cov_series is not None:
                cov_aligned = df['gene_name'].map(cov_series).fillna(0.0)
                cov_scores = cov_aligned.abs()
                if args.sampling_strategy == "uniform":
                    cov_indices = get_stratified_indices(cov_scores, n_bins=10, seed=args.seed)
                    c_name = "Control Covariance (Uniform)"
                else:
                    cov_indices = cov_scores.sort_values(ascending=False).index.values
                    c_name = "Control Covariance (Strongest)"
                strategies.append(StaticStrategy(total_genes, args, cov_indices, name=c_name))

        # 4. External Perturbation (Target Specific)
        if adata_ext:
            ext_series = compute_external_perturbation_effect(adata_ext, target_name, args.target_label, args.control_label)
            if ext_series is not None:
                ext_aligned = df['gene_name'].map(ext_series).fillna(0.0)
                ext_scores = ext_aligned.abs()
                if args.sampling_strategy == "uniform":
                    ext_idxs = get_stratified_indices(ext_scores, n_bins=10, seed=args.seed)
                    e_name = "External Perturbation (Uniform)"
                else:
                    ext_idxs = ext_scores.sort_values(ascending=False).index.values
                    e_name = "External Perturbation (Strongest)"
                strategies.append(StaticStrategy(total_genes, args, ext_idxs, name=e_name))

        # 5. GP / Active
        if shared_learner:
            shared_learner.reset()
            # Static GP
            gp_idxs = cov_indices if cov_indices is not None else rnd_idxs
            gp_name = "ControlCovariance + GP" if cov_indices is not None else "Random + GP"
            strategies.append(StaticGPStrategy(total_genes, args, gp_idxs, shared_learner, name=gp_name))
            
            # Active Strategies
            if args.run_active_leverage:
                shared_learner.reset()
                strategies.append(HighLeverageStrategy(total_genes, args, shared_learner, prior_indices=cov_indices))
            if args.run_active_uncertainty:
                shared_learner.reset()
                strategies.append(UncertaintyStrategy(total_genes, args, shared_learner, prior_indices=None))
            if args.run_active_diversity:
                shared_learner.reset()
                strategies.append(DiversityStrategy(total_genes, args, shared_learner, prior_indices=cov_indices))
            if args.run_active_pca:
                shared_learner.reset()
                strategies.append(PCUncertaintyStrategy(total_genes, args, shared_learner, prior_indices=cov_indices))
            if args.run_active_var_reduction:
                shared_learner.reset()
                strategies.append(VarianceReductionStrategy(total_genes, args, shared_learner, prior_indices=cov_indices))

        # --- EXECUTE LOOP ---
        mse_histories = {}
        target_results = [] # Store local results for the immediate summary print
        
        for strat in strategies:
            # Reset learner state
            if hasattr(strat, 'learner') and strat.learner is not None:
                strat.learner.reset()
                
            res = run_simulation_strategy(
                strat, df, total_genes, args.batch_size, args.p_threshold,
                target_name, y_true, ground_truth_p,
                max_batches=args.max_batches, print_every=args.print_every, silent=silent
            )
            results_list.append(res)
            target_results.append(res)
            
            # Plotting (Only for main target)
            if is_plot_target:
                mse_histories[strat.name] = res['history_mse']
                plot_pvalue_history(res['history_obs'], f"{strat.name}_Obs", args.output_dir, target_name, ground_truth_p)
                plot_pvalue_history(res['history_imp'], f"{strat.name}_Imp", args.output_dir, target_name, ground_truth_p)

        # --- IMMEDIATE SUMMARY (Only for main target) ---
        if is_plot_target:
            plot_mse_comparison(mse_histories, args.output_dir)
            
            print("\n" + "="*50)
            print(f"SUMMARY FOR {target_name}: Batches needed for Significance")
            print("="*50)
            print(f"{'Strategy':<40} | {'ObservedGenes':<18} | {'ImputedGenes':<18}")
            print("-" * 82)

            def fmt(val): 
                # run_simulation_strategy returns (max_batches + 1) if fail.
                # If we passed None for max_batches, it returns (total_batches + 1).
                limit = args.max_batches if args.max_batches else total_genes
                return str(val) if val <= limit else "> Max"

            for res in target_results:
                s_name = res['strategy']
                obs = res['batches_obs']
                imp = res['batches_imp']
                print(f"{s_name:<40} | {fmt(obs):<18} | {fmt(imp):<18}")
            print("-" * 82)
            
            # Print Diagnostics
            print("\n" + "="*85)
            print("DIAGNOSTIC REPORT: Average Quality of Selected Genes")
            print("="*85)
            print(f"{'Strategy':<40} | {'Avg |LoF|':<12} | {'Avg |Beta|':<12} | {'Avg Connectivity':<16}")
            print("-" * 85)
            
            for res in target_results:
                 s_name = res['strategy']
                 d = res['avg_diag']
                 conn_str = f"{d['conn']:.4f}" if d['conn'] > 0 else "N/A"
                 # Note: avg_diag contains 'hba1' key but it represents the current target's Beta
                 print(f"{s_name:<40} | {d['lof']:.4f}       | {d['hba1']:.4f}       | {conn_str:<16}")
            print("-" * 85)


    # ==========================================
    # FINAL AGGREGATE SUMMARY
    # ==========================================
    print_final_summary(results_list)

if __name__ == "__main__":
    main()
