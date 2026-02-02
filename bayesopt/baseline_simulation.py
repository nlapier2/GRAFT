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
from active_strategies import StaticStrategy, StaticGPStrategy, HighLeverageStrategy


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
    parser.add_argument("--emb_metric", type=str, default="cosine", help="Metric for embedding kernels (cosine/rbf).")
    parser.add_argument("--kernel_weight_gamma", type=float, default=1.0, help="Gamma parameter for kernel alignment weights.")
    parser.add_argument("--kernel_agg", type=str, default="mean", choices=["mean", "wmean"], help="GP Kernel aggregation method.")
    parser.add_argument("--gp_noise_var", type=float, default=0.01, help="GP noise variance (lambda).")
    parser.add_argument("--gp_recompute_freq", type=int, default=5, help="How often to re-weight GP kernels (batches).")
    
    # --- Active Learning Arguments ---
    parser.add_argument("--run_active_leverage", action="store_true", help="Run the Active High Leverage strategy.")
    parser.add_argument("--acq_beta", type=float, default=1.0, help="Beta parameter for acquisition (mean vs std trade-off).")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to run (optional limit).")

    return parser.parse_args()


def run_simulation_strategy(strategy, df_master, total_genes, batch_size, p_threshold, max_batches=None, print_every=10):
    """
    Runs the simulation using a Strategy object.
    
    Args:
        strategy: An instance of BaseStrategy (e.g., StaticStrategy).
        df_master: The canonical DataFrame (unsorted, index 0..N-1) containing Truth.
        total_genes: N.
    """
    print(f"\nRunning Strategy: {strategy.name}")
    print(f"{'Batch':<8} | {'Revealed':<8} | {'Corr (ObsGenes)':<15} | {'P (ObsGenes)':<15} | {'Corr (ImpGenes)':<15} | {'P (ImpGenes)':<15}")
    print("-" * 75)

    # Truth Vectors (aligned to 0..N indices)
    all_lof_true = df_master['LoF_gamma'].values
    all_hba1_true = df_master['HBA1_beta'].values
    
    # State
    revealed_mask = np.zeros(total_genes, dtype=bool)
    n_revealed = 0
    
    sig_batch_obs = None
    sig_batch_imp = None

    history_obs = []
    history_imp = []
    history_mse = []
    
    batch_idx = 0
    
    while n_revealed < total_genes:
        # Check max batches
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Reached max_batches ({max_batches}). Stopping.")
            break

        batch_idx += 1
        
        # 1. Ask Strategy for next batch indices
        # Pass the current mask and the KNOWN HBA1 values aligned to that mask
        new_indices = strategy.select_next_batch(batch_size, revealed_mask, all_hba1_true[revealed_mask])
        
        if len(new_indices) == 0:
            break # No more genes to select
            
        # 2. Reveal Data
        revealed_mask[new_indices] = True
        n_revealed = np.sum(revealed_mask)
        
        # Get Observed Data Subsets
        lof_known = all_lof_true[revealed_mask]
        hba1_known = all_hba1_true[revealed_mask]
        
        # 3. Notify Strategy (for Active Learning updates)
        # Passing just the NEW data logic or FULL known data logic depends on implementation.
        # Here we pass the indices and values of the NEW batch if needed, 
        # but typically the strategy might just use the FULL known set next time.
        strategy.update(new_indices, all_hba1_true[new_indices])

        # ==========================================
        # Metric 1: Observed Correlation
        # ==========================================
        if n_revealed >= 2 and np.std(lof_known) > 0 and np.std(hba1_known) > 0:
            corr_obs, p_obs = stats.pearsonr(lof_known, hba1_known)
        else:
            corr_obs, p_obs = 0.0, 1.0
            
        if sig_batch_obs is None and p_obs < p_threshold:
            sig_batch_obs = batch_idx
        history_obs.append(p_obs)

        # ==========================================
        # Metric 2: Imputed Correlation
        # ==========================================
        # Check if Strategy offers a prediction (Active GP), otherwise use Mean Imputation
        prediction = strategy.predict(revealed_mask, hba1_known)
        
        if prediction is not None:
            # Strategy provided full imputed vector
            hba1_full_hybrid = prediction
        else:
            # Fallback: Mean Imputation
            if len(hba1_known) > 0:
                mean_val = np.mean(hba1_known)
            else:
                mean_val = 0.0
            
            hba1_full_hybrid = np.copy(all_hba1_true) # Start with truth...
            hba1_full_hybrid[~revealed_mask] = mean_val # ...overwrite unknown with mean
            
        if np.std(all_lof_true) > 0 and np.std(hba1_full_hybrid) > 0:
            corr_imp, p_imp = stats.pearsonr(all_lof_true, hba1_full_hybrid)
        else:
            corr_imp, p_imp = 0.0, 1.0

        if sig_batch_imp is None and p_imp < p_threshold:
            sig_batch_imp = batch_idx
        history_imp.append(p_imp)

        # Metric 3: MSE
        mse = np.mean((all_hba1_true - hba1_full_hybrid) ** 2)
        history_mse.append(mse)

        # Print
        if batch_idx == 1 or batch_idx % print_every == 0 or n_revealed == total_genes:
            print(f"{batch_idx:<8} | {n_revealed:<8} | {corr_obs:+.4f}     | {p_obs:.2e}    | {corr_imp:+.4f}     | {p_imp:.2e}")

    return sig_batch_obs, sig_batch_imp, history_obs, history_imp, history_mse


def plot_pvalue_history(p_values, method_name, output_dir, true_p_val=None):
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
    df = df.dropna(subset=['LoF_gamma', 'HBA1_beta']).reset_index(drop=True)
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

    # --- Calculate Ground Truth P-Value ---
    # This is the target p-value if we perfectly measured the entire dataset.
    if df['LoF_gamma'].std() > 0 and df['HBA1_beta'].std() > 0:
        _, ground_truth_p = stats.pearsonr(df['LoF_gamma'], df['HBA1_beta'])
    else:
        ground_truth_p = 1.0
    print(f"Ground Truth Correlation P-value (All Genes): {ground_truth_p:.2e}")

    # Dictionary to store MSE histories for comparison plot
    all_mse_histories = {}

    # ---------------------------------------------------------
    # Strategy 1: GammaMagnitude Sampling
    # ---------------------------------------------------------
    # Sort indices by absolute LoF_gamma (descending)
    mag_indices = df['LoF_gamma'].abs().sort_values(ascending=False).index.values
    
    strat_mag = StaticStrategy(total_genes, args, mag_indices, name="GammaMagnitude Sorting")
    
    mag_obs, mag_imp, mag_hist_obs, mag_hist_imp, mag_mse = run_simulation_strategy(
        strat_mag, df, total_genes, args.batch_size, args.p_threshold, args.max_batches, args.print_every
    )
    all_mse_histories["GammaMagnitude"] = mag_mse

    plot_pvalue_history(mag_hist_obs, "GammaMagnitude_ObservedGenes", args.output_dir, ground_truth_p)
    plot_pvalue_history(mag_hist_imp, "GammaMagnitude_ImputedGenes", args.output_dir, ground_truth_p)

    # ---------------------------------------------------------
    # Strategy 2: Random Sampling
    # ---------------------------------------------------------
    # Random indices
    rnd_indices = df.sample(frac=1, random_state=args.seed).index.values
    
    strat_rnd = StaticStrategy(total_genes, args, rnd_indices, name="Random Sampling")
    
    rnd_obs, rnd_imp, rnd_hist_obs, rnd_hist_imp, rnd_mse = run_simulation_strategy(
        strat_rnd, df, total_genes, args.batch_size, args.p_threshold, args.max_batches, args.print_every
    )
    all_mse_histories["Random"] = rnd_mse

    plot_pvalue_history(rnd_hist_obs, "Random_ObservedGenes", args.output_dir, ground_truth_p)
    plot_pvalue_history(rnd_hist_imp, "Random_ImputedGenes", args.output_dir, ground_truth_p)

    # ---------------------------------------------------------
    # Strategy 3: Control Covariance (Optional)
    # ---------------------------------------------------------
    # We will store the indices for use in Strategy 5 if available
    cov_indices = None 
    
    if args.control_h5ad:
        cov_series = compute_control_covariance(
            args.control_h5ad, "HBA1", args.target_label, args.control_label
        )

        if cov_series is not None:
            # Map to DF to get aligned values
            # Fill missing with 0.0
            cov_aligned = df['gene_name'].map(cov_series).fillna(0.0)
            
            # Sort indices by absolute covariance
            cov_indices = cov_aligned.abs().sort_values(ascending=False).index.values
            
            strat_cov = StaticStrategy(total_genes, args, cov_indices, name="Control Covariance")

            cov_obs, cov_imp, cov_hist_obs, cov_hist_imp, cov_mse = run_simulation_strategy(
                strat_cov, df, total_genes, args.batch_size, args.p_threshold, args.max_batches, args.print_every
            )
            all_mse_histories["ControlCovariance"] = cov_mse

            plot_pvalue_history(cov_hist_obs, "ControlCovariance_ObservedGenes", args.output_dir, ground_truth_p)
            plot_pvalue_history(cov_hist_imp, "ControlCovariance_ImputedGenes", args.output_dir, ground_truth_p)

    # ---------------------------------------------------------
    # Strategy 4: External Perturbation Effect (Optional)
    # ---------------------------------------------------------
    if args.external_h5ad:
        ext_series = compute_external_perturbation_effect(
            args.external_h5ad, "HBA1", args.target_label, args.control_label
        )

        if ext_series is not None:
            ext_aligned = df['gene_name'].map(ext_series).fillna(0.0)
            ext_indices = ext_aligned.abs().sort_values(ascending=False).index.values
            
            strat_ext = StaticStrategy(total_genes, args, ext_indices, name="External Perturbation")

            ext_obs, ext_imp, ext_hist_obs, ext_hist_imp, ext_mse = run_simulation_strategy(
                strat_ext, df, total_genes, args.batch_size, args.p_threshold, args.max_batches, args.print_every
            )
            all_mse_histories["ExternalPerturbation"] = ext_mse

            plot_pvalue_history(ext_hist_obs, "ExternalPerturbation_ObservedGenes", args.output_dir, ground_truth_p)
            plot_pvalue_history(ext_hist_imp, "ExternalPerturbation_ImputedGenes", args.output_dir, ground_truth_p)

    # ---------------------------------------------------------
    # Shared GP Initialization (Run once if any GP strategy is needed)
    # ---------------------------------------------------------
    shared_learner = None
    has_external_data = (args.external_list or args.external_h5ad or args.embeddings_yaml)
    
    # We initialize the learner if we have external data AND (we run the default GP strategy OR the optional active one)
    # Note: Strategy 5 runs automatically if data is present. Strategy 6 is optional.
    if ActiveGPLearner is not None and has_external_data:
        print("\nInitializing Shared Active GP Learner...")
        # Learner uses gene names matching df index order.
        # This loads the heavy kernels ONCE.
        shared_learner = ActiveGPLearner(df['gene_name'].values, args)

    # ---------------------------------------------------------
    # Strategy 5: GP Imputation (Covariance/Random + GP Prediction)
    # ---------------------------------------------------------
    gp_obs, gp_imp = None, None
    gp_strat_name = "GP Imputation"
    
    if shared_learner is not None:
        # Select Order: Prioritize Control Covariance, fallback to Random
        if cov_indices is not None:
            gp_indices = cov_indices
            gp_strat_name = "Control Covariance + GP Imputation"
            plot_name_obs = "CovGP_ObservedGenes"
            plot_name_imp = "CovGP_ImputedGenes"
            mse_key = "ControlCovariance_GP"
        else:
            print("Warning: Control Covariance not available. Falling back to Random Sampling for GP.")
            gp_indices = rnd_indices
            gp_strat_name = "Random Sampling + GP Imputation"
            plot_name_obs = "RandomGP_ObservedGenes"
            plot_name_imp = "RandomGP_ImputedGenes"
            mse_key = "Random_GP"
            
        # Use StaticGPStrategy to wrap the static list + learner
        # Ensure learner starts clean
        shared_learner.reset()
        strat_gp = StaticGPStrategy(total_genes, args, gp_indices, shared_learner, name=gp_strat_name)

        gp_obs, gp_imp, gp_hist_obs, gp_hist_imp, gp_mse = run_simulation_strategy(
            strat_gp, df, total_genes, args.batch_size, args.p_threshold, args.max_batches, args.print_every
        )
        
        all_mse_histories[mse_key] = gp_mse
        plot_pvalue_history(gp_hist_obs, plot_name_obs, args.output_dir, ground_truth_p)
        plot_pvalue_history(gp_hist_imp, plot_name_imp, args.output_dir, ground_truth_p)

    # ---------------------------------------------------------
    # Strategy 6: Active High Leverage (Optional)
    # ---------------------------------------------------------
    lev_obs, lev_imp = None, None
    if args.run_active_leverage:
        if shared_learner is None:
            print("\nError: Active High Leverage requires ActiveGPLearner and external data/embeddings.")
        else:
            print("\nRunning Active High Leverage Strategy...")
            # Reset the learner to clear weights learned during Strategy 5
            shared_learner.reset()
            
            strat_lev = HighLeverageStrategy(total_genes, args, shared_learner)
            
            lev_obs, lev_imp, lev_hist_obs, lev_hist_imp, lev_mse = run_simulation_strategy(
                strat_lev, df, total_genes, args.batch_size, args.p_threshold, args.max_batches, args.print_every
            )
            
            all_mse_histories["Active_HighLeverage"] = lev_mse
            plot_pvalue_history(lev_hist_obs, "HighLeverage_ObservedGenes", args.output_dir, ground_truth_p)
            plot_pvalue_history(lev_hist_imp, "HighLeverage_ImputedGenes", args.output_dir, ground_truth_p)

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
    print(f"{'Strategy':<40} | {'ObservedGenes':<18} | {'ImputedGenes':<18}")
    print("-" * 82)

    def fmt(val): return str(val) if val else "> Max Batches"

    print(f"{'GammaMagnitude Sorting':<40} | {fmt(mag_obs):<18} | {fmt(mag_imp):<18}")
    print(f"{'Random Sampling':<40} | {fmt(rnd_obs):<18} | {fmt(rnd_imp):<18}")
    if args.control_h5ad and cov_obs is not None:
        print(f"{'Control Covariance Sorting':<40} | {fmt(cov_obs):<18} | {fmt(cov_imp):<18}")
    if args.external_h5ad and ext_obs is not None:
        print(f"{'External Perturbation Sorting':<40} | {fmt(ext_obs):<18} | {fmt(ext_imp):<18}")
    if gp_obs is not None:
        print(f"{gp_strat_name:<40} | {fmt(gp_obs):<18} | {fmt(gp_imp):<18}")
    if lev_obs is not None:
        print(f"{'Active High Leverage':<40} | {fmt(lev_obs):<18} | {fmt(lev_imp):<18}")
    print("-" * 82)

if __name__ == "__main__":
    main()