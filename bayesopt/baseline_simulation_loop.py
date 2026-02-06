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
    parser = argparse.ArgumentParser(description="Active Learning Baselines: Multi-Target Simulation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the 'Wide' TSV file containing gene_name, LoF_gamma, and multiple _beta columns.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of genes to reveal in each batch.")
    parser.add_argument("--print_every", type=int, default=10, help="Print progress every N batches.")
    parser.add_argument("--p_threshold", type=float, default=0.05, help="P-value threshold for statistical significance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--center_data", action="store_true", help="Center LoF_gamma and Beta targets at 0.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots.")
    
    # Target Selection
    parser.add_argument("--plot_target", type=str, default="HBA1", help="The specific target gene to generate plots and detailed logs for.")

    # Arguments for Control Strategy
    parser.add_argument("--control_h5ad", type=str, default=None, help="Path to H5AD file containing control cells.")
    parser.add_argument("--target_label", type=str, default="target_gene", help="Obs column name identifying perturbation/control status.")
    parser.add_argument("--control_label", type=str, default="", help="Value in target_label that identifies control cells.")

    # Argument for External Perturbation Strategy
    parser.add_argument("--external_h5ad", type=str, default=None, help="Path to 'external' pseudobulked H5AD.")

    # --- GP Imputation Arguments ---
    parser.add_argument("--external_list", type=str, default="", help="List of external h5ad files for GP kernel.")
    parser.add_argument("--embeddings_yaml", type=str, default="", help="Single YAML file defining pathway/embedding sources.")
    parser.add_argument("--emb_metric", type=str, default="cosine", help="Metric for embedding kernels (cosine/rbf).")
    parser.add_argument("--kernel_weight_gamma", type=float, default=1.0, help="Gamma parameter for kernel alignment weights.")
    parser.add_argument("--kernel_agg", type=str, default="mean", choices=["mean", "wmean"], help="GP Kernel aggregation method.")
    parser.add_argument("--gp_noise_var", type=float, default=0.01, help="GP noise variance (lambda).")
    parser.add_argument("--gp_recompute_freq", type=int, default=5, help="How often to re-weight GP kernels.")
    
    # --- Active Learning Arguments ---
    parser.add_argument("--run_active_leverage", action="store_true", help="Run Active High Leverage strategy.")
    parser.add_argument("--run_active_uncertainty", action="store_true", help="Run Active Uncertainty strategy.")
    parser.add_argument("--run_active_diversity", action="store_true", help="Run Active Diversity strategy.")
    parser.add_argument("--run_active_pca", action="store_true", help="Run Active PC-Uncertainty strategy.")
    parser.add_argument("--run_active_var_reduction", action="store_true", help="Run Active Stepwise Variance Reduction strategy.")

    parser.add_argument("--acq_beta", type=float, default=1.0, help="Beta parameter for acquisition.")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to run.")
    parser.add_argument("--pca_recompute_freq", type=int, default=1, help="Frequency to recompute PCA.")
    parser.add_argument("--pca_top_k", type=int, default=50, help="Number of PCs for uncertainty.")
    parser.add_argument("--stepwise_subset_size", type=int, default=400, help="Working Set size for variance reduction.")

    # --- Static Strategy Options ---
    parser.add_argument("--imputation_method", type=str, default="mean", choices=["mean", "zero"], help="Imputation method for static strategies.")
    parser.add_argument("--sampling_strategy", type=str, default="strongest", choices=["strongest", "uniform"], help="Order for static strategies.")
    parser.add_argument("--static_only", action="store_true", help="Skip all active learning strategies.")

    return parser.parse_args()


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def load_control_adata(h5ad_path, obs_label, control_val):
    """Loads and filters the control AnnData once."""
    if not h5ad_path or not os.path.exists(h5ad_path):
        return None
        
    print(f"Loading control H5AD: {h5ad_path} ...")
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"Error loading H5AD: {e}")
        return None

    # Filter for control cells
    if control_val is not None and str(control_val).strip() != "":
        if obs_label in adata.obs.columns:
            c_val_lower = str(control_val).lower()
            mask = adata.obs[obs_label].astype(str).str.lower() == c_val_lower
            adata = adata[mask].copy()
            print(f"  -> Retained {adata.n_obs} control cells.")
        else:
            print(f"  -> Warning: '{obs_label}' not found. Using all cells.")
    
    if adata.n_obs < 5:
        print("  -> Error: Too few control cells.")
        return None
        
    return adata


def get_covariance_for_target(adata, target_gene):
    """Computes covariance vector for a specific target gene from pre-loaded adata."""
    if adata is None or target_gene not in adata.var_names:
        return None

    # Extract Target Vector
    target_idx = adata.var_names.get_loc(target_gene)
    X = adata.X
    
    if sp.issparse(X):
        # Dense extraction of just the target column is fast
        y_vec = X[:, target_idx].toarray().flatten()
    else:
        y_vec = X[:, target_idx]
        
    y_centered = y_vec - np.mean(y_vec)
    
    # We need full matrix densification for covariance with ALL genes?
    # Or chunked? For <30k genes, dense is usually fine (approx 2-4GB RAM).
    if sp.issparse(X):
        try:
            X = X.toarray()
        except MemoryError:
            print("Error: Control matrix too large for dense covariance.")
            return None
            
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean[None, :]
    
    N = adata.n_obs
    covariances = np.dot(X_centered.T, y_centered) / (N - 1)
    
    return pd.Series(covariances, index=adata.var_names)


def load_external_adata(h5ad_path, obs_label, control_val):
    """Loads external perturbation H5AD once."""
    if not h5ad_path or not os.path.exists(h5ad_path):
        return None
    print(f"Loading external H5AD: {h5ad_path} ...")
    try:
        adata = ad.read_h5ad(h5ad_path)
        if obs_label not in adata.obs.columns:
            print(f"  -> Error: '{obs_label}' not found.")
            return None
        return adata
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_perturbation_effect_for_target(adata, target_gene, obs_label, control_val):
    """Computes perturbation effect vector for a target from pre-loaded adata."""
    if adata is None or target_gene not in adata.var_names:
        return None
        
    gene_idx = adata.var_names.get_loc(target_gene)
    X_vec = adata.X[:, gene_idx]
    
    if sp.issparse(X_vec):
        X_vec = X_vec.toarray().flatten()
    else:
        X_vec = np.asarray(X_vec).flatten()

    obs_vals = adata.obs[obs_label].astype(str)
    
    # Identify Control Mean
    is_ctrl = obs_vals == str(control_val)
    if not is_ctrl.any():
        is_ctrl = obs_vals.str.lower() == str(control_val).lower()
        
    if not is_ctrl.any():
        return None
        
    ctrl_mean = np.mean(X_vec[is_ctrl])
    
    # Group by perturbation
    df_temp = pd.DataFrame({'pert': obs_vals, 'expr': X_vec})
    df_pert = df_temp[~is_ctrl]
    
    pert_means = df_pert.groupby('pert')['expr'].mean()
    abs_deltas = (pert_means - ctrl_mean).abs()
    
    return abs_deltas


def get_stratified_indices(scores, n_bins=10, seed=42):
    """Stratified sampling helper."""
    try:
        bins = pd.qcut(scores, q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        try:
            bins = pd.cut(scores, bins=n_bins, labels=False, duplicates='drop')
        except ValueError:
            return scores.index.values
    
    if hasattr(bins, 'categories'):
        n_actual = len(bins.categories)
    else:
        n_actual = int(bins.max()) + 1
    
    df_temp = pd.DataFrame({'score': scores, 'bin': bins})
    rng = np.random.default_rng(seed)
    queues = []
    max_len = 0
    
    for b in range(n_actual):
        indices = df_temp[df_temp['bin'] == b].index.values
        if len(indices) > 0:
            rng.shuffle(indices)
            queues.append(list(indices))
            if len(indices) > max_len:
                max_len = len(indices)
    
    stratified_order = []
    for i in range(max_len):
        for q in queues:
            if i < len(q):
                stratified_order.append(q[i])
                
    return np.array(stratified_order, dtype=int)


# =============================================================================
# SIMULATION LOGIC
# =============================================================================

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
    
    batch_idx = 0
    
    while n_revealed < total_genes:
        if max_batches is not None and batch_idx >= max_batches:
            if not silent: print(f"Reached max_batches ({max_batches}). Stopping.")
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

        # 4. Metric: Observed
        if n_revealed >= 2 and np.std(lof_known) > 0 and np.std(y_known) > 0:
            corr_obs, p_obs = stats.pearsonr(lof_known, y_known)
        else:
            corr_obs, p_obs = 0.0, 1.0
            
        if sig_batch_obs is None and p_obs < p_threshold:
            sig_batch_obs = batch_idx
        
        final_p_obs = p_obs
        history_obs.append(p_obs)

        # 5. Metric: Imputed
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

        # MSE
        mse = np.mean((y_true_values - y_full_hybrid) ** 2)
        history_mse.append(mse)

        if not silent:
            if batch_idx == 1 or batch_idx % print_every == 0 or n_revealed == total_genes:
                print(f"{batch_idx:<8} | {n_revealed:<8} | {corr_obs:+.4f}     | {p_obs:.2e}  | {corr_imp:+.4f}     | {p_imp:.2e}")

    # --- Compile Stats ---
    # Log10 P-value stats
    nlog10_true = -np.log10(ground_truth_p) if ground_truth_p > 1e-300 else 300
    nlog10_pred_obs = -np.log10(final_p_obs) if final_p_obs > 1e-300 else 300
    nlog10_pred_imp = -np.log10(final_p_imp) if final_p_imp > 1e-300 else 300
    
    return {
        "gene": target_gene_name,
        "strategy": strategy.name,
        "true_p": ground_truth_p,
        "final_p_obs": final_p_obs,
        "final_p_imp": final_p_imp,
        "batches_obs": sig_batch_obs if sig_batch_obs else (max_batches + 1),
        "batches_imp": sig_batch_imp if sig_batch_imp else (max_batches + 1),
        "success_obs": (sig_batch_obs is not None),
        "success_imp": (sig_batch_imp is not None),
        "error_nlog10_obs": (nlog10_pred_obs - nlog10_true), # Bias
        "error_nlog10_imp": (nlog10_pred_imp - nlog10_true),
        "abs_error_nlog10_obs": abs(nlog10_pred_obs - nlog10_true),
        "abs_error_nlog10_imp": abs(nlog10_pred_imp - nlog10_true),
        "history_obs": history_obs,
        "history_imp": history_imp,
        "history_mse": history_mse
    }


def plot_pvalue_history(p_values, method_name, output_dir, target_name, true_p_val=None):
    if not p_values: return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    batches = range(1, len(p_values) + 1)
    min_p = 1e-20
    capped_p_values = [max(p, min_p) for p in p_values]
    nlog10_p = [-np.log10(p) for p in capped_p_values]
    
    if true_p_val is not None:
        ref_p = max(true_p_val, min_p)
        label_text = "Ground Truth"
    else:
        ref_p = capped_p_values[-1]
        label_text = "Final"

    ref_nlog10 = -np.log10(ref_p)
    thresh_nlog10 = -np.log10(0.05)
    
    plt.figure(figsize=(10, 6))
    plt.plot(batches, nlog10_p, label='-log10(p-value)', linewidth=2)
    plt.axhline(y=thresh_nlog10, color='r', linestyle='--', alpha=0.7, label='Marginal Sig (0.05)')
    plt.axhline(y=ref_nlog10, color='g', linestyle='--', alpha=0.7, label=f'{label_text} P-value')
    
    plt.title(f"{target_name}: {method_name}")
    plt.xlabel("Batches")
    plt.ylabel("-log10(p-value)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    safe_m = method_name.replace(" ", "_").replace("(", "").replace(")", "")
    safe_t = target_name.replace("/", "_")
    out_path = os.path.join(output_dir, f"{safe_t}_{safe_m}.png")
    plt.savefig(out_path)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

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
    adata_ctrl = load_control_adata(args.control_h5ad, args.target_label, args.control_label)
    adata_ext = load_external_adata(args.external_h5ad, args.target_label, args.control_label)
    
    # Initialize Shared GP Learner Once
    shared_learner = None
    has_external_data = (args.external_list or args.external_h5ad or args.embeddings_yaml)
    
    if not args.static_only and ActiveGPLearner is not None and has_external_data:
        print("\nInitializing Shared Active GP Learner (Kernel Computation)...")
        # Ensure we drop rows with NaN in LoF to match indices
        # NOTE: We must ensure df indices align with learner.
        # We'll drop NAs relative to LoF globally first.
        df = df.dropna(subset=['LoF_gamma']).reset_index(drop=True)
        shared_learner = ActiveGPLearner(df['gene_name'].values, args)

    total_genes = len(df)
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

        # Extract vectors for this target (drop NaNs specific to this beta)
        # Note: We need to maintain alignment with shared_learner (which used full df).
        # Imputation: Fill NaNs in y_true with 0 OR skip?
        # Simulation requires Truth. If Truth is NaN, we can't simulate.
        # However, we cannot drop rows now or indices will shift relative to GP Kernel.
        # Solution: Mask out NaNs in 'revealed_mask' effectively?
        # Simpler: Fill NaNs with 0 for simulation "Truth" but warn? 
        # Or better: Just use the subset where both are valid for correlation calc?
        # Let's assume input is cleaned. If not, fillna(0).
        y_true = df[target_col].fillna(0.0).values
        
        # Calculate Ground Truth
        if df['LoF_gamma'].std() > 0 and np.std(y_true) > 0:
            _, ground_truth_p = stats.pearsonr(df['LoF_gamma'], y_true)
        else:
            ground_truth_p = 1.0
            
        if not silent:
            print(f"Ground Truth P-value: {ground_truth_p:.2e}")

        # --- PREPARE STRATEGIES ---
        strategies = []
        
        # 1. GammaMagnitude
        mag_scores = df['LoF_gamma'].abs()
        if args.sampling_strategy == "uniform":
            mag_idxs = get_stratified_indices(mag_scores, n_bins=10, seed=args.seed)
            mag_name = "GammaMagnitude (Uniform)"
        else:
            mag_idxs = mag_scores.sort_values(ascending=False).index.values
            mag_name = "GammaMagnitude (Strongest)"
        strategies.append(StaticStrategy(total_genes, args, mag_idxs, name=mag_name))
        
        # 2. Random
        # Use target-specific seed to vary random sampling? Or fixed? Fixed allows comparison.
        rnd_idxs = df.sample(frac=1, random_state=args.seed).index.values
        strategies.append(StaticStrategy(total_genes, args, rnd_idxs, name="Random Sampling"))
        
        # 3. Control Covariance
        cov_indices = None
        if adata_ctrl:
            cov_series = get_covariance_for_target(adata_ctrl, target_name)
            if cov_series is not None:
                # Align to DF
                cov_aligned = df['gene_name'].map(cov_series).fillna(0.0)
                cov_scores = cov_aligned.abs()
                if args.sampling_strategy == "uniform":
                    cov_indices = get_stratified_indices(cov_scores, n_bins=10, seed=args.seed)
                    c_name = "Control Covariance (Uniform)"
                else:
                    cov_indices = cov_scores.sort_values(ascending=False).index.values
                    c_name = "Control Covariance (Strongest)"
                strategies.append(StaticStrategy(total_genes, args, cov_indices, name=c_name))

        # 4. External Perturbation
        if adata_ext:
            ext_series = get_perturbation_effect_for_target(adata_ext, target_name, args.target_label, args.control_label)
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

        # --- EXECUTE ---
        mse_histories = {}
        
        for strat in strategies:
            res = run_simulation_strategy(
                strat, df, total_genes, args.batch_size, args.p_threshold,
                target_name, y_true, ground_truth_p,
                max_batches=args.max_batches, print_every=args.print_every, silent=silent
            )
            results_list.append(res)
            
            # Plotting for the main target
            if is_plot_target:
                mse_histories[strat.name] = res['history_mse']
                plot_pvalue_history(res['history_obs'], f"{strat.name}_Obs", args.output_dir, target_name, ground_truth_p)
                plot_pvalue_history(res['history_imp'], f"{strat.name}_Imp", args.output_dir, target_name, ground_truth_p)

    # ==========================================
    # FINAL SUMMARY REPORT
    # ==========================================
    df_res = pd.DataFrame(results_list)
    
    # Split into Correlated (Marginally Significant Ground Truth) vs Uncorrelated
    df_corr = df_res[df_res['true_p'] < 0.05]
    df_null = df_res[df_res['true_p'] >= 0.05]
    
    def print_group_summary(name, sub_df):
        print(f"\n\n>>> SUMMARY: {name} Genes (Count: {len(sub_df['gene'].unique())})")
        if sub_df.empty:
            print("No genes in this category.")
            return

        # Group by Strategy
        grp = sub_df.groupby('strategy')
        
        print(f"{'Strategy':<35} | {'Succ% (Obs)':<10} | {'Bias (Obs)':<10} | {'MAE (Obs)':<10} | {'Succ% (Imp)':<10} | {'Bias (Imp)':<10} | {'MAE (Imp)':<10}")
        print("-" * 115)
        
        for strat, g in grp:
            # Stats
            succ_obs = (g['success_obs'].sum() / len(g)) * 100
            bias_obs = g['error_nlog10_obs'].mean()
            mae_obs = g['abs_error_nlog10_obs'].mean()
            
            succ_imp = (g['success_imp'].sum() / len(g)) * 100
            bias_imp = g['error_nlog10_imp'].mean()
            mae_imp = g['abs_error_nlog10_imp'].mean()
            
            print(f"{strat:<35} | {succ_obs:6.1f}%    | {bias_obs:6.2f}     | {mae_obs:6.2f}     | {succ_imp:6.1f}%    | {bias_imp:6.2f}     | {mae_imp:6.2f}")

    print_group_summary("TRUE CORRELATED", df_corr)
    print_group_summary("UNCORRELATED (NULL)", df_null)
    print("\n" + "="*60)


if __name__ == "__main__":
    main()