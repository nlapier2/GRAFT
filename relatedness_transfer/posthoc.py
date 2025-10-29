#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import anndata as ad

# local imports from your repo
from utils import (
    collapse_to_pseudobulk,
    compute_deltas,
    apply_confidence_boost,
)
from transforms import (
    subspace_boost,
    sharpen_effects,
)


def set_random_seed(seed: int):
    """Minimal local seed helper (mirrors transfer_main-style seeding)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    return rng


def compute_ctrl_mean_singlecell(adata_in, target_label, control_label):
    """
    Compute control mean directly from the *input* single-cell/pseudobulk AnnData.
    This is mostly for provenance. The real ctrl_mean we apply to deltas below
    will come from the pseudobulk summary after collapse_to_pseudobulk().
    """
    ctrl_mask = (adata_in.obs[target_label].astype(str).values == control_label)
    if not np.any(ctrl_mask):
        raise ValueError(f"No control cells found with label '{control_label}' in input AnnData.")
    X = np.asarray(adata_in.X, dtype=np.float32)
    return X[ctrl_mask].mean(axis=0).astype(np.float32)


def build_boosted_deltas(
    pb_adata,
    perts,
    delta_mat,
    ctrl_mean_vec,
    boost_pcs,
    boost_gamma,
    conf_boost_alpha,
    conf_shrink_alpha,
    conf_min_var,
    conf_max_var,
    sharpen_mode,
    sharpen_gamma,
    sharpen_topk_frac,
    sharpen_alpha,
    sharpen_beta,
    sharpen_sigmoid_B,
    sharpen_preserve_q,
):
    """
    Given pseudobulk deltas (pert - ctrl) for each perturbation, apply the same
    style of post-hoc transforms we use after KRR:
      1. subspace_boost()
      2. apply_confidence_boost()
      3. reconstruct pred expression = ctrl_mean + boosted_delta
      4. sharpen_effects() on expression
      5. final_delta = sharpened_expr - ctrl_mean
    """

    # --- 1. Subspace boost on the delta space ---
    # In transfer_main you boost predicted deltas using PCs from training-set deltas.
    # We don't have a separate training set here, so we just use delta_mat itself
    # to define that subspace (so it's "self-referential PCs").
    boosted_delta = subspace_boost(
        Y_U_hat_delta=delta_mat.copy(),
        Y_O_delta=delta_mat.copy(),
        k=max(0, int(boost_pcs)),
        gamma=float(boost_gamma),
    )

    # --- 2. Confidence-based amplification ---
    # We don't have true pred_delta_var here (that's produced by KRR),
    # but apply_confidence_boost() wants one so it can scale by confidence.
    # We'll feed a dummy "all-min-var" matrix, which basically says:
    #   confidence = 1 everywhere
    #   => scale ~ 1 + conf_boost_alpha
    # Shrinkage still works if conf_shrink_alpha > 0.
    dummy_var = np.full_like(boosted_delta, fill_value=conf_min_var, dtype=np.float32)

    boosted_delta2 = apply_confidence_boost(
        pred_delta_mat=boosted_delta,
        pred_delta_var=dummy_var,
        conf_boost_alpha=conf_boost_alpha,
        conf_shrink_alpha=conf_shrink_alpha,
        conf_min_var=conf_min_var,
        conf_max_var=conf_max_var,
    )

    # --- 3. Reconstruct predicted expression from boosted deltas ---
    pred_expr = ctrl_mean_vec[None, :] + boosted_delta2  # shape (P,G)

    # --- 4. Optional nonlinear sharpening in expression space ---
    # sharpen_effects() expects full predicted expression, and returns a new
    # predicted expression (ctrl + sharpened_delta). After that we recover the
    # sharpened delta again.
    pred_expr_sharp = sharpen_effects(
        pred_mat=pred_expr,
        ctrl_mean=ctrl_mean_vec,
        mode=sharpen_mode,
        gamma=sharpen_gamma,
        topk_frac=sharpen_topk_frac,
        alpha=sharpen_alpha,
        beta=sharpen_beta,
        sigmoid_B=sharpen_sigmoid_B,
        preserve_q=sharpen_preserve_q,
    )

    final_delta = pred_expr_sharp - ctrl_mean_vec[None, :]  # (P,G)
    return final_delta.astype(np.float32), pred_expr_sharp.astype(np.float32)


def synthesize_single_cells(
    adata_in,
    target_label,
    control_label,
    perts,
    final_delta,
    n_cells_per_pert,
    rng,
):
    """
    Build a synthetic single-cell AnnData:
      - For each perturbation p (non-control):
          sample n_cells_per_pert control cells with replacement,
          add delta[p] to each sampled control cell's expression,
          copy that control cell's obs row but overwrite target_label with p.
      - ALSO append ALL original control cells unchanged, so the output object
        contains both predicted perturbed cells and real controls.
    """
    X_all = np.asarray(adata_in.X, dtype=np.float32)
    obs_all = adata_in.obs.copy()
    G = X_all.shape[1]

    # identify control cells in the original data
    ctrl_mask = (obs_all[target_label].astype(str).values == control_label)
    if not np.any(ctrl_mask):
        raise ValueError(f"No control cells found with label '{control_label}' for synthesis.")

    ctrl_X_full = X_all[ctrl_mask, :]                 # all real control cells
    ctrl_obs_full = obs_all[ctrl_mask].copy()         # their full metadata

    # subset for sampling (same as full controls, but we'll index with rng)
    ctrl_X = ctrl_X_full
    ctrl_obs = ctrl_obs_full.reset_index(drop=True)

    synth_X_rows = []
    synth_obs_rows = []

    # generate synthetic cells for each non-control perturbation
    for p_idx, p in enumerate(perts):
        d_vec = final_delta[p_idx].reshape(1, G)  # (1,G)

        # pick control cells with replacement
        idx_sample = rng.integers(low=0, high=ctrl_X.shape[0], size=n_cells_per_pert)
        sampled_ctrl_X = ctrl_X[idx_sample, :]  # (n_cells_per_pert, G)

        # apply delta to each sampled control cell
        synth_expr = sampled_ctrl_X + d_vec  # broadcast add

        synth_X_rows.append(synth_expr.astype(np.float32))

        # carry obs metadata from sampled controls, overwrite perturbation label
        sampled_ctrl_obs = ctrl_obs.iloc[idx_sample].copy()
        sampled_ctrl_obs[target_label] = p
        synth_obs_rows.append(sampled_ctrl_obs)

    # concatenate all synthetic perturbed cells (if any)
    if len(synth_X_rows) == 0:
        synth_X_cat = np.zeros((0, adata_in.n_vars), dtype=np.float32)
        synth_obs_cat = pd.DataFrame({target_label: []})
    else:
        synth_X_cat = np.vstack(synth_X_rows).astype(np.float32)
        synth_obs_cat = pd.concat(synth_obs_rows, axis=0).reset_index(drop=True)

    # NOW append *all original control cells unchanged* to the bottom
    X_out = np.vstack([synth_X_cat, ctrl_X_full]).astype(np.float32)
    obs_out = pd.concat([synth_obs_cat, ctrl_obs_full.reset_index(drop=True)], axis=0).reset_index(drop=True)

    # Build final AnnData
    adata_out = ad.AnnData(
        X=X_out,
        obs=obs_out,
        var=adata_in.var.copy(),
        uns=adata_in.uns.copy(),
    )
    adata_out.var_names = adata_in.var_names.copy()
    return adata_out


def run(args):
    rng = set_random_seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. Load input AnnData
    # -------------------------------------------------------------------------
    print(f"[posthoc] Loading {args.adata_in}")
    adata_in = ad.read_h5ad(args.adata_in)

    # -------------------------------------------------------------------------
    # 2. Collapse to pseudobulk and compute deltas vs control
    #    (works whether adata_in is already pseudo or still single cell)
    # -------------------------------------------------------------------------
    print("[posthoc] Collapsing to per-pert pseudobulk and computing deltas")
    pb_adata = collapse_to_pseudobulk(adata_in, args.target_label)
    deltas_dict, ctrl_mean_vec = compute_deltas(
        pb_adata,
        target_label=args.target_label,
        control_label=args.control_label,
    )
    # organize perts + delta_mat in a consistent order
    perts = sorted(deltas_dict.keys())
    if len(perts) == 0:
        print("[posthoc] No non-control perturbations found. Exiting with empty output.")
        empty = ad.AnnData(
            X=np.zeros((0, adata_in.n_vars), dtype=np.float32),
            obs=pd.DataFrame({args.target_label: []}),
            var=adata_in.var.copy(),
            uns=adata_in.uns.copy(),
        )
        empty.var_names = adata_in.var_names.copy()
        empty.write_h5ad(args.adata_out)
        print(f"[posthoc] Wrote empty {args.adata_out}")
        return

    delta_mat = np.stack([np.asarray(deltas_dict[p]).astype(np.float32).ravel() for p in perts], axis=0)
    ctrl_mean_vec = np.asarray(ctrl_mean_vec, dtype=np.float32).ravel()

    # -------------------------------------------------------------------------
    # 3. Boost / confidence-scale / sharpen in the same spirit as transfer_main
    # -------------------------------------------------------------------------
    print("[posthoc] Applying post-hoc transforms")
    final_delta, pred_expr_sharp = build_boosted_deltas(
        pb_adata,
        perts,
        delta_mat,
        ctrl_mean_vec,
        boost_pcs=args.boost_pcs,
        boost_gamma=args.boost_gamma,
        conf_boost_alpha=args.conf_boost_alpha,
        conf_shrink_alpha=args.conf_shrink_alpha,
        conf_min_var=args.conf_min_var,
        conf_max_var=args.conf_max_var,
        sharpen_mode=args.sharpen_mode,
        sharpen_gamma=args.sharpen_gamma,
        sharpen_topk_frac=args.sharpen_topk_frac,
        sharpen_alpha=args.sharpen_alpha,
        sharpen_beta=args.sharpen_beta,
        sharpen_sigmoid_B=args.sharpen_sigmoid_B,
        sharpen_preserve_q=args.sharpen_preserve_q,
    )

    # -------------------------------------------------------------------------
    # 4. Generate synthetic single-cell AnnData via control-cell sampling
    # -------------------------------------------------------------------------
    print("[posthoc] Synthesizing single-cell predictions")
    adata_synth = synthesize_single_cells(
        adata_in,
        target_label=args.target_label,
        control_label=args.control_label,
        perts=perts,
        final_delta=final_delta,
        n_cells_per_pert=args.n_cells_per_pert,
        rng=rng,
    )

    # Store breadcrumbs in .uns so you can track provenance later
    adata_synth.uns["posthoc_params"] = {
        "boost_pcs": args.boost_pcs,
        "boost_gamma": args.boost_gamma,
        "conf_boost_alpha": args.conf_boost_alpha,
        "conf_shrink_alpha": args.conf_shrink_alpha,
        "conf_min_var": args.conf_min_var,
        "conf_max_var": args.conf_max_var,
        "sharpen_mode": args.sharpen_mode,
        "sharpen_gamma": args.sharpen_gamma,
        "sharpen_topk_frac": args.sharpen_topk_frac,
        "sharpen_alpha": args.sharpen_alpha,
        "sharpen_beta": args.sharpen_beta,
        "sharpen_sigmoid_B": args.sharpen_sigmoid_B,
        "sharpen_preserve_q": args.sharpen_preserve_q,
        "n_cells_per_pert": args.n_cells_per_pert,
        "seed": args.seed,
    }
    adata_synth.uns["posthoc_perts"] = perts
    adata_synth.uns["posthoc_ctrl_mean"] = ctrl_mean_vec.astype(np.float32)

    # Also stash the final pseudobulk predictions we generated (after sharpening)
    # so you can inspect them quickly without re-running.
    # Shape: (P,G) rows aligned to 'posthoc_perts'
    adata_synth.uns["posthoc_pseudobulk_expr"] = pred_expr_sharp.astype(np.float32)

    # -------------------------------------------------------------------------
    # 5. Write output
    # -------------------------------------------------------------------------
    print(f"[posthoc] Writing {args.adata_out}")
    adata_synth.write_h5ad(args.adata_out)
    print("[posthoc] Done.")


def make_arg_parser():
    ap = argparse.ArgumentParser(
        description="Post-hoc transform of perturbation deltas + synthetic single-cell generation."
    )

    # I/O
    ap.add_argument("--adata_in", type=str, required=True,
                    help="Input AnnData (.h5ad) containing control + pert cells (single-cell or pseudobulk).")
    ap.add_argument("--adata_out", type=str, required=True,
                    help="Output AnnData (.h5ad) with synthesized single-cell predictions per perturbation.")

    # Labels
    ap.add_argument("--target_label", type=str, default="target_gene",
                    help="obs column indicating perturbation identity.")
    ap.add_argument("--control_label", type=str, default="non-targeting",
                    help="Value in target_label column denoting control cells.")
    ap.add_argument("--n_cells_per_pert", type=int, default=200,
                    help="How many synthetic cells to generate per perturbation.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for control resampling.")

    # Subspace boosting knobs (transforms.subspace_boost)
    ap.add_argument("--boost_pcs", type=int, default=0,
                    help="Top-k PCs to boost in delta space (0 = off).")
    ap.add_argument("--boost_gamma", type=float, default=1.0,
                    help="Boost strength for subspace_boost.")

    # Confidence-based amplification knobs (utils.apply_confidence_boost)
    ap.add_argument("--conf_boost_alpha", type=float, default=0.0,
                    help="Amplify high-confidence genes (>=0).")
    ap.add_argument("--conf_shrink_alpha", type=float, default=0.0,
                    help="Shrink low-confidence genes (>=0). Set 0 to disable shrinking.")
    ap.add_argument("--conf_min_var", type=float, default=1e-6,
                    help="Lower bound on variance for confidence mapping.")
    ap.add_argument("--conf_max_var", type=float, default=1.0,
                    help="Upper bound on variance for confidence mapping.")

    # Expression-space sharpening knobs (transforms.sharpen_effects)
    ap.add_argument("--sharpen_mode", type=str, default="none",
                    choices=["none", "power", "topk", "sigmoid"],
                    help="Nonlinear sharpening mode to apply after reconstruction.")
    ap.add_argument("--sharpen_gamma", type=float, default=1.5,
                    help="Exponent gamma for mode='power'.")
    ap.add_argument("--sharpen_topk_frac", type=float, default=0.1,
                    help="Fraction of genes to amplify for mode='topk'.")
    ap.add_argument("--sharpen_alpha", type=float, default=0.3,
                    help="Amplification factor for top genes in mode='topk'.")
    ap.add_argument("--sharpen_beta", type=float, default=0.2,
                    help="Shrink factor for non-top genes in mode='topk'.")
    ap.add_argument("--sharpen_sigmoid_B", type=float, default=0.7,
                    help="Slope B for mode='sigmoid'.")
    ap.add_argument("--sharpen_preserve_q", type=float, default=0.95,
                    help="Quantile of |delta| magnitude to preserve when rescaling (power/sigmoid).")

    return ap


if __name__ == "__main__":
    args = make_arg_parser().parse_args()
    run(args)
