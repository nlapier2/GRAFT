"""
predict.py (Refactored for Streaming and Gene Unscrambling)

This script performs inference using a trained GRAFT model, leveraging the
same robust, streaming data pipeline as train_gnn.py.

Workflow:
1.  Loads a trained model and all required artifacts.
2.  Accepts a --template-h5ad file containing new cells to predict on, and a
    --controls-dataset-id to specify which set of trained controls to use for matching.
3.  Bypasses the index parquet filtering to allow prediction on new, unseen cells.
4.  Processes data in chunks, generating normalized predictions.
5.  Re-noises the combined predictions into count space.
6.  "Unscrambles" the genes by projecting the predicted matrix back to the
    original gene order of the template file.
7.  Saves the final AnnData object.
"""
import argparse
import os
import sys
from pathlib import Path

import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from typing import Dict
import gc
from scipy import sparse
import tempfile
import yaml

# Ensure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Model and data components
from graft.data.dataset import GraftStreamingConfig, GraftStreamingDataset, ControlANN
from graft.data.samplers import BalancedRoundRobin
from graft.models.gnn_core import StatePropagator
from graft.models.step0 import StepZeroClamp
from graft.models.heads import MediatedHead, TrueSparseDirectHead
from graft.utils.re_noise import ReNoiser, write_anndata
from graft.utils.chunk_preprocess import load_gene_list, _build_projection
from scripts.build_index import main as build_index_main
from train_gnn import make_prediction


def build_U(path: str, device: torch.device) -> torch.Tensor:
    """Helper to load the U matrix."""
    arr = np.load(path, allow_pickle=False)
    return torch.tensor(arr, dtype=torch.float32, device=device)

def load_renoise_params(path: str) -> dict:
    """Loads the .npz file from prepare_renoise_params.py into a dict."""
    npz_file = np.load(path)
    params = {'theta': npz_file['theta'], 'L_ref': float(npz_file['L_ref'])}
    lib_sizes = {k.replace('lib_sizes_', ''): v for k, v in npz_file.items() if k.startswith('lib_sizes_')}
    params['lib_sizes_by_dataset'] = lib_sizes
    return params

def to_device(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            # Handle multi-dimensional arrays for control latents/expressions
            if v.ndim > 1 and v.dtype.kind in ('f',):
                 dtype = torch.float32
            elif v.dtype.kind in ("f",):
                dtype = torch.float32
            elif v.dtype.kind in ("i", "u", "b"):
                dtype = torch.int64
            else:
                continue # Skip non-numeric types like string arrays
            out[k] = torch.as_tensor(v, dtype=dtype, device=device)
    return out

def main():
    parser = argparse.ArgumentParser(description="Generate predictions from a trained GRAFT model.")
    parser.add_argument("--config", required=True, help="Path to the training config YAML (e.g., graft_smoke.yaml).")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--renoise-params", required=True, help="Path to the renoise_params.npz file.")
    parser.add_argument("--template-h5ad", required=True, help="Path to an AnnData file with new cells to predict.")
    parser.add_argument("--controls-dataset-id", required=True, help="The dataset_id from training to use for control matching.")
    parser.add_argument("--output-h5ad", required=True, help="Path to save the final predicted AnnData object.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for processing predictions.")
    parser.add_argument("--skip-renoise", action="store_true", help="If set, skip the re-noising step and output normalized values.")
    args = parser.parse_args()

    # --- 1. Load Configs and Artifacts ---
    print("Loading configuration and artifacts...")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    paths, model_cfg, train_cfg = cfg["paths"], cfg["model"], cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gene_list = load_gene_list(paths["gene_list_tsv"])
    
    control_ann = ControlANN.load(paths["control_index_dir"], paths["control_z_npz"], paths["control_xbar_npz"], paths["index_parquet"])
    U = build_U(paths["factor_U"], device=device)
    renoise_data = load_renoise_params(args.renoise_params)

    # Sanitize renoise keys
    final_lib_sizes = {k.replace('_', '-'): v for k, v in renoise_data['lib_sizes_by_dataset'].items()}
    renoiser = ReNoiser(L_ref=renoise_data['L_ref'], theta=renoise_data['theta'], lib_sizes_by_dataset=final_lib_sizes)

    with open(paths["datasets_yaml"], "r") as _f:
        _y = yaml.safe_load(_f)
    _ds_map = _y.get("datasets", {})
    train_graft_true = {str(k) for k, v in _ds_map.items()
                        if isinstance(v, dict) and bool(v.get("train_graft", True))}
    # get dataset_ids that actually made it into the scVI input (after your CT filter)
    _scvi_ref = ad.read_h5ad(paths["scvi_input_h5ad"], backed="r")
    scvi_ds_ids = set(_scvi_ref.obs["dataset_id"].astype(str).unique())
    # intersection in sorted order to reproduce training-time mapping
    train_env_ds_ids = sorted(train_graft_true & scvi_ds_ids)
    if not train_env_ds_ids:
        raise ValueError("No datasets remain after (train_graft:true ∩ scVI-input) intersection.")
    if args.controls_dataset_id not in train_env_ds_ids:
        raise ValueError(
            f"controls-dataset-id '{args.controls_dataset_id}' is not in the training env set "
            "(likely filtered out by scVI-input cell-type filter). Choose one of: "
            + ", ".join(train_env_ds_ids)
        )
    n_envs_train = len(train_env_ds_ids)

    # --- preload control pools for random sampling baseline ---
    rng = np.random.default_rng(12345)
    k_controls = int(cfg["training"].get("k_controls", 16))
    G = len(gene_list)
    # Load z_ctrl from the NPZ produced by train_scvi.py
    with np.load(paths["control_z_npz"], allow_pickle=False) as zfile:
        Z_ctrl = np.asarray(zfile["z"], dtype=np.float32, order="C")
    with np.load(paths["control_xbar_npz"], allow_pickle=False) as xfile:
        key = "xbar" if "xbar" in xfile.files else "X"
        if key == "X":
            print("[warn] control_xbar_npz does not contain 'xbar' key, using 'X' instead.")
        XBAR_ctrl = np.asarray(xfile[key], dtype=np.float32, order="C")

    # --- 2. Set up Streaming Data Loader for Prediction ---
    print(f"Setting up streaming loader for prediction using controls from: {args.controls_dataset_id}")
    template_adata = ad.read_h5ad(args.template_h5ad, backed='r')
    original_genes = template_adata.var_names.astype(str).tolist()
    original_obs = template_adata.obs.copy()

    # CORRECTED LOGIC: Create a temporary YAML where the user-provided dataset_id
    # points to the template H5AD file.
    temp_index_file = None
    temp_yaml_file = None
    with open(paths['datasets_yaml'], 'r') as f:
        datasets_yaml_data = yaml.safe_load(f)
    
    # Isolate the config for the target dataset and point it to the template file
    target_ds_config = datasets_yaml_data['datasets'][args.controls_dataset_id]
    temp_config = {
        'defaults': datasets_yaml_data['defaults'],
        'datasets': {args.controls_dataset_id: target_ds_config.copy()}
    }
    temp_config['datasets'][args.controls_dataset_id]['raw_path'] = args.template_h5ad

    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as tmp_yaml:
        yaml.dump(temp_config, tmp_yaml)
        temp_yaml_file = tmp_yaml.name

    # Create a temporary file path for the index parquet
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_idx:
        temp_index_file = tmp_idx.name
    
    print(f"Running build_index on template file...")
    # Call the main function from build_index.py
    build_index_main(
        yaml_path=temp_yaml_file,
        out_index=temp_index_file,
        gene_list_path=paths["gene_list_tsv"]
    )
    
    # Sampler will now correctly yield the ID that the loader can find.
    sampler = BalancedRoundRobin(dataset_ids=[args.controls_dataset_id], steps=None)
    
    ds_cfg = GraftStreamingConfig(
        datasets_yaml=temp_yaml_file, index_parquet=temp_index_file, #paths["index_parquet"],
        gene_list_tsv=paths["gene_list_tsv"], scvi_model_dir=paths["scvi_model_dir"],
        scvi_input_h5ad=paths["scvi_input_h5ad"], control_index_dir=paths["control_index_dir"],
        control_z_npz=paths["control_z_npz"], control_xbar_npz=paths["control_xbar_npz"],
        batch_size=args.batch_size, chunk_size=train_cfg.get("chunk_size", 50000),
        k_controls=train_cfg["k_controls"], oversample=train_cfg["oversample"],
        match_within=train_cfg.get("match_within", "dataset"), forward_batch_size=train_cfg["forward_batch_size"],
        include_controls_in_query=False,
        filter_by_index=True,  # Ok with temp index since we want all cells in the template
        use_log1p_target=bool(train_cfg.get("use_log1p_target", False)),
    )
    dataset = GraftStreamingDataset(ds_cfg, sampler)

    # --- 3. Lazy Model Instantiation & Loading ---
    prop, step0, head_med, head_dir = None, None, None, None
    env_to_code = None
    
    # --- 4. Prediction Loop ---
    expected = dataset.by_ds[args.controls_dataset_id].shape[0]  # perturbed-only pool
    print(f"Generating predictions for {expected} cells...")
    all_preds_normalized, all_cell_ids = [], []
    
    data_iterator = iter(dataset)

    with torch.no_grad(), tqdm(total=expected) as pbar:
        while len(all_cell_ids) < expected:
            try:
                batch = next(data_iterator)
                n = batch["z_q"].shape[0]
                z_idx = rng.integers(0, Z_ctrl.shape[0], size=n, endpoint=False)
                x_idx = rng.integers(0, XBAR_ctrl.shape[0], size=(n, k_controls), endpoint=False)
                batch["z_ctrl"] = Z_ctrl[z_idx].astype(np.float32, copy=False)           # (N, z_dim)
                batch["xbar_ctrl"] = XBAR_ctrl[x_idx].astype(np.float32, copy=False)     # (N, k_controls, G)

            except StopIteration:
                break

            if prop is None:
                print("First batch received. Lazily initializing model...")
                z_dim = batch["z_q"].shape[1]
                G = len(gene_list)

                prop = StatePropagator(z_dim=z_dim, n_envs=n_envs_train, n_genes=G, **model_cfg["propagator"]).to(device)
                step0 = StepZeroClamp(z_dim=z_dim, n_labs=n_envs_train, **model_cfg["step0"]).to(device)
                head_med = MediatedHead(z_dim=z_dim, **model_cfg["mediated"]).to(device)
                head_dir = TrueSparseDirectHead(z_dim=z_dim, n_genes=G, **model_cfg["direct"]).to(device)
                # head_dir = SparseDirectHead(z_dim=z_dim, G=G, **model_cfg["direct"]).to(device)

                checkpoint = torch.load(args.checkpoint, map_location=device)
                prop.load_state_dict(checkpoint["models"]["prop"])
                step0.load_state_dict(checkpoint["models"]["step0"])
                head_med.load_state_dict(checkpoint["models"]["head_med"])
                head_dir.load_state_dict(checkpoint["models"]["head_dir"])
                prop.eval(); step0.eval(); head_med.eval(); head_dir.eval()
            
            # make predictions
            tb = to_device(batch, device)
            y_pred, x_clamped_authoritative, dx_dir, dx_med, z_ref, eff = make_prediction(tb, step0, head_med, head_dir, prop, U)

            all_preds_normalized.append(y_pred.cpu().numpy())
            all_cell_ids.extend(batch['cell_ids'])

            pbar.update(len(batch['cell_ids']))
        
    # local ids for perturbed cells in the same order as predictions
    pert_local_ids_in_order = [cid.split("::", 1)[1] for cid in all_cell_ids]

    # --- 5. Assemble, Re-noise, and Unscramble (Rewritten for Memory Efficiency) ---
    print("Finalizing predictions: re-noising and unscrambling genes...")
    final_normalized_matrix = np.vstack(all_preds_normalized)
    del all_preds_normalized
    gc.collect()

    # Reconstruct final obs dataframe
    processed_local_ids = [cid.split('::')[1] for cid in all_cell_ids]
    final_obs = original_obs.loc[processed_local_ids]

    if train_cfg.get("use_log1p_target", False):
        final_counts_canonical = sparse.csr_matrix(final_normalized_matrix)
    else:
        # Re-noise to get a dense matrix of canonical counts
        print('Re-noising to obtain counts in canonical gene space...')
        final_counts_canonical = renoiser.sample_counts(
            xbar=final_normalized_matrix,
            dataset_ids=[args.controls_dataset_id] * len(final_obs),
            mode="empirical",
            chunk_size=5000
        )
        del final_normalized_matrix # Free memory
        gc.collect()

    # ---- KEY CHANGE: Convert to sparse BEFORE unscrambling ----
    print("Converting to sparse format before gene unscrambling...")
    # Using int16 if counts are within range, otherwise int32
    # dtype = np.int16 if final_counts_canonical.max() < 32767 else np.int32
    sparse_counts_canonical = sparse.csr_matrix(final_counts_canonical) #, dtype=dtype)
    del final_counts_canonical # Free the dense canonical matrix
    gc.collect()

    # Build the back-projection matrix to map from canonical to original gene order
    back_projection = _build_projection(gene_list, original_genes)

    # Perform the unscrambling in sparse format. The result is also sparse and memory-efficient.
    print("Unscrambling genes in sparse format...")
    final_counts_unscrambled_sparse = sparse_counts_canonical @ back_projection

    print("Adding control cells back into the final output...")
    # The write_anndata helper can accept the sparse matrix directly
    idx_tmp = pd.read_parquet(temp_index_file)
    ctrl_global_ids = set(idx_tmp.loc[idx_tmp["is_control"].astype(bool), "cell_id"].astype(str))
    ctrl_local_ids_in_template = [lid for lid in template_adata.obs_names.astype(str)
                                  if f"{args.controls_dataset_id}::{lid}" in ctrl_global_ids]
    ctrl_view = template_adata[ctrl_local_ids_in_template, :]
    if train_cfg.get("use_log1p_target", False):
        ctrl_view = ctrl_view.to_memory()
        sc.pp.normalize_total(ctrl_view, target_sum=1e4, key_added="ncounts")
        sc.pp.log1p(ctrl_view)
    ctrl_X = ctrl_view.X
    ctrl_counts_unscrambled = ctrl_X if sparse.issparse(ctrl_X) else sparse.csr_matrix(ctrl_X)

    from scipy.sparse import vstack as sp_vstack
    final_counts_unscrambled = sp_vstack([final_counts_unscrambled_sparse, ctrl_counts_unscrambled], format="csr")
    final_obs = pd.concat(
        [original_obs.loc[pert_local_ids_in_order], original_obs.loc[ctrl_local_ids_in_template]],
        axis=0
    )

    print(f"Writing final predicted AnnData to {args.output_h5ad}...")
    output_adata = write_anndata(
        counts=final_counts_unscrambled,
        var_names=original_genes,
        obs_df=final_obs
    )

    if temp_index_file and os.path.exists(temp_index_file):
        os.remove(temp_index_file)
    if temp_yaml_file and os.path.exists(temp_yaml_file):
        os.remove(temp_yaml_file)

    output_dir = os.path.dirname(args.output_h5ad)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    output_adata.write_h5ad(args.output_h5ad, compression="lzf")
    print("\nPrediction complete.")

if __name__ == "__main__":
    main()

