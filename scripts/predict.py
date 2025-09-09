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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from typing import Dict
import gc
from scipy import sparse

# Ensure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Model and data components
from graft.data.dataset import GraftStreamingConfig, GraftStreamingDataset, ControlANN
from graft.data.samplers import BalancedRoundRobin
from graft.models.gnn_core import StatePropagator
from graft.models.step0 import StepZeroClamp
from graft.models.heads import MediatedHead, SparseDirectHead
from graft.utils.re_noise import ReNoiser, write_anndata
from graft.utils.chunk_preprocess import load_gene_list, _build_projection


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
    
    # --- 2. Set up Streaming Data Loader for Prediction ---
    print(f"Setting up streaming loader for prediction using controls from: {args.controls_dataset_id}")
    template_adata = ad.read_h5ad(args.template_h5ad, backed='r')
    original_genes = template_adata.var_names.astype(str).tolist()
    original_obs = template_adata.obs.copy()

    # CORRECTED LOGIC: Create a temporary YAML where the user-provided dataset_id
    # points to the template H5AD file.
    temp_yaml_path = "temp_predict_datasets.yaml"
    with open(paths['datasets_yaml'], 'r') as f:
        datasets_yaml_data = yaml.safe_load(f)
    
    # Ensure the target dataset exists in the config before overwriting
    if args.controls_dataset_id not in datasets_yaml_data['datasets']:
        raise KeyError(f"Provided --controls-dataset-id '{args.controls_dataset_id}' not found in {paths['datasets_yaml']}")
    
    datasets_yaml_data['datasets'][args.controls_dataset_id]['raw_path'] = args.template_h5ad
    
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(datasets_yaml_data, f)
    
    # Sampler will now correctly yield the ID that the loader can find.
    sampler = BalancedRoundRobin(dataset_ids=[args.controls_dataset_id], steps=None)
    
    ds_cfg = GraftStreamingConfig(
        datasets_yaml=temp_yaml_path, index_parquet=paths["index_parquet"],
        gene_list_tsv=paths["gene_list_tsv"], scvi_model_dir=paths["scvi_model_dir"],
        scvi_input_h5ad=paths["scvi_input_h5ad"], control_index_dir=paths["control_index_dir"],
        control_z_npz=paths["control_z_npz"], control_xbar_npz=paths["control_xbar_npz"],
        batch_size=args.batch_size, chunk_size=train_cfg.get("chunk_size", 50000),
        k_controls=train_cfg["k_controls"], oversample=train_cfg["oversample"],
        match_within=train_cfg.get("match_within", "dataset"), forward_batch_size=train_cfg["forward_batch_size"],
        include_controls_in_query=True,
        filter_by_index=False # <-- IMPORTANT: Do not filter for unseen cells
    )
    dataset = GraftStreamingDataset(ds_cfg, sampler)

    # --- 3. Lazy Model Instantiation & Loading ---
    prop, step0, head_med, head_dir = None, None, None, None
    env_to_code = None
    
    # --- 4. Prediction Loop ---
    print(f"Generating predictions for {template_adata.n_obs} cells...")
    all_preds_normalized, all_cell_ids = [], []
    
    data_iterator = iter(dataset)

    with torch.no_grad(), tqdm(total=template_adata.n_obs) as pbar:
        while len(all_cell_ids) < template_adata.n_obs:
            try:
                batch = next(data_iterator)
            except StopIteration:
                break

            if prop is None:
                print("First batch received. Lazily initializing model...")
                z_dim = batch["z_q"].shape[1]
                G = len(gene_list)
                scvi_ref_adata = ad.read_h5ad(paths['scvi_input_h5ad'], backed='r')
                ds_ids = sorted(list(scvi_ref_adata.obs['dataset_id'].astype(str).unique()))
                n_envs = len(ds_ids)
                env_to_code = {dsid: i for i, dsid in enumerate(ds_ids)}

                prop = StatePropagator(z_dim=z_dim, n_envs=n_envs, n_genes=G, **model_cfg["propagator"]).to(device)
                step0 = StepZeroClamp(z_dim=z_dim, n_labs=n_envs, **model_cfg["step0"]).to(device)
                head_med = MediatedHead(z_dim=z_dim, **model_cfg["mediated"]).to(device)
                head_dir = SparseDirectHead(z_dim=z_dim, G=G, **model_cfg["direct"]).to(device)

                checkpoint = torch.load(args.checkpoint, map_location=device)
                prop.load_state_dict(checkpoint["models"]["prop"])
                step0.load_state_dict(checkpoint["models"]["step0"])
                head_med.load_state_dict(checkpoint["models"]["head_med"])
                head_dir.load_state_dict(checkpoint["models"]["head_dir"])
                prop.eval(); step0.eval(); head_med.eval(); head_dir.eval()
            
            # make predictions
            tb = to_device(batch, device)
            x0 = tb["xbar_ctrl"].mean(dim=1)

            # --- 1. Calculate clamp effectiveness based on PRE-perturbation state z_q
            x_clamped_authoritative, eff = step0(x0, tb["z_q"], tb["env_code"], tb["target_idx"])

            # --- 2. Propagate state, now conditioned on the effectiveness of the initial hit
            z_ref = prop(tb["z_q"], eff=eff, target_idx=tb["target_idx"], env_codes=tb["env_code"])

            # --- 3. Predict downstream changes from the new state z_ref
            m = head_med(z_ref)
            dx_med = m @ U
            dx_dir = head_dir(z_ref)
            y_pred_downstream = x0 + dx_med + dx_dir

            # --- 4. Surgically intervene to enforce the Step-0 clamp on the final prediction
            y_pred = y_pred_downstream.clone()
            mask = tb["target_idx"] >= 0
            if torch.any(mask):
                rows_to_update = torch.nonzero(mask, as_tuple=False).view(-1)
                cols_to_update = tb["target_idx"][mask]
                clamped_values = x_clamped_authoritative[rows_to_update, cols_to_update]
                y_pred[rows_to_update, cols_to_update] = clamped_values

            all_preds_normalized.append(y_pred.cpu().numpy())
            all_cell_ids.extend(batch['cell_ids'])

            pbar.update(len(batch['cell_ids']))

    os.remove(temp_yaml_path)

    # --- 5. Assemble, Re-noise, and Unscramble (Rewritten for Memory Efficiency) ---
    print("Finalizing predictions: re-noising and unscrambling genes...")
    final_normalized_matrix = np.vstack(all_preds_normalized)
    del all_preds_normalized
    gc.collect()

    # Reconstruct final obs dataframe
    processed_local_ids = [cid.split('::')[1] for cid in all_cell_ids]
    final_obs = original_obs.loc[processed_local_ids]

    # Re-noise to get a dense matrix of canonical counts
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
    dtype = np.int16 if final_counts_canonical.max() < 32767 else np.int32
    sparse_counts_canonical = sparse.csr_matrix(final_counts_canonical, dtype=dtype)
    del final_counts_canonical # Free the dense canonical matrix
    gc.collect()

    # Build the back-projection matrix to map from canonical to original gene order
    back_projection = _build_projection(gene_list, original_genes)

    # Perform the unscrambling in sparse format. The result is also sparse and memory-efficient.
    print("Unscrambling genes in sparse format...")
    final_counts_unscrambled_sparse = sparse_counts_canonical @ back_projection

    print(f"Writing final predicted AnnData to {args.output_h5ad}...")
    # The write_anndata helper can accept the sparse matrix directly
    output_adata = write_anndata(
        counts=final_counts_unscrambled_sparse,
        var_names=original_genes,
        obs_df=final_obs
    )

    output_dir = os.path.dirname(args.output_h5ad)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    output_adata.write_h5ad(args.output_h5ad, compression="lzf")
    print("\nPrediction complete.")

if __name__ == "__main__":
    main()

