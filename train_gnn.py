#!/usr/bin/env python3
"""
train_gnn.py (reconciled and refactored for efficient iteration)
=========================

This script combines the modern streaming data pipeline with the original,
correct multi-component model architecture and detailed loss calculations.

Refactoring:
- Implements efficient iteration by creating a single persistent data iterator.
- The sampler logic is passed into the dataset class to manage stateful chunk loading.
- Training loop calls next(iterator) instead of re-creating the iterator each step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import anndata as ad # Added import for pre-calculation

# --- Imports from the data pipeline ---
from graft.data.dataset import GraftStreamingConfig, GraftStreamingDataset
from graft.data.samplers import make_dataset_chooser, estimate_dataset_sizes

# --- Imports from the model and loss definitions ---
from graft.models.gnn_core import StatePropagator
from graft.models.step0 import StepZeroClamp
from graft.models.heads import MediatedHead, TrueSparseDirectHead
from graft.losses.distribution import sliced_wasserstein, mmd_rbf, energy_distance
from graft.losses.consistency import target_knockdown_consistency
from graft.losses.invariance import risk_extrapolation, irmv1_penalty
from graft.utils.common import seed_everything


# --- Initialize accumulators outside the loop ---
timings = {
    "data_fetch": [],
    "model_forward": [],
    "loss_backward": []
}


def set_seed(seed: int = 1337):
    # Consolidating seed setting
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

# --- Helper functions ---
def build_U(path: str, device: torch.device) -> torch.Tensor:
    arr = np.load(path, allow_pickle=False)
    return torch.tensor(arr, dtype=torch.float32, device=device)

def make_optimizer(params, cfg: Dict[str, Any]):
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 2e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    raise ValueError(f"Unknown optimizer {name}")

def pick_dist_fn(name: str):
    name = (name or "swd").lower()
    return {"swd": sliced_wasserstein, "sliced": sliced_wasserstein,
            "mmd": mmd_rbf, "rbf": mmd_rbf,
            "energy": energy_distance, "ed": energy_distance}.get(name, sliced_wasserstein)

def save_checkpoint(model_dict: Dict[str, nn.Module], opt: optim.Optimizer, step: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    
    state_dict = {name: module.state_dict() for name, module in model_dict.items()}
    
    ckpt = {
        "models": state_dict,
        "opt": opt.state_dict(),
        "step": step,
    }
    torch.save(ckpt, outdir / f"ckpt_{step:07d}.pt")


def make_prediction(tb, step0, head_med, head_dir, prop, U):    # Placeholder for prediction logic
    x0 = tb["xbar_ctrl"].mean(dim=1)  # Average across k neighbors -> shape (B, G)
    # define pre-perturbation state as mean embedding across k most similar controls
    z0 = tb["z_ctrl"].mean(dim=1) if tb["z_ctrl"].dim() == 3 else tb["z_ctrl"]  # (B, d)

    # Calculate clamp effectiveness based on PRE-perturbation state z_ctrl
    x_clamped_authoritative, eff = step0(x0, z0, tb["env_code"], tb["target_idx"])

    # Predict direct effects on other genes
    dx_dir = head_dir(z0, target_idx=tb["target_idx"], eff=eff)

    # Propagate state, now conditioned on the effectiveness of the initial hit
    z_ref = prop(z0, eff=eff, target_idx=tb["target_idx"], env_codes=tb["env_code"])

    # Predict downstream changes from the new state z_ref
    m = head_med(z_ref)
    dx_med = m @ U
    y_pred_downstream = x0 + dx_med + dx_dir

    # Surgically intervene to enforce the Step-0 clamp on the final prediction
    y_pred = y_pred_downstream.clone()
    mask = tb["target_idx"] >= 0
    if torch.any(mask):
        rows_to_update = torch.nonzero(mask, as_tuple=False).view(-1)
        cols_to_update = tb["target_idx"][mask]
        clamped_values = x_clamped_authoritative[rows_to_update, cols_to_update]
        y_pred[rows_to_update, cols_to_update] = clamped_values
    return y_pred, x_clamped_authoritative, dx_dir, dx_med, z_ref, eff


def compute_losses(dist_fn, y_pred, y_true, dx_dir, U_t, tb, w_dist, w_rex, w_cons, w_l1, w_orth, device):
    per_env_losses = []
    unique_envs_in_batch = torch.unique(tb["env_code"])
    for env_code in unique_envs_in_batch:
        mask = (tb["env_code"] == env_code)
        if mask.sum() > 0:
            loss = dist_fn(y_pred[mask], y_true[mask])
            per_env_losses.append(loss)
            
    loss_dist = w_dist * (torch.stack(per_env_losses).mean() if per_env_losses else torch.tensor(0.0, device=device))
    loss_rex  = w_rex  * (risk_extrapolation(per_env_losses) if len(per_env_losses) > 1 else torch.tensor(0.0, device=device))
    loss_cons = w_cons * target_knockdown_consistency(y_pred, y_true, tb["target_idx"], mode="mse")
    loss_l1   = w_l1   * dx_dir.abs().mean()
    loss_orth = torch.tensor(0.0, device=device)
    if w_orth > 0.0:
        loss_orth = w_orth * ((dx_dir @ U_t) ** 2).mean()
    total_loss = loss_dist + loss_rex + loss_cons + loss_l1 + loss_orth
    return total_loss, loss_dist, loss_rex, loss_cons, loss_l1, loss_orth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to train config YAML")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    train_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(int(train_cfg.get("seed", 1337)))

    # --- Data and Sampler Setup Refactoring ---
    # 1. Pre-calculate dataset sizes and IDs for sampler configuration.
    # This logic mimics the filtering performed inside GraftStreamingDataset.__init__
    # to ensure the sampler operates on the correct set of cells.
    print("[info] Pre-calculating dataset sizes for sampler...")
    index_all = pd.read_parquet(paths["index_parquet"])
    
    # Filter index by datasets present in the scVI reference AnnData
    scvi_reference_adata = ad.read_h5ad(paths["scvi_input_h5ad"], backed="r")
    valid_scvi_datasets = set(scvi_reference_adata.obs['dataset_id'].astype(str).unique())
    query_pool = index_all[index_all["dataset_id"].isin(valid_scvi_datasets)].copy()
    
    # Filter to non-controls if specified by config
    include_controls_in_query = bool(train_cfg.get("include_controls_in_query", False))
    if not include_controls_in_query and "is_control" in query_pool.columns:
        query_pool = query_pool[~query_pool["is_control"].astype(bool)]

    with open(paths["datasets_yaml"], "r") as f:
        _yaml = yaml.safe_load(f)
    _ds_map = _yaml.get("datasets", {})
    allowed_for_gnn = {str(k) for k, v in _ds_map.items()
                    if isinstance(v, dict) and bool(v.get("train_graft", True))}
    if len(allowed_for_gnn) == 0:
        raise ValueError("No datasets have train_graft: true in datasets.yaml")
    before = query_pool["dataset_id"].nunique()
    query_pool = query_pool[query_pool["dataset_id"].isin(allowed_for_gnn)].copy()
    after = query_pool["dataset_id"].nunique()
    if after == 0:
        raise ValueError("After applying train_graft filter, no queryable cells remain.")

    # Calculate sizes and get dataset IDs from the filtered query pool
    sizes = query_pool.groupby("dataset_id").size().to_dict()
    ds_ids = sorted(list(sizes.keys()))
    if not ds_ids:
        raise ValueError("No queryable cells found after filtering by control status and scVI reference datasets.")

    # 2. Create Sampler (Chooser)
    total_steps = int(train_cfg.get("epochs", 1)) * int(train_cfg.get("steps_per_epoch", 1000))
    chooser = make_dataset_chooser(
        dataset_ids=ds_ids, sizes=sizes,
        policy=str(train_cfg.get("sampler_policy", "weighted")),
        weight_mode=str(train_cfg.get("sampler_weight_mode", "sqrt")),
        steps=total_steps,
        seed=int(train_cfg.get("seed", 1337)),
        priority=train_cfg.get("priority"),
    )

    # 3. Initialize Streaming Dataset with Sampler
    ds_cfg = GraftStreamingConfig(
        datasets_yaml=paths["datasets_yaml"],
        index_parquet=paths["index_parquet"],
        gene_list_tsv=paths["gene_list_tsv"],
        scvi_model_dir=paths["scvi_model_dir"],
        scvi_input_h5ad=paths["scvi_input_h5ad"],
        control_index_dir=paths["control_index_dir"],
        control_z_npz=paths["control_z_npz"],
        control_xbar_npz=paths["control_xbar_npz"],
        batch_size=int(train_cfg.get("batch_size", 2048)),
        chunk_size=int(train_cfg.get("chunk_size", 50000)),
        k_controls=int(train_cfg.get("k_controls", 16)),
        oversample=int(train_cfg.get("oversample", 5)),
        match_within=str(train_cfg.get("match_within", "dataset")),
        forward_batch_size=int(train_cfg.get("forward_batch_size", 4096)),
        include_controls_in_query=include_controls_in_query,
        use_log1p_target=bool(train_cfg.get("use_log1p_target", False)),
    )
    # Pass the chooser to the dataset constructor as required by dataset.py refactoring
    ds = GraftStreamingDataset(ds_cfg, chooser)
    ds.env_codes = {dsid: i for i, dsid in enumerate(sorted(ds_ids))}
    n_envs = len(ds_ids)

    # --- Model Instantiation ---
    G = len(ds.gene_list)
    F = int(model_cfg["mediated"]["F"])
    prop: nn.Module = None
    step0: nn.Module = None
    head_med: nn.Module = None
    head_dir: nn.Module = None
    opt: optim.Optimizer = None
    
    U = build_U(paths["factor_U"], device=device)
    U_t = U.transpose(0, 1)  # (G, F)

    # --- Loss Configuration ---
    dist_fn = pick_dist_fn(loss_cfg["distribution"].get("type", "swd"))
    w_dist = float(loss_cfg["distribution"].get("weight", 1.0))
    w_rex  = float(loss_cfg["rex"].get("weight", 0.1))
    w_cons = float(loss_cfg["consistency"].get("weight", 0.5))
    w_l1   = float(loss_cfg["direct"].get("l1", 1e-4))
    w_orth = float(loss_cfg["direct"].get("orth_to_U", 0.0))

    outdir = Path(paths.get("output_dir", "./gnn_out"))
    outdir.mkdir(parents=True, exist_ok=True)
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 500))

    # --- Main Training Loop (Refactored Iterator Usage) ---
    print("Starting training loop...")
    # Create one persistent iterator from the dataset object
    data_iterator = iter(ds)
    
    for step in range(total_steps):
        start_fetch = time.time()
        try:
            batch = next(data_iterator)
        except StopIteration:
            print(f"Data iterator exhausted at step {step}. Re-initializing for next epoch.")
            # Re-create sampler for the next epoch if steps > number of batches in one epoch.
            current_epoch = (step // train_cfg.get("steps_per_epoch", 1000)) + 1
            new_seed = int(train_cfg.get("seed", 1337)) + current_epoch
            chooser = make_dataset_chooser(
                dataset_ids=ds_ids, sizes=sizes,
                policy=str(train_cfg.get("sampler_policy", "weighted")),
                weight_mode=str(train_cfg.get("sampler_weight_mode", "sqrt")),
                steps=total_steps - step, # Or keep total_steps if running for fixed epochs regardless
                seed=new_seed,
                priority=train_cfg.get("priority"),
            )
            ds.sampler = chooser # Update dataset's internal sampler
            data_iterator = iter(ds)
            batch = next(data_iterator)

        # Lazily create models and optimizer once we know z_dim from the first batch
        if prop is None:
            z_dim = batch["z_q"].shape[1]
            print(f"First batch received. Inferred z_dim={z_dim}, G={G}, F={F}")
            
            prop = StatePropagator(z_dim=z_dim, n_envs=n_envs, n_genes=G, **model_cfg["propagator"]).to(device)
            step0 = StepZeroClamp(z_dim=z_dim, n_labs=n_envs, **model_cfg["step0"]).to(device)
            head_med = MediatedHead(z_dim=z_dim, **model_cfg["mediated"]).to(device)
            head_dir = TrueSparseDirectHead(z_dim=z_dim, n_genes=G, **model_cfg["direct"]).to(device)
            # head_dir = SparseDirectHead(z_dim=z_dim, G=G, **model_cfg["direct"]).to(device)

            all_params = list(prop.parameters()) + list(step0.parameters()) + list(head_med.parameters()) + list(head_dir.parameters())
            opt = make_optimizer(all_params, cfg.get("optim", {}))
        
        prop.train(); step0.train(); head_med.train(); head_dir.train()
        tb = to_device(batch, device)
        y_true = tb["xbar_q"]
        timings["data_fetch"].append(time.time() - start_fetch)

        # Calculate x0 as the mean expression of the matched controls (k neighbors)
        start_forward = time.time()
        y_pred, x_clamped_authoritative, dx_dir, dx_med, z_ref, eff = make_prediction(tb, step0, head_med, head_dir, prop, U)
        timings["model_forward"].append(time.time() - start_forward)
        
        # --- Loss Calculation ---
        start_backward = time.time()

        total_loss, loss_dist, loss_rex, loss_cons, loss_l1, loss_orth = compute_losses(
            dist_fn, y_pred, y_true, dx_dir, U_t, tb, w_dist, w_rex, w_cons, w_l1, w_orth, device
        )

        opt.zero_grad()
        total_loss.backward()
        all_params = list(prop.parameters()) + list(step0.parameters()) + list(head_med.parameters()) + list(head_dir.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        opt.step()
        timings["loss_backward"].append(time.time() - start_backward)

        step_index = step + 1 # for 1-based indexing in logs
        if step_index % log_every == 0:
            log_data = {
                "step": step_index, "dataset": batch["dataset_id"], "loss": float(total_loss.item()),
                "dist": float(loss_dist.item()), "rex": float(loss_rex.item()),
                "cons": float(loss_cons.item()),
                "l1": float(loss_l1.item()), "orth": float(loss_orth.item()),
            }
            print(json.dumps(log_data))
            del log_data

        if step_index % ckpt_every == 0:
            model_dict = {"prop": prop, "step0": step0, "head_med": head_med, "head_dir": head_dir}
            save_checkpoint(model_dict, opt, step_index, outdir)
            del model_dict
        
        # Cleanup
        del batch, tb, y_true, y_pred, dx_dir, dx_med, z_ref, eff
        del total_loss, loss_dist, loss_rex, loss_cons, loss_l1, loss_orth
        if step % 50 == 0:
            torch.cuda.empty_cache()
    
    # Final save
    step_index = step + 1
    model_dict = {"prop": prop, "step0": step0, "head_med": head_med, "head_dir": head_dir}
    save_checkpoint(model_dict, opt, step_index, outdir)
    print(f"[done] total steps: {step_index}, final models saved to {outdir}")

    if timings["data_fetch"]:
        avg_fetch_time = sum(timings["data_fetch"]) / len(timings["data_fetch"])
        avg_forward_time = sum(timings["model_forward"]) / len(timings["model_forward"])
        avg_backward_time = sum(timings["loss_backward"]) / len(timings["loss_backward"])

        print(f"\nAverage time per step:")
        print(f"  Data Fetching & Prep: {avg_fetch_time:.4f} seconds")
        print(f"  Model Forward Pass:   {avg_forward_time:.4f} seconds")
        print(f"  Loss & Backward Pass: {avg_backward_time:.4f} seconds")

if __name__ == "__main__":
    main()
