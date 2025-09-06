#!/usr/bin/env python3
"""
train_gnn.py (reconciled)
=========================

This script combines the modern streaming data pipeline with the original,
correct multi-component model architecture and detailed loss calculations.

- Streams data via GraftStreamingDataset and samples via make_dataset_chooser.
- Instantiates the full model: Propagator -> StepZeroClamp -> Mediated/Direct Heads.
- Computes the complete, multi-part loss function as defined in the original script.
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

# --- Imports from the new data pipeline ---
from graft.data.dataset import GraftStreamingConfig, GraftStreamingDataset
from graft.data.samplers import make_dataset_chooser, estimate_dataset_sizes

# --- Imports from the original, correct model and loss definitions ---
from graft.models.gnn_core import StatePropagator
from graft.models.step0 import StepZeroClamp
from graft.models.heads import MediatedHead, SparseDirectHead
from graft.losses.distribution import sliced_wasserstein, mmd_rbf, energy_distance
from graft.losses.consistency import target_knockdown_consistency
from graft.losses.invariance import risk_extrapolation, irmv1_penalty
from graft.utils.common import seed_everything


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

# --- Helper functions from the old script ---
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
    data_cfg = cfg.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(int(train_cfg.get("seed", 1337)))

    # --- New Streaming Dataset Setup ---
    ds_cfg = GraftStreamingConfig(
        datasets_yaml=paths["datasets_yaml"],
        index_parquet=paths["index_parquet"],
        gene_list_tsv=paths["gene_list_tsv"],
        scvi_model_dir=paths["scvi_model_dir"],
        scvi_input_h5ad=paths["scvi_input_h5ad"],
        control_index_dir=paths["control_index_dir"],
        control_z_npz=paths["control_z_npz"],
        batch_size=int(train_cfg.get("batch_size", 2048)),
        chunk_size=int(train_cfg.get("chunk_size", 50000)),
        k_controls=int(train_cfg.get("k_controls", 16)),
        oversample=int(train_cfg.get("oversample", 5)),
        match_within=str(train_cfg.get("match_within", "dataset")),
        forward_batch_size=int(train_cfg.get("forward_batch_size", 4096)),
        include_controls_in_query=bool(train_cfg.get("include_controls_in_query", False)),
    )
    ds = GraftStreamingDataset(ds_cfg)
    ds_ids = ds.get_dataset_ids()
    ds_id_to_code = {dsid: i for i, dsid in enumerate(sorted(ds_ids))}
    n_envs = len(ds_ids)

    # --- Restored Model Instantiation ---
    G = len(ds.gene_list)
    F = int(model_cfg["mediated"]["F"])
    # Defer z_dim until first batch
    
    prop: nn.Module = None
    step0: nn.Module = None
    head_med: nn.Module = None
    head_dir: nn.Module = None
    opt: optim.Optimizer = None
    
    U = build_U(paths["factor_U"], device=device)
    U_t = U.t().contiguous()

    # --- Restored Loss Configuration ---
    dist_fn = pick_dist_fn(loss_cfg["distribution"].get("type", "swd"))
    w_dist = float(loss_cfg["distribution"].get("weight", 1.0))
    w_rex  = float(loss_cfg["rex"].get("weight", 0.1))
    w_irm  = float(loss_cfg["irm"].get("weight", 0.0))
    irm_target_only = bool(loss_cfg["irm"].get("use_target_only", True))
    w_cons = float(loss_cfg["consistency"].get("weight", 0.5))
    w_l1   = float(loss_cfg["direct"].get("l1", 1e-4))
    w_orth = float(loss_cfg["direct"].get("orth_to_U", 0.0))

    # --- New Sampler Setup ---
    sizes = estimate_dataset_sizes(ds.by_ds)
    total_steps = int(train_cfg.get("epochs", 1)) * int(train_cfg.get("steps_per_epoch", 1000))
    chooser = make_dataset_chooser(
        dataset_ids=ds_ids, sizes=sizes,
        policy=str(train_cfg.get("sampler_policy", "weighted")),
        weight_mode=str(train_cfg.get("sampler_weight_mode", "sqrt")),
        steps=total_steps,
        seed=int(train_cfg.get("seed", 1337)),
    )

    outdir = Path(paths.get("output_dir", "./gnn_out"))
    outdir.mkdir(parents=True, exist_ok=True)
    step = 0
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 500))

    # --- Main Training Loop (New Data Loading + Old Model/Loss Logic) ---
    print("Starting training loop...")
    for dsid in chooser:
        batch = next(iter(ds.iter_batches([dsid])), None)
        if batch is None:
            continue

        # Lazily create models and optimizer once we know z_dim from the first batch
        if prop is None:
            z_dim = batch["z_q"].shape[1]
            print(f"First batch received. Inferred z_dim={z_dim}, G={G}, F={F}")
            
            prop = StatePropagator(z_dim=z_dim, n_envs=n_envs, n_genes=G, **model_cfg["propagator"]).to(device)
            step0 = StepZeroClamp(z_dim=z_dim, n_labs=n_envs, **model_cfg["step0"]).to(device)
            head_med = MediatedHead(z_dim=z_dim, **model_cfg["mediated"]).to(device)
            head_dir = SparseDirectHead(z_dim=z_dim, G=G, **model_cfg["direct"]).to(device)

            all_params = list(prop.parameters()) + list(step0.parameters()) + list(head_med.parameters()) + list(head_dir.parameters())
            opt = make_optimizer(all_params, cfg.get("optim", {}))
        
        prop.train(); step0.train(); head_med.train(); head_dir.train()
        tb = to_device(batch, device)
        y_true = tb["xbar_q"]

        # Calculate x0 as the mean expression of the matched controls (k neighbors)
        if "xbar_ctrl" in tb and tb["xbar_ctrl"].shape[1] > 0:
            x0 = tb["xbar_ctrl"].mean(dim=1)  # Average across k neighbors -> shape (B, G)
        else:
            x0 = y_true # Fallback if no controls were found for this batch

        z_ref = prop(tb["z_q"], target_idx=tb["target_idx"], env_codes=tb["env_code"])
        x_clamp, _ = step0(x0, z_ref, tb["env_code"], tb["target_idx"])

        m = head_med(z_ref)
        dx_med = m @ U
        dx_dir = head_dir(z_ref)
        y_pred = x_clamp + dx_med + dx_dir
        
        # --- Restored Loss Calculation ---
        per_env_losses = []
        unique_envs_in_batch = torch.unique(tb["env_code"])
        for env_code in unique_envs_in_batch:
            mask = (tb["env_code"] == env_code)
            if mask.sum() > 0:
                loss = dist_fn(y_pred[mask], y_true[mask])
                per_env_losses.append(loss)
        
        loss_dist = w_dist * (torch.stack(per_env_losses).mean() if per_env_losses else torch.tensor(0.0, device=device))
        loss_rex  = w_rex  * (risk_extrapolation(per_env_losses) if len(per_env_losses) > 1 else torch.tensor(0.0, device=device))
        loss_irm  = w_irm  * irmv1_penalty(y_pred, y_true, tb["env_code"], use_target_only=irm_target_only, target_idx=tb["target_idx"]) if w_irm > 0 else torch.tensor(0.0, device=device)
        loss_cons = w_cons * target_knockdown_consistency(y_pred, y_true, tb["target_idx"], mode="mse")
        loss_l1   = w_l1   * dx_dir.abs().mean()
        loss_orth = torch.tensor(0.0, device=device)
        if w_orth > 0.0:
            loss_orth = w_orth * ((dx_dir @ U) ** 2).mean()

        total_loss = loss_dist + loss_rex + loss_irm + loss_cons + loss_l1 + loss_orth

        opt.zero_grad()
        total_loss.backward()
        all_params = list(prop.parameters()) + list(step0.parameters()) + list(head_med.parameters()) + list(head_dir.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, 5.0)
        opt.step()

        step += 1
        if step % log_every == 0:
            log_data = {
                "step": step, "dataset": dsid, "loss": float(total_loss.item()),
                "dist": float(loss_dist.item()), "rex": float(loss_rex.item()),
                "irm": float(loss_irm.item()), "cons": float(loss_cons.item()),
                "l1": float(loss_l1.item()), "orth": float(loss_orth.item()),
            }
            print(json.dumps(log_data))

        if step % ckpt_every == 0:
            model_dict = {"prop": prop, "step0": step0, "head_med": head_med, "head_dir": head_dir}
            save_checkpoint(model_dict, opt, step, outdir)

        if step >= total_steps:
            break

    model_dict = {"prop": prop, "step0": step0, "head_med": head_med, "head_dir": head_dir}
    save_checkpoint(model_dict, opt, step, outdir)
    print(f"[done] total steps: {step}, final models saved to {outdir}")

if __name__ == "__main__":
    main()