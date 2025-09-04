#!/usr/bin/env python3
"""
train_gnn.py (streaming)
========================

Trainer wired to the new streaming data pipeline:

- Streams raw H5AD per-dataset in backed mode via GraftStreamingDataset
- Computes scVI z and normalized x̄ on the fly per mini-batch
- Matches controls via a prebuilt ANN index over control z
- Feeds batches into the gated-MLP "GNN-ish" core model
- Optimizes a simple consistency loss by default, with hooks for invariance / distribution

Config YAML (example):
----------------------
paths:
  datasets_yaml: artifacts/datasets.yaml
  index_parquet: artifacts/cell_index.parquet
  gene_list_tsv: artifacts/gene_list.tsv
  scvi_model_dir: artifacts/scvi_K562
  control_index_dir: artifacts/control_index
  control_z_npz: artifacts/scvi_z_controls.npz
  output_dir: artifacts/gnn_runs/run1

training:
  epochs: 5
  steps_per_epoch: 2000
  batch_size: 2048
  chunk_size: 50000
  lr: 1.0e-3
  weight_decay: 0.0
  sampler_policy: "weighted"       # "balanced" | "weighted"
  sampler_weight_mode: "sqrt"      # "uniform" | "count" | "sqrt"
  seed: 1337
  log_every: 50
  ckpt_every: 500

loss:
  w_consistency: 1.0
  w_invariance: 0.0
  w_distribution: 0.0
  distribution: "energy"           # "energy" | "swd"

model:
  dim_z: 32                        # inferred at runtime if None
  hidden: 256
  depth: 2
  dropout: 0.0

Notes:
- Invariance and distribution losses are optional; start with consistency for smoke tests.
- The dataset enforces environment = dataset_id. The sampler picks which dataset to draw next.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from graft.data.dataset import GraftStreamingConfig, GraftStreamingDataset
from graft.data.samplers import make_dataset_chooser, estimate_dataset_sizes

from graft.losses.distribution import energy_distance, sliced_wasserstein_distance
from graft.losses.invariance import rex_penalty, irm_penalty


# Core model
from graft.models.gnn_core import GraftCore


def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            dtype = torch.float32 if v.dtype.kind in ("f",) else torch.int64 if v.dtype.kind in ("i", "u", "b") else None
            if dtype is None:
                continue
            out[k] = torch.as_tensor(v, dtype=dtype, device=device)
    return out


def build_model(dim_z: int, dim_g: int, cfg_model: Dict[str, Any], device: torch.device) -> nn.Module:
    model = GraftCore(
        dim_z=dim_z,
        dim_g=dim_g,
        hidden=int(cfg_model.get("hidden", 256)),
        depth=int(cfg_model.get("depth", 2)),
        dropout=float(cfg_model.get("dropout", 0.0)),
    )
    return model.to(device)


def compute_losses(
    pred_xbar: torch.Tensor,
    true_xbar: torch.Tensor,
    env_code: torch.Tensor,
    loss_cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Compose total loss from consistency + optional invariance/distribution."""
    losses: Dict[str, torch.Tensor] = {}
    w_cons = float(loss_cfg.get("w_consistency", 1.0))
    w_inv = float(loss_cfg.get("w_invariance", 0.0))
    w_dist = float(loss_cfg.get("w_distribution", 0.0))
    dist_kind = str(loss_cfg.get("distribution", "energy"))

    # Consistency: L1
    l_cons = torch.mean(torch.abs(pred_xbar - true_xbar))
    losses["consistency"] = l_cons

    # Invariance (REx as default if available, otherwise skip)
    if w_inv > 0:
        # Group by env within the batch
        envs = env_code.detach().cpu().numpy()
        uniq = np.unique(envs)
        per_env = []
        for e in uniq:
            m = (env_code == int(e)).float().view(-1, 1)
            # mean absolute error per env
            num = torch.sum(m * torch.abs(pred_xbar - true_xbar))
            den = torch.clamp(torch.sum(m), min=1.0)
            per_env.append(num / den)
        if per_env:
            per_env_losses = torch.stack(per_env, dim=0)
            l_inv = rex_penalty(per_env_losses)
            losses["invariance"] = l_inv
        else:
            losses["invariance"] = torch.tensor(0.0, device=pred_xbar.device)
    else:
        losses["invariance"] = torch.tensor(0.0, device=pred_xbar.device)

    # Distributional
    if w_dist > 0:
        if dist_kind == "energy":
            l_dist = energy_distance(pred_xbar, true_xbar)
        elif dist_kind == "swd":
            l_dist = sliced_wasserstein_distance(pred_xbar, true_xbar, n_projections=64)
        else:
            l_dist = torch.tensor(0.0, device=pred_xbar.device)
        losses["distribution"] = l_dist
    else:
        losses["distribution"] = torch.tensor(0.0, device=pred_xbar.device)

    total = w_cons * losses["consistency"] + w_inv * losses["invariance"] + w_dist * losses["distribution"]
    losses["total"] = total
    return losses


def save_checkpoint(model: nn.Module, opt: optim.Optimizer, step: int, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(train_cfg.get("seed", 1337)))

    # Build streaming dataset
    ds_cfg = GraftStreamingConfig(
        datasets_yaml=paths["datasets_yaml"],
        index_parquet=paths["index_parquet"],
        gene_list_tsv=paths["gene_list_tsv"],
        scvi_model_dir=paths["scvi_model_dir"],
        control_index_dir=paths["control_index_dir"],
        control_z_npz=paths["control_z_npz"],
        batch_size=int(train_cfg.get("batch_size", 2048)),
        chunk_size=int(train_cfg.get("chunk_size", 50000)),
        k_controls=int(train_cfg.get("k_controls", 16)) if "k_controls" in train_cfg else 16,
        oversample=int(train_cfg.get("oversample", 5)) if "oversample" in train_cfg else 5,
        match_within=str(train_cfg.get("match_within", "dataset")),
        forward_batch_size=int(train_cfg.get("forward_batch_size", 4096)) if "forward_batch_size" in train_cfg else 4096,
        include_controls_in_query=bool(train_cfg.get("include_controls_in_query", False)),
    )
    ds = GraftStreamingDataset(ds_cfg)

    # Infer dims
    dim_g = len(ds.gene_list)
    # dim_z is unknown until we pull first batch; we can defer model build after first batch

    # Sampler / chooser over dataset_ids
    ds_ids = ds.get_dataset_ids()
    sizes = estimate_dataset_sizes(ds.by_ds)
    total_steps = int(train_cfg.get("epochs", 1)) * int(train_cfg.get("steps_per_epoch", 1000))
    chooser = make_dataset_chooser(
        dataset_ids=ds_ids,
        sizes=sizes,
        policy=str(train_cfg.get("sampler_policy", "weighted")),
        weight_mode=str(train_cfg.get("sampler_weight_mode", "sqrt")),
        steps=total_steps,
        shuffle_each_epoch=bool(train_cfg.get("shuffle_each_epoch", False)),
        seed=int(train_cfg.get("seed", 1337)),
    )

    # Optimizer will be created after we see the first batch (to know dim_z)
    model: nn.Module = None  # type: ignore
    opt: optim.Optimizer = None  # type: ignore

    outdir = Path(paths.get("output_dir", "./gnn_out"))
    outdir.mkdir(parents=True, exist_ok=True)

    step = 0
    log_every = int(train_cfg.get("log_every", 50))
    ckpt_every = int(train_cfg.get("ckpt_every", 500))

    # Training loop
    for dsid in chooser:
        # Ask dataset to yield ONE mini-batch from this dataset_id
        batch = None
        for b in ds.iter_batches([dsid]):
            batch = b
            break
        if batch is None:
            continue

        # Lazily create model/optimizer once we know dim_z
        if model is None:
            dim_z = batch["z_q"].shape[1]
            model = build_model(dim_z=dim_z, dim_g=dim_g, cfg_model=model_cfg, device=device)
            opt = optim.Adam(model.parameters(),
                             lr=float(train_cfg.get("lr", 1e-3)),
                             weight_decay=float(train_cfg.get("weight_decay", 0.0)))

        model.train()
        tb = to_device(batch, device)
        z_q = tb["z_q"]            # (B, d)
        xbar_q = tb["xbar_q"]      # (B, G)
        # target_idx, env_code etc. are available in tb if needed by your model

        # Forward
        if hasattr(model, "forward"):
            pred_xbar = model(z_q)  # GraftCore maps z->gene residuals or direct xbar; adapt as needed
            if isinstance(pred_xbar, tuple):
                pred_xbar = pred_xbar[0]
        else:
            raise RuntimeError("Model has no forward().")

        # Loss
        losses = compute_losses(pred_xbar, xbar_q, tb.get("env_code", torch.zeros(z_q.size(0), dtype=torch.long, device=device)), loss_cfg)

        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        step += 1
        if step % log_every == 0:
            msg = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
            msg["step"] = step
            msg["dataset"] = dsid
            print(json.dumps(msg))

        if step % ckpt_every == 0:
            save_checkpoint(model, opt, step, outdir)

        # Stop after total_steps
        if step >= total_steps:
            break

    # final checkpoint
    save_checkpoint(model, opt, step, outdir)
    print(f"[done] total steps: {step}, saved to {outdir}")

if __name__ == "__main__":
    main()
