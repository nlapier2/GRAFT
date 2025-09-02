#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, math
from typing import Dict, Any, Optional, List

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from graft.models.gnn_core import StatePropagator
from graft.models.step0 import StepZeroClamp
from graft.models.heads import MediatedHead, SparseDirectHead

from graft.losses.distribution import sliced_wasserstein, mmd_rbf, energy_distance
from graft.losses.consistency import target_knockdown_consistency
from graft.losses.invariance import risk_extrapolation, irmv1_penalty

from graft.utils.common import read_parquet_indexed, encode_categories, seed_everything

try:
    import anndata as ad
    import scvi
except Exception:
    scvi = None
    ad = None


class PerDatasetDecoder:
    def __init__(self, model_dir: str, transform_batch: Optional[str] = None):
        if scvi is None:
            raise RuntimeError("scvi/anndata not available.")
        self.model_dir = model_dir
        self.transform_batch = transform_batch
        self.cache: Dict[str, tuple] = {}

    def _get(self, dataset_id: str, h5ad_path: str):
        key = str(dataset_id)
        if key in self.cache:
            return self.cache[key]
        adata = ad.read_h5ad(h5ad_path)
        model = scvi.model.SCVI.load(self.model_dir, adata=None).load_query_data(adata)
        model.eval()
        self.cache[key] = (adata, model)
        return self.cache[key]

    @torch.no_grad()
    def xbar(self, dataset_id: str, h5ad_path: str, row_idx: np.ndarray, device: torch.device) -> torch.Tensor:
        adata, model = self._get(dataset_id, h5ad_path)
        xbar = model.get_normalized_expression(adata[row_idx], transform_batch=self.transform_batch)
        return torch.tensor(xbar.values, dtype=torch.float32, device=device)


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


def pick_dist(name: str):
    name = (name or "swd").lower()
    return {"swd": sliced_wasserstein, "sliced": sliced_wasserstein,
            "mmd": mmd_rbf, "rbf": mmd_rbf,
            "energy": energy_distance, "ed": energy_distance}.get(name, sliced_wasserstein)


def train(cfg: Dict[str, Any]) -> None:
    seed_everything(int(cfg.get("seed", 13)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = int(cfg["data"]["n_genes"])
    F = int(cfg["model"]["mediated"]["F"])
    z_dim = int(cfg["model"]["z_dim"])

    datasets_cfg = cfg["paths"]["datasets"]
    index_parquet = cfg["paths"]["index_parquet"]
    df_index = read_parquet_indexed(index_parquet)
    ds_ids = [d["id"] for d in datasets_cfg]
    ds_id_to_code = {ds: i for i, ds in enumerate(ds_ids)}
    n_envs = len(ds_ids)

    decoder = PerDatasetDecoder(model_dir=cfg["paths"]["scvi_model_dir"],
                                transform_batch=cfg["data"].get("transform_batch"))

    U = build_U(cfg["paths"]["factor_U"], device=device)   # (F, G)
    U_t = U.t().contiguous()                               # (G, F)

    prop = StatePropagator(
        z_dim=z_dim,
        hidden=int(cfg["model"]["propagator"].get("hidden", 256)),
        layers=int(cfg["model"]["propagator"].get("layers", 2)),
        steps=int(cfg["model"]["propagator"].get("steps", 2)),
        dropout=float(cfg["model"]["propagator"].get("dropout", 0.0)),
        use_env_film=bool(cfg["model"]["propagator"].get("use_env_film", True)),
        use_target_cond=bool(cfg["model"]["propagator"].get("use_target_cond", True)),
        target_embed_dim=int(cfg["model"]["propagator"].get("target_embed_dim", 32)),
        n_envs=n_envs,
        n_genes=G,
    ).to(device)

    step0 = StepZeroClamp(
        z_dim=z_dim,
        n_labs=n_envs,
        hidden=int(cfg["model"]["step0"].get("hidden", 64)),
        init_eff=float(cfg["model"]["step0"].get("init_eff", 0.9)),
        mode=str(cfg["model"]["step0"].get("mode", "down")),
    ).to(device)

    head_med = MediatedHead(
        z_dim=z_dim,
        F=F,
        hidden=int(cfg["model"]["mediated"].get("hidden", 256)),
        use_factor_feats=bool(cfg["model"]["mediated"].get("use_factor_feats", False)),
        a_dim=int(cfg["model"]["mediated"].get("a_dim", 0)),
        nonneg=bool(cfg["model"]["mediated"].get("nonneg", True)),
        dropout=float(cfg["model"]["mediated"].get("dropout", 0.0)),
    ).to(device)

    head_dir = SparseDirectHead(
        z_dim=z_dim,
        G=G,
        hidden=int(cfg["model"]["direct"].get("hidden", 256)),
        dropout=float(cfg["model"]["direct"].get("dropout", 0.0)),
        bound=None,
    ).to(device)

    opt = make_optimizer(
        list(prop.parameters()) + list(step0.parameters()) + list(head_med.parameters()) + list(head_dir.parameters()),
        cfg.get("optim", {})
    )

    dist_fn = pick_dist(cfg["loss"]["distribution"].get("type", "swd"))
    w_dist = float(cfg["loss"]["distribution"].get("weight", 1.0))
    w_rex  = float(cfg["loss"]["rex"].get("weight", 0.1))
    w_irm  = float(cfg["loss"]["irm"].get("weight", 0.0))
    irm_target_only = bool(cfg["loss"]["irm"].get("use_target_only", True))
    w_cons = float(cfg["loss"]["consistency"].get("weight", 0.5))
    w_l1   = float(cfg["loss"]["direct"].get("l1", 1e-4))
    w_orth = float(cfg["loss"]["direct"].get("orth_to_U", 0.0))

    parts = []
    for ds in datasets_cfg:
        z_df = read_parquet_indexed(ds["z_parquet"])
        z_df["__dataset_id__"] = ds["id"]
        parts.append(z_df)
    Z = pd.concat(parts, axis=0, join="inner")
    Z = Z.loc[Z.index.intersection(read_parquet_indexed(index_parquet).index)].copy()

    train_mask = read_parquet_indexed(index_parquet).loc[Z.index].get("is_train", True).values if "is_train" in read_parquet_indexed(index_parquet).columns else np.ones(len(Z), dtype=bool)
    Z_train = Z.loc[train_mask]

    batch_size = int(cfg.get("batch_size", 1024))
    epochs = int(cfg.get("epochs", 10))

    def df_to_tensor_batch(df_batch: pd.DataFrame) -> Dict[str, torch.Tensor]:
        z_cols = [c for c in df_batch.columns if c.startswith("z")]
        z = torch.tensor(df_batch[z_cols].values, dtype=torch.float32, device=device)
        env = torch.tensor([ds_id_to_code[s] for s in df_batch["__dataset_id__"].values], dtype=torch.long, device=device)
        idx_df = read_parquet_indexed(index_parquet).loc[df_batch.index]
        tgt = torch.tensor(idx_df.get("target_gene_idx", pd.Series(-1, index=idx_df.index)).fillna(-1).astype(int).values, dtype=torch.long, device=device)
        return {"z": z, "env": env, "target_idx": tgt, "cell_ids": df_batch.index.values}

    def fetch_xbars(df_batch: pd.DataFrame) -> (torch.Tensor, torch.Tensor):
        x0_list, yt_list = [], []
        for ds in datasets_cfg:
            msk = (df_batch["__dataset_id__"].values == ds["id"])
            if not msk.any(): 
                continue
            cell_ids = df_batch.index.values[msk]
            adata, model = decoder._get(ds["id"], ds["h5ad"])
            pos = {k: i for i, k in enumerate(adata.obs_names.tolist())}
            rows = np.array([pos[c] for c in cell_ids], dtype=int)
            yt = decoder.xbar(ds["id"], ds["h5ad"], rows, device)
            if "knn_parquet" in ds and ds["knn_parquet"] and os.path.exists(ds["knn_parquet"]):
                knn = read_parquet_indexed(ds["knn_parquet"])
                ctrl_ids = knn.loc[cell_ids, "ctrl_id"].values
                rows0 = np.array([pos[c] for c in ctrl_ids], dtype=int)
                x0 = decoder.xbar(ds["id"], ds["h5ad"], rows0, device)
            else:
                x0 = yt.clone()
            x0_list.append(x0); yt_list.append(yt)
        return torch.cat(x0_list, 0), torch.cat(yt_list, 0)

    for epoch in range(1, epochs + 1):
        prop.train(); step0.train(); head_med.train(); head_dir.train()
        idx = np.arange(len(Z_train)); np.random.shuffle(idx)
        n_steps = int(math.ceil(len(idx) / batch_size))
        sums = dict(dist=0.0, rex=0.0, irm=0.0, cons=0.0, l1=0.0, orth=0.0)

        for s in range(n_steps):
            bs = idx[s*batch_size:(s+1)*batch_size]
            df_b = Z_train.iloc[bs]
            batch = df_to_tensor_batch(df_b)

            x0, y_true = fetch_xbars(df_b)
            z_ref = prop(batch["z"], target_idx=batch["target_idx"], env_codes=batch["env"])
            x_clamp, eff = step0(x0, z_ref, batch["env"], batch["target_idx"])
            m = head_med(z_ref); dx_med = m @ U_t
            dx_dir = head_dir(z_ref)
            y_pred = x_clamp + dx_med + dx_dir

            per_env = []
            for ds in ds_ids:
                msk = (df_b["__dataset_id__"].values == ds)
                if not msk.any(): 
                    continue
                per_env.append(pick_dist(cfg["loss"]["distribution"]["type"])(y_pred[msk], y_true[msk]))

            loss_dist = w_dist * (torch.stack(per_env).mean() if per_env else y_pred.new_tensor(0.0))
            loss_rex  = w_rex  * (risk_extrapolation(per_env, unbiased=False) if len(per_env) > 1 else y_pred.new_tensor(0.0))
            loss_irm  = w_irm  * irmv1_penalty(y_pred, y_true, batch["env"], use_target_only=irm_target_only, target_idx=batch["target_idx"]) if w_irm > 0 else y_pred.new_tensor(0.0)
            loss_cons = w_cons * target_knockdown_consistency(y_pred, y_true, batch["target_idx"], mode="mse")
            loss_l1   = w_l1   * dx_dir.abs().mean()
            loss_orth = y_pred.new_tensor(0.0)
            if w_orth > 0.0:
                loss_orth = w_orth * ((dx_dir @ U) ** 2).mean()

            loss = loss_dist + loss_rex + loss_irm + loss_cons + loss_l1 + loss_orth

            optim_params = list(prop.parameters()) + list(step0.parameters()) + list(head_med.parameters()) + list(head_dir.parameters())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, 5.0)
            opt.step()

            sums["dist"] += float(loss_dist.item())
            sums["rex"]  += float(loss_rex.item())
            sums["irm"]  += float(loss_irm.item())
            sums["cons"] += float(loss_cons.item())
            sums["l1"]   += float(loss_l1.item())
            sums["orth"] += float(loss_orth.item())

        denom = max(n_steps, 1)
        print(f"[epoch {epoch:03d}] dist={sums['dist']/denom:.4f} rex={sums['rex']/denom:.4f} irm={sums['irm']/denom:.4f} cons={sums['cons']/denom:.4f} l1={sums['l1']/denom:.4f} orth={sums['orth']/denom:.4f}")
    print("Training finished.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/gnn_v1.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
