#!/usr/bin/env python3
"""
Train GRAFT GNN (dataset-centric v1, per-dataset shards)

Key changes vs. earlier skeleton:
- No merged scvi_input H5AD. We keep **per-dataset** shards:
    * z parquet per dataset (from encode_query_z.py)
    * original dataset H5AD path (to decode normalized means on the fly via scVI.load_query_data)
    * optional precomputed kNN neighbors per dataset (from build_knn_controls.py)
- The trainer aggregates z across datasets into a single index=cell_id DataFrame,
  but decodes x̄ directly from each dataset's H5AD chunk-by-chunk at batch time.

Config shape (see YAML example):
paths:
  index_parquet: artifacts/cell_index.parquet
  scvi_model_dir: artifacts/scvi_K562_controls/scvi_K562
  datasets:
    - id: K562_ReplogleWeissman
      z_parquet: artifacts/z_by_dataset/K562_ReplogleWeissman.parquet
      h5ad: /data/K562_ReplogleWeissman.h5ad
      knn_parquet: artifacts/knn_by_dataset/K562_ReplogleWeissman.knn.parquet  # optional
    - id: DixitRegev
      z_parquet: artifacts/z_by_dataset/DixitRegev.parquet
      h5ad: /data/DixitRegev.h5ad

  train_perturb: artifacts/train_rows.parquet
  val_perturb: artifacts/val_rows.parquet

model:
  n_programs: 256
  hidden: 256
  gnn_layers: 2
  use_factor_features: false
  transform_batch: null          # optional reference batch category for decoding

training:
  mode: distribution             # or "paired_controls"
  batch_size: 2048
  steps_per_epoch: 200
  max_epochs: 50
  lr: 1.0e-3
  dist_loss_weight: 1.0
  kd_consistency_weight: 0.5
  rex_weight: 0.1
  l1_direct_weight: 0.0
  interleave_global: false
  p_global: 0.2
  seed: 13

Notes:
- For paired_controls mode, if a dataset has knn_parquet, we use rank-0 neighbors;
  if missing, we fallback to random in-dataset controls.
- Decoding x̄ uses scVI.load(model_dir) once, then load_query_data on the needed subset
  of a dataset H5AD per batch. To speed this up further, add a query cache later.
"""

from __future__ import annotations
import os, sys, math, argparse, gc
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
import scvi
import yaml

from graft.data.dataset import GraftDataset
from graft.data.samplers import DatasetBalancedSampler, InterleavedGlobalSampler
from graft.models.step0 import StepZeroClamp
from graft.models.gnn_core import StatePropagator
from graft.models.heads import MediatedHead, SparseDirectHead
from graft.losses.distribution import sliced_wasserstein
from graft.losses.invariance import risk_extrapolation
from graft.losses.consistency import target_knockdown_consistency


# ------------------ Helpers ------------------
def load_U(path: str) -> torch.Tensor:
    U = np.load(path).astype(np.float32)
    if U.shape[0] < U.shape[1]:
        U = U.T  # (G, F)
    return torch.from_numpy(U)


def _resolve_index(df: pd.DataFrame) -> pd.DataFrame:
    if "cell_id" in df.columns:
        df = df.set_index("cell_id", drop=True)
    return df


def load_multi_z(datasets_cfg, needed_cell_ids: pd.Index | None = None) -> pd.DataFrame:
    """
    Concatenate per-dataset z parquets into a single DataFrame (index=cell_id).
    If needed_cell_ids is provided, filter rows to that set for efficiency.
    """
    frames = []
    for entry in datasets_cfg:
        zp = entry["z_parquet"]
        Z = pd.read_parquet(zp)
        Z = _resolve_index(Z)
        if needed_cell_ids is not None:
            keep = Z.index.intersection(needed_cell_ids)
            Z = Z.loc[keep]
        frames.append(Z.astype("float32"))
    Zall = pd.concat(frames, axis=0)
    Zall = Zall.loc[~Zall.index.duplicated(keep="first")]
    return Zall


class PerDatasetDecoder:
    """
    Minimal decoder that uses a control-trained scVI model to decode normalized means
    for arbitrary subsets of cells from per-dataset H5AD shards.
    """
    def __init__(self, scvi_model_dir: str, datasets_cfg, transform_batch=None, forward_batch_size: int = 4096):
        self.base_model = scvi.model.SCVI.load(scvi_model_dir, adata=None)
        self.ds_to_h5 = {d["id"]: d["h5ad"] for d in datasets_cfg}
        self.transform_batch = transform_batch
        self.forward_bs = forward_batch_size

    def get_xbar(self, ds_id: str, cell_ids: np.ndarray) -> np.ndarray:
        """
        Decode normalized means for given cell_ids belonging to dataset ds_id.
        We load the dataset H5AD, subset to requested ids, attach via load_query_data, then decode.
        """
        # Load dataset adata and subset to cell_ids
        A = ad.read_h5ad(self.ds_to_h5[ds_id])
        # Align and subset
        # Ensure requested ids exist
        ids_set = set(A.obs_names.astype(str))
        have = np.array([cid for cid in cell_ids if cid in ids_set], dtype=str)
        if len(have) == 0:
            # nothing to decode
            return np.zeros((0, A.n_vars), dtype=np.float32)
        Aq = A[have].copy()
        qmodel = self.base_model.load_query_data(Aq, inplace=False)
        X = qmodel.get_normalized_expression(transform_batch=self.transform_batch, batch_size=self.forward_bs)
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32, copy=False)
        else:
            X = X.astype(np.float32).values
        del qmodel, A, Aq
        gc.collect()
        return X


def load_knn_map(datasets_cfg) -> dict[str, pd.DataFrame]:
    """
    Load optional knn neighbor tables per dataset (long form).
    Returns dict ds_id -> DataFrame or {} if none.
    """
    out = {}
    for d in datasets_cfg:
        p = d.get("knn_parquet", None)
        if p and os.path.exists(p):
            df = pd.read_parquet(p)
            out[d["id"]] = df
    return out


def sample_matched_controls(ds_id: str, pert_cell_ids: np.ndarray, knn_tables: dict, train_df: pd.DataFrame) -> np.ndarray:
    """
    For each perturbed cell id, pick a matched control id within the same dataset.
    Prefer rank-0 from precomputed KNN if available; else fall back to random control from that dataset.
    Returns an array of control ids aligned to pert_cell_ids.
    """
    out = np.empty(len(pert_cell_ids), dtype=object)
    knn = knn_tables.get(ds_id, None)
    # pool of controls in-train for fallback
    pool = train_df[(train_df["dataset_id"].astype(str) == ds_id) & (train_df["is_control"])].index.values
    if len(pool) == 0:
        # fallback global controls
        pool = train_df[train_df["is_control"]].index.values
    if len(pool) == 0:
        # no controls at all
        return np.array([None] * len(pert_cell_ids), dtype=object)

    if knn is not None and not knn.empty:
        # make a quick lookup of rank-0 neighbor
        g = knn[knn["rank"] == 0].set_index("cell_id")["control_id"]
        for i, cid in enumerate(pert_cell_ids):
            if cid in g.index:
                out[i] = g.loc[cid]
            else:
                out[i] = np.random.choice(pool)
    else:
        # pure random fallback
        for i in range(len(pert_cell_ids)):
            out[i] = np.random.choice(pool)
    return out.astype(str)


# ------------------ Model wrapper ------------------
class GraftCore(nn.Module):
    def __init__(self, z_dim: int, G: int, U: torch.Tensor, hidden=256, gnn_layers=2, use_factor=False, a_dim=0, n_envs=1):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)  # freeze for v1
        self.propagator = StatePropagator(z_dim, hidden=hidden, layers=gnn_layers)
        self.med = MediatedHead(z_dim, F=U.size(1), hidden=hidden, use_factor_feats=use_factor, a_dim=a_dim)
        self.dir = SparseDirectHead(z_dim, G=G, hidden=hidden)
        self.step0 = StepZeroClamp(z_dim=z_dim, n_labs=n_envs, hidden=64, init_eff=0.9)

    def forward(self, z, x0, env_codes, target_idx, a=None):
        z_ref = self.propagator(z)
        x_clamped, eff = self.step0(x0, z_ref, env_codes, target_idx)
        m = self.med(z_ref, a)
        dx_med = torch.matmul(m, self.U.T)
        dx_dir = self.dir(z_ref)
        x_pred = x_clamped + dx_med + dx_dir
        return x_pred, {"eff": eff, "m": m, "dx_med": dx_med, "dx_dir": dx_dir, "x_clamped": x_clamped}


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load index & splits
    index_df = pd.read_parquet(cfg["paths"]["index_parquet"])
    index_df = _resolve_index(index_df)
    train_df = pd.read_parquet(cfg["paths"]["train_perturb"])
    train_df = _resolve_index(train_df)
    val_df   = pd.read_parquet(cfg["paths"]["val_perturb"])
    val_df   = _resolve_index(val_df)

    # Only keep rows that exist in index
    train_df = train_df.loc[train_df.index.intersection(index_df.index)]
    val_df   = val_df.loc[val_df.index.intersection(index_df.index)]

    # Load per-dataset z and concatenate only for rows we need (train+val)
    needed_ids = train_df.index.union(val_df.index)
    Zall = load_multi_z(cfg["paths"]["datasets"], needed_cell_ids=needed_ids)

    # Get gene list from first dataset h5ad and assert compatibility
    first_h5ad = cfg["paths"]["datasets"][0]["h5ad"]
    A0 = ad.read_h5ad(first_h5ad, backed="r")
    genes = A0.var_names.to_numpy()
    # (Optional) check others match
    for d in cfg["paths"]["datasets"][1:]:
        Ai = ad.read_h5ad(d["h5ad"], backed="r")
        if Ai.n_vars != len(genes) or not np.array_equal(Ai.var_names.to_numpy(), genes):
            print(f"[warn] Gene list mismatch in dataset {d['id']}; ensure harmonized gene maps prior to training.")
        del Ai
    del A0

    # Wrap datasets (dataset-centric)
    train = GraftDataset(train_df, Zall, meta=None, genes=genes, dataset_col="dataset_id")
    val   = GraftDataset(val_df,   Zall, meta=None, genes=genes, dataset_col="dataset_id")
    print(f"[info] Train rows: {train.n_cells} across {train.n_datasets} datasets")
    print(f"[info] Val rows:   {val.n_cells} across {val.n_datasets} datasets")

    # Sampler
    ds_map = train.split_by_dataset()
    if cfg["training"].get("interleave_global", False):
        sampler = InterleavedGlobalSampler(ds_map, batch_size=cfg["training"]["batch_size"],
                                           p_global=cfg["training"].get("p_global", 0.2),
                                           seed=cfg["training"].get("seed", 13))
    else:
        sampler = DatasetBalancedSampler(ds_map, batch_size=cfg["training"]["batch_size"],
                                         with_replacement_small=True,
                                         seed=cfg["training"].get("seed", 13))

    # Decoder & knn maps
    decoder = PerDatasetDecoder(
        scvi_model_dir=cfg["paths"]["scvi_model_dir"],
        datasets_cfg=cfg["paths"]["datasets"],
        transform_batch=cfg["model"].get("transform_batch", None),
        forward_batch_size=cfg["training"].get("forward_batch_size", 4096),
    )
    knn_tables = load_knn_map(cfg["paths"]["datasets"])

    # Model
    U = load_U(cfg["paths"]["factor_U"]).to(device)
    model = GraftCore(
        z_dim=Zall.shape[1],
        G=len(genes),
        U=U,
        hidden=cfg["model"]["hidden"],
        gnn_layers=cfg["model"]["gnn_layers"],
        use_factor=cfg["model"].get("use_factor_features", False),
        a_dim=0,
        n_envs=train.n_datasets,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    # Weights
    w_dist = float(cfg["training"].get("dist_loss_weight", 1.0))
    w_kd   = float(cfg["training"].get("kd_consistency_weight", 0.5))
    w_rex  = float(cfg["training"].get("rex_weight", 0.1))
    l1_direct = float(cfg["training"].get("l1_direct_weight", 0.0))

    train_mode = cfg["training"].get("mode", "distribution")
    steps_per_epoch = int(cfg["training"].get("steps_per_epoch", 200))
    max_epochs = int(cfg["training"].get("max_epochs", 50))

    # Main loop
    for epoch in range(1, max_epochs + 1):
        model.train()
        sampler.set_seed(epoch)
        running = defaultdict(float)

        for step, idx in zip(range(steps_per_epoch), sampler):
            b = train.batch_dict(idx)
            z_b = torch.from_numpy(b["z"]).to(device)
            ds_codes = torch.from_numpy(b["dataset_codes"]).long().to(device)
            tgt_idx = torch.from_numpy(b["target_idx"]).long().to(device)
            ds_labels = train.datasets[idx]  # str labels aligned to idx
            cell_ids = b["cell_ids"]         # str cell ids aligned to idx

            # Build x0,y
            if train_mode == "paired_controls":
                # split per-dataset to decode in fewer calls
                x0_list = []
                y_list = []
                order_mask = []
                is_ctrl = train.is_control[idx]
                pert_mask = ~is_ctrl
                idx_arr = np.asarray(idx, dtype=int)

                # For perturbed: match controls via knn (or random fallback)
                pert_ids = cell_ids[pert_mask]
                pert_ds  = ds_labels[pert_mask]
                # Group by dataset for decoding
                for ds in np.unique(pert_ds):
                    m = (pert_ds == ds)
                    matched_ctrl_ids = sample_matched_controls(ds, pert_ids[m], knn_tables, train.df)
                    # decode matched controls (x0) and perturbed (y) for this subset
                    x0_ds = decoder.get_xbar(ds, matched_ctrl_ids)
                    y_ds  = decoder.get_xbar(ds, pert_ids[m])
                    x0_list.append(x0_ds)
                    y_list.append(y_ds)
                    order_mask.append(("pert", ds, m))

                # For controls: x0=y=control
                ctrl_ids = cell_ids[is_ctrl]
                ctrl_ds  = ds_labels[is_ctrl]
                for ds in np.unique(ctrl_ds):
                    m = (ctrl_ds == ds)
                    xc = decoder.get_xbar(ds, ctrl_ids[m])
                    x0_list.append(xc)
                    y_list.append(xc)
                    order_mask.append(("ctrl", ds, m))

                # Stitch into batch order
                B, G = len(idx_arr), len(genes)
                x0 = np.zeros((B, G), dtype=np.float32)
                y  = np.zeros((B, G), dtype=np.float32)
                cursor = 0
                # Fill pert subsets first in same order as we appended
                for kind, ds, m in order_mask:
                    nrows = np.sum(m)
                    block_x = x0_list[cursor]
                    block_y = y_list[cursor]
                    cursor += 1
                    # place into correct positions within the full batch
                    if kind == "pert":
                        # places correspond to positions where pert_mask True and in this ds
                        x0[pert_mask][m, :] = block_x
                        y[pert_mask][m, :]  = block_y
                    else:
                        x0[is_ctrl][m, :] = block_x
                        y[is_ctrl][m, :]  = block_y

            else:
                # distribution mode: decode each dataset slice for the same cells
                B = len(idx)
                Gdim = len(genes)
                x0 = np.zeros((B, Gdim), dtype=np.float32)
                y  = np.zeros((B, Gdim), dtype=np.float32)
                for ds in np.unique(ds_labels):
                    m = (ds_labels == ds)
                    ids_ds = cell_ids[m]
                    X = decoder.get_xbar(ds, ids_ds)
                    x0[m, :] = X
                    y[m,  :] = X

            x0_t = torch.from_numpy(x0).to(device)
            y_t  = torch.from_numpy(y).to(device)

            # Forward
            y_pred, aux = model(z_b, x0_t, ds_codes, tgt_idx)

            # Per-dataset losses
            per_ds_losses = []
            for ds_int in torch.unique(ds_codes).tolist():
                m = (ds_codes == ds_int)
                if torch.count_nonzero(m) > 1:
                    per_ds_losses.append(sliced_wasserstein(y_pred[m], y_t[m], n_proj=32))
                else:
                    per_ds_losses.append(torch.nn.functional.mse_loss(y_pred[m], y_t[m]))
            L_dist = torch.stack(per_ds_losses).mean()
            L_rex  = torch.stack(per_ds_losses).var(unbiased=False)
            L_kd   = target_knockdown_consistency(y_pred, y_t, tgt_idx, weight=1.0)
            L_l1   = l1_direct * torch.mean(torch.abs(aux["dx_dir"]))

            L = w_dist * L_dist + w_kd * L_kd + w_rex * L_rex + L_l1
            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running["loss"] += float(L)
            running["dist"] += float(L_dist)
            running["kd"]   += float(L_kd)
            running["rex"]  += float(L_rex)
            running["l1"]   += float(L_l1)

        steps = float(steps_per_epoch)
        print(f"[epoch {epoch:03d}] loss={running['loss']/steps:.4f} "
              f"(dist={running['dist']/steps:.4f}, kd={running['kd']/steps:.4f}, "
              f"rex={running['rex']/steps:.4f}, l1={running['l1']/steps:.4f})")

        # ---------- Validation (light) ----------
        model.eval()
        with torch.no_grad():
            val_map = val.split_by_dataset()
            val_sampler = DatasetBalancedSampler(val_map, batch_size=min(len(val.df), cfg['training']['batch_size']))
            val_running = defaultdict(float)
            for step, vidx in zip(range(min(steps_per_epoch//4, 50)), val_sampler):
                b = val.batch_dict(vidx)
                z_b = torch.from_numpy(b["z"]).to(device)
                ds_codes = torch.from_numpy(b["dataset_codes"]).long().to(device)
                tgt_idx = torch.from_numpy(b["target_idx"]).long().to(device)
                ds_labels = val.datasets[vidx]
                cell_ids = b["cell_ids"]

                # decode per dataset
                B = len(vidx); Gdim = len(genes)
                x0 = np.zeros((B, Gdim), dtype=np.float32)
                y  = np.zeros((B, Gdim), dtype=np.float32)
                for ds in np.unique(ds_labels):
                    m = (ds_labels == ds)
                    ids_ds = cell_ids[m]
                    X = decoder.get_xbar(ds, ids_ds)
                    x0[m, :] = X
                    y[m,  :] = X

                x0_t = torch.from_numpy(x0).to(device)
                y_t  = torch.from_numpy(y).to(device)

                y_pred, _ = model(z_b, x0_t, ds_codes, tgt_idx)
                per_ds_losses = []
                for ds_int in torch.unique(ds_codes).tolist():
                    m = (ds_codes == ds_int)
                    if torch.count_nonzero(m) > 1:
                        per_ds_losses.append(sliced_wasserstein(y_pred[m], y_t[m], n_proj=32))
                    else:
                        per_ds_losses.append(torch.nn.functional.mse_loss(y_pred[m], y_t[m]))
                L_dist = torch.stack(per_ds_losses).mean()
                L_kd = target_knockdown_consistency(y_pred, y_t, tgt_idx, weight=1.0)
                L_rex = torch.stack(per_ds_losses).var(unbiased=False)
                L = w_dist * L_dist + w_kd * L_kd + w_rex * L_rex

                val_running["loss"] += float(L)
                val_running["dist"] += float(L_dist)
                val_running["kd"]   += float(L_kd)
                val_running["rex"]  += float(L_rex)

            denom = max(1.0, float(min(steps_per_epoch//4, 50)))
            print(f"          val: loss={val_running['loss']/denom:.4f} "
                  f"(dist={val_running['dist']/denom:.4f}, kd={val_running['kd']/denom:.4f}, "
                  f"rex={val_running['rex']/denom:.4f})")

    print("[DONE] Training finished.")
    # Optional: save weights
    # torch.save(model.state_dict(), cfg["paths"].get("gnn_ckpt", "artifacts/gnn_core.pt"))


if __name__ == "__main__":
    main()
