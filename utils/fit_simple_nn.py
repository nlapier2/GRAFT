#!/usr/bin/env python3
"""
learn_prototype_deltas.py

Learn one delta vector per perturbation (in log1p space) and reconstruct:
- Controls: copied verbatim to output.
- Perturbed: x_hat = control_anchor + delta_hat[pert_id],
  where control_anchor is mean of k-NN controls (in a compact SVD space).

Assumes input .X is already log1p-normalized, and outputs log1p as well.

Example:
  python learn_prototype_deltas.py \
    --input input.h5ad \
    --output output_pred.h5ad \
    --label-col target_gene \
    --control-col is_control \
    --svd-components 64 \
    --k-ctrl 8 \
    --epochs 20 \
    --batch-size 128 \
    --loss huber \
    --hvg-top 0
"""

import argparse
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# --------------------------
# Helpers
# --------------------------

def infer_controls(obs: pd.DataFrame, label_col: str):
    # Heuristics if no explicit control column is provided
    if label_col not in obs.columns:
        # All NaN labels treated as control
        return obs.index.isin(obs.index[obs.index == "__NONE__"])  # no-op (all False)
    s = obs[label_col].astype(str)
    ctrl_tokens = {"control", "ctrl", "ntc", "none", "nan", "", "non-targeting", "neg", "negctrl", "neg_control"}
    return s.str.lower().isin(ctrl_tokens) | s.isna()

def build_hvg_mask(A: ad.AnnData, n_top: int):
    import scanpy as sc
    A_tmp = A.copy()
    # A already log1p; don't re-normalize; just compute HVGs
    sc.pp.highly_variable_genes(A_tmp, n_top_genes=n_top, flavor="seurat_v3", inplace=True)
    mask = A_tmp.var["highly_variable"].to_numpy()
    return mask

def csr_row_mean(mat_csr: sparse.csr_matrix, rows: np.ndarray) -> np.ndarray:
    """Return dense 1D mean over given row indices (log1p space)."""
    sub = mat_csr[rows]                     # (k, G) CSR
    # sum along axis=0 -> numpy matrix; divide by k; convert to 1D float32
    m = (sub.sum(axis=0) / float(len(rows))).A1.astype(np.float32, copy=False)
    return m

# --------------------------
# Dataset
# --------------------------

class PerturbedDataset(Dataset):
    def __init__(self, X_log_csr, Z_feat, knn_ctrl, ctrl_idx, pert_idx, pert_ids, k_ctrl, hvg_mask=None):
        self.X = X_log_csr
        self.Z = Z_feat
        self.knn = knn_ctrl
        self.ctrl_idx = ctrl_idx
        self.pert_idx = pert_idx
        self.pert_ids = pert_ids.astype(np.int64)
        self.k = int(k_ctrl)
        self.hvg_mask = hvg_mask

    def __len__(self):
        return len(self.pert_idx)

    def __getitem__(self, i):
        rid = self.pert_idx[i]
        # find control neighbors using SVD features
        z = self.Z[rid : rid + 1]  # (1, d)
        nn_dists, nn_pos = self.knn.kneighbors(z, n_neighbors=self.k, return_distance=True)
        # indices in the original matrix
        neigh = self.ctrl_idx[nn_pos[0]]
        anchor = csr_row_mean(self.X, neigh)            # (G,)
        x_pert = self.X[rid].toarray().astype(np.float32).ravel()
        delta_target = x_pert - anchor
        if self.hvg_mask is not None:
            delta_target = delta_target[self.hvg_mask]
            anchor = anchor[self.hvg_mask]
        return (anchor, delta_target, self.pert_ids[i], rid)

# --------------------------
# Model
# --------------------------

class ProtoDelta(nn.Module):
    def __init__(self, n_perts: int, out_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_perts, out_dim)  # δ_p in log1p space
        nn.init.zeros_(self.emb.weight)

    def forward(self, pert_ids):
        return self.emb(pert_ids)  # (B, G_eff)

# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label-col", default="target_gene", help="obs column with perturbation labels")
    ap.add_argument("--control-col", default=None, help="optional boolean obs column marking controls")
    ap.add_argument("--svd-components", type=int, default=64, help="components for kNN space")
    ap.add_argument("--k-ctrl", type=int, default=8, help="# nearest controls to anchor each perturbed cell")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--loss", choices=["l1", "huber"], default="huber")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hvg-top", type=int, default=0, help="train loss on top-N HVGs (0 = all genes)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load (assumes .X already log1p)
    A = ad.read_h5ad(args.input)
    obs = A.obs.copy()
    var = A.var.copy()
    genes = A.var_names.astype(str)
    Xlog = A.X.tocsr().astype(np.float32) if sparse.issparse(A.X) else sparse.csr_matrix(A.X.astype(np.float32))
    N, G = Xlog.shape

    # Identify controls and perturbed
    if args.control_col is not None and args.control_col in obs.columns:
        ctrl_mask = obs[args.control_col].astype(bool).to_numpy()
    else:
        ctrl_mask = infer_controls(obs, args.label_col).to_numpy()

    if args.label_col not in obs.columns:
        raise ValueError(f"obs['{args.label_col}'] is required to identify perturbations.")

    labels = obs[args.label_col].astype("category")
    # Treat controls as a special label, but exclude them from the learnable set
    ctrl_idxs = np.where(ctrl_mask)[0]
    pert_idxs = np.where(~ctrl_mask)[0]
    if len(pert_idxs) == 0 or len(ctrl_idxs) == 0:
        raise ValueError("Need both control and perturbed cells present.")

    # Make a label mapping that excludes the control class
    cats = labels.cat.categories.tolist()
    # Map pert labels -> 0..P-1 ; controls (any value) -> -1 (unused)
    label_codes = labels.cat.codes.to_numpy()
    # Find which codes are present among perturbed rows:
    pert_codes = np.unique(label_codes[pert_idxs])
    # Build compact map: old_code -> new_code (0..P-1), controls -> -1
    old2new = {int(c): i for i, c in enumerate(sorted(int(c) for c in pert_codes))}
    pert_id_vec = np.full(N, -1, dtype=np.int64)
    for i in pert_idxs:
        pert_id_vec[i] = old2new[int(label_codes[i])]
    n_perts = len(old2new)
    print(f"[info] N={N}, G={G}, controls={len(ctrl_idxs)}, perturbed={len(pert_idxs)}, n_perts={n_perts}")

    # Build SVD features for kNN (fit on controls; transform all)
    comps = max(1, int(args.svd_components))
    print(f"[info] TruncatedSVD (fit on controls) with {comps} comps ...")
    svd = TruncatedSVD(n_components=min(comps, min(len(ctrl_idxs), G) - 1))
    Z_ctrl = svd.fit_transform(Xlog[ctrl_idxs])  # (Nc, d)
    Z_all = svd.transform(Xlog)                  # (N, d)

    # kNN on controls
    knn = NearestNeighbors(n_neighbors=min(args.k_ctrl, len(ctrl_idxs)), algorithm="auto", n_jobs=-1)
    knn.fit(Z_ctrl)

    # HVG mask (optional) — used for the loss only; predictions are full-G
    hvg_mask = None
    if args.hvg_top and args.hvg_top > 0:
        try:
            import scanpy as sc
            A_copy = ad.AnnData(X=Xlog, obs=obs.copy(), var=var.copy())
            sc.pp.highly_variable_genes(A_copy, n_top_genes=args.hvg_top, flavor="seurat_v3", inplace=True)
            hvg_mask = A_copy.var["highly_variable"].to_numpy()
            print(f"[info] Using HVG mask with {hvg_mask.sum()} genes for the loss.")
        except Exception as e:
            print(f"[warn] HVG computation failed ({e}); proceeding without HVG mask.")

    # Dataset / DataLoader over perturbed cells
    ds = PerturbedDataset(
        X_log_csr=Xlog,
        Z_feat=Z_all,
        knn_ctrl=knn,
        ctrl_idx=ctrl_idxs,
        pert_idx=pert_idxs,
        pert_ids=pert_id_vec[pert_idxs],
        k_ctrl=args.k_ctrl,
        hvg_mask=hvg_mask
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    # Model (δ per perturbation)
    G_eff = int(hvg_mask.sum()) if hvg_mask is not None else G
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtoDelta(n_perts=n_perts, out_dim=G_eff).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.loss == "huber":
        loss_fn = nn.SmoothL1Loss(reduction="mean")
    else:
        loss_fn = nn.L1Loss(reduction="mean")

    # Train
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        nsteps = 0
        for anchor_np, delta_t_np, pid, _ in dl:
            anchor = (anchor_np if isinstance(anchor_np, torch.Tensor) else torch.from_numpy(anchor_np)).to(device, non_blocking=True).float()
            delta_t = (delta_t_np if isinstance(delta_t_np, torch.Tensor) else torch.from_numpy(delta_t_np)).to(device, non_blocking=True).float()
            pid = pid.to(device)

            delta_hat = model(pid)
            loss = loss_fn(delta_hat, delta_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item())
            nsteps += 1
        print(f"[epoch {epoch}] loss={total / max(1, nsteps):.5f}")

    # --------------------------
    # Prediction & Write Output
    # --------------------------
    model.eval()

    # Prepare CSR builder
    indptr = [0]
    indices = []
    data = []

    # Build quick lookup for control features (already fit on Z_ctrl)
    # We reuse the same kNN; for each row, query neighbors and anchor
    batch_rows = 256

    with torch.no_grad():
        for i0 in range(0, N, batch_rows):
            i1 = min(N, i0 + batch_rows)
            rows = np.arange(i0, i1)
            # classify rows as control vs perturbed
            ctrl_mask_batch = ctrl_mask[rows]
            rows_ctrl = rows[ctrl_mask_batch]
            rows_pert = rows[~ctrl_mask_batch]

            # Controls: copy input sparsity & data
            for r in rows_ctrl:
                start, end = Xlog.indptr[r], Xlog.indptr[r+1]
                row_idx = Xlog.indices[start:end]
                row_dat = Xlog.data[start:end]
                indices.extend(row_idx.tolist())
                data.extend(row_dat.tolist())
                indptr.append(indptr[-1] + (end - start))

            # Perturbed: anchor + learned delta
            if len(rows_pert) > 0:
                # knn on SVD features
                Zp = Z_all[rows_pert]
                nn_dists, nn_pos = knn.kneighbors(Zp, n_neighbors=args.k_ctrl, return_distance=True)
                # predict per row
                for jj, r in enumerate(rows_pert):
                    neigh = Xlog[ctrl_idxs[nn_pos[jj]]]
                    # anchor (dense)
                    anchor = (neigh.sum(axis=0) / float(neigh.shape[0])).A1.astype(np.float32, copy=False)
                    if hvg_mask is not None:
                        anchor_eff = anchor[hvg_mask]
                    else:
                        anchor_eff = anchor
                    pid = pert_id_vec[r]
                    delta = model.emb.weight[pid].cpu().numpy().astype(np.float32, copy=False)
                    yhat = anchor_eff + delta
                    # merge back to full G
                    if hvg_mask is not None:
                        full = anchor.copy()
                        full[hvg_mask] = yhat
                        yhat = full
                    # sparsify with small threshold to keep file size reasonable
                    # keep entries > 1e-8
                    nz = np.where(yhat > 1e-8)[0]
                    indices.extend(nz.tolist())
                    data.extend(yhat[nz].tolist())
                    indptr.append(indptr[-1] + len(nz))

    X_out = sparse.csr_matrix((np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)),
                              shape=(N, G))

    A_out = ad.AnnData(
        X=X_out,
        obs=obs,
        var=var,
        uns=(A.uns.copy() if A.uns is not None else None),
        obsm=(A.obsm.copy() if A.obsm is not None else None),
        varm=(A.varm.copy() if A.varm is not None else None),
        layers=({k: v.copy() for k, v in (A.layers.items() if A.layers is not None else [])}),
    )
    A_out.var_names = genes

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    A_out.write_h5ad(args.output, compression="lzf")
    print(f"[done] wrote predictions to: {args.output}")

if __name__ == "__main__":
    main()
