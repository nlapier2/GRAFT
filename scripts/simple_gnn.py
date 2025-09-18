#!/usr/bin/env python3
# gnn_fit_panel.py
import argparse, math, os, sys, random
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

# ----------------------------
# Utilities
# ----------------------------
def to_numpy(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)

def build_target_to_gene_index(adata: ad.AnnData, target_label: str) -> Dict[str, int]:
    """
    Map each perturbation label to a gene index IF the label is a gene present in var_names.
    Non-gene labels will be ignored (they can still be part of the training set, but Step0 will
    clamp nothing for that sample). For your panel, we expect labels == gene symbols.
    """
    varset = set(adata.var_names)
    t2i = {}
    for t in adata.obs[target_label].unique():
        if t in varset:
            t2i[t] = int(np.where(adata.var_names == t)[0][0])
    return t2i

def sample_minibatch(
    X_ctrl: np.ndarray,
    X_pert: np.ndarray,
    pert_labels: np.ndarray,
    control_label: str,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Returns batch_x_ctrl (B,G), batch_x_pert (B,G), batch_targets (list of labels).
    Matches each perturbed cell to a random control cell.
    """
    B = batch_size
    # indices for perturbed cells (exclude controls)
    pert_mask = pert_labels != control_label
    pert_idx = np.where(pert_mask)[0]
    if len(pert_idx) < B:
        choice = rng.choice(pert_idx, size=B, replace=True)
    else:
        choice = rng.choice(pert_idx, size=B, replace=False)
    # random controls
    ctrl_idx = np.where(pert_labels == control_label)[0]
    rand_ctrl = rng.choice(ctrl_idx, size=B, replace=True)
    bx_ctrl = torch.from_numpy(X_ctrl[rand_ctrl]).float()
    bx_pert = torch.from_numpy(X_pert[choice]).float()
    btargets = pert_labels[choice].tolist()
    return bx_ctrl, bx_pert, btargets

def make_base_adjacency(G: int, self_loops: bool = True) -> torch.Tensor:
    """
    Dense fully-connected adjacency (uniform), normalized row-wise.
    We'll mask rows per-sample to forbid inbound messages to the target.
    """
    A = torch.ones(G, G)
    if not self_loops:
        A.fill_diagonal_(0.0)
    # row-normalize so each node aggregates an average of neighbors
    A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
    return A

# Add near your utilities
def collapse_to_pseudobulk(adata, target_label: str):
    """Return a new AnnData with one row per label (perturbation + control)."""
    import pandas as pd
    X = to_numpy(adata.X).astype(np.float32)
    labels = adata.obs[target_label].astype(str).values
    df = pd.DataFrame(X, columns=adata.var_names).groupby(labels).mean()
    from anndata import AnnData
    ad_bulk = AnnData(df.values.astype(np.float32))
    ad_bulk.var_names = adata.var_names.copy()
    ad_bulk.obs[target_label] = df.index.astype(str)
    return ad_bulk

# ----------------------------
# Model: Step0 + MPNN + Readout
# ----------------------------
class Step0Clamp(nn.Module):
    """
    Simple Step-0: clamp the target node toward an anchor 'tau' with learnable efficacy alpha in (0,1).
    For CRISPRi-like behavior, tau=0.0 (in normalized space).
    """
    def __init__(self, tau: float = 0.0, num_perts: int = None):
        super().__init__()
        # global alpha by default; if num_perts is given, use per-pert embedding for alpha
        if num_perts is not None:
            self.alpha_table = nn.Embedding(num_perts, 1)
            nn.init.zeros_(self.alpha_table.weight)  # sigmoid(0)=0.5
        else:
            self.alpha_table = None
            self.logit_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid -> ~0.5 initially
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x_ctrl: torch.Tensor, target_idx: torch.Tensor, pert_rowidx: torch.Tensor = None) -> torch.Tensor:
        """
        x_ctrl: (B,G)
        target_idx: (B,) int tensor with -1 when unknown (i.e., target label not a gene)
        """
        B, G = x_ctrl.shape
        x0 = x_ctrl.clone()
        if self.alpha_table is not None and pert_rowidx is not None:
            alpha = torch.sigmoid(self.alpha_table(pert_rowidx)).view(-1)  # (B,)
        else:
            alpha = torch.sigmoid(self.logit_alpha).expand(B)  # (B,)
        if (target_idx >= 0).any():
            bmask = (target_idx >= 0)
            rows = torch.arange(B, device=x_ctrl.device)[bmask]
            cols = target_idx[bmask]
            # x_t := (1 - alpha) * x_ctrl_t + alpha * tau
            x0[rows, cols] = (1.0 - alpha[bmask]) * x_ctrl[rows, cols] + alpha[bmask] * self.tau
        return x0

class MPNNLayer(nn.Module):
    """
    Basic MPNN layer with dense adjacency.
    h_in -> aggregate (A @ h_in) -> update with residual
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.upd = nn.Linear(2 * hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, A_batch: torch.Tensor, h_t_frozen: torch.Tensor) -> torch.Tensor:
        """
        h:        (B,G,C)
        A_batch:  (B,G,G) row-normalized, with row[target]=0 for each sample
        h_t_frozen: (B,1,C) the clamped target embedding to re-impose after update
        """
        # messages
        m = torch.matmul(A_batch, self.msg(h))  # (B,G,C)
        h_new = self.act(self.upd(torch.cat([h, m], dim=-1)))  # (B,G,C)
        # residual
        h_out = h + h_new
        # re-impose frozen target state
        # gather: replace the row corresponding to target with frozen
        # h_t_frozen is provided already extracted as h[:, t, :].unsqueeze(1) after Step-0 embed
        # We assume caller already zeroed inbound to t in A_batch.
        # Concatenate by slicing to avoid scatter for speed on small G
        # (But we need indices; we’ll do it in the caller for clarity.)
        return h_out

class GeneMPNN(nn.Module):
    def __init__(self, G: int, hidden: int = 128, T: int = 2, tau: float = 0.0, num_perts: int = None):
        super().__init__()
        self.G = G
        self.hidden = hidden
        self.T = T
        # per-node input is scalar (expression); use a shared linear to lift to hidden
        self.embed = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([MPNNLayer(hidden) for _ in range(T)])
        self.readout = nn.Linear(hidden, 1)
        # simple perturbation embedding for FiLM + per-pert alpha in Step-0
        self.pert_emb = nn.Embedding(num_perts, hidden) if num_perts is not None else None
        self.film_gamma = nn.Linear(hidden, hidden) if self.pert_emb is not None else None
        self.film_beta  = nn.Linear(hidden, hidden) if self.pert_emb is not None else None
        self.step0 = Step0Clamp(tau=tau, num_perts=num_perts)

    def forward(self, x_ctrl: torch.Tensor, target_idx: torch.Tensor, A_base: torch.Tensor, pert_rowidx: torch.Tensor = None) -> torch.Tensor:
        """
        x_ctrl: (B,G)
        target_idx: (B,) int tensor with -1 where unknown
        A_base: (G,G) dense row-normalized base adjacency
        """
        device = x_ctrl.device
        B, G = x_ctrl.shape
        assert G == self.G

        # Step-0 clamp in expression space (per-pert alpha if available)
        x0 = self.step0(x_ctrl, target_idx, pert_rowidx=pert_rowidx)  # (B,G)

        # Initial hidden state (shared 1->hidden linear applied per gene)
        h = self.embed(x0.unsqueeze(-1))  # (B,G,hidden)
        # FiLM condition on perturbation embedding (broadcast across genes)
        if self.pert_emb is not None and pert_rowidx is not None:
            e = self.pert_emb(pert_rowidx)                 # (B,hidden)
            gamma = torch.tanh(self.film_gamma(e)).unsqueeze(1)  # (B,1,hidden)
            beta  = self.film_beta(e).unsqueeze(1)               # (B,1,hidden)
            h = h * (1 + gamma) + beta

        # Prepare per-sample adjacency (block inbound to target)
        # Start from base A, then zero the row 't' per sample.
        A_batch = A_base.unsqueeze(0).repeat(B, 1, 1).to(device)  # (B,G,G)
        # keep a copy of target embeddings to re-impose after each layer
        # If target_idx == -1, we won’t freeze anything; we’ll handle with a mask.
        freeze_mask = (target_idx >= 0)
        if freeze_mask.any():
            rows = torch.arange(B, device=device)[freeze_mask]
            cols = target_idx[freeze_mask]
            A_batch[rows, cols, :] = 0.0  # zero inbound to target (row=t)

        # Save the frozen target embedding (after Step-0 embed)
        # If some samples lack known target, we’ll just skip the replacement.
        h_t0 = torch.zeros(B, 1, self.hidden, device=device)
        if freeze_mask.any():
            h_t0[freeze_mask] = h[rows, cols].unsqueeze(1)

        # Run T layers with reimposition of target state
        for layer in self.layers:
            h = layer(h, A_batch, h_t0)
            if freeze_mask.any():
                # put frozen target embedding back
                h[rows, cols] = h_t0[freeze_mask, 0]

        # Readout back to expression space
        y = self.readout(h).squeeze(-1)  # (B,G)
        return y, x0  # return x0 for optional locality loss

# ----------------------------
# Losses
# ----------------------------
def mse_loss(yhat, y):
    return F.mse_loss(yhat, y)

def target_consistency_loss(yhat, x_ctrl, target_idx, mode="knockdown", margin=0.0):
    """
    Encourage correct direction at the target:
      knockdown: yhat[t] <= x_ctrl[t] - margin
      activation: yhat[t] >= x_ctrl[t] + margin
    """
    if (target_idx < 0).sum() == target_idx.numel():
        return yhat.new_tensor(0.0)
    rows = torch.arange(target_idx.numel(), device=yhat.device)[target_idx >= 0]
    cols = target_idx[target_idx >= 0]
    y_t = yhat[rows, cols]
    x_t = x_ctrl[rows, cols]
    if mode == "activation":
        # hinge: max(0, (x+margin) - y)
        return F.relu((x_t + margin) - y_t).mean()
    # default knockdown
    return F.relu(y_t - (x_t - margin)).mean()

def locality_damping(yhat, x0, target_idx, k_mask=None, weight=1.0):
    """
    Penalize changes far from target. Simplest form: L1 over all non-target genes.
    You can pass a boolean k-hop mask (B,G) with True where penalty applies less (or zero near t).
    For now, just exclude the target index itself.
    """
    B, G = yhat.shape
    loss = 0.0
    for b in range(B):
        t = int(target_idx[b].item())
        if t >= 0:
            mask = torch.ones(G, dtype=torch.bool, device=yhat.device)
            mask[t] = False
            loss = loss + (yhat[b, mask] - x0[b, mask]).abs().mean()
    return (loss / max((target_idx >= 0).sum().item(), 1)) * weight

# ----------------------------
# Training
# ----------------------------
def train(
    adata: ad.AnnData,
    target_label: str,
    control_label: str,
    hidden: int = 128,
    T: int = 2,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_target: float = 0.1,
    weight_local: float = 0.0,
    seed: int = 0,
    tau: float = 0.0,
    device: str = "cuda",
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Data arrays
    X = to_numpy(adata.X).astype(np.float32)  # assume normalized/log space already
    labels = adata.obs[target_label].astype(str).values
    G = adata.n_vars

    # Index pools
    ctrl_mask = labels == control_label
    if ctrl_mask.sum() == 0:
        raise ValueError("No control cells found.")
    # perturbed pool includes all non-controls (even if target gene not found)
    pert_mask = ~ctrl_mask
    if pert_mask.sum() == 0:
        raise ValueError("No perturbed cells found.")

    X_ctrl = X  # we’ll pick rows via indices
    X_pert = X
    pert_labels = labels

    # Map perturbation label -> gene index (for Step-0); unknown => -1
    t2gi = build_target_to_gene_index(adata, target_label)
    # Precompute a tensor of target indices per cell
    tgt_idx = np.full(adata.n_obs, -1, dtype=np.int64)
    for i, lab in enumerate(labels):
        tgt_idx[i] = t2gi.get(lab, -1)

    # Model
    # Build stable mapping from label -> embedding row
    pert_names_unique = sorted(set(labels.tolist()))
    pert2row = {p: i for i, p in enumerate(pert_names_unique)}
    num_perts = len(pert_names_unique)

    # Model
    model = GeneMPNN(G=G, hidden=hidden, T=T, tau=tau, num_perts=num_perts).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Base adjacency (fully connected, row-normalized)
    A_base = make_base_adjacency(G, self_loops=True).to(device)

    # Simple schedule
    steps_per_epoch = math.ceil(pert_mask.sum() / batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"mse": 0.0, "targ": 0.0, "loc": 0.0, "tot": 0.0}
        for step in range(steps_per_epoch):
            bx_ctrl, bx_pert, btargets = sample_minibatch(
                X_ctrl=X_ctrl, X_pert=X_pert, pert_labels=pert_labels,
                control_label=control_label, batch_size=batch_size, rng=rng
            )
            # per-sample target index tensor
            tidx = torch.tensor([t2gi.get(t, -1) for t in btargets], dtype=torch.long)

            bx_ctrl = bx_ctrl.to(device)
            bx_pert = bx_pert.to(device)
            tidx = tidx.to(device)

            # per-sample perturbation row indices for embedding / FiLM
            pert_rowidx = torch.tensor([pert2row[t] for t in btargets], dtype=torch.long, device=device)
            yhat, x0 = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx)

            loss_mse = mse_loss(yhat - x0, bx_pert - x0)
            loss_t = target_consistency_loss(yhat, bx_ctrl, tidx, mode="knockdown", margin=0.0)
            loss_loc = locality_damping(yhat, x0, tidx, weight=1.0) if weight_local > 0 else yhat.new_tensor(0.0)

            loss = loss_mse + weight_target * loss_t + weight_local * loss_loc

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running["mse"]  += float(loss_mse.item())
            running["targ"] += float(loss_t.item())
            running["loc"]  += float(loss_loc.item())
            running["tot"]  += float(loss.item())

        denom = max(steps_per_epoch, 1)
        print(f"[epoch {epoch:03d}] "
              f"mse={running['mse']/denom:.5f}  "
              f"targ={running['targ']/denom:.5f}  "
              f"loc={running['loc']/denom:.5f}  "
              f"total={running['tot']/denom:.5f}")

    return model


@torch.no_grad()
def predict_all_perturbations(
    adata: ad.AnnData,
    model: nn.Module,
    target_label: str,
    control_label: str,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 0,
):
    """
    For every perturbed cell, match a random control, run model, and collect predictions.
    Returns:
      pred_mat: (N_pert, G) predicted expressions (aligned to perturbed rows)
      true_mat: (N_pert, G) true perturbed expressions
      pert_names: list[str] of length N_pert (labels for each row)
      ctrl_mean: (G,) global control pseudobulk (mean of all control cells)
    """
    rng = np.random.default_rng(seed)
    X = to_numpy(adata.X).astype(np.float32)
    labels = adata.obs[target_label].astype(str).values
    G = adata.n_vars

    # pools
    ctrl_idx = np.where(labels == control_label)[0]
    pert_idx = np.where(labels != control_label)[0]
    if len(ctrl_idx) == 0 or len(pert_idx) == 0:
        raise ValueError("Need both control and perturbed cells for evaluation.")

    # control pseudobulk (global)
    ctrl_mean = X[ctrl_idx].mean(axis=0)

    # target mapping (label -> gene index), -1 if not a gene in panel
    t2gi = build_target_to_gene_index(adata, target_label)

    # adjacency
    A_base = make_base_adjacency(G, self_loops=True).to(device)

    # allocate
    Np = len(pert_idx)
    pred_mat = np.zeros((Np, G), dtype=np.float32)
    true_mat = X[pert_idx]  # (Np,G)
    pert_names = labels[pert_idx].tolist()

    # stable mapping label -> row index (should match training order if same labels set)
    pert_names_unique = sorted(set(labels.tolist()))
    pert2row = {p: i for i, p in enumerate(pert_names_unique)}

    # batched forward with random control matches
    model.eval()
    for start in range(0, Np, batch_size):
        end = min(start + batch_size, Np)
        b_idx = np.arange(start, end)
        # random controls (with replacement)
        rand_ctrl = rng.choice(ctrl_idx, size=len(b_idx), replace=True)

        bx_ctrl = torch.from_numpy(X[rand_ctrl]).float().to(device)
        tidx = torch.tensor([t2gi.get(p, -1) for p in pert_names[start:end]], dtype=torch.long, device=device)
        pert_rowidx = torch.tensor([pert2row[p] for p in pert_names[start:end]], dtype=torch.long, device=device)

        yhat, _ = model(bx_ctrl, tidx, A_base, pert_rowidx=pert_rowidx)
        pred_mat[b_idx] = yhat.detach().cpu().numpy()

    return pred_mat, true_mat, pert_names, ctrl_mean


def evaluate_model(
    adata: ad.AnnData,
    model: nn.Module,
    target_label: str,
    control_label: str,
    device: str = "cuda",
    batch_size: int = 256,
    seed: int = 0,
):
    """
    Computes:
      - per-perturbation MAE
      - knockdown efficiency (abs & %) for true vs predicted at the target gene
      - perturbation similarity: mean & min pairwise Pearson corr between predicted mean effect vectors
      - PDS (Perturbation Discrimination Score): mean over perturbations
    Prints a concise report and returns a dict with all metrics.
    """
    pred_mat, true_mat, pert_names, ctrl_mean = predict_all_perturbations(
        adata, model, target_label, control_label, device=device, batch_size=batch_size, seed=seed
    )
    G = adata.n_vars
    df_obs = adata.obs
    labels = df_obs[target_label].astype(str).values

    # group indices by perturbation (excluding control)
    perts = sorted(set(pert_names))
    # target mapping
    t2gi = build_target_to_gene_index(adata, target_label)

    # per-pert pseudobulks (pred & true) and MAE
    pred_bulk = {}
    true_bulk = {}
    mae_per_pert = {}

    # map pert_names (length Np) to row indices for quick grouping
    from collections import defaultdict
    rows_by_pert = defaultdict(list)
    for i, p in enumerate(pert_names):
        rows_by_pert[p].append(i)

    for p in perts:
        rows = rows_by_pert[p]
        yhat_p = pred_mat[rows]  # (n_p, G)
        ytrue_p = true_mat[rows] # (n_p, G)
        pred_bulk[p] = yhat_p.mean(axis=0)
        true_bulk[p] = ytrue_p.mean(axis=0)
        mae_per_pert[p] = np.mean(np.abs(yhat_p - ytrue_p))

    # knockdown efficiency at target gene (abs & %), true vs predicted
    # uses GLOBAL control pseudobulk as the "control" reference
    eps = 1e-8
    kd_eff = {}  # p -> dict
    for p in perts:
        t = t2gi.get(p, -1)
        if t < 0:
            kd_eff[p] = {"target_gene": None,
                         "true_abs": np.nan, "true_pct": np.nan,
                         "pred_abs": np.nan, "pred_pct": np.nan}
            continue
        ctrl_t = float(ctrl_mean[t])
        true_t = float(true_bulk[p][t])
        pred_t = float(pred_bulk[p][t])

        # absolute "knockdown" (positive if below control)
        true_abs = ctrl_t - true_t
        pred_abs = ctrl_t - pred_t
        # percentage relative to control level
        true_pct = true_abs / (ctrl_t + eps)
        pred_pct = pred_abs / (ctrl_t + eps)

        kd_eff[p] = {"target_gene": adata.var_names[t],
                     "true_abs": true_abs, "true_pct": true_pct,
                     "pred_abs": pred_abs, "pred_pct": pred_pct}

    # perturbation similarity (correlations between predicted mean effect vectors)
    # use predicted (pred_bulk[p] - ctrl_mean) as effect vector
    effect_vecs = []
    for p in perts:
        effect_vecs.append(pred_bulk[p] - ctrl_mean)
    effect_mat = np.stack(effect_vecs, axis=0)  # (K,G)
    # pairwise Pearson correlation matrix
    K = effect_mat.shape[0]
    # normalize
    em = effect_mat - effect_mat.mean(axis=1, keepdims=True)
    denom = np.sqrt((em ** 2).sum(axis=1, keepdims=True)) + 1e-8
    emn = em / denom
    corr_mat = emn @ emn.T  # (K,K)
    # take upper triangle excluding diagonal
    iu = np.triu_indices(K, k=1)
    mean_corr = float(corr_mat[iu].mean()) if iu[0].size > 0 else np.nan
    min_corr = float(corr_mat[iu].min()) if iu[0].size > 0 else np.nan

    # PDS (Perturbation Discrimination Score)
    # Distance between predicted pseudobulk for p and true pseudobulks for t
    # Exclude the target gene of *each* perturbation in the distance (both p's and t's, if present).
    # Rank of true t==p among all t (ascending distance). PDS_p = 1 - (rank-1)/(K-1).
    # Overall PDS = mean over p.
    # Build target indexes per perturbation (or -1 if N/A)
    t_idx_per_pert = {p: t2gi.get(p, -1) for p in perts}
    true_bulk_mat = np.stack([true_bulk[p] for p in perts], axis=0)  # (K,G)
    pred_bulk_mat = np.stack([pred_bulk[p] for p in perts], axis=0)  # (K,G)

    # precompute masks per pair to exclude targets
    PDS_scores = []
    for i, p in enumerate(perts):
        # distances to every t
        dists = []
        for j, tname in enumerate(perts):
            mask = np.ones(G, dtype=bool)
            ti = t_idx_per_pert[p]
            tj = t_idx_per_pert[tname]
            if ti >= 0: mask[ti] = False
            if tj >= 0: mask[tj] = False
            # L1 distance over masked genes
            d = np.abs(pred_bulk_mat[i, mask] - true_bulk_mat[j, mask]).sum()
            dists.append(d)
        dists = np.asarray(dists)
        # rank of the true target (j==i) in ascending distances
        order = np.argsort(dists)
        rank = int(np.where(order == i)[0][0]) + 1  # 1-based
        Kp = len(perts)
        PDS_p = 1.0 if Kp == 1 else (1.0 - (rank - 1) / (Kp - 1))
        PDS_scores.append(PDS_p)

    PDS_mean = float(np.mean(PDS_scores)) if len(PDS_scores) > 0 else np.nan

    # ---- Print concise report ----
    print("\n=== Evaluation ===")
    print(f"Per-perturbation MAE (mean ± sd): {np.mean(list(mae_per_pert.values())):.5f} ± {np.std(list(mae_per_pert.values())):.5f}")
    print(f"Perturbation similarity (pred mean effects): mean corr={mean_corr:.4f}, min corr={min_corr:.4f}")
    print(f"PDS (mean over perts): {PDS_mean:.4f}")
    print("\nKnockdown efficiency per perturbation (target gene, true_abs, true_pct, pred_abs, pred_pct):")
    # show a few lines sorted by true_abs descending
    preview = sorted(kd_eff.items(), key=lambda kv: (np.nan_to_num(kv[1]['true_abs'], nan=-1e9)), reverse=True)
    for p, d in preview[: min(10, len(preview))]:
        tg = d['target_gene'] or "N/A"
        print(f"  {p:20s}  tg={tg:12s}  true_abs={d['true_abs']:.4f}  true_pct={d['true_pct']:.2%}  "
              f"pred_abs={d['pred_abs']:.4f}  pred_pct={d['pred_pct']:.2%}")

    return {
        "mae_per_pert": mae_per_pert,
        "kd_eff": kd_eff,
        "mean_corr_pred_effects": mean_corr,
        "min_corr_pred_effects": min_corr,
        "PDS_mean": PDS_mean,
        "PDS_scores": dict(zip(perts, PDS_scores)),
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Step0-aware MPNN to fit perturbed gene vectors on a small panel.")
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--target_label", default="target_gene", help="obs column with perturbation labels.")
    ap.add_argument("--control_label", default="non-targeting", help="label value for control cells.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--T", type=int, default=2, help="Number of message-passing steps.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_target", type=float, default=0.1)
    ap.add_argument("--weight_local", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.0, help="Step-0 anchor (e.g., 0.0 for CRISPRi).")
    ap.add_argument("--use_pseudobulk", action="store_true",
                    help="Collapse to one mean row per perturbation (incl. control).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    adata = ad.read_h5ad(args.in_h5ad)
    if args.use_pseudobulk:
        adata = collapse_to_pseudobulk(adata, args.target_label)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if sparse.isspmatrix(adata.X) and not sparse.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()  # nicer slicing, though we load to numpy anyway

    model = train(
        adata=adata,
        target_label=args.target_label,
        control_label=args.control_label,
        hidden=args.hidden,
        T=args.T,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_target=args.weight_target,
        weight_local=args.weight_local,
        seed=args.seed,
        tau=args.tau,
        device=args.device,
    )

    # Evaluate on the same panel (training fit quality)
    _ = evaluate_model(
        adata=adata,
        model=model,
        target_label=args.target_label,
        control_label=args.control_label,
        device=args.device,
        batch_size=512,
        seed=args.seed,
    )

    # Optional: save weights
    out_path = os.path.splitext(args.in_h5ad)[0] + f".mpnn_hidden{args.hidden}_T{args.T}.pt"
    torch.save({"state_dict": model.state_dict(),
                "G": model.G,
                "hidden": model.hidden,
                "T": model.T}, out_path)
    print(f"[done] saved model to {out_path}")

if __name__ == "__main__":
    main()
