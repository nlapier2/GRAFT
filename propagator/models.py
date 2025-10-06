#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
 
class AveragePredictor(nn.Module):
    """
    Baseline: compute the average delta (perturbed - control_mean) on the TRAIN pseudobulk
    and predict that same delta for every test perturbation.
    Final prediction = control_mean + avg_delta.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("ctrl_mean", None)   # (G,)
        self.register_buffer("avg_delta", None)   # (G,)

    @torch.no_grad()
    def fit(self, adata, target_label: str, control_label: str, device: str = "cpu"):
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X = X.astype("float32", copy=False)
        labels = adata.obs[target_label].astype(str).values
        ctrl_idx = (labels == control_label)
        pert_idx = ~ctrl_idx
        if not ctrl_idx.any() or not pert_idx.any():
            raise ValueError("AveragePredictor.fit: need both control and perturbed rows in pseudobulk.")
        ctrl_mean = X[ctrl_idx].mean(axis=0)                           # (G,)
        deltas = X[pert_idx] - ctrl_mean[None, :]                      # (P,G)
        avg_delta = deltas.mean(axis=0)                                # (G,)
        self.ctrl_mean = torch.tensor(ctrl_mean, device=device)
        self.avg_delta = torch.tensor(avg_delta, device=device)

    @torch.no_grad()
    def predict(self, n_rows: int) -> torch.Tensor:
        """
        Return a (n_rows, G) tensor of predictions = ctrl_mean + avg_delta, repeated.
        Assumes fit() was called.
        """
        if (self.ctrl_mean is None) or (self.avg_delta is None):
            raise RuntimeError("AveragePredictor: call fit() before predict().")
        y = self.ctrl_mean + self.avg_delta                             # (G,)
        return y.unsqueeze(0).repeat(n_rows, 1)                         # (n_rows, G)


# -------------------------
# Target-spike + sparse propagation (Q/K -> TopK)
# -------------------------
class PropSpikeModel(nn.Module):
    """
    Step-0: seed = s_{p} * one_hot(target_gene)
    Propagate with a learned, sparse, row-stochastic W (CSR), built from Q/K Top-K scores.
    For Step-0, we keep W context-agnostic (global). You can FiLM it later by dataset/celltype if desired.
    """
    def __init__(
        self,
        num_genes: int,
        num_datasets: int,
        num_celltypes: int,
        gene_emb_dim: int = 64,
        qk_dim: int = 32,
        topk: int = 64,
        alpha: float = 0.3,
        T: int = 2,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.G = num_genes
        self.DS = num_datasets
        self.CT = num_celltypes
        self.h = gene_emb_dim
        self.dk = qk_dim
        self.topk = topk
        self.alpha = alpha
        self.T = T
        self.tau = temperature

        # Gene embeddings and Q/K projections
        self.gene_emb = nn.Embedding(self.G, self.h)
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        self.Q = nn.Linear(self.h, self.dk, bias=False)
        self.K = nn.Linear(self.h, self.dk, bias=False)

        # Simple scalar amplitude head s_p from (pert gene embedding, dataset, celltype)
        # Keep minimal for now; you can expand later.
        self.ds_emb = nn.Embedding(self.DS, 16)
        self.ct_emb = nn.Embedding(self.CT, 16)
        amp_in = self.h + 16 + 16
        self.amp = nn.Sequential(
            nn.Linear(amp_in, 64), nn.GELU(),
            nn.Linear(64, 1)  # signed scalar; CRISPRi should learn negative
        )

        # CSR graph buffers: rowptr (G+1), colind (E), values (E)
        self.register_buffer("csr_rowptr", None)
        self.register_buffer("csr_colind", None)
        self.register_buffer("csr_values", None)

        self.device = device
        self.to(device)

    @torch.no_grad()
    def build_sparse_graph(self, chunk_rows: int = 2048):
        """
        Build CSR from Q/K Top-K scores (global, context-agnostic).
        Scores s_ij = <Q u_i, K u_j> / sqrt(dk).
        We never materialize GxG: process in row chunks, keep Top-K per row.
        """
        G = self.G
        K = min(int(self.topk), max(G - 1, 1))  # TopK ≤ G-1, at least 1
        u = self.gene_emb.weight                        # (G,h)
        q = self.Q(u)                                   # (G,dk)
        k = self.K(u)                                   # (G,dk)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        rowptr = [0]
        colind = []
        values = []
        scale = (self.dk ** -0.5)

        for start in range(0, G, chunk_rows):
            end = min(G, start + chunk_rows)
            # (rows, dk) @ (dk, G) -> (rows, G)
            scores = (q[start:end] @ k.t()) * scale
            # prevent self-loop by setting diag to -inf in the relevant slice
            idx_rows = torch.arange(start, end, device=scores.device)
            scores[torch.arange(end - start), idx_rows] = float("-inf")
            # Top-K per row
            vals, idx = torch.topk(scores, k=K, dim=1)
            # Row-softmax with temperature
            vals = (vals / max(self.tau, 1e-6)).softmax(dim=1)  # (rows,K)
            # Append
            colind.append(idx)
            values.append(vals)
            for _ in range(end - start):
                rowptr.append(rowptr[-1] + K)

        colind = torch.cat(colind, dim=0)                                             # (G, K)
        if colind.dim() == 2:
            colind = colind.reshape(-1)                                               # (G*K,)
        colind = colind.to(device=self.device, dtype=torch.int64)

        values = torch.cat(values, dim=0).to(device=self.device, dtype=torch.float32) # (G, K)
        values = values.reshape(-1)                                             # (G*K,)
        rowptr = torch.tensor(rowptr, dtype=torch.int64, device=self.device)     # (G+1,)

        self.csr_rowptr = rowptr.contiguous()
        self.csr_colind = colind.contiguous()
        self.csr_values = values.contiguous()

    def _spmm(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B,G)  →  returns (B,G) equal to W @ y^T, transposed back.
        Implements CSR SpMM without torch.sparse.* kernels:
        out[b, i] = sum_{e in row i} values[e] * y[b, colind[e]]
        """
        assert self.csr_rowptr is not None, "Call build_sparse_graph() first."
        B, G = y.shape
        device = y.device
        dtype  = y.dtype

        rowptr = self.csr_rowptr.to(device)
        col    = self.csr_colind.to(device)
        w      = self.csr_values.to(device, dtype=torch.float32)  # keep weights fp32
        # degrees per row, shape (G,)
        deg = (rowptr[1:] - rowptr[:-1]).to(torch.long)
        # rows index per edge, shape (E,)
        rows = torch.repeat_interleave(torch.arange(G, device=device, dtype=torch.long), deg)
        # gather source features: y[:, col] → (B, E)
        src = y.index_select(dim=1, index=col)                     # (B, E)
        # weighted messages
        msg = src * w.unsqueeze(0)                                  # (B, E)
        # scatter-add into output along rows dim=1
        out = torch.zeros(B, G, device=device, dtype=dtype)
        out.scatter_add_(dim=1, index=rows.unsqueeze(0).expand(B, -1), src=msg)
        return out

    def forward(
        self,
        pert_idx: torch.Tensor,          # (B,) gene indices (long)
        dset_idx: torch.Tensor,          # (B,) dataset indices (long)
        ct_idx: torch.Tensor,            # (B,) celltype indices (long)
    ) -> torch.Tensor:
        """
        Returns predicted Δμ_hat: (B,G)
        """
        B, G = pert_idx.shape[0], self.G
        # step-0 amplitude s_{p}
        e_p = self.gene_emb(pert_idx)             # (B,h)
        e_d = self.ds_emb(dset_idx)               # (B,16)
        e_c = self.ct_emb(ct_idx)                 # (B,16)
        amp_in = torch.cat([e_p, e_d, e_c], dim=-1)    # (B, h+32)
        s = self.amp(amp_in).squeeze(-1)          # (B,)
        # seed: s * one_hot(p)
        y = torch.zeros(B, G, device=self.device, dtype=torch.float32)
        y[torch.arange(B, device=self.device), pert_idx] = s
        # Propagate with damped fixed-point
        for _ in range(self.T):
            y = (1.0 - self.alpha) * y + self.alpha * self._spmm(y)
            # Freeze target row influence by zeroing outbound from target row? (Optional later)
            # For now, we rely on no self-loop + damping.
        return y
