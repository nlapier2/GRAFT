#!/usr/bin/env python3
import math
from typing import Optional
import torch
import torch.nn as nn


# ----------------------------
# Model: Step0 + MPNN + Readout
# ----------------------------
class Step0Head(nn.Module):
    """
    Predict efficacy alpha_t from the target embedding and APPLY the counts-space clamp.
    Returns the clamped input x0 and the efficacy vector alpha_t.
    """
    def __init__(self, tau: float = 0.0, d: int = 64):
        super().__init__()
        self.alpha_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.alpha_mean_train: torch.Tensor | None = None  # set externally if available
        self.tau = tau

    def forward(
        self,
        x_ctrl: torch.Tensor,             # (B,G), log1p
        target_idx: torch.Tensor,         # (B,), -1 for control
        e_t: torch.Tensor,                # (B,d) target embedding
        alpha_cap: float = 1.0,
        mean_shrink: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Learn alpha_t = sigmoid(MLP(e_t)), optionally shrink to train mean,
        then clamp the target coordinate in counts space and map back to log1p.
        """
        # 1) predict efficacy
        alpha_t = torch.sigmoid(self.alpha_head(e_t)).squeeze(-1)     # (B,)
        if self.alpha_mean_train is not None and mean_shrink > 0.0:
            alpha_t = (1 - mean_shrink) * alpha_t + mean_shrink * self.alpha_mean_train
        alpha_t = alpha_t.clamp(0.0, float(alpha_cap))

        # 2) counts-space clamp at the target
        x0 = x_ctrl.clone()
        mask = (target_idx >= 0)
        if mask.any():
            rows = torch.arange(x_ctrl.shape[0], device=x_ctrl.device)[mask]
            cols = target_idx[mask]
            counts = torch.expm1(x0[rows, cols].clamp_min(0.0))
            m = (1.0 - alpha_t[mask]).clamp(0.0, 1.0)
            new_counts = counts * m
            x0[rows, cols] = torch.log1p(new_counts.clamp_min(0.0))
        return x0, alpha_t

class PrototypeGenerator(nn.Module):
    """
    Gene-conditioned generator:
      - learns a per-gene embedding e_t in R^d
      - maps e_t to a prototype mean-effect vector b_t in R^G
      - predicts a Step-0 efficacy alpha_t in (0,1) from e_t (and optional meta covariates)
    """
    def __init__(self, G: int, d: int = 64, extra_cond_dim: int = 0):
        super().__init__()
        self.E = nn.Embedding(G, d)
        nn.init.normal_(self.E.weight, std=0.02)
        # MLP to produce a hidden code from [e_t, alpha, (optional z_d, z_ct)]
        self.phi = nn.Sequential(
            nn.Linear(d + 1 + extra_cond_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.W_out = nn.Linear(d, G, bias=False)
        # kept for backward-compat; may not be used when external alpha is provided
        self.alpha_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

    def forward(self, target_idx: torch.LongTensor, meta: torch.Tensor | None = None, 
                external_alpha: torch.Tensor | None = None, z_extra: torch.Tensor | None = None):
        """
        target_idx: (B,) gene indices (>=0); if -1, we still produce something but it won't be used.
        meta: optional (B, M) covariates; ignored in this minimal version.
        Returns:
          b: (B,G) prototype vector
          alpha: (B,) efficacy in (0,1)
          e_t: (B,d) gene embeddings
        """
        e_t = self.E(torch.clamp(target_idx, min=0))    # (B,d)
        alpha = (torch.sigmoid(self.alpha_head(e_t)).squeeze(-1)
                 if external_alpha is None else external_alpha)       # (B,)
        # Nonlinear conditioning on efficacy (+ optional dataset/celltype embedding)
        if z_extra is None:
            z = torch.cat([e_t, alpha.unsqueeze(-1)], dim=-1)         # (B,d+1)
        else:
            z = torch.cat([e_t, alpha.unsqueeze(-1), z_extra], dim=-1) # (B,d+1+extra)
        h = self.phi(z)                                               # (B,d)
        b = self.W_out(h)                                             # (B,G)
        return b, alpha, e_t


class MPNNLayer(nn.Module):
    """
    Basic MPNN layer with dense adjacency.
    h_in -> aggregate (A @ h_in) -> update with residual
    """
    def __init__(self, hidden_dim: int, mode: str = "concat"):
        super().__init__()
        self.msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mode = mode
        if self.mode == "concat":
            self.upd = nn.Linear(2 * hidden_dim, hidden_dim)
        else:  # gated add message rather than concat
            self.upd = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.beta = nn.Parameter(torch.tensor(0.5))  # start with a gentle mix of neighbor info
        self.gamma = nn.Parameter(torch.tensor(1.0)) # gate on message path

    def forward(self, h: torch.Tensor, A_batch: torch.Tensor, h_t_frozen: torch.Tensor) -> torch.Tensor:
        """
        h:        (B,G,C)
        A_batch:  (B,G,G) row-normalized, with row[target]=0 for each sample
        h_t_frozen: (B,1,C) the clamped target embedding to re-impose after update
        """
        # messages
        M = self.msg(h)                                     # (B,G,C)
        B, G, C = M.shape
        Agg = torch.mm(A_batch, M.permute(1,0,2).reshape(G, B*C)) \
                .reshape(G, B, C).permute(1,0,2).contiguous()     # (B,G,C)
        m = self.beta * Agg
        if self.mode == "concat":
            h_new = self.act(self.upd(torch.cat([h, m], dim=-1)))  # (B,G,C)
        else:  # add and gate
            h_new = self.act(self.upd(h + self.gamma * m))
        # residual
        h_out = self.norm(h + h_new)
        # re-impose frozen target state
        # gather: replace the row corresponding to target with frozen
        # h_t_frozen is provided already extracted as h[:, t, :].unsqueeze(1) after Step-0 embed
        # We assume caller already zeroed inbound to t in A_batch.
        # Concatenate by slicing to avoid scatter for speed on small G
        # (But we need indices; we’ll do it in the caller for clarity.)
        return h_out
    
class SparseSpMPLayer(nn.Module):
    """
    Sparse MPNN: messages = SpMM(A, msg(h)); update = GELU(upd([h, beta*messages])); residual+norm.
    Uses learnable per-edge logits (passed in) and row-normalizes weights each forward.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg = nn.Linear(in_dim, out_dim, bias=False)       # like dense.msg
        self.upd = nn.Linear(2 * out_dim, out_dim, bias=True)   # like dense.upd
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.tensor(0.5))             # gate like dense.beta

    def forward(self, h: torch.Tensor,
                rowptr: torch.Tensor, colind: torch.Tensor, rows: torch.Tensor,
                edge_logit: torch.Tensor) -> torch.Tensor:
        """
        h: (B,G,D)
        rowptr:(G+1,), colind:(E,), rows:(E,), edge_logit:(E,)
        """
        B, G, D = h.shape

        # 1) Per-node message transform
        M = self.msg(h)                                         # (B,G,D)

        # 2) Build row-normalized edge weights from logits
        w = torch.sigmoid(edge_logit)                           # (E,)
        row_sums = torch.zeros(G, dtype=w.dtype, device=w.device)
        row_sums.index_add_(0, rows, w)
        w = w / row_sums[rows].clamp_min(1e-8)

        # 3) SpMM in fp32 for cuSPARSE compatibility: Agg = A @ M
        M2 = M.permute(1, 0, 2).reshape(G, B * D)               # (G, B*D)
        A32 = torch.sparse_csr_tensor(rowptr, colind, w.to(torch.float32),
                                      size=(G, G), device=h.device, dtype=torch.float32)
        Agg2 = torch.sparse.mm(A32, M2.to(torch.float32))       # (G, B*D)
        Agg = Agg2.reshape(G, B, D).permute(1, 0, 2)            # (B,G,D)

        # 4) Gate + update on concat([h, m])
        m = self.beta * Agg
        h_new = self.act(self.upd(torch.cat([h, m], dim=-1)))   # (B,G,D)

        # 5) Residual + norm (dropout like dense)
        out = self.dropout(h_new)
        return self.norm(h + out)

class GeneMPNN(nn.Module):
    def __init__(self, G: int, A_base: torch.Tensor, device: str = "cuda", hidden: int = 128, T: int = 2, tau: float = 0.0, alpha_cap: float = 1.0, node_dim: int = 128,
                 prior_dim: int | None = None, dset_vocab: int = 0, dset_dim: int = 0, ct_vocab: int = 0, ct_dim: int = 0, proj_dim: int = 128,
                 use_sparse_topk: bool = False, topk_keep: int = 12, num_tokens: int = 0, token_dim: int = 0, learn_dense_edges: bool = False,
                 num_extra_perts: int = 0):
        super().__init__()
        self.G = G
        self.register_buffer('A_base', A_base.to(device), persistent=False)
        # Sparse CSR adjacency (set later via set_sparse_A)
        self.A_csr = None  # will become a torch.sparse_csr_tensor
        # Candidate CSR buffers for sparse path (filled later)
        self.register_buffer('csr_rowptr', None, persistent=False)
        self.register_buffer('csr_colind', None, persistent=False)
        self.register_buffer('csr_values', None, persistent=False)
        self.edge_logit = None                             # nn.Parameter of shape (E,)
        self.hidden = hidden
        self.T = T
        self.alpha_cap = alpha_cap
        self.proj_dim = proj_dim

        # variables for sparse message passing layers
        self.use_sparse_topk = use_sparse_topk
        self.learn_dense_edges = learn_dense_edges
        self.topk_keep = topk_keep
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        # per-gene node embeddings (used by all nodes, every batch)
        self.node_dim = node_dim
        self.node_E = nn.Embedding(G, self.node_dim)
        nn.init.normal_(self.node_E.weight, std=0.02)
        self.num_extra_perts = int(num_extra_perts)
        self.extra_E = nn.Embedding(self.num_extra_perts, node_dim) if self.num_extra_perts > 0 else None
        # Optional small projector: prior (R) -> node_dim
        self.prior_proj = nn.Linear(prior_dim, self.node_dim, bias=False) if prior_dim is not None else None
        # input becomes [log1p expression, node embedding] per gene
        self.embed = nn.Linear(1 + self.node_dim, hidden)

        # set layers
        self.layers = nn.ModuleList()
        for _ in range(T):
            if self.use_sparse_topk:
                self.layers.append(SparseSpMPLayer(hidden, hidden))
            else:
                self.layers.append(MPNNLayer(hidden, mode="concat"))

        self.readout = nn.Linear(hidden, 1)
        # Optional dataset/cell-type embeddings
        self.dset_E = nn.Embedding(dset_vocab, dset_dim) if dset_vocab > 0 and dset_dim > 0 else None
        self.ct_E   = nn.Embedding(ct_vocab,   ct_dim)   if ct_vocab   > 0 and ct_dim   > 0 else None
        extra_cond_dim = (dset_dim if self.dset_E is not None else 0) + (ct_dim if self.ct_E is not None else 0)
        # FiLM will see [e_t, alpha, (z_d), (z_ct)]
        self.film_gamma = nn.Linear(self.node_dim + 1 + extra_cond_dim, hidden)
        self.film_beta  = nn.Linear(self.node_dim + 1 + extra_cond_dim, hidden)
        # Step-0 head: predict efficiency and apply the clamp
        self.step0_head = Step0Head(tau=tau, d=self.node_dim)
        # gene-conditioned prototype & alpha; FiLM from gene embedding
        self.proto = PrototypeGenerator(G=G, d=self.node_dim, extra_cond_dim=extra_cond_dim)
        # Tie prototype embedding to the node embedding so both share the same e_t
        self.proto.E = self.node_E
        # --- contrastive projection head: R^G -> R^{proj_dim} ---
        self.delta_proj = nn.Sequential(
            nn.Linear(self.G, self.proj_dim),
            nn.GELU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )
        # --- NEW: context→query head (used as QUERIES for contrast) ---
        # Reuse existing embeddings if present; else make a tiny target embedding.
        if not hasattr(self, "node_E"):
            # Fallback target embedding (one row per gene/target)
            self.target_E = nn.Embedding(self.G, self.hidden)
            target_dim = self.hidden
        else:
            target_dim = self.node_E.embedding_dim
        dset_dim = getattr(getattr(self, "dset_E", None), "embedding_dim", 0)
        ct_dim   = getattr(getattr(self, "ct_E",   None), "embedding_dim", 0)
        ctx_dim  = target_dim + dset_dim + ct_dim + 1  # +1 for alpha scalar
        self.query_mlp = nn.Sequential(
            nn.Linear(ctx_dim, self.proj_dim),
            nn.GELU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )

        self.dense_edge_logit = None
        if self.learn_dense_edges:
            G = self.A_base.shape[0]
            # initialize from A_base (clipped to (0,1) then inverse-sigmoid)
            with torch.no_grad():
                prior = self.A_base.clamp(1e-6, 1.0 - 1e-6).float()
                init = torch.log(prior / (1.0 - prior))
            self.dense_edge_logit = nn.Parameter(init)  # (G, G)

    @torch.no_grad()
    def init_from_prior(self, W_meta: torch.Tensor):
        """
        W_meta: (R, G) tensor matching var_names order. Projects columns to node_dim and uses
        as an initialization for node_E. Requires prior_proj to be present.
        """
        assert self.prior_proj is not None, "prior_proj not initialized (set prior_dim when constructing GeneMPNN)."
        # Project each gene's R-dim vector to node_dim
        # W_meta.T: (G, R) -> (G, node_dim)
        E0 = self.prior_proj(W_meta.T)  # (G, node_dim)
        self.node_E.weight.copy_(E0)

    def set_candidate_csr(self, rowptr: torch.Tensor, colind: torch.Tensor, values: Optional[torch.Tensor] = None):
        # rowptr: (G+1,), colind: (E,), values: (E,) or None
        self.csr_rowptr = rowptr
        self.csr_colind = colind
        self.csr_values = values

    def dense_edge_l1(self) -> torch.Tensor:
        """Mean L1 of row-normalized dense edge weights σ(logit)."""
        if self.dense_edge_logit is None:
            return torch.tensor(0.0, device=self.A_base.device)
        w = torch.sigmoid(self.dense_edge_logit)             # (G,G)
        row_sums = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_sums
        return w_norm.abs().mean()

    def set_sparse_A(self, rowptr: torch.Tensor, colind: torch.Tensor, values: torch.Tensor):
        """
        Register CSR structure and initialize learnable per-edge weights from 'values'.
        rowptr:(G+1,), colind:(E,), values:(E,) – typically row-normalized priors in (0,1].
        """
        assert rowptr.ndim == 1 and colind.ndim == 1 and values.ndim == 1
        G = rowptr.numel() - 1
        E = colind.numel()
        assert int(rowptr[-1]) == E, "rowptr[-1] must equal number of edges"
        self.node_count = G
        # Store CSR structure
        self.csr_rowptr = rowptr
        self.csr_colind = colind
        # Precompute 'rows' index per edge for fast segment ops
        rows = torch.repeat_interleave(torch.arange(G, device=rowptr.device), rowptr[1:] - rowptr[:-1])
        self.csr_rows = rows
        # Initialize learnable edge logits from prior values (clamped to (0,1))
        v = values.clamp_(1e-6, 1 - 1e-6).float()
        self.edge_logit = nn.Parameter(torch.log(v / (1 - v)))  # inverse sigmoid
        # (Optional) keep a non-learnable A for sanity checks; not used in forward once weights are learnable.
        self.A_csr = None

    def edge_l1(self) -> torch.Tensor:
        """Mean L1 of current edge weights σ(logit). Handy for regularization."""
        return torch.sigmoid(self.edge_logit).abs().mean()

    def forward(self, x_ctrl: torch.Tensor, target_idx: torch.Tensor, A_base: torch.Tensor, pert_rowidx: torch.Tensor = None,
                dset_idx: torch.Tensor | None = None, ct_idx: torch.Tensor | None = None, pidx: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_ctrl: (B,G)
        target_idx: (B,) int tensor with -1 when the perturbation is NOT one of the G genes
        pidx:       (B,) int tensor with -1 when not a “non-gene” perturbation; otherwise index in extra_E
        A_base: (G,G) dense row-normalized base adjacency
        """
        device = x_ctrl.device
        B, G = x_ctrl.shape
        assert G == self.G

        # --- Step-0: learn efficacy and apply counts-space clamp in one place ---
        e_gene = self.node_E(torch.clamp(target_idx, min=0))                       # (B,64)
        if (pidx is not None) and (self.extra_E is not None):
            mask_extra = (pidx >= 0)                                            # (B,)
            e_extra = self.extra_E(torch.clamp(pidx, min=0))
            e_t = torch.where(mask_extra.unsqueeze(-1), e_extra, e_gene)        # (B, node_dim)
        else:
            mask_extra = torch.zeros(B, dtype=torch.bool, device=device)
            e_t = e_gene
        # hand mean alpha to the head if we have it
        if hasattr(self, "alpha_mean_train"):
            self.step0_head.alpha_mean_train = self.alpha_mean_train
        # Step-0: predict alpha from e_t but CLAMP only where target_idx>=0. Set alpha=0 for non-gene rows.
        x0, alpha_t = self.step0_head(x_ctrl, target_idx, e_t, alpha_cap=self.alpha_cap)  # clamps only rows with target_idx>=0
        alpha_t = torch.where(target_idx >= 0, alpha_t, torch.zeros_like(alpha_t))

        # Initial hidden state (shared 1->hidden linear applied per gene)
        # assemble per-node embeddings for all genes
        idx_all = torch.arange(G, device=device)
        node_feats = self.node_E(idx_all).unsqueeze(0).expand(B, G, -1)  # (B,G,node_dim)
        x_in = torch.cat([x0.unsqueeze(-1), node_feats], dim=-1)         # (B,G,1+node_dim)
        h = self.embed(x_in)
        # FiLM conditions on both the identity (e_t) and strength (alpha_t) of the hit
        # Gather optional conditioning from dataset / cell-type
        z_list = []
        if self.dset_E is not None and dset_idx is not None:
            z_list.append(self.dset_E(dset_idx))                      # (B, dset_dim)
        if self.ct_E is not None and ct_idx is not None:
            z_list.append(self.ct_E(ct_idx))                          # (B, ct_dim)
        z_extra = torch.cat(z_list, dim=-1) if len(z_list) > 0 else None
        # FiLM conditions on [e_t, alpha_t, (z_d), (z_ct)]
        e_aug = torch.cat([e_t, alpha_t.unsqueeze(-1)], dim=-1) if z_extra is None else torch.cat([e_t, alpha_t.unsqueeze(-1), z_extra], dim=-1)
        gamma = torch.tanh(self.film_gamma(e_aug)).unsqueeze(1)       # (B,1,H)
        beta  = self.film_beta(e_aug).unsqueeze(1)                    # (B,1,H)
        g1 = gamma.add(1)          # (B,1,H)  small; avoids building (B,G,H)
        h.mul_(g1).add_(beta)      # in-place on h; uses broadcast, no extra buffers

        # Use a single shared adjacency (G,G) for the whole batch.
        A_batch = A_base.to(device)  # (G,G), requires_grad=False

        # Prepare to re-impose the frozen target state after each layer.
        freeze_mask = (target_idx >= 0)
        if freeze_mask.any():
            rows = torch.arange(B, device=device)[freeze_mask]
            cols = target_idx[freeze_mask]

        # Save the frozen target embedding (after Step-0 embed)
        # If some samples lack known target, we’ll just skip the replacement.
        h_t0 = torch.zeros(B, 1, self.hidden, device=device)
        if freeze_mask.any():
            h_t0[freeze_mask] = h[rows, cols].unsqueeze(1)

        # Run T layers with reimposition of target state
        for layer in self.layers:
            if freeze_mask.any():
                h_t_prev = h[rows, cols].clone()     # snapshot target rows BEFORE this layer

            if self.use_sparse_topk:
                assert (self.csr_rowptr is not None) and (self.edge_logit is not None), \
                    "Sparse SpMM active but CSR or edge weights are not set. Call set_sparse_A()."
                h = layer(h, self.csr_rowptr, self.csr_colind, self.csr_rows, self.edge_logit)
            else:
                if self.learn_dense_edges:
                    # Build row-normalized per-edge weights W (G,G), then apply to all items in the batch
                    W = torch.softmax(self.dense_edge_logit, dim=1)  # (G,G)
                    A_eff = (A_batch * W).to(A_batch.dtype)              # (G,G), keep 2-D
                    h = layer(h, A_eff, h_t0)
                else:
                    h = layer(h, A_batch, h_t0)

            if freeze_mask.any():
                # put frozen target embedding back
                h[rows, cols] = h_t_prev    # h_t0[freeze_mask, 0]

        # Readout back to expression space
        y = self.readout(h).squeeze(-1)  # (B,G)
        # add gene-conditioned prototype mean-effect (now nonlinear in alpha_t and optional z_extra)
        b_proto, _, _ = self.proto(target_idx, meta=None, external_alpha=alpha_t, z_extra=z_extra)
        y = y + b_proto
        # Preserve Step-0 at the target: y_t := x0_t
        freeze_mask = (target_idx >= 0)
        if freeze_mask.any():
            rows = torch.arange(B, device=device)[freeze_mask]
            cols = target_idx[freeze_mask]
            y[rows, cols] = x0[rows, cols]
        return y, x0, alpha_t

    # --- helper to project deltas and L2-normalize for cosine similarity ---
    @torch.no_grad()
    def project_key(self, delta_vec: torch.Tensor) -> torch.Tensor:
        """
        Args: delta_vec: (..., G)
        Returns: (..., D) L2-normalized key embedding (no grad)
        """
        k = self.delta_proj(delta_vec)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-8)
        return k

    def project_query(self, delta_vec: torch.Tensor) -> torch.Tensor:
        """
        Same as project_key but with grad (for queries from predicted deltas).
        """
        q = self.delta_proj(delta_vec)
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
        return q

    # === Contrastive KEY projection (delta → emb) ===
    @torch.no_grad()
    def project_key(self, delta_vec: torch.Tensor) -> torch.Tensor:
        k = self.delta_proj(delta_vec)
        return torch.nn.functional.normalize(k, dim=-1, eps=1e-8)

    # === Contrastive QUERY from CONTEXT (target/dset/ct + alpha) ===
    def project_query_from_context(self,
                                   target_idx: torch.Tensor,
                                   alpha: torch.Tensor,
                                   dset_idx: torch.Tensor | None = None,
                                   ct_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
          target_idx: (B,) long
          alpha:      (B,) float
          dset_idx:   (B,) long or None
          ct_idx:     (B,) long or None
        Returns:
          q: (B, proj_dim) L2-normalized
        """
        parts = []
        # target (prefer node_E if it's a real embedding; else fallback)
        node_emb = getattr(self, "node_E", None)
        if isinstance(node_emb, torch.nn.Embedding):
            parts.append(node_emb(target_idx))          # (B, target_dim)
        else:
            parts.append(self.target_E(target_idx))     # fallback created in __init__

        # dataset embedding (optional)
        dset_emb = getattr(self, "dset_E", None)
        if (dset_idx is not None) and isinstance(dset_emb, torch.nn.Embedding):
            parts.append(dset_emb(dset_idx))

        # cell-type embedding (optional)
        ct_emb = getattr(self, "ct_E", None)
        if (ct_idx is not None) and isinstance(ct_emb, torch.nn.Embedding):
            parts.append(ct_emb(ct_idx))
        parts.append(alpha.unsqueeze(-1))  # (B,1)
        ctx = torch.cat(parts, dim=-1)
        q = self.query_mlp(ctx)
        return torch.nn.functional.normalize(q, dim=-1, eps=1e-8)
    
class SparseTopKAttentionLayer(nn.Module):
    """
    Message passing over a candidate CSR graph with learned attention and hard Top-K per node.
    Optional low-rank global mixer via R global tokens (genes <-> tokens).
    Memory-lean: no (B,G,G) tensors; no expansion of FiLM-like broadcasts.
    """
    def __init__(self, in_dim, out_dim, topk_keep=12, token_dim=0, num_tokens=0, dropout=0.0):
        super().__init__()
        self.topk_keep = topk_keep
        self.num_tokens = num_tokens
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Node projections for attention & messages
        self.q = nn.Linear(in_dim, out_dim, bias=False)
        self.k = nn.Linear(in_dim, out_dim, bias=False)
        self.v = nn.Linear(in_dim, out_dim, bias=False)
        self.out = nn.Linear(out_dim, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)

        # Optional global tokens (low-rank mixer)
        if num_tokens > 0:
            self.token_dim = token_dim if token_dim > 0 else out_dim
            self.tokens = nn.Parameter(torch.randn(num_tokens, self.token_dim) / math.sqrt(self.token_dim))
            self.t_read_q = nn.Linear(out_dim, self.token_dim, bias=False)   # genes -> tokens (Q)
            self.t_read_k = nn.Linear(out_dim, self.token_dim, bias=False)   # genes -> tokens (K)
            self.t_write_k = nn.Linear(self.token_dim, out_dim, bias=False)  # tokens -> genes (K)
            self.t_write_v = nn.Linear(self.token_dim, out_dim, bias=False)  # tokens -> genes (V)
            self.lambda_tokens = nn.Parameter(torch.tensor(0.2))             # mixing strength
        else:
            self.token_dim = 0

        self.act = nn.GELU()

    @torch.no_grad()
    def _topk_mask(self, row_ptr, col_idx, scores, k):
        """
        Given edge scores per row (flattened in CSR order), select Top-K edges per row.
        Returns a boolean mask over edges (same shape as col_idx) indicating which to keep.
        """
        device = scores.device
        E = col_idx.numel()
        G = row_ptr.numel() - 1
        keep_mask = torch.zeros(E, dtype=torch.bool, device=device)

        # We do a simple loop over rows (G) — works fine on GPU at 18k with k<<G.
        # Each row r: edges in [row_ptr[r]:row_ptr[r+1])
        for r in range(G):
            s = row_ptr[r].item()
            e = row_ptr[r + 1].item()
            if e <= s:
                continue
            kk = min(k, e - s)
            vals = scores[s:e]
            top_idx = torch.topk(vals, kk, largest=True, sorted=False).indices + s
            keep_mask[top_idx] = True
        return keep_mask

    def forward(self, h, csr_rowptr, csr_colind, csr_values=None, tokens_enabled=True):
        """
        h: (B,G,D)
        csr_rowptr: (G+1,)
        csr_colind: (E,)
        csr_values: (E,) or None
        """
        B, G, D = h.shape
        device = h.device

        Q = self.q(h)             # (B,G,D)
        K = self.k(h)             # (B,G,D)
        V = self.v(h)             # (B,G,D)

        out = torch.zeros(B, G, D, device=device, dtype=h.dtype)

        # ---- streaming per row; no (B,E,*) allocations ----
        sqrtD = math.sqrt(D)
        k_keep = self.topk_keep

        for r in range(G):
            s = int(csr_rowptr[r])
            e = int(csr_rowptr[r + 1])
            if e <= s:
                continue

            cols = csr_colind[s:e]            # (Er,)
            # gather just for this row's edges
            Kr = K[:, cols, :]                # (B,Er,D)
            Vr = V[:, cols, :]                # (B,Er,D)
            Qr = Q[:, r, :].unsqueeze(1)      # (B,1,D)

            # attention logits: (B,Er)
            att = (Qr * Kr).sum(-1) / sqrtD

            # optional prior bias
            if csr_values is not None:
                att = att + csr_values[s:e].unsqueeze(0)

            # select Top-K per row, using batch-mean scores to avoid storing (B,Er) twice
            scores_mean = att.mean(dim=0)     # (Er,)
            kk = min(k_keep, e - s)
            top_idx = torch.topk(scores_mean, kk, largest=True, sorted=False).indices  # (kk,)
            att_k = att[:, top_idx]           # (B,kk)
            Vr_k  = Vr[:, top_idx, :]         # (B,kk,D)

            # softmax weights within the kept set
            att_k = att_k - att_k.max(dim=1, keepdim=True).values
            w = torch.softmax(att_k, dim=1)   # (B,kk)

            # weighted sum -> (B,1,D) -> (B,D)
            out[:, r, :] = torch.bmm(w.unsqueeze(1), Vr_k).squeeze(1)

        # ---- optional global tokens (see memory-lean tweaks below) ----
        if tokens_enabled and self.num_tokens > 0:
            T = self.num_tokens
            Td = self.token_dim if self.token_dim > 0 else D

            Qg = self.t_read_q(h)            # (B,G,Td)
            Kg = self.t_read_k(h)            # (B,G,Td)

            # Shared read attention over genes per batch: (B,G)
            att_t = torch.softmax((Qg * Kg).sum(-1), dim=1)  # (B,G)

            # tokens param: (T,Td) -> (B,T,Td) view
            tokens = self.tokens.unsqueeze(0).expand(B, T, -1)  # (B,T,Td)

            # Memory-lean read: use einsum to avoid building (B,T,G)
            # agg = (B,Td) summary from genes, then broadcast to T tokens
            agg = torch.einsum('bg,bgd->bd', att_t, Qg)        # (B,Td)
            tokens = tokens + agg.unsqueeze(1)                 # (B,T,Td)

            # Write: tokens -> genes
            Kt = self.t_write_k(tokens)          # (B,T,D)
            Vt = self.t_write_v(tokens)          # (B,T,D)
            logits_gt = torch.einsum('bgd,btd->bgt', h, Kt) / math.sqrt(D)  # (B,G,T)
            att_gt = torch.softmax(logits_gt, dim=-1)                        # (B,G,T)
            glob = torch.einsum('bgt,btd->bgd', att_gt, Vt)                  # (B,G,D)

            out = out + torch.tanh(self.lambda_tokens) * glob

        out = self.out(out)
        out = self.dropout(out)
        out = self.norm(h + out)
        return out
