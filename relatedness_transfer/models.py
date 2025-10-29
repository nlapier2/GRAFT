# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def directional_under_loss(y_true, y_pred, margin_scale=1.0, mag_thresh=0.0):
    """
    Penalize under-shooting true effect magnitude in the correct direction,
    but do not penalize overshooting in the correct direction.

    y_true: (P,G) tensor
    y_pred: (P,G) tensor
    margin_scale: allow "close enough" if we hit margin_scale * |y_true|
    mag_thresh: ignore tiny |y_true| values (below this absolute magnitude)
    """
    # sign of the real effect
    sign = torch.sign(y_true)  # (+1, 0, -1)

    # how far we moved in the correct direction
    margin = sign * y_pred  # positive if we move with the correct sign

    # how far we *should* move to avoid penalty
    target_mag = margin_scale * torch.abs(y_true)

    # how much we're still missing
    under_mag = target_mag - margin  # we want this <= 0

    # only care about genes with |y_true| above threshold
    mask = (torch.abs(y_true) > mag_thresh).float()

    under_loss = torch.clamp(under_mag, min=0.0) * mask
    # mean over (non-masked) entries
    denom = mask.sum().clamp(min=1.0)
    return under_loss.sum() / denom


def attention_stats(attn):
    """
    Returns:
      row_entropy: (P,) entropy per row
      topw:        (P,) top weight per row
      secondw:     (P,) second-highest weight per row
    """
    attn_clamped = torch.clamp(attn, 1e-8, 1.0)
    row_entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=1)

    top2_vals, _ = torch.topk(attn_clamped, k=2, dim=1)
    topw    = top2_vals[:, 0]
    secondw = top2_vals[:, 1]
    return row_entropy, topw, secondw



class AttentionRetriever(nn.Module):
    """
    Single-head attention over perturbations.

    We learn an embedding vector e_p for each perturbation p in the external dataset.
    For a query perturbation u, we compute attention weights over all donor perturbations p:

        score(u,p) = e_u · e_p
        alpha(u,·) = softmax(score(u,·), mask_self=True)

    and reconstruct u's external delta as:
        recon[u] = sum_p alpha(u,p) * Delta_src[p]

    where Delta_src[p] is the (G,)-dim effect vector for perturbation p in the EXTERNAL dataset.

    Phase 1 goal:
    - Train these embeddings to reconstruct each perturbation's external profile using
      other perturbations.
    - Then *reuse* alpha(u,p) to mix TARGET training deltas Y_O[p] and predict u in target.
    """

    def __init__(self, n_perts: int, embed_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_perts, embed_dim)
        # init embeddings small so softmax doesn't start ultra-peaky
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, Delta_src: torch.Tensor, exclude_self: bool = True):
        """
        Reconstruct ALL perturbations' external deltas from attention over ALL perturbations.

        Delta_src: (P, G) tensor of external deltas, row p = delta for pert p.
        exclude_self: if True, we mask self-attention (alpha(u,u)=0), then renormalize.

        Returns:
            recon: (P, G) predicted deltas for each perturbation
            attn:  (P, P) attention weights alpha(u,p) AFTER masking+renorm
        """
        # embeddings (P, H)
        E = self.emb.weight  # (P, H)
        # raw scores = E @ E^T  -> (P,P)
        scores = E @ E.T  # (P, P)

        if exclude_self:
            # mask diagonal so a perturbation can't just copy itself
            mask = torch.eye(scores.shape[0], device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(mask, float("-inf"))

        # row-wise softmax -> attention weights over donors
        attn = F.softmax(scores, dim=1)  # (P,P)

        # recon[u] = sum_p attn[u,p] * Delta_src[p]
        recon = attn @ Delta_src  # (P,G)

        return recon, attn

    @torch.no_grad()
    def get_attention_weights(self, exclude_self: bool = True):
        """
        Convenience method: just return the (P,P) attention matrix alpha(u,p)
        without also doing reconstruction.

        Returns:
            attn: (P,P) tensor on CPU
        """
        E = self.emb.weight  # (P,H)
        scores = E @ E.T  # (P,P)
        if exclude_self:
            mask = torch.eye(scores.shape[0], device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=1)
        return attn.detach().cpu()


def train_attention_retriever(
    Delta_src: np.ndarray,
    donor_idx: np.ndarray,       # NEW: indices of perts observed in target train (O_int)
    transfer_idx: np.ndarray,    # NEW: indices of perts we must predict (U_int \ O_int)
    embed_dim: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 0,
):
    """
    Train a single-head AttentionRetriever to reconstruct each external perturbation
    from other external perturbations.

    Args:
        Delta_src: (P, G) numpy array of external perturbation DELTAS (pseudobulk - control).
                   Row order defines the perturbation index. We'll keep that order consistent.
        embed_dim: embedding dimensionality for perturbation embeddings
        epochs:    simple number of full passes; dataset is tiny so full-batch is fine
        lr:        learning rate
        entropy_reg: coefficient to encourage *peaky* attention (low entropy),
                     which tends to give more distinctive signatures -> higher PDS.
                     Higher entropy_reg -> stronger push to be peaky.
        device:    "cuda" or "cpu"
        seed:      RNG seed for reproducibility

    Returns:
        model (AttentionRetriever) on CPU, in eval() mode
        final_attn (P,P) attention weights (torch.Tensor on CPU)
        recon_src (P,G) reconstruction of Delta_src by the trained model (numpy)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    P, G = Delta_src.shape
    Delta_t = torch.tensor(Delta_src, dtype=torch.float32, device=device)  # (P,G)

    donor_idx_t = torch.tensor(donor_idx, dtype=torch.long, device=device)
    transfer_idx_t = torch.tensor(transfer_idx, dtype=torch.long, device=device)

    model = AttentionRetriever(n_perts=P, embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()

        # -------- Two passes through the same embedding --------
        # donors can self-copy (stabilize anchors)
        recon_self,   attn_self   = model(Delta_t, exclude_self=False)  # (P,G),(P,P)
        # transfers must build themselves from others
        recon_noself, attn_noself = model(Delta_t, exclude_self=True)   # (P,G),(P,P)

        # -------- Build donor-only attention for transfers (simulate real inference) --------
        # Mask attention columns down to donors only, then renormalize row-wise.
        donor_mask = torch.zeros_like(attn_noself)          # (P,P)
        donor_mask[:, donor_idx_t] = 1.0
        attn_restrict = attn_noself * donor_mask            # zero out non-donor columns
        row_sums = attn_restrict.sum(dim=1, keepdim=True).clamp(min=1e-8)
        attn_restrict = attn_restrict / row_sums            # (P,P), rows now sum to 1 over donors

        # donor-only reconstruction for everyone (P,G)
        recon_restrict = attn_restrict @ Delta_t

        # -------- Loss 1: donor anchoring --------
        # Donor perts (the ones we actually observe in target) SHOULD keep a strong,
        # high-amplitude, directionally-correct signature. We allow self-copy here.
        loss_donor = directional_under_loss(
            y_true = Delta_t[donor_idx_t],
            y_pred = recon_self[donor_idx_t],
            margin_scale = 1.0,
            mag_thresh   = 0.1,
        )

        # -------- Loss 2: transfer supervision under true constraint --------
        # Transfer perts must be reconstructable *only from donor perts*, without self.
        # This is exactly what inference will do in the target domain.
        loss_transfer = directional_under_loss(
            y_true = Delta_t[transfer_idx_t],
            y_pred = recon_restrict[transfer_idx_t],
            margin_scale = 1.0,
            mag_thresh   = 0.1,
        )

        # -------- Loss 3: attention regularization on transfer rows --------
        # We regularize the donor-restricted attention for ONLY the transfer perts.
        # Goal:
        #   - avoid totally diffuse attention (entropy high),
        #   - avoid collapsing to a single donor (second weight ~0).
        if transfer_idx_t.numel() > 0:
            row_entropy, topw, secondw = attention_stats(attn_restrict[transfer_idx_t])
            loss_entropy   = row_entropy.mean()         # penalize diffuse "use everyone"
            penalty_tooone = (1.0 - secondw).mean()     # penalize pure 1-donor collapse
        else:
            # no transfers? then these terms are 0
            loss_entropy   = torch.tensor(0.0, device=device)
            penalty_tooone = torch.tensor(0.0, device=device)

        entropy_w = 0.1
        tooone_w  = 0.1
        loss_reg  = entropy_w * loss_entropy + tooone_w * penalty_tooone

        # -------- Total loss and step --------
        loss = loss_donor + loss_transfer + loss_reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 50 == 0 or ep == 0:
            with torch.no_grad():
                print(
                    f"[ATTN TRAIN] ep {ep+1} "
                    f"donor={loss_donor.item():.4f} "
                    f"transfer={loss_transfer.item():.4f} "
                    f"reg={loss_reg.item():.4f} "
                    f"total={loss.item():.4f}"
                )

    # -------- After training, produce both attention variants on CPU --------
    model.eval()
    with torch.no_grad():
        attn_allow_self = model.get_attention_weights(exclude_self=False).numpy().astype(np.float32)
        attn_no_self    = model.get_attention_weights(exclude_self=True).numpy().astype(np.float32)

    # Move model to CPU so caller can save/pickle
    model_cpu = AttentionRetriever(n_perts=P, embed_dim=embed_dim)
    model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()

    return model_cpu, attn_allow_self, attn_no_self


def build_target_predictions_from_attention(
    attn_no_self: np.ndarray,
    perts_all: list[str],
    O: list[str],
    U: list[str],
    Y_O: np.ndarray,
):
    """
    Use learned attention weights to predict DELTAS in the TARGET domain.

    Inputs:
        attn_weights: (P,P) attention matrix alpha(u,p) from train_attention_retriever(),
                      where rows/cols are aligned with perts_all.
                      Row = query perturbation u, col = donor perturbation p.
        perts_all:    list of ALL perturbations in the external space in the order
                      used to build Delta_src. Usually O + unseen-U.
        O:            list of observed target perts (train split, minus control)
        U:            list of eval/held-out perts we want to predict
        Y_O:          (|O|, G) target DELTA matrix for observed perts O
                      (pseudobulk - global_control_mean), same G as eval gene space.

    Output:
        Y_hat_U:      (|U|, G) predicted DELTAS for each u in U, in TARGET space.

    How it works:
        - For each u in U, we look at the learned attention row alpha(u,*).
        - We only keep donor weights corresponding to perts in O (the ones we actually saw in target).
        - Renormalize those weights over O so they sum to 1.
        - Form a convex (or near-convex) combo of the TARGET deltas Y_O.

    This is the key transfer step: "use externally learned mixing recipe, but mix *target* deltas".
    """
    idx = {p: i for i, p in enumerate(perts_all)}
    iO = np.array([idx[p] for p in O], dtype=int)  # donor indices in perts_all order
    iU = np.array([idx[p] for p in U], dtype=int)  # transfer/eval perts

    P = len(perts_all)
    G = Y_O.shape[1]

    # attn_no_self is (P,P); we now:
    #   - zero out columns not in O,
    #   - renormalize rows,
    #   - then for each u in U, take that row and mix Y_O.
    attn_full = attn_no_self.astype(np.float32, copy=True)  # (P,P)

    donor_mask = np.zeros_like(attn_full, dtype=np.float32)
    donor_mask[:, iO] = 1.0
    attn_restrict = attn_full * donor_mask  # zero out non-O columns

    row_sums = attn_restrict.sum(axis=1, keepdims=True) + 1e-8
    attn_restrict /= row_sums  # now each row is renormalized over O only

    # Now gather predictions for U:
    # For each u_idx in iU, we want weights over O (not over all P), so slice and matmul.
    # Extract just the donor columns for those rows:
    A_UO = attn_restrict[np.ix_(iU, iO)]   # (|U|, |O|)

    # Mix target deltas Y_O (|O|,G)
    Y_hat_U = A_UO @ Y_O                   # (|U|, G)

    return Y_hat_U


def attn_predict_from_external(
    adata_source,
    adata_train,
    adata_eval,
    target_label: str,
    control_label: str,
    ctrl_mean_target: np.ndarray | None = None,
    embed_dim: int = 64,
    epochs: int = 200,
    lr: float = 1e-3,
    entropy_reg: float = 1e-2,
    device: str = "cuda",
    seed: int = 0,
    boost_pcs: int = 0,
    boost_gamma: float = 0.6,
):
    """
    High-level wrapper that mimics krr_predict_from_external() but uses the
    learned attention retriever instead of a fixed kernel smoother.

    This does:
      1. Define O, U, perts_all the same way krr_predict_from_external does.
      2. Build external pseudobulk deltas for perts_all.
      3. Train AttentionRetriever on external deltas to learn attention weights.
      4. Build target deltas Y_O for O from adata_train (pseudobulk - ctrl_mean_target).
      5. Use the learned attention to synthesize Y_hat_U for U in TARGET space.
      6. Optionally apply subspace boosting (same idea as KRR tail).
      7. Add back ctrl_mean_target to get predicted EXPRESSION.
      8. Return (pred_mat, true_mat, pert_names, ctrl_mean_target).

    NOTE:
    - We do *not* yet do target-gene overwrite, clamping, confidence boosting,
      etc. You can apply those same transforms afterward in transfer_main
      exactly like you already do for KRR outputs.
    - We assume adata_source and adata_train/adata_eval are already gene-aligned
      (you're running --intersect_genes in Phase 1).

    Shapes:
      pred_mat: (|U|, G)
      true_mat: (|U|, G)   (from adata_eval)
      pert_names: list[str] length |U|
      ctrl_mean_target: (G,)
    """

    # ----- control mean from TRAIN split -----
    if ctrl_mean_target is None:
        train_mask = np.asarray(adata_train.obs[target_label] == control_label)
        ctrl_mean_target = np.asarray(adata_train.X)[train_mask].mean(axis=0).reshape(-1)

    G = adata_train.n_vars

    # perts in source (excluding control)
    P_src = (
        adata_source.obs[target_label]
        .astype(str)
        .unique()
        .tolist()
    )
    P_src = [p for p in P_src if p != control_label]

    # observed perts in target train
    O = (
        adata_train.obs[target_label]
        .astype(str)
        .unique()
        .tolist()
    )
    O = [p for p in O if p != control_label]

    # eval perts in target eval
    U = (
        adata_eval.obs[target_label]
        .astype(str)
        .unique()
        .tolist()
    )
    U = [p for p in U if p != control_label]

    # restrict everything to perturbations that actually exist in source, too
    O_int = [p for p in O if p in P_src]
    U_int = [p for p in U if p in P_src]

    # union order for training the attention model
    perts_all = O_int + [p for p in U_int if p not in O_int]

    # map perturbation -> row index in perts_all
    idx_map = {p: i for i, p in enumerate(perts_all)}

    donor_idx = np.array([idx_map[p] for p in O_int], dtype=int)
    transfer_idx = np.array([idx_map[p] for p in U_int if p not in O_int], dtype=int)

    # map perturbation -> external delta vector (pseudobulk - control) in SOURCE
    # We assume adata_source is already pseudobulked (1 row per pert),
    # just like in your current pipeline before KRR.
    # We'll compute deltas the same way krr_predict_from_external() does:
    def _ctrl_mean_src(adata_src):
        mask_c = np.asarray(adata_src.obs[target_label] == control_label)
        return np.asarray(adata_src.X)[mask_c].mean(axis=0).reshape(-1)

    ctrl_mean_src = _ctrl_mean_src(adata_source)

    def _pseudobulk_row(adataX, pert_name: str):
        v = adataX[adataX.obs[target_label] == pert_name].X
        v = np.asarray(v).reshape(-1, G).mean(axis=0)
        return v

    # external deltas for each perturbation in perts_all
    Delta_src_rows = []
    for p in perts_all:
        v = _pseudobulk_row(adata_source, p)  # now guaranteed non-empty
        Delta_src_rows.append(v - ctrl_mean_src)
    Delta_src = np.stack(Delta_src_rows, axis=0).astype(np.float32)

    print("ATTN DEBUG] Delta_src shape:", Delta_src.shape)
    print("ATTN DEBUG] Delta_src any NaN:", np.isnan(Delta_src).any(), 
        "any Inf:", np.isinf(Delta_src).any())

    row_norms = np.linalg.norm(Delta_src, axis=1)
    print("ATTN DEBUG] Delta_src row L2 norms (min/median/max):",
        float(row_norms.min()), float(np.median(row_norms)), float(row_norms.max()))

    pairwise_corr = np.corrcoef(Delta_src)
    offdiag = pairwise_corr - np.eye(pairwise_corr.shape[0])
    print("ATTN DEBUG] Delta_src offdiag corr stats (min/median/max):",
        float(offdiag.min()), float(np.median(offdiag)), float(offdiag.max()))

    # ----- train the attention retriever on external -----
    model, attn_allow_self, attn_no_self = train_attention_retriever(
        Delta_src=Delta_src,
        donor_idx=donor_idx,
        transfer_idx=transfer_idx,
        embed_dim=embed_dim,
        epochs=epochs,
        lr=lr,
        device=device,
        seed=seed,
    )

    # We'll mainly use attn_no_self for prediction of held-out perts.
    attn_np_no_self = attn_no_self.astype(np.float32)

    def summarize_attn(attn_mat, label):
        row_entropy = -(attn_mat * np.log(np.clip(attn_mat, 1e-12, 1.0))).sum(axis=1)
        print(f"[ATTN DEBUG] {label} row_entropy min/med/max:",
              float(row_entropy.min()), float(np.median(row_entropy)), float(row_entropy.max()))

        topw = attn_mat.max(axis=1)
        print(f"[ATTN DEBUG] {label} top weight min/med/max:",
              float(topw.min()), float(np.median(topw)), float(topw.max()))

        frac_10 = (attn_mat > 0.10).mean(axis=1)
        print(f"[ATTN DEBUG] {label} frac donors >10% weight min/med/max:",
              float(frac_10.min()), float(np.median(frac_10)), float(frac_10.max()))

    summarize_attn(attn_np_no_self, "no_self(restricted-before-pred)")

    # ----- build target deltas Y_O using TRAIN set -----
    def _delta_mat(adataX, perts: list[str]):
        rows = []
        for p in perts:
            v = _pseudobulk_row(adataX, p)
            rows.append(v - ctrl_mean_target)
        return np.stack(rows, axis=0).astype(np.float32)

    Y_O = _delta_mat(adata_train, O_int)  # (|O|, G)

    pair_corr_YO = np.corrcoef(Y_O)
    offdiag_YO = pair_corr_YO - np.eye(pair_corr_YO.shape[0])
    print("[ATTN DEBUG] Y_O offdiag corr stats (min/med/max):",
        float(offdiag_YO.min()), float(np.median(offdiag_YO)), float(offdiag_YO.max()))

    # ----- predict deltas for U in TARGET space using learned attention -----
    Y_hat_U = build_target_predictions_from_attention(
        attn_no_self=attn_np_no_self,
        perts_all=perts_all,
        O=O_int,
        U=U_int,
        Y_O=Y_O,
    )

    print("[ATTN DEBUG] Y_hat_U shape:", Y_hat_U.shape)

    # how different are predicted deltas across perts?
    pair_corr_pred = np.corrcoef(Y_hat_U)
    offdiag_pred = pair_corr_pred - np.eye(pair_corr_pred.shape[0])

    print("[ATTN DEBUG] Y_hat_U offdiag corr stats (min/med/max):",
        float(offdiag_pred.min()), float(np.median(offdiag_pred)), float(offdiag_pred.max()))

    row_norms_pred = np.linalg.norm(Y_hat_U, axis=1)
    print("[ATTN DEBUG] Y_hat_U row L2 norms (min/med/max):",
        float(row_norms_pred.min()), float(np.median(row_norms_pred)), float(row_norms_pred.max()))

    # ----- optional subspace boost (same spirit as KRR tail) -----
    if boost_pcs and boost_pcs > 0 and boost_gamma > 0:
        # subspace boosting needs Y_O to define "real" target-space perturbation variation.
        # We'll just inline a tiny version here to avoid circular imports.
        Yc = Y_O - Y_O.mean(axis=0, keepdims=True)  # (|O|,G)
        # economy SVD
        U_svd, S_svd, Vt_svd = np.linalg.svd(Yc, full_matrices=False)
        k = min(boost_pcs, Vt_svd.shape[0])
        V_top = Vt_svd[:k, :]                         # (k,G)
        proj = Y_hat_U @ V_top.T @ V_top              # project onto top-k PCs
        Y_hat_U = Y_hat_U + boost_gamma * proj

    # ----- convert to EXPRESSION levels; add back ctrl_mean_target -----
    pred_mat = Y_hat_U + ctrl_mean_target[None, :]  # (|U|,G)

    # ----- build true_mat for these U perts from adata_eval -----
    true_rows = []
    for p in U_int:
        v = _pseudobulk_row(adata_eval, p)
        true_rows.append(v)
    true_mat = np.stack(true_rows, axis=0).astype(np.float32)  # (|U|,G)

    return pred_mat, true_mat, U, ctrl_mean_target, O_int
