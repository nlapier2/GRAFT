import warnings
# Suppress annoying FutureWarning from scanpy
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import scanpy as sc
import anndata
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from sklearn.decomposition import PCA
from torchdiffeq import odeint
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment

# --- Import from your existing models file ---
from models import ConditionalFlowModel, ConditionalFlowFiLM
from flow_utils import *
from load_pathways import *

def calculate_pca_metrics(pca, val_data):
    """
    Calculates and prints the reconstruction R^2 (single-cell and pseudobulk)
    for the PCA transformation itself.
    """
    print("\n--- PCA Baseline Metrics ---")
    # "Encode" and then "decode" the validation data
    latent_data = pca.transform(val_data)
    reconstructed_data = pca.inverse_transform(latent_data)

    # Calculate per-cell R^2
    r2_per_cell = r2_score(val_data, reconstructed_data, multioutput='variance_weighted')
    print(f"PCA Reconstruction R^2 (per-cell, variance-weighted): {r2_per_cell:.4f}")

    # Calculate pseudobulk R^2
    real_pseudobulk = val_data.sum(axis=0)
    recon_pseudobulk = reconstructed_data.sum(axis=0)
    r2_pseudobulk = r2_score(real_pseudobulk, recon_pseudobulk)
    print(f"PCA Reconstruction R^2 (pseudobulk): {r2_pseudobulk:.4f}")

def main(args):
    """
    Main function to run the PCA + Flow model diagnostic test.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 1. Data Preparation ---
    print("\n--- Phase 1: Preparing Data ---")
    adata_full = sc.read_h5ad(args.input)
    if args.target_label not in adata_full.obs:
        raise ValueError(f"Column '{args.target_label}' not found in adata.obs")
    sc.pp.normalize_total(adata_full, inplace=True)
    sc.pp.log1p(adata_full)
    # We don't pseudobulk in this script
    args.use_pseudobulk = False
    args.remove_non_gene_perts = False
    adata_train, adata_test, _ = train_test_split(args, adata_full, pb_target=None)  # uses seed / pct / external test
    print(f"[Split] train={adata_train.n_obs} rows, test={(adata_test.n_obs if adata_test is not None else 0)} rows.")  # :contentReference[oaicite:2]{index=2}
    # Matrix views
    X_train = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else adata_train.X
    X_test  = (adata_test.X.toarray()  if (adata_test is not None and hasattr(adata_test.X, "toarray")) else
               (adata_test.X if adata_test is not None else None))

    n = X_train.shape[0]
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(0.9 * n)
    train_indices = idx[:split]
    val_indices   = idx[split:]
    train_data = X_train[train_indices]
    val_data   = X_train[val_indices]

    if args.no_pca:
        # --- 2. Bypass PCA ---
        print("\n--- Phase 2: Bypassing PCA (Flow on Gene Space) ---")
        # Override n_latent to be the number of genes
        args.n_latent = train_data.shape[1]

        # Create a dummy PCA-like object that just passes data through
        class PCAPassThrough:
            def transform(self, data): return data
            def inverse_transform(self, data): return data
        pca = PCAPassThrough()
        train_latent = train_data
        print(f"Flow model will be trained on {args.n_latent} genes directly.")
    else:
        # --- 2. PCA "Encoder/Decoder" ---
        print(f"\n--- Phase 2: Fitting PCA with {args.n_latent} components ---")
        pca = PCA(n_components=args.n_latent, random_state=0)
        pca.fit(train_data)                          # fit on TRAIN (all perts)
        train_latent = pca.transform(train_data)     # (N_train, L)
        print(f"Data transformed to PCA space. Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        # --- Whiten PCA latents (mean/std on TRAIN only) ---
        mu = train_latent.mean(axis=0, keepdims=True)
        std = train_latent.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1e-8, std)
        train_latent = (train_latent - mu) / std
        # Whiten VAL using same stats
        val_latent_w = (pca.transform(val_data) - mu) / std
        mu_all, std_all = mu.copy(), std.copy()  # for all later unwhitening

    # ---------------------------
    # Global label vocabulary (train ∪ test) so unseen perts have embeddings
    # ---------------------------
    ser_train = adata_train.obs[args.target_label].astype('category')
    if adata_test is not None:
        # Build a union of categories deterministically
        cats = sorted(set(ser_train.astype(str).tolist()) |
                      set(adata_test.obs[args.target_label].astype(str).tolist()))
    else:
        cats = sorted(set(ser_train.astype(str).tolist()))
    label_to_idx = {lab: i for i, lab in enumerate(cats)}
    n_perts = len(cats)
    print(f"[Labels] global vocab size = {n_perts} (includes held-out perts).")
    # Codes for TRAIN and VAL (TRAIN rows only)
    p_idx_all_train = np.array(
        [label_to_idx[s] for s in adata_train.obs[args.target_label].astype(str).values],
        dtype=np.int64,
    )
    p_idx_train = p_idx_all_train[train_indices]
    p_idx_val   = p_idx_all_train[val_indices]

    # ---------------------------
    # Pathway → gene-feature matrix → compressed gene embeddings (ALL genes)
    # ---------------------------
    gene_names = list(adata_full.var_names)  # reference order for all genes
    if args.pathway_cfg:
        print(f"\n--- Loading pathway sources from: {args.pathway_cfg} ---")
        srcs = load_pathway_sources(args.pathway_cfg)  # dict[name] -> {file,gene_col,pathway_col,format}
        if not srcs:
            print("[Pathways] No sources found in YAML; continuing without pathway features.")
            gene_emb = None
        else:
            # Build and concatenate pathway matrices (genes x pathways)
            mats = []
            for name, meta in srcs.items():
                pm = make_pathway_matrix(
                    file_name=meta['file'],
                    gene_col=meta['gene_col'],
                    pathway_col=meta['pathway_col'],
                    format=meta['format'],
                    var_names=gene_names,
                )  # shape: (n_genes, n_feats_name)
                print(f"[Pathways] Loaded '{name}' with shape {pm.shape}")
                mats.append(pm)
            feat_df = pd.concat(mats, axis=1) if len(mats) > 1 else mats[0]
            # Ensure gene row order matches var_names
            feat_df = feat_df.loc[gene_names]
            feat_mat = feat_df.to_numpy(dtype=np.float32)   # (n_genes, n_total_feats)
            # Compress to emb_dim using PCA over features
            print(f"[Pathways] Compressing {feat_mat.shape[1]} features → {args.emb_dim} dims via PCA")
            pca_features = PCA(n_components=min(args.emb_dim, min(feat_mat.shape)-1), random_state=0)
            gene_emb = pca_features.fit_transform(feat_mat)  # (n_genes, emb_dim_eff)
            # If emb_dim > computed (edge-case for tiny feats), pad with zeros
            if gene_emb.shape[1] < args.emb_dim:
                pad = np.zeros((gene_emb.shape[0], args.emb_dim - gene_emb.shape[1]), dtype=np.float32)
                gene_emb = np.hstack([gene_emb.astype(np.float32), pad])
            else:
                gene_emb = gene_emb.astype(np.float32)
            print(f"[Pathways] Gene embedding matrix: {gene_emb.shape}")
    else:
        print("[Pathways] No --pathway_cfg provided; will initialize embeddings randomly.")
        gene_emb = None

    # Map perturbation labels (which are target genes) → rows in gene_emb (or zeros)
    # Ensures **every label** (incl. unseen at train) has an embedding vector.
    per_label_init = np.zeros((n_perts, args.emb_dim), dtype=np.float32)
    if gene_emb is not None:
        gene_to_row = {g:i for i,g in enumerate(gene_names)}
        for lab, idx_lab in label_to_idx.items():
            if lab == args.control_label:
                per_label_init[idx_lab, :] = 0.0  # control embedding = zero
            else:
                ridx = gene_to_row.get(lab, None)
                per_label_init[idx_lab, :] = gene_emb[ridx, :] if ridx is not None else 0.0
    # If no pathway features loaded, we leave zeros; the embedding layer will remain trainable.

    train_dataset = TensorDataset(
        torch.tensor(train_latent, dtype=torch.float32),
        torch.tensor(p_idx_train, dtype=torch.long),
    )

    # Optional: class-balanced sampling could be added later; shuffle is fine to start.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # --- 3. Flow Model Training ---
    print("\n--- Phase 3: Training SINGLE CONDITIONAL Flow on PCA Latent (whitened) ---")
    flow_model = ConditionalFlowModel(
        n_latent=args.n_latent, n_hidden=args.n_hidden, n_perts=n_perts, emb_dim=args.emb_dim
    ).to(DEVICE)
    # flow_model = ConditionalFlowFiLM(
    #     n_latent=args.n_latent, n_hidden=args.n_hidden, n_perts=n_perts, emb_dim=args.emb_dim
    # ).to(DEVICE)
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=args.lr)

    # Initialize perturbation embedding table from pathway-derived gene embeddings (if available)
    if per_label_init is not None:
        with torch.no_grad():
            init_t = torch.from_numpy(per_label_init).to(flow_model.pert_embed.weight.device)
            if init_t.shape[1] != flow_model.pert_embed.embedding_dim:
                # Safety (shouldn’t hit because we used args.emb_dim): project or pad
                emb_dim = flow_model.pert_embed.embedding_dim
                if init_t.shape[1] > emb_dim:
                    init_t = init_t[:, :emb_dim]
                else:
                    pad = torch.zeros(init_t.size(0), emb_dim - init_t.size(1), device=init_t.device)
                    init_t = torch.cat([init_t, pad], dim=1)
            flow_model.pert_embed.weight.data.copy_(init_t)
        print("[Init] Perturbation embeddings initialized from pathway features (zeros for control/missing).")

    train_losses = []
    for epoch in range(args.epochs):
        flow_model.train()
        running_train_loss = 0.0
        for z1_batch, p_batch in train_loader:
            z1_batch = z1_batch.to(DEVICE)           # (B, L)
            p_batch  = p_batch.to(DEVICE)            # (B,)

            # Flow Matching loss (noise -> latent μ) conditioned on perturbation
            z0_batch = torch.randn_like(z1_batch)
            t = torch.rand(z1_batch.size(0), device=DEVICE)
            zt_batch = (1 - t.unsqueeze(1)) * z0_batch + t.unsqueeze(1) * z1_batch
            v_target = z1_batch - z0_batch
            v_pred = flow_model(zt_batch, t, p_batch)
            loss = F.mse_loss(v_pred, v_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}], Flow Loss: {avg_train_loss:.4f}")

    # --- 4. Validation: Generative Quality Check ---
    print("\n--- Phase 4: Validating Generative Quality ---")
    calculate_pca_metrics(pca, val_data)
    flow_model.eval()
    with torch.no_grad():
        # For quick VAL diagnostics, generate exactly |VAL| samples with matching labels
        n_val = val_data.shape[0]
        z0_gen = torch.randn(n_val, args.n_latent, device=DEVICE)
        p_val_t = torch.tensor(p_idx_val, dtype=torch.long, device=DEVICE)
        def ode_func(t, z):
            # Integrate conditionally; broadcast t and use p_val labels
            return flow_model(z, t.expand(z.size(0)), p_val_t)
        t_span = torch.tensor([0.0, 1.0], device=DEVICE)
        z1_gen_w = odeint(ode_func, z0_gen, t_span, method='dopri5')[1]   # (N_val, L) whitened
        generated_cells = pca.inverse_transform(z1_gen_w.cpu().numpy() * std + mu)
        generated_cells = np.maximum(generated_cells, 0.0)

    # (1) Rowwise (reference only, usually negative). Slice because we oversampled.
    n_val = val_data.shape[0]
    r2_per_cell_gen = r2_score(val_data, generated_cells[:n_val], multioutput='variance_weighted')

    # (2) kNN-matched R^2 in WHITENED PCA space (many-to-one)
    real_lat = pca.transform(val_data)
    real_lat_w = (real_lat - mu) / std
    gen_lat_w  = z1_gen_w.cpu().numpy()       # whitened VAL samples
    nn = NearestNeighbors(n_neighbors=1).fit(gen_lat_w)
    dist, idx = nn.kneighbors(real_lat_w)
    gen_matched_knn = generated_cells[idx[:,0]]
    r2_matched_knn = r2_score(val_data, gen_matched_knn, multioutput='variance_weighted')
    print(f"Generative R^2 (per-cell, variance-weighted) [kNN-matched]: {r2_matched_knn:.4f}")

    # Calculate pseudobulk R^2 for generated cells
    real_pseudobulk = val_data.sum(axis=0)
    gen_pseudobulk = generated_cells.sum(axis=0)
    r2_pseudobulk_gen = r2_score(real_pseudobulk, gen_pseudobulk)
    print(f"Generative R^2 (pseudobulk): {r2_pseudobulk_gen:.4f}")

    # Quantitative Check: Gene-Gene Correlation
    # Use np.nan_to_num to handle cases where a gene has zero variance
    real_corr = np.nan_to_num(np.corrcoef(val_data, rowvar=False))
    fake_corr = np.nan_to_num(np.corrcoef(generated_cells, rowvar=False))
    corr_score, _ = pearsonr(real_corr[np.triu_indices_from(real_corr, k=1)], fake_corr[np.triu_indices_from(fake_corr, k=1)])
    print(f"Generative Gene-Gene Correlation Score (Pearson R): {corr_score:.4f}")

    # (4) Distributional sanity: MMD on PCA latents (RBF kernel)
    def _rbf(x, y, gamma):
        # x: (n,d), y: (m,d)
        xx = (x**2).sum(1, keepdims=True)
        yy = (y**2).sum(1, keepdims=True)
        xy = x @ y.T
        d2 = xx - 2*xy + yy.T
        return np.exp(-gamma * d2)
    # median heuristic for gamma
    # Use same whitened spaces for MMD
    sub = min(2048, real_lat_w.shape[0], gen_lat_w.shape[0])
    pair_d2 = np.sum((real_lat_w[:sub] - gen_lat_w[:sub])**2, axis=1)
    median_d2 = np.median(pair_d2)
    gamma = 1.0 / (median_d2 + 1e-8)
    Kxx = _rbf(real_lat_w, real_lat_w, gamma)
    Kyy = _rbf(gen_lat_w,  gen_lat_w,  gamma)
    Kxy = _rbf(real_lat_w, gen_lat_w,  gamma)
    mmd2 = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
    print(f"MMD^2 (whitened PCA latent, RBF kernel): {mmd2:.6f}")

    # Create UMAP
    # For UMAP, subsample generated cells to match #real (for a balanced plot)
    if args.plot_file is not None:
        gen_for_plot = generated_cells
        combined_data = np.concatenate([val_data, gen_for_plot], axis=0)
        source_labels = ['Real'] * val_data.shape[0] + ['Generated'] * gen_for_plot.shape[0]
        adata_combined = anndata.AnnData(combined_data, obs={'source': pd.Categorical(source_labels)})
        
        sc.pp.neighbors(adata_combined, use_rep='X')
        sc.tl.umap(adata_combined)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sc.pl.umap(adata_combined, color='source', ax=ax, show=False, title="PCA + Flow Generative Quality")
        plt.savefig(args.plot_file)
        plt.close(fig)
        print(f"Generative UMAP for PCA baseline saved to '{args.plot_file}'")

    # -------------------------
    # Phase 5: EVALUATION on TEST (and optionally TRAIN) using your flow_utils
    #          + optional predicted/true .h5ads
    # -------------------------
    @torch.no_grad()
    def predict_for_split(adata_split):
        """Generate one sample per NON-control row, in the SAME ROW ORDER as adata_split[nonctrl]."""
        labels = adata_split.obs[args.target_label].astype(str).values
        nonctrl_mask = labels != args.control_label
        idx_non = np.nonzero(nonctrl_mask)[0]
        pert_names = labels[nonctrl_mask].tolist()  # row-aligned list
        # global control mean from TRAIN (not split)
        ctrl_mean = to_numpy(adata_train[adata_train.obs[args.target_label]==args.control_label].X).mean(axis=0)

        N_non = len(idx_non)
        G = adata_split.n_vars
        pred_mat = np.empty((N_non, G), dtype=np.float32)

        # iterate labels in order of first appearance among non-controls
        seen, ordered_labels = set(), []
        for lab in pert_names:
            if lab not in seen:
                seen.add(lab)
                ordered_labels.append(lab)

        for lab in ordered_labels:
            # positions within the non-control slice where this label occurs
            sel = np.where(labels[idx_non] == lab)[0]
            Ng = sel.size
            if Ng == 0:
                continue
            pid = label_to_idx.get(lab, 0)
            z0 = torch.randn(Ng, args.n_latent, device=DEVICE)
            pids = torch.full((Ng,), pid, dtype=torch.long, device=DEVICE)
            def ode_func_g(t, z):
                return flow_model(z, t.expand(z.size(0)), pids)
            tspan = torch.tensor([0.0, 1.0], device=DEVICE)
            z1w = odeint(ode_func_g, z0, tspan, method='dopri5')[1].cpu().numpy()   # (Ng, L)
            Xg  = pca.inverse_transform(z1w * std + mu)                              # (Ng, G)
            Xg  = np.maximum(Xg, 0.0)
            pred_mat[sel, :] = Xg

        true_mat = to_numpy(adata_split[nonctrl_mask].X).astype(np.float32, copy=False)
        return (pred_mat, true_mat, pert_names, ctrl_mean)

    if adata_test is not None:
        print("\n=== Final Evaluation on HELD-OUT TEST perturbations ===")
        pred_bundle = predict_for_split(adata_test)
        evaluate_model(adata=adata_test, args=args, pred_bundle=pred_bundle)
        if args.out_pred_h5ad:
            print(f"\n💾 Writing TEST predicted+true h5ads → {args.out_pred_h5ad}")
            adata_test_dense = adata_test.copy()
            adata_test_dense.X = to_numpy(adata_test_dense.X).astype(np.float32, copy=False)
            write_pred_true_h5ads(eval_adata=adata_test_dense, pred_bundle=pred_bundle,
                                  out_pred_h5ad=args.out_pred_h5ad,
                                  target_label=args.target_label,
                                  control_label=args.control_label)

    if args.eval_on_train:
        print("\n=== Evaluation on TRAIN split (fit on TRAIN) ===")
        pred_bundle_tr = predict_for_split(adata_train)
        evaluate_model(adata=adata_train, args=args, pred_bundle=pred_bundle_tr)
        if args.out_pred_h5ad_train:
            print(f"\n💾 Writing TRAIN predicted+true h5ads → {args.out_pred_h5ad_train}")
            adata_train_dense = adata_train.copy()
            adata_train_dense.X = to_numpy(adata_train_dense.X).astype(np.float32, copy=False)
            write_pred_true_h5ads(eval_adata=adata_train_dense, pred_bundle=pred_bundle_tr,
                                  out_pred_h5ad=args.out_pred_h5ad_train,
                                  target_label=args.target_label,
                                  control_label=args.control_label)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a PCA + Flow model diagnostic test.")
    # Required Arguments
    parser.add_argument('--input', type=str, required=True, help="Path to the input h5ad file.")
    # Model & Training Arguments
    parser.add_argument('--no_pca', action='store_true', help='Bypass PCA and run flow model directly on gene space.')
    parser.add_argument('--n_latent', type=int, default=64, help='Number of principal components (dimensionality of the latent space).')
    parser.add_argument('--n_hidden', type=int, default=256, help='Number of hidden units in the Flow Model.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--train_split', type=float, default=0.8, help="Fraction of data to use for training (0.0 to 1.0).")
    parser.add_argument('--emb_dim', type=int, default=256, help='Perturbation embedding dimension for the conditional flow.')
    parser.add_argument('--pathway_cfg', type=str, default='', help='YAML with pathway sources (file, gene_col, pathway_col, format).')
    # I/O Arguments
    parser.add_argument('--plot_file', type=str, default=None, help='Path to save the output generative UMAP plot.')
    parser.add_argument('--output_h5ad', type=str, default='conditional_generated.h5ad', help='ONE .h5ad stacking ORIGINAL and per-perturbation GENERATED cells.')
    parser.add_argument('--target_label', type=str, default='target_gene', help='The column name in adata.obs that contains perturbation information.')
    parser.add_argument('--control_label', type=str, default='non-targeting', help='The value in the target_label column that indicates a control cell.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_pct_perts', type=float, default=0.0, help='Fraction of *perturbation labels* to hold out (excl. control); 0.0 means no holdout.')
    parser.add_argument('--test_h5ad', type=str, default='', help='Optional external TEST .h5ad; if set, overrides --test_pct_perts.')
    parser.add_argument('--eval_on_train', action='store_true', help='Also evaluate on TRAIN split.')
    parser.add_argument('--out_pred_h5ad', type=str, default='', help='If set, write predicted+true .h5ads for TEST split (pred file and pred.true.h5ad).')
    parser.add_argument('--out_pred_h5ad_train', type=str, default='', help='If set, also write predicted+true .h5ads for TRAIN split.')

    args = parser.parse_args()
    main(args)
