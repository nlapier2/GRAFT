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
from models import FlowModel

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
    adata = sc.read_h5ad(args.input)
    control_adata = adata[adata.obs[args.target_label] == args.control_label].copy()
    print(f"Found {control_adata.n_obs} control cells.")
    
    sc.pp.normalize_total(control_adata)
    sc.pp.log1p(control_adata)
    print("Data normalized and log-transformed.")

    full_data_matrix = control_adata.X.toarray() if hasattr(control_adata.X, "toarray") else control_adata.X
    
    # Create train/validation split from the numpy matrix
    train_size = int(args.train_split * len(full_data_matrix))
    val_size = len(full_data_matrix) - train_size
    train_indices, val_indices = random_split(range(len(full_data_matrix)), [train_size, val_size])
    
    train_data = full_data_matrix[train_indices.indices]
    val_data = full_data_matrix[val_indices.indices]

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
        pca = PCA(n_components=args.n_latent)
        # Fit PCA ONLY on the training data to avoid data leakage
        pca.fit(train_data)
        train_latent = pca.transform(train_data)
        print(f"Data transformed to PCA space. Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        # --- Whiten PCA latents (mean/std on TRAIN only) ---
        mu = train_latent.mean(axis=0, keepdims=True)
        std = train_latent.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1e-8, std)
        train_latent = (train_latent - mu) / std

    # Convert to PyTorch Tensor for training
    train_dataset = TensorDataset(torch.tensor(train_latent, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # --- 3. Flow Model Training ---
    print("\n--- Phase 3: Training Flow Model on PCA Latent Space ---")
    flow_model = FlowModel(n_latent=args.n_latent, n_hidden=args.n_hidden).to(DEVICE)
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=args.lr)

    train_losses = []
    for epoch in range(args.epochs):
        flow_model.train()
        running_train_loss = 0.0
        for z1_batch, in train_loader:
            z1_batch = z1_batch.to(DEVICE)
            
            # Flow Matching loss (noise -> PCA data)
            z0_batch = torch.randn_like(z1_batch)
            t = torch.rand(z1_batch.size(0), device=DEVICE)
            zt_batch = (1 - t.unsqueeze(1)) * z0_batch + t.unsqueeze(1) * z1_batch
            v_target = z1_batch - z0_batch
            v_pred = flow_model(zt_batch, t)
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
        # Generate an oversampled pool in WHITENED latent space
        n_val = val_data.shape[0]
        n_gen = int(max(1, args.oversample_factor) * n_val)
        z0_gen = torch.randn(n_gen, args.n_latent).to(DEVICE)
        
        def ode_func(t, z):
            t_batch = t.expand(z.size(0))
            return flow_model(z, t_batch)
        
        t_span = torch.tensor([0.0, 1.0], device=DEVICE)
        z1_gen_w = odeint(ode_func, z0_gen, t_span, method='dopri5')[1]
        # Unwhiten to PCA space, then inverse PCA to gene space
        z1_gen = z1_gen_w.cpu().numpy()
        generated_cells = pca.inverse_transform(z1_gen * std + mu)

    # (1) Rowwise (reference only, usually negative). Slice because we oversampled.
    n_val = val_data.shape[0]
    r2_per_cell_gen = r2_score(val_data, generated_cells[:n_val], multioutput='variance_weighted')

    # (2) kNN-matched R^2 in WHITENED PCA space (many-to-one)
    real_lat = pca.transform(val_data)
    real_lat_w = (real_lat - mu) / std
    gen_lat_w  = z1_gen  # already in whitened space
    nn = NearestNeighbors(n_neighbors=1).fit(gen_lat_w)
    dist, idx = nn.kneighbors(real_lat_w)
    gen_matched_knn = generated_cells[idx[:,0]]
    r2_matched_knn = r2_score(val_data, gen_matched_knn, multioutput='variance_weighted')
    print(f"Generative R^2 (per-cell, variance-weighted) [kNN-matched]: {r2_matched_knn:.4f}")

    # (3) One-to-one assignment (Hungarian) in WHITENED PCA space
    #     To keep memory bounded with oversampling, use only the nearest n_val generated points
    #     (optional refinement; keeps the spirit of 1-1 matching)
    nearest_cols = idx[:,0]                      # best candidate per real cell
    unique_cols = np.unique(nearest_cols)        # prune to unique candidates
    gen_sub = generated_cells[unique_cols]
    gen_sub_w = gen_lat_w[unique_cols]
    # Build cost between all real and pruned generated set
    cost = ((real_lat_w[:,None,:] - gen_sub_w[None,:,:])**2).sum(axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    gen_matched_hungarian = gen_sub[col_ind]
    r2_matched_h = r2_score(val_data[row_ind], gen_matched_hungarian, multioutput='variance_weighted')
    print(f"Generative R^2 (per-cell, variance-weighted) [Hungarian 1-1]: {r2_matched_h:.4f}")


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
    gen_for_plot = generated_cells[:val_data.shape[0]]
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

    if args.output_h5ad:
        print(f"\n--- Phase 5: Saving generated cells to '{args.output_h5ad}' ---")
        # Get the PCA reconstructions of the validation set
        pca_reconstructed_cells = pca.inverse_transform(pca.transform(val_data))

        # Combine all data matrices
        combined_output_data = np.concatenate([val_data, pca_reconstructed_cells, generated_cells], axis=0)

        # Create corresponding observation metadata
        source_labels = ['real'] * val_data.shape[0] + ['pca_reconstructed'] * pca_reconstructed_cells.shape[0] + ['flow_generated'] * generated_cells.shape[0]
        output_obs = pd.DataFrame({'source': source_labels})
        output_adata = anndata.AnnData(X=combined_output_data, obs=output_obs, var=control_adata.var)
        output_adata.write(args.output_h5ad)
        print("Saved.")

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
    parser.add_argument('--oversample_factor', type=float, default=1.0, help='How many generated samples relative to #val cells.')
    # I/O Arguments
    parser.add_argument('--plot_file', type=str, default='pca_flow_umap.png', help='Path to save the output generative UMAP plot.')
    parser.add_argument('--output_h5ad', type=str, default=None, help='Path to save an anndata file with generated cells for inspection.')
    parser.add_argument('--target_label', type=str, default='target_gene', help='The column name in adata.obs that contains perturbation information.')
    parser.add_argument('--control_label', type=str, default='non-targeting', help='The value in the target_label column that indicates a control cell.')
    
    args = parser.parse_args()
    main(args)
