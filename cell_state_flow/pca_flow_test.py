import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import scanpy as sc
import anndata
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import os
import pandas as pd
from sklearn.decomposition import PCA
from torchdiffeq import odeint

# --- Import from your existing models file ---
from models import FlowModel

# Suppress annoying FutureWarning from scanpy
warnings.filterwarnings('ignore', category=FutureWarning)

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

    # --- 2. PCA "Encoder/Decoder" ---
    print(f"\n--- Phase 2: Fitting PCA with {args.n_latent} components ---")
    pca = PCA(n_components=args.n_latent)
    # Fit PCA ONLY on the training data to avoid data leakage
    pca.fit(train_data)
    
    # "Encode" the training data into the PCA latent space
    train_latent = pca.transform(train_data)
    
    # Convert to PyTorch Tensor for training
    train_dataset = TensorDataset(torch.tensor(train_latent, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Data transformed to PCA space. Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

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
    flow_model.eval()
    with torch.no_grad():
        # Generate samples in the latent space
        z0_gen = torch.randn(val_data.shape[0], args.n_latent).to(DEVICE)
        
        def ode_func(t, z):
            t_batch = t.expand(z.size(0))
            return flow_model(z, t_batch)
        
        t_span = torch.tensor([0.0, 1.0], device=DEVICE)
        z1_gen = odeint(ode_func, z0_gen, t_span, method='dopri5')[1]
        
        # "Decode" the generated latent vectors back to gene space using PCA
        generated_cells = pca.inverse_transform(z1_gen.cpu().numpy())

    # Create UMAP
    combined_data = np.concatenate([val_data, generated_cells], axis=0)
    source_labels = ['Real'] * val_data.shape[0] + ['Generated'] * generated_cells.shape[0]
    adata_combined = anndata.AnnData(combined_data, obs={'source': pd.Categorical(source_labels)})
    
    sc.pp.neighbors(adata_combined, use_rep='X')
    sc.tl.umap(adata_combined)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_combined, color='source', ax=ax, show=False, title="PCA + Flow Generative Quality")
    plt.savefig(args.plot_file)
    plt.close(fig)
    print(f"Generative UMAP for PCA baseline saved to '{args.plot_file}'")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a PCA + Flow model diagnostic test.")
    # Required Arguments
    parser.add_argument('--input', type=str, required=True, help="Path to the input h5ad file.")
    # Model & Training Arguments
    parser.add_argument('--n_latent', type=int, default=64, help='Number of principal components (dimensionality of the latent space).')
    parser.add_argument('--n_hidden', type=int, default=256, help='Number of hidden units in the Flow Model.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--train_split', type=float, default=0.8, help="Fraction of data to use for training (0.0 to 1.0).")
    # I/O Arguments
    parser.add_argument('--plot_file', type=str, default='pca_flow_umap.png', help='Path to save the output generative UMAP plot.')
    parser.add_argument('--target_label', type=str, default='target_gene', help='The column name in adata.obs that contains perturbation information.')
    parser.add_argument('--control_label', type=str, default='non-targeting', help='The value in the target_label column that indicates a control cell.')
    
    args = parser.parse_args()
    main(args)
