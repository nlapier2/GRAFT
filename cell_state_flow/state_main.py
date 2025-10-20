import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import scanpy as sc
import anndata
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
from sklearn.metrics import r2_score
from torchdiffeq import odeint
from scipy.stats import pearsonr

from models import *

def vae_loss_function(x_hat, x, mu, log_var, beta):
    """
    Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
    """
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Normalize KL divergence by batch size to make it independent of batch size
    kl_divergence /= x.size(0)

    return reconstruction_loss + beta * kl_divergence

def calculate_reconstruction_r2(model, dataloader, device):
    """Calculates the gene-wise R^2 score for the model's reconstructions."""
    model.eval()
    all_original = []
    all_reconstructed = []
    with torch.no_grad():
        for data_batch, in dataloader:
            data_batch = data_batch.to(device)
            x_hat, _, _ = model(data_batch)
            all_original.append(data_batch.cpu().numpy())
            all_reconstructed.append(x_hat.cpu().numpy())
    
    original_matrix = np.concatenate(all_original, axis=0)
    reconstructed_matrix = np.concatenate(all_reconstructed, axis=0)
    
    r2 = r2_score(original_matrix, reconstructed_matrix, multioutput='variance_weighted')
    print(f"\n--- Validation Metric: Reconstruction R^2 ---")
    print(f"Mean R^2 (variance-weighted): {r2:.4f}")

def calculate_pseudobulk_r2(model, dataloader, device):
    """Calculates the R^2 score on the pseudobulked profiles."""
    model.eval()
    all_original = []
    all_reconstructed = []
    with torch.no_grad():
        for data_batch, in dataloader:
            data_batch = data_batch.to(device)
            x_hat, _, _ = model(data_batch)
            all_original.append(data_batch.cpu().numpy())
            all_reconstructed.append(x_hat.cpu().numpy())
    
    # Create full matrices from batches
    original_matrix = np.concatenate(all_original, axis=0)
    reconstructed_matrix = np.concatenate(all_reconstructed, axis=0)
    
    # Create pseudobulk profiles by summing over cells
    original_pseudobulk = original_matrix.sum(axis=0)
    reconstructed_pseudobulk = reconstructed_matrix.sum(axis=0)
    
    # Calculate R^2 on the pseudobulk vectors
    r2 = r2_score(original_pseudobulk, reconstructed_pseudobulk)
    print(f"Pseudobulk R^2: {r2:.4f}")

def plot_latent_umap(model, dataloader, device, plot_file):
    """
    Generates a UMAP comparing the latent space of original vs. reconstructed data.
    """
    model.eval()
    print(f"\n--- Validation Plot: Generating UMAP ---")
    all_original_z = []
    all_reconstructed_z = []
    
    with torch.no_grad():
        for data_batch, in dataloader:
            data_batch = data_batch.to(device)
            
            # Get reconstructions
            x_hat, _, _ = model(data_batch)
            
            # Encode original and reconstructed data to get their latent representations
            mu_original, _ = torch.chunk(model.encoder(data_batch), 2, dim=-1)
            mu_reconstructed, _ = torch.chunk(model.encoder(x_hat), 2, dim=-1)
            
            all_original_z.append(mu_original.cpu().numpy())
            all_reconstructed_z.append(mu_reconstructed.cpu().numpy())

    # Concatenate all batches
    original_z = np.concatenate(all_original_z, axis=0)
    reconstructed_z = np.concatenate(all_reconstructed_z, axis=0)
    
    # Combine for UMAP
    combined_z = np.concatenate([original_z, reconstructed_z], axis=0)
    
    # Create an anndata object for UMAP
    n_obs = original_z.shape[0]
    source_labels = ['Original'] * n_obs + ['Reconstructed'] * n_obs
    latent_adata = anndata.AnnData(combined_z, obs={'source': pd.Categorical(source_labels)})

    # Compute and plot UMAP
    sc.pp.neighbors(latent_adata, use_rep='X')
    sc.tl.umap(latent_adata)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(latent_adata, color='source', ax=ax, show=False, title="Latent Space UMAP: Original vs. Reconstructed")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close(fig)
    print(f"UMAP plot saved to '{plot_file}'")

def plot_reconstruction_umap(model, dataloader, device, plot_file):
    """
    Generates a UMAP comparing the original vs. reconstructed data in gene space.
    """
    model.eval()
    print(f"\n--- Validation Plot: Generating Gene Space Reconstruction UMAP ---")
    all_original_x = []
    all_reconstructed_x = []

    with torch.no_grad():
        for data_batch, in dataloader:
            data_batch = data_batch.to(device)
            x_hat, _, _ = model(data_batch)
            all_original_x.append(data_batch.cpu().numpy())
            all_reconstructed_x.append(x_hat.cpu().numpy())

    # Concatenate all batches
    original_x = np.concatenate(all_original_x, axis=0)
    reconstructed_x = np.concatenate(all_reconstructed_x, axis=0)

    # Combine for UMAP
    combined_x = np.concatenate([original_x, reconstructed_x], axis=0)
    n_obs = original_x.shape[0]
    source_labels = ['Original'] * n_obs + ['Reconstructed'] * n_obs
    adata_recon = anndata.AnnData(combined_x, obs={'source': pd.Categorical(source_labels)})

    sc.pp.neighbors(adata_recon, use_rep='X')
    sc.tl.umap(adata_recon)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_recon, color='source', ax=ax, show=False, title="Gene Space UMAP: Original vs. Reconstructed")
    plt.savefig(plot_file)
    plt.close(fig)
    print(f"Gene space reconstruction UMAP plot saved to '{plot_file}'")

def generate_samples(model, n_samples, device):
    """Generates new cell samples using the trained flow model."""
    model.eval()
    with torch.no_grad():
        # 1. Sample z0 from the prior (Gaussian noise)
        # Infer n_latent from the model architecture
        n_latent = model.flow_model.net[0].in_features - model.flow_model.time_embed.dim
        z0 = torch.randn(n_samples, n_latent).to(device)

        # Define the ODE function for the solver, which expects inputs (t, z)
        def ode_func(t, z):
            # Our model expects a batch of t values, so we expand the scalar t
            t_batch = t.expand(z.size(0))
            return model.flow_model(z, t_batch)

        # 2. Integrate from t=0 to t=1 to generate z1
        t_span = torch.tensor([0.0, 1.0], device=device)
        z1 = odeint(ode_func, z0, t_span, method='dopri5')[1] # We only need the final state at t=1

        # 3. Decode z1 to get the generated cell expression
        x_fake = model.decoder(z1)

    return x_fake.cpu().numpy()

def validate_generative_quality(model, adata_control, device, plot_file):
    """Performs visual and quantitative checks on the generative quality."""
    print("\n--- Phase 2.2: Validating Generative Quality ---")
    n_samples = adata_control.n_obs

    # 1. Generate fake samples
    fake_data = generate_samples(model, n_samples, device)

    # 2. Visual Check: UMAP
    if hasattr(adata_control.X, "toarray"):
        real_data = adata_control.X.toarray()
    else:
        real_data = adata_control.X

    combined_data = np.concatenate([real_data, fake_data], axis=0)
    source_labels = ['Real'] * n_samples + ['Generated'] * n_samples
    adata_combined = anndata.AnnData(combined_data, obs={'source': pd.Categorical(source_labels)})

    sc.pp.neighbors(adata_combined, use_rep='X')
    sc.tl.umap(adata_combined)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_combined, color='source', ax=ax, show=False, title="Generative Quality: Real vs. Generated Cells")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close(fig)
    print(f"Generative UMAP plot saved to '{plot_file}'")

    # 3. Quantitative Check: Gene-Gene Correlation
    real_corr = np.corrcoef(real_data, rowvar=False)
    fake_corr = np.corrcoef(fake_data, rowvar=False)
    corr_score, _ = pearsonr(real_corr[np.triu_indices_from(real_corr, k=1)], fake_corr[np.triu_indices_from(fake_corr, k=1)])
    print(f"Gene-Gene Correlation Score (Pearson R): {corr_score:.4f}")

def plot_latent_generative_umap(model, dataloader, device, plot_file):
    """
    Generates a UMAP comparing the latent space of real encoded data vs. data
    generated by the flow model (before decoding). This is a key diagnostic plot.
    """
    model.eval()
    print(f"\n--- Diagnostic Plot: Generating Latent Space Generative UMAP ---")

    # 1. Generate latent vectors from the flow model (z_gen)
    n_samples = sum(len(batch[0]) for batch in dataloader)
    n_latent = model.flow_model.net[0].in_features - model.flow_model.time_embed.dim
    z0 = torch.randn(n_samples, n_latent).to(device)

    def ode_func(t, z):
        t_batch = t.expand(z.size(0))
        return model.flow_model(z, t_batch)

    t_span = torch.tensor([0.0, 1.0], device=device)
    with torch.no_grad():
        z_gen = odeint(ode_func, z0, t_span, method='dopri5')[1].cpu().numpy()

    # 2. Get latent vectors from real data (z_real)
    all_real_z = []
    with torch.no_grad():
        for data_batch, in dataloader:
            data_batch = data_batch.to(device)
            mu_real, _ = torch.chunk(model.encoder(data_batch), 2, dim=-1)
            all_real_z.append(mu_real.cpu().numpy())
    z_real = np.concatenate(all_real_z, axis=0)

    # 3. Combine, create AnnData, and plot UMAP
    combined_z = np.concatenate([z_real, z_gen], axis=0)
    source_labels = ['Real Latent'] * len(z_real) + ['Generated Latent'] * len(z_gen)
    adata_latent_gen = anndata.AnnData(combined_z, obs={'source': pd.Categorical(source_labels)})

    sc.pp.neighbors(adata_latent_gen, use_rep='X')
    sc.tl.umap(adata_latent_gen)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_latent_gen, color='source', ax=ax, show=False, title="Latent Space Diagnostic: Real vs. Generated")
    plt.savefig(plot_file)
    plt.close(fig)
    print(f"Latent generative UMAP plot saved to '{plot_file}'")

    print(f"\n--- Diagnostic Plot: Generating Latent Space Generative UMAP ---")

    # 1. Generate latent vectors from the flow model (z_gen)
    n_samples = sum(len(batch[0]) for batch in dataloader)
    n_latent = model.flow_model.net[0].in_features - model.flow_model.time_embed.dim
    z0 = torch.randn(n_samples, n_latent).to(device)

    def ode_func(t, z):
        t_batch = t.expand(z.size(0))
        return model.flow_model(z, t_batch)

    t_span = torch.tensor([0.0, 1.0], device=device)
    with torch.no_grad():
        z_gen = odeint(ode_func, z0, t_span, method='dopri5')[1].cpu().numpy()

    # 2. Get latent vectors from real data (z_real)
    all_real_z = []
    with torch.no_grad():
        for data_batch, in dataloader:
            data_batch = data_batch.to(device)
            mu_real, _ = torch.chunk(model.encoder(data_batch), 2, dim=-1)
            all_real_z.append(mu_real.cpu().numpy())
    z_real = np.concatenate(all_real_z, axis=0)

    # 3. Combine, create AnnData, and plot UMAP
    combined_z = np.concatenate([z_real, z_gen], axis=0)
    source_labels = ['Real Latent'] * len(z_real) + ['Generated Latent'] * len(z_gen)
    adata_latent_gen = anndata.AnnData(combined_z, obs={'source': pd.Categorical(source_labels)})

    sc.pp.neighbors(adata_latent_gen, use_rep='X')
    sc.tl.umap(adata_latent_gen)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_latent_gen, color='source', ax=ax, show=False, title="Latent Space Diagnostic: Real vs. Generated")
    plt.savefig(plot_file)
    plt.close(fig)
    print(f"Latent generative UMAP plot saved to '{plot_file}'")

def main(args):
    """
    Main function to run the VAE training pipeline.
    """
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Running with arguments: {args}")

    # --- 1.1: Data Preparation ---
    print("\n--- Phase 1.1: Preparing Data ---")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Data file not found at: {args.input}")
    
    adata = sc.read_h5ad(args.input)

    # Filter for control cells
    control_adata = adata[adata.obs[args.target_label] == args.control_label].copy()
    print(f"Found {control_adata.n_obs} control cells.")
    
    if control_adata.n_obs == 0:
        raise ValueError(f"No control cells found with label '{args.control_label}' in column '{args.target_label}'. Check your arguments.")

    # Normalize and log-transform
    sc.pp.normalize_total(control_adata)
    sc.pp.log1p(control_adata)
    print("Data normalized and log-transformed.")

    # Convert to PyTorch Tensor
    # Handle both sparse and dense arrays
    if hasattr(control_adata.X, "toarray"):
        data_matrix = control_adata.X.toarray()
    else:
        data_matrix = control_adata.X
        
    data_tensor = torch.tensor(data_matrix, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)

    # Create train/validation split
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")


    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # --- 2.1: VAE + Flow Model Training ---
    print("\n--- Phase 2.1: Training VAE + Flow Model ---")
    n_genes = data_tensor.shape[1]
    model = VAEFlowModel(n_genes=n_genes, n_latent=args.n_latent, n_hidden=args.n_hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # --- Training Step ---
        model.train()
        running_train_loss = 0.0
        running_train_vae_loss = 0.0
        running_train_flow_loss = 0.0
        for data_batch, in train_loader:
            data_batch = data_batch.to(DEVICE)

            # Forward pass
            x_hat, mu, log_var = model(data_batch)

            # Calculate individual losses
            vae_loss = calculate_vae_loss(x_hat, data_batch, mu, log_var, args.beta)
            if epoch < args.vae_warmup_epochs:
                total_loss = vae_loss
                flow_loss = torch.tensor(0.0) # For logging purposes
            else:
                # We use mu as z1, the stable center of the encoded distribution
                flow_loss = calculate_flow_matching_loss(model.flow_model, mu)
                # Combine into a single loss
                total_loss = calculate_composite_loss(vae_loss, flow_loss, args.gamma)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_train_loss += total_loss.item()
            running_train_vae_loss += vae_loss.item()
            running_train_flow_loss += flow_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_vae_loss = running_train_vae_loss / len(train_loader)
        avg_train_flow_loss = running_train_flow_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Step ---
        model.eval()
        running_val_loss = 0.0
        running_val_vae_loss = 0.0
        running_val_flow_loss = 0.0
        with torch.no_grad():
            for data_batch, in val_loader:
                data_batch = data_batch.to(DEVICE)
                x_hat, mu, log_var = model(data_batch)

                # Calculate individual losses for validation
                vae_loss = calculate_vae_loss(x_hat, data_batch, mu, log_var, args.beta)
                if epoch < args.vae_warmup_epochs:
                    total_loss = vae_loss
                    flow_loss = torch.tensor(0.0)
                else:
                    flow_loss = calculate_flow_matching_loss(model.flow_model, mu)
                    total_loss = calculate_composite_loss(vae_loss, flow_loss, args.gamma)

                running_val_loss += total_loss.item()
                running_val_vae_loss += vae_loss.item()
                running_val_flow_loss += flow_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_vae_loss = running_val_vae_loss / len(val_loader)
        avg_val_flow_loss = running_val_flow_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        stage = "Warm-up" if epoch < args.vae_warmup_epochs else "Joint"
        print(f"Epoch [{epoch+1}/{args.epochs}, {stage}], "
              f"Train Total: {avg_train_loss:.4f} (VAE: {avg_train_vae_loss:.4f}, Flow: {avg_train_flow_loss:.4f}), "
              f"Val Total: {avg_val_loss:.4f} (VAE: {avg_val_vae_loss:.4f}, Flow: {avg_val_flow_loss:.4f})")

    # --- Plotting and Saving Loss Curves ---
    print(f"\nTraining complete. Saving loss curves to '{args.plot_file}'...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('VAE Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.plot_file)
    plt.close()

    # --- 1.3: Validation Metrics ---
    # Use the validation set for R^2 calculation
    calculate_reconstruction_r2(model, val_loader, DEVICE)
    calculate_pseudobulk_r2(model, val_loader, DEVICE)
    
    # Use the validation set for UMAP visualization
    plot_latent_umap(model, val_loader, DEVICE, args.umap_plot_file)
    plot_reconstruction_umap(model, val_loader, DEVICE, args.reconstruction_plot_file)

    # --- 2.2: Generative Quality Validation ---
    validate_generative_quality(model, control_adata, DEVICE, args.generative_plot_file)
    # Add the call to the new diagnostic plot function
    plot_latent_generative_umap(model, val_loader, DEVICE, args.latent_generative_plot_file)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VAE on single-cell data.")
    
    # Required Arguments
    parser.add_argument('--input', type=str, required=True, help='Path to the input .h5ad file.')
    
    # Optional Arguments
    parser.add_argument('--plot_file', type=str, default='loss_curves.png', help='Path to save the output loss curve plot.')
    parser.add_argument('--umap_plot_file', type=str, default='umap.png', help='Path to save the output UMAP plot.')
    parser.add_argument('--reconstruction_plot_file', type=str, default='reconstruction_umap.png', help='Path to save the gene space reconstruction UMAP plot.')
    parser.add_argument('--generative_plot_file', type=str, default='generative_umap.png', help='Path to save the generative quality UMAP plot.')
    parser.add_argument('--latent_generative_plot_file', type=str, default='latent_generative_umap.png', help='Path to save the latent space generative quality UMAP plot.')
    parser.add_argument('--target_label', type=str, default='target_gene', help='The column name in adata.obs that contains perturbation information.')
    parser.add_argument('--control_label', type=str, default='non-targeting', help='The value in the target_label column that indicates a control cell.')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of the data to use for the training set.')
    parser.add_argument('--n_latent', type=int, default=64, help='Dimensionality of the latent space.')
    parser.add_argument('--n_hidden', type=int, default=256, help='Number of nodes in the hidden layers.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--vae_warmup_epochs', type=int, default=25, help='Number of epochs to train only the VAE before joint training.')
    parser.add_argument('--beta', type=float, default=1e-5, help='Weight for the KL divergence term in the VAE loss.')
    parser.add_argument('--gamma', type=float, default=1.0, help='Weight for the Flow Matching loss term.')

    args = parser.parse_args()
    main(args)
