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

# --- 1.2: VAE Model Architecture ---

class VAE(nn.Module):
    """A Variational Autoencoder implemented in vanilla PyTorch."""
    def __init__(self, n_genes, n_latent=128, n_hidden=512):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_latent * 2) # Outputs mu and log_var
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_genes)
        )

    def reparameterize(self, mu, log_var):
        """
        Performs the reparameterization trick to allow for backpropagation.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Defines the forward pass of the VAE.
        """
        # Encode
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

def vae_loss_function(x_hat, x, mu, log_var, beta):
    """
    Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
    """
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Normalize KL divergence by batch size to make it independent of batch size
    kl_divergence /= x.size(0)

    return reconstruction_loss + beta * kl_divergence

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

    # --- 1.2: VAE Training ---
    print("\n--- Phase 1.2: Training VAE ---")
    n_genes = data_tensor.shape[1]
    model = VAE(n_genes=n_genes, n_latent=args.n_latent, n_hidden=args.n_hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # --- Training Step ---
        model.train()
        running_train_loss = 0.0
        for data_batch, in train_loader:
            data_batch = data_batch.to(DEVICE)

            # Forward pass
            x_hat, mu, log_var = model(data_batch)
            loss = vae_loss_function(x_hat, data_batch, mu, log_var, args.beta)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Step ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data_batch, in val_loader:
                data_batch = data_batch.to(DEVICE)
                x_hat, mu, log_var = model(data_batch)
                loss = vae_loss_function(x_hat, data_batch, mu, log_var, args.beta)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VAE on single-cell data.")
    
    # Required Arguments
    parser.add_argument('--input', type=str, required=True, help='Path to the input .h5ad file.')
    
    # Optional Arguments
    parser.add_argument('--plot_file', type=str, default='loss_curves.png', help='Path to save the output loss curve plot.')
    parser.add_argument('--target_label', type=str, default='target_gene', help='The column name in adata.obs that contains perturbation information.')
    parser.add_argument('--control_label', type=str, default='non-targeting', help='The value in the target_label column that indicates a control cell.')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of the data to use for the training set.')
    parser.add_argument('--n_latent', type=int, default=64, help='Dimensionality of the latent space.')
    parser.add_argument('--n_hidden', type=int, default=256, help='Number of nodes in the hidden layers.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--beta', type=float, default=0.0005, help='Weight for the KL divergence term in the VAE loss.')
    
    args = parser.parse_args()
    main(args)
