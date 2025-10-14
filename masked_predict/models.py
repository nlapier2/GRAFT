# In masked_main.py, after write_influence_scores_csv()

import argparse, math, os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from collections import defaultdict
from sklearn.metrics import pairwise_distances

import torch
from torch.utils.data import TensorDataset, DataLoader

from utils import to_numpy
from masked_main import write_influence_scores_csv

class PerturbationAutoencoder(torch.nn.Module):
    """
    An MLP autoencoder that learns to reconstruct perturbation response vectors,
    conditioned on the identity of the perturbed gene.
    """
    def __init__(self, num_genes: int, pert_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.pert_embedding = torch.nn.Embedding(num_genes, pert_embed_dim)
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_genes + pert_embed_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, num_genes),
        )

    def forward(self, x_delta: torch.Tensor, pert_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_delta: (B, G) tensor of (masked) response vectors.
            pert_idx: (B,) tensor of perturbation indices.
        """
        pert_emb = self.pert_embedding(pert_idx)
        
        # Concatenate the response vector with the perturbation embedding
        net_input = torch.cat([x_delta, pert_emb], dim=1)
        
        hidden = self.encoder(net_input)
        reconstruction = self.decoder(hidden)
        return reconstruction

def train_mlp_autoencoder(
    delta_vectors: torch.Tensor,
    pert_indices: torch.Tensor,
    model: PerturbationAutoencoder,
    args,
) -> PerturbationAutoencoder:
    """Trains the PerturbationAutoencoder on masked response vectors."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    
    dataset = TensorDataset(delta_vectors, pert_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print("  Training on {} perturbation response vectors...".format(len(dataset)))
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for delta_batch, p_idx_batch in loader:
            delta_batch = delta_batch.to(args.device)
            p_idx_batch = p_idx_batch.to(args.device)
            
            # Mask 15% of the genes in the response vector
            mask = (torch.rand(delta_batch.shape, device=args.device) > 0.15)
            delta_masked = delta_batch * mask
            
            # Reconstruct the full vector
            y_pred = model(delta_masked, p_idx_batch)
            
            # Loss is only calculated on the genes that were masked
            loss = loss_fn(y_pred[~mask], delta_batch[~mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  [Epoch {epoch:02d}] AE Reconstruction Loss: {avg_loss:.6f}")
            
    return model.eval()

@torch.no_grad()
def evaluate_mlp_reconstruction(
    delta_vectors: torch.Tensor,
    pert_indices: torch.Tensor,
    model: PerturbationAutoencoder,
    args,
):
    """
    Evaluates the model's ability to predict held-out gene responses.
    For each perturbation, it masks one gene at a time and records the prediction.
    """
    model.eval()
    num_perts, num_genes = delta_vectors.shape
    
    # We will store the true values and our model's predictions for the masked genes
    all_true_masked_values = []
    all_pred_masked_values = []
    
    for i in range(num_perts):
        # Get the response vector and index for one perturbation
        delta_true = delta_vectors[i:i+1].to(args.device)
        p_idx = pert_indices[i:i+1].to(args.device)
        
        # Leave-one-out: predict each gene's response given all others
        for j in range(num_genes):
            delta_masked = delta_true.clone()
            delta_masked[:, j] = 0.0 # Mask one gene
            
            y_pred = model(delta_masked, p_idx)
            
            all_true_masked_values.append(delta_true[0, j].item())
            all_pred_masked_values.append(y_pred[0, j].item())
            
    # Calculate Pearson correlation across all leave-one-out predictions
    true_vals = np.array(all_true_masked_values)
    pred_vals = np.array(all_pred_masked_values)
    
    corr = np.corrcoef(true_vals, pred_vals)[0, 1]
    
    print("\n📊 Evaluation of Held-Out Gene Response Prediction:")
    print(f"  - Pearson Correlation: {corr:.4f}")
    return corr

def run_mlp_autoencoder_flow(args, adata_train: ad.AnnData):
    """
    Orchestrates the training and evaluation of the interventional MLP autoencoder.
    """
    if not args.use_pseudobulk:
        raise ValueError("--model_type 'mlp_ae' requires the --use_pseudobulk flag.")

    # 1. Prepare data: Create response vectors (delta_p)
    print("\n=== Preparing Response Vectors from Pseudobulk Data ===")
    labels = adata_train.obs[args.target_label].astype(str)
    control_vec = to_numpy(adata_train[labels == args.control_label].X).mean(axis=0)
    
    pert_labels = sorted({p for p in labels if p != args.control_label})
    pert_map = {label: i for i, label in enumerate(pert_labels)}
    
    delta_vectors = []
    pert_indices = []
    
    for p_label, p_idx in pert_map.items():
        pert_vec = to_numpy(adata_train[labels == p_label].X).mean(axis=0)
        delta_vectors.append(pert_vec - control_vec)
        pert_indices.append(p_idx)
        
    delta_vectors = torch.from_numpy(np.array(delta_vectors, dtype=np.float32))
    pert_indices = torch.from_numpy(np.array(pert_indices, dtype=np.int64))

    # 2. Initialize and train the model
    print("\n=== 1. Training Interventional MLP Autoencoder (mlp_ae) ===")
    num_genes = adata_train.n_vars
    num_perts = len(pert_labels)
    
    model = PerturbationAutoencoder(
        num_genes=num_genes,
        pert_embed_dim=args.pert_embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    
    model = train_mlp_autoencoder(delta_vectors, pert_indices, model, args)

    # 3. Evaluate the model's ability to predict held-out responses
    print("\n=== 2. Evaluating Gene Response Reconstruction ===")
    evaluate_mlp_reconstruction(delta_vectors, pert_indices, model, args)

    if args.out_influence_csv:
        mlp_influence_matrix = compute_mlp_influence_matrix(
            model=model,
            delta_vectors=delta_vectors, # <-- Pass the full delta tensor
            num_perts=len(pert_labels),
            device=args.device
        )
        write_influence_scores_csv(
            influence_matrix=mlp_influence_matrix,
            adata=adata_train,
            output_path=args.out_influence_csv
        )
    return model

def compute_mlp_influence_matrix(
    model: PerturbationAutoencoder,
    delta_vectors: torch.Tensor, # <-- Pass in the data to compute the mean
    num_perts: int,
    device: str,
) -> np.ndarray:
    """
    Computes a (G, G) influence matrix from the trained MLP autoencoder
    using the Gradient Saliency method at the mean response vector.
    """
    print("\n🧠 Computing MLP influence matrix via Gradient Saliency...")
    model.eval()
    num_genes = delta_vectors.shape[1]

    # 1. Compute the baseline: the mean response vector across all perturbations
    delta_mean = delta_vectors.mean(dim=0, keepdim=True).to(device)
    delta_mean.requires_grad = True # IMPORTANT: Track gradients with respect to the input

    # 2. Use an average perturbation embedding as a neutral context
    avg_pert_idx = torch.arange(num_perts, device=device)
    avg_pert_embedding = model.pert_embedding(avg_pert_idx).mean(dim=0, keepdim=True)

    # 3. Get the baseline prediction
    net_input = torch.cat([delta_mean, avg_pert_embedding], dim=1)
    hidden = model.encoder(net_input)
    y_pred = model.decoder(hidden)

    influence_matrix = np.zeros((num_genes, num_genes))

    # 4. Iterate through each OUTPUT gene to compute its gradient w.r.t. all INPUT genes
    for i in range(num_genes):
        if model.training: model.zero_grad() # Zero out gradients in model, just in case
        if delta_mean.grad is not None:
            delta_mean.grad.zero_() # Zero out gradients on the input tensor

        # Backpropagate from the i-th output neuron.
        # retain_graph=True is crucial because we perform multiple backward passes
        # on the same computational graph.
        y_pred[0, i].backward(retain_graph=True)

        # The gradient delta_mean.grad now contains d(output_i) / d(input_j) for all j.
        # This vector is the influence of all input genes on the i-th output gene.
        grad_vector = delta_mean.grad.squeeze().cpu().numpy()
        influence_matrix[i, :] = grad_vector # This becomes the i-th ROW of the matrix

    print("   ...Done.")
    return influence_matrix