import argparse, math, os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from collections import defaultdict
from sklearn.metrics import pairwise_distances

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import to_numpy


class CausalGNN(nn.Module):
    """
    A GNN that simulates effect propagation using a dynamic, dot-product
    attention mechanism to compute context-aware gene relatedness.
    """
    def __init__(self, num_genes: int, embed_dim: int, hidden_dim: int,
                 num_steps: int, damping_factor: float = 1.0):
        super().__init__()
        self.num_genes = num_genes
        self.num_steps = num_steps
        self.damping_factor = damping_factor
        self.embed_dim = embed_dim

        # Shared embeddings for all genes in the dataset
        self.shared_embedding = nn.Embedding(num_genes, embed_dim)

        # Head to predict gene-specific knockdown effectiveness
        self.effectiveness_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # --- NEW: Linear layers for Query and Key projections ---
        # The input to these will be the gene's contextual state (embedding + delta)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.message_proj = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, pert_idx: torch.Tensor, control_expr_at_pert: torch.Tensor) -> torch.Tensor:
        """
        Predicts the full delta vector by simulating effect propagation with attention.
        """
        B = pert_idx.shape[0]
        device = pert_idx.device
        G = self.num_genes
        D = self.embed_dim

        # --- Initial State Calculation (same as before) ---
        pert_emb = self.shared_embedding(pert_idx)
        knockdown_alpha = self.effectiveness_head(pert_emb).squeeze(-1)
        initial_effect = -knockdown_alpha * control_expr_at_pert

        delta_state = torch.zeros(B, G, device=device)
        updates = torch.zeros(B, G, device=device)
        updates.scatter_(1, pert_idx.unsqueeze(1), initial_effect.unsqueeze(1))
        delta_state = delta_state + updates
        
        # Expand embeddings for batch operations
        all_embeddings = self.shared_embedding.weight.unsqueeze(0).expand(B, -1, -1)

        # --- Propagation Loop with Dynamic Attention ---
        for _ in range(self.num_steps):
            # 1. Create a contextual state representation for each gene
            # Combine the gene's fixed identity (embedding) with its dynamic state (delta)
            # Shape: (B, G, D)
            current_state_repr = all_embeddings + delta_state.unsqueeze(-1)

            # 2. Project to Query, Key, and VALUE vectors
            Q = self.q_proj(current_state_repr)
            K = self.k_proj(current_state_repr)
            # V represents the learnable MESSAGE each gene sends out
            V = self.v_proj(current_state_repr)

            # 3. Calculate attention scores and weights
            attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # 4. Aggregate the high-dimensional Value vectors using attention
            # attn_weights @ V -> (B, G, G) @ (B, G, D) -> (B, G, D)
            messages_high_dim = attn_weights @ V

            # 5. Project the high-dimensional messages down to a 1D update signal
            # (B, G, D) -> (B, G, 1) -> (B, G)
            messages_1d = self.message_proj(messages_high_dim).squeeze(-1)

            # 6. Update state and set the new "wave"
            delta_state = delta_state + self.damping_factor * messages_1d
            updates = self.damping_factor * messages_1d
            
        return delta_state

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
        # write_influence_scores_csv(
        #     influence_matrix=mlp_influence_matrix,
        #     adata=adata_train,
        #     output_path=args.out_influence_csv
        # )
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

# In your models.py file

import torch
import torch.nn as nn

# ... (your other model classes like PerturbationAutoencoder) ...

class DualHeadAutoencoder(nn.Module):
    """
    A multi-task model with a shared gene embedding layer and two heads:
    1. Reconstruction Head: Predicts masked genes in a response vector.
    2. Prediction Head: Predicts the full response vector from a gene's embedding.
    """
    def __init__(self, num_genes: int, pert_embed_dim: int, hidden_dim: int):
        super().__init__()
        # Core Component: A single, shared embedding for ALL genes
        self.shared_embedding = nn.Embedding(num_genes, pert_embed_dim)
        
        # Head 1: Reconstruction (Encoder-Decoder)
        self.recon_encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.GELU(),
        )
        self.recon_decoder = nn.Sequential(
            nn.Linear(hidden_dim, num_genes)
        )
        
        # Head 2: Perturbation Prediction (MLP)
        self.pred_head = nn.Sequential(
            nn.Linear(pert_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_genes)
        )

    def forward(self, delta_masked: torch.Tensor, pert_idx: torch.Tensor):
        """Forward pass for multi-task training."""
        # --- Task 1: Reconstruction ---
        h_context = self.recon_encoder(delta_masked)
        recon_output = self.recon_decoder(h_context)
        
        # --- Task 2: Prediction ---
        pert_emb = self.shared_embedding(pert_idx)
        pred_output = self.pred_head(pert_emb)
        
        return recon_output, pred_output

    def predict(self, pert_idx: torch.Tensor):
        """Forward pass for zero-shot prediction (uses only the prediction head)."""
        pert_emb = self.shared_embedding(pert_idx)
        pred_output = self.pred_head(pert_emb)
        return pred_output
