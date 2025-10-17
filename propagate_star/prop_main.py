#!/usr/bin/env python3
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

from utils import *
from losses import *
from models import *


def parse_arguments():
    ap = argparse.ArgumentParser(description="Step0-aware MPNN to fit perturbed gene vectors on a small panel.")
    # Basic and I/O options
    ap.add_argument("--in_h5ad", required=True, help="Small input AnnData object.")
    ap.add_argument("--out_pred_h5ad", type=str, default="",
                    help="If set, write an AnnData with predictions for the evaluation split.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_label", default="target_gene", help="obs column with perturbation labels.")
    ap.add_argument("--control_label", default="non-targeting", help="label value for control cells.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out_influence_csv", type=str, default="",
                    help="If set, write a CSV with directed influence scores for each gene pair.")
    
    # masked model options
    ap.add_argument("--model_type", type=str, default="linear_mgm", 
                    choices=["linear_mgm", "mlp_ae", "dual_head_mt", "causal_gnn"],
                    help="Which modeling approach to run. 'linear_mgm' for observational masked modeling, "
                         "'mlp_ae' for interventional autoencoder on response vectors.")
    ap.add_argument("--pert_embed_dim", type=int, default=32,
                    help="Dimension of the learnable embedding for each perturbation (for mlp_ae).")
    ap.add_argument("--hidden_dim", type=int, default=256,
                    help="Hidden dimension of the MLP autoencoder (for mlp_ae).")
    ap.add_argument("--weight_recon", type=float, default=1.0,
                    help="Weight for the reconstruction loss in the dual-head model.")
    ap.add_argument("--weight_pred", type=float, default=1.0,
                    help="Weight for the prediction loss in the dual-head model.")
    ap.add_argument("--gnn_steps", type=int, default=2,
                    help="Number of propagation steps (T) for the CausalGNN.")
    ap.add_argument("--damping_factor", type=float, default=1.0,
                    help="Damping factor for GNN message updates (1.0 = no damping).")

    # Train/test split and eval options
    ap.add_argument("--test_pct_perts", type=float, default=0.0,
                    help="Fraction of perturbation labels (excluding control) to hold out for testing. 0.0 = no holdout.")
    ap.add_argument("--test_h5ad", type=str, default="",
                    help="Optional path to a separate test AnnData. If set, overrides --test_pct_perts.")
    ap.add_argument('--remove_non_gene_perts', action='store_true', help='Remove non-gene perturbation labels')
    ap.add_argument('--eval_on_train', action='store_true', help='Evaluate on training set in addition to test set')
    ap.add_argument('--write_test', action='store_true', help='Write true test set')

    # Pseudobulk and batching options
    ap.add_argument("--use_pseudobulk", action="store_true",
                    help="Collapse to one mean row per perturbation (incl. control).")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--single_pert_batches", action="store_true",
                    help="If set, each batch contains cells from a single perturbation label.")

    # Model architecture options
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--T", type=int, default=2, help="Number of message-passing steps.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.0, help="Step-0 anchor (e.g., 0.0 for CRISPRi).")
    ap.add_argument("--node_dim", type=int, default=128, help="Dimensionality of gene node embeddings.")

    # Loss function options
    ap.add_argument("--weight_target", type=float, default=0.1)
    ap.add_argument("--weight_local", type=float, default=0.0)
    ap.add_argument("--weight_mse", type=float, default=0.0, help="Weight for per-cell MSE loss.")
    ap.add_argument("--weight_proto", type=float, default=0.2, help="Weight for prototype loss.")
    ap.add_argument("--weight_dist", type=float, default=1.0, help="Weight for distribution loss.")
    ap.add_argument("--dist_loss", choices=["none","mmd","swd","energy"], default="mmd",
                    help="Distribution loss between predicted and true deltas per perturbation.")
    ap.add_argument("--swd_projections", type=int, default=128, help="Num random projections for SWD.")

    # Pretraining options
    ap.add_argument("--pretrain_pseudobulk", type=str, default="",
                    help="Path to a pseudobulk .h5ad for Stage-1 pretraining; empty = skip Stage-1")
    ap.add_argument("--pretrain_pseudobulk_list", type=str, default="",
                        help="Text file with one pseudobulk .h5ad path per line; blank/comment lines ignored")
    ap.add_argument("--include_target_pseudobulk", action="store_true",
                        help="Also pseudobulk the target dataset and include it in Stage-1 pretraining")
    ap.add_argument("--pretrain_epochs", type=int, default=10,
                    help="Epochs to run Stage-1 pseudobulk pretraining")
    ap.add_argument("--dset_embed_dim", type=int, default=0,
                    help="Dimensionality of dataset embedding (if enabled)")
    ap.add_argument("--ct_embed_dim", type=int, default=0,
                    help="Dimensionality of cell_type embedding (if enabled)")
    ap.add_argument("--missing_gene_fill", type=str, default="nan", choices=["nan", "-1"],
                        help="Placeholder used in pseudobulk for missing genes; masked in Stage-1 losses")
    args = ap.parse_args()
    return args

def train_causal_gnn(
    delta_vectors: torch.Tensor,
    pert_indices: torch.Tensor,
    control_vec: torch.Tensor, # <-- Add control_vec here
    model: 'CausalGNN',
    args,
) -> 'CausalGNN':
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    
    # The dataset now includes the control expression for each perturbation
    control_exprs_for_perts = control_vec[pert_indices]
    dataset = TensorDataset(delta_vectors, pert_indices, control_exprs_for_perts)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print("  Training on {} perturbation response vectors...".format(len(dataset)))
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for delta_batch, p_idx_batch, control_expr_batch in loader:
            delta_batch = delta_batch.to(args.device)
            p_idx_batch = p_idx_batch.to(args.device)
            control_expr_batch = control_expr_batch.to(args.device) # <-- Get from loader
            
            # Pass the control expression to the model
            pred_delta = model(p_idx_batch, control_expr_batch)
            
            # Calculate simple MSE loss
            loss = loss_fn(pred_delta, delta_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == args.epochs:
            avg_loss = total_loss / len(loader)
            print(f"  [Epoch {epoch:02d}] Prediction Loss: {avg_loss:.6f}")
            
    return model.eval()

@torch.no_grad()
def predict_with_causal_gnn(
    model: 'CausalGNN',
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    args,
) -> tuple:
    print("\n=== Predicting responses for held-out perturbations using CausalGNN ===")
    model.eval()
    
    ctrl_mean = to_numpy(adata_train[adata_train.obs[args.target_label] == args.control_label].X).mean(axis=0)
    gene_to_idx = {name: i for i, name in enumerate(adata_train.var_names)}
    test_perts = sorted({p for p in adata_test.obs[args.target_label].unique() if p != args.control_label})
    ctrl_mean_tensor = torch.from_numpy(ctrl_mean).to(args.device)

    pred_deltas = []
    for p_label in test_perts:
        p_idx_val = gene_to_idx[p_label]
        p_idx = torch.tensor([p_idx_val], device=args.device)
        
        # Get the control expression for this specific gene
        control_expr_for_pert = ctrl_mean_tensor[p_idx_val]
        
        # Pass it to the model
        pred_delta = model(p_idx, control_expr_for_pert.unsqueeze(0))
        pred_deltas.append(pred_delta.squeeze().cpu().numpy())
        
    pred_delta_mat = np.array(pred_deltas)
    pred_mat = pred_delta_mat + ctrl_mean
    
    test_pert_mask = adata_test.obs[args.target_label].isin(test_perts)
    true_mat = to_numpy(adata_test[test_pert_mask].X)
    pert_names = adata_test[test_pert_mask].obs[args.target_label].tolist()
    
    print("   ...Done. Predicted effects for {} perturbations.".format(len(test_perts)))
    return (pred_mat, true_mat, pert_names, ctrl_mean)

@torch.no_grad()
def print_causal_gnn_diagnostics(
    model: 'CausalGNN',
    adata_train: ad.AnnData,
    args,
):
    """
    Computes and prints useful diagnostics from a trained CausalGNN model,
    including statistics about the learned relatedness weights.
    """
    print("\n" + "="*25)
    print("🔬 CAUSAL GNN DIAGNOSTICS")
    print("="*25)
    model.eval()
    
    # 1. Re-compute the (G, G) relatedness matrix R
    all_embeddings = model.shared_embedding.weight
    G, D = all_embeddings.shape
    source_embs = all_embeddings.repeat(G, 1)
    target_embs = all_embeddings.repeat_interleave(G, dim=0)
    embedding_pairs = torch.cat([source_embs, target_embs], dim=1)
    relatedness_scores = model.relatedness_mlp(embedding_pairs)
    R = relatedness_scores.view(G, G) # R_ij is relatedness of source j to target i

    # 2. Exclude self-relatedness from all stats
    no_self_loop_mask = ~torch.eye(G, dtype=torch.bool, device=R.device)

    # --- Stats for ALL gene pairs ---
    all_relatedness = R[no_self_loop_mask]
    mean_relatedness_all = all_relatedness.mean().item()
    max_relatedness_all = all_relatedness.max().item()
    
    print("\n--- Relatedness Weights (All Genes) ---")
    print(f"  - Mean (off-diagonal): {mean_relatedness_all:.4f}")
    print(f"  - Max (off-diagonal):  {max_relatedness_all:.4f}")

    # --- Stats for UNPERTURBED target genes ---
    train_perts = {p for p in adata_train.obs[args.target_label] if p in adata_train.var_names}
    unperturbed_mask = torch.ones(G, dtype=torch.bool, device=R.device)
    gene_to_idx = {name: i for i, name in enumerate(adata_train.var_names)}
    for p in train_perts:
        unperturbed_mask[gene_to_idx[p]] = False
        final_unperturbed_mask = unperturbed_mask.view(-1, 1) & no_self_loop_mask
    unperturbed_relatedness = R[final_unperturbed_mask]
    
    # Select rows corresponding to unperturbed genes, then apply off-diagonal mask
    unperturbed_relatedness = R[final_unperturbed_mask]
    if unperturbed_relatedness.numel() > 0:
        mean_relatedness_unp = unperturbed_relatedness.mean().item()
        max_relatedness_unp = unperturbed_relatedness.max().item()
        print("\n--- Relatedness Weights (Unperturbed Target Genes Only) ---")
        print(f"  - Mean (off-diagonal): {mean_relatedness_unp:.4f}")
        print(f"  - Max (off-diagonal):  {max_relatedness_unp:.4f}")

    # --- Top 5 most related gene pairs ---
    top_vals, top_indices_flat = torch.topk(all_relatedness, 5)
    # Need to map flat indices back to (row, col)
    row_indices, col_indices = np.unravel_index(top_indices_flat.cpu().numpy(), (G, G))
    
    print("\n--- Top 5 Most Related Pairs (Source -> Target) ---")
    gene_names = adata_train.var_names
    for i in range(5):
        target_gene = gene_names[row_indices[i]]
        source_gene = gene_names[col_indices[i]]
        print(f"  1. {source_gene} -> {target_gene} (Score: {top_vals[i]:.4f})")
        
    # --- Effectiveness Head Diagnostics ---
    all_embs = model.shared_embedding.weight
    effectiveness = model.effectiveness_head(all_embs).squeeze()
    print("\n--- Knockdown Effectiveness Head ---")
    print(f"  - Mean predicted alpha: {effectiveness.mean().item():.3f}")
    print(f"  - Min predicted alpha:  {effectiveness.min().item():.3f}")
    print(f"  - Max predicted alpha:  {effectiveness.max().item():.3f}")
    print("="*25 + "\n")

def run_causal_gnn_flow(args, adata_train: ad.AnnData, adata_test: ad.AnnData):
    """Orchestrates the training and prediction for the CausalGNN model."""
    if not args.use_pseudobulk:
        raise ValueError("--model_type 'causal_gnn' requires the --use_pseudobulk flag.")

    print("\n=== Training CausalGNN Model ===")
    labels = adata_train.obs[args.target_label].astype(str)
    control_vec = to_numpy(adata_train[labels == args.control_label].X).mean(axis=0)
    control_vec_tensor = torch.from_numpy(control_vec)
    
    gene_to_idx = {name: i for i, name in enumerate(adata_train.var_names)}
    train_perts = sorted({p for p in labels if p != args.control_label and p in gene_to_idx})
    
    delta_vectors = [to_numpy(adata_train[labels == p].X).mean(axis=0) - control_vec for p in train_perts]
    pert_indices = [gene_to_idx[p] for p in train_perts]
    
    delta_vectors = torch.from_numpy(np.array(delta_vectors, dtype=np.float32))
    pert_indices = torch.from_numpy(np.array(pert_indices, dtype=np.int64))
    
    model = CausalGNN(
        num_genes=adata_train.n_vars,
        embed_dim=args.pert_embed_dim,
        hidden_dim=args.hidden_dim,
        num_steps=args.gnn_steps,
        damping_factor=args.damping_factor,
    ).to(args.device)
    
    model = train_causal_gnn(delta_vectors, pert_indices, control_vec_tensor, model, args)

    print_causal_gnn_diagnostics(model, adata_train, args)

    if adata_test is not None:
        pred_bundle = predict_with_causal_gnn(model, adata_train, adata_test, args)
        
        print("\n=== Final Evaluation on Held-Out Perturbations ===")
        evaluate_model(adata=adata_test, args=args, pred_bundle=pred_bundle)

        if args.out_pred_h5ad:
            print(f"\n💾 Writing prediction and true AnnData files to {args.out_pred_h5ad}...")
            write_pred_true_h5ads(
                eval_adata=adata_test,
                pred_bundle=pred_bundle,
                out_pred_h5ad=args.out_pred_h5ad,
                target_label=args.target_label,
                control_label=args.control_label,
            )
    
    if args.eval_on_train:
        print("\n=== Evaluation on Training Set ===")
        pred_bundle_train = predict_with_causal_gnn(model, adata_train, adata_train, args)
        evaluate_model(adata=adata_train, args=args, pred_bundle=pred_bundle_train)


class LinearReconstructor(torch.nn.Module):
    """A simple linear model to predict masked genes from unmasked ones."""
    def __init__(self, num_genes: int):
        super().__init__()
        self.reconstruct = torch.nn.Linear(num_genes, num_genes)
        # Crucially, zero out the diagonal. A gene cannot predict itself.
        self.reconstruct.weight.data.fill_diagonal_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce the diagonal constraint throughout training
        self.reconstruct.weight.data.fill_diagonal_(0)
        return self.reconstruct(x)

def train_linear_model(
    adata: ad.AnnData,
    epochs: int,
    lr: float,
    batch_size: int,
    mask_prob: float = 0.15,
    device: str = "cuda",
) -> LinearReconstructor:
    """Trains the LinearReconstructor using a masked gene modeling objective."""
    G = adata.n_vars
    X = to_numpy(adata.X).astype(np.float32)
    
    model = LinearReconstructor(num_genes=G).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            
            # Create mask: True means we KEEP the gene, False means we MASK it
            mask = (torch.rand(x_batch.shape, device=device) > mask_prob)
            x_masked = x_batch * mask

            # Predict the full vector
            y_pred = model(x_masked)
            
            # Loss is only calculated on the genes that were MASKED
            loss = loss_fn(y_pred[~mask], x_batch[~mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  [Epoch {epoch:02d}] MGM Pre-training Loss: {avg_loss:.6f}")

    return model.eval()


@torch.no_grad()
def analyze_asymmetry_vs_effects(
    model: LinearReconstructor,
    adata: ad.AnnData,
    target_label: str,
    control_label: str,
):
    """
    Analyzes the trained linear model's weights.
    1. Computes the Directed Influence Asymmetry Score for all gene pairs.
    2. For each perturbation, correlates the learned influence vector with the true effect vector.
    """
    # 1. Extract the learned weight matrix W (G_i <- G_j)
    W = model.reconstruct.weight.data.cpu().numpy()  # Shape (G_out, G_in)
    
    # 2. Get true perturbation effects
    labels = adata.obs[target_label].astype(str)
    ctrl_mean = to_numpy(adata[labels == control_label].X).mean(axis=0)
    perts = sorted({p for p in labels if p != control_label})
    t2gi = build_target_to_gene_index(adata, target_label)

    effect_vectors = {}
    for p in perts:
        if p in t2gi:
            pert_mean = to_numpy(adata[labels == p].X).mean(axis=0)
            effect_vectors[p] = pert_mean - ctrl_mean

    # 3. Correlate learned influence with true effects
    correlations = {}
    for pert, true_effect in effect_vectors.items():
        target_idx = t2gi[pert]
        
        # The model's prediction of how perturbing `target_idx` affects all other genes
        # is given by the `target_idx`-th column of the weight matrix W.
        learned_influence = W[:, target_idx]
        
        # We don't care about the self-effect
        true_effect[target_idx] = 0
        learned_influence[target_idx] = 0
        
        # Calculate Pearson correlation
        corr = np.corrcoef(true_effect, learned_influence)[0, 1]
        if not np.isnan(corr):
            correlations[pert] = corr

    # 4. Report results
    if not correlations:
        print("Could not compute any correlations. Check if perturbations are in the dataset.")
        return

    corr_values = np.array(list(correlations.values()))
    print(f"\n📊 Correlation between Learned Influence and True Effects (over {len(corr_values)} perturbations):")
    print(f"  - Mean Correlation:   {corr_values.mean():.4f}")
    print(f"  - Median Correlation: {np.median(corr_values):.4f}")
    print(f"  - Std Dev:            {corr_values.std():.4f}")
    
    # Show top 5 most consistent perturbations
    print("\n  Top 5 perturbations (by correlation):")
    sorted_corrs = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
    for p, c in sorted_corrs[:5]:
        print(f"    - {p:<15}: {c:.4f}")

@torch.no_grad()
def write_influence_scores_csv(
    influence_matrix: np.ndarray, # <-- Changed from model to matrix
    adata: ad.AnnData,
    output_path: str,
    epsilon: float = 1e-8,
):
    """
    Calculates the directed influence asymmetry score from a pre-computed
    (G, G) influence matrix and saves it to a CSV file.
    """
    print(f"\n📝 Calculating asymmetry and writing influence scores to {output_path}...")
    
    # The first dimension is the target (output) and the second is the source (input)
    # So, W[i, j] is the influence of j on i.
    W = influence_matrix
    
    # The rest of the function remains the same!
    W_t = W.T
    numerator = W - W_t
    denominator = W + W_t + epsilon
    asymmetry_matrix = numerator / denominator
    
    influence_df = pd.DataFrame(
        asymmetry_matrix,
        index=adata.var_names,
        columns=adata.var_names,
    )
    influence_df.index.name = "TargetGene"
    influence_df.columns.name = "SourceGene"

    influence_df.to_csv(output_path)
    print(f"   ...Done. Matrix shape: {influence_df.shape}")

@torch.no_grad()
def predict_held_out_perts(
    model,
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    args,
) -> tuple:
    """
    Uses the trained autoencoder to predict response vectors for held-out perturbations.
    """
    print("\n=== Stage 2: Predicting responses for held-out perturbations ===")
    model.eval()
    num_genes = adata_train.n_vars
    
    # 1. Get the global control mean from the TRAINING data
    train_labels = adata_train.obs[args.target_label].astype(str)
    ctrl_mean = to_numpy(adata_train[train_labels == args.control_label].X).mean(axis=0)

    # 2. Prepare a neutral input by encoding a zero-vector
    zero_delta = torch.zeros(1, num_genes, device=args.device)
    # Use a dummy embedding for the encoder pass (it will be replaced at the decoder)
    dummy_pert_idx = torch.tensor([0], device=args.device)
    dummy_pert_emb = model.pert_embedding(dummy_pert_idx)
    
    net_input = torch.cat([zero_delta, dummy_pert_emb], dim=1)
    h_neutral = model.encoder(net_input)

    # 3. Get embeddings for the HELD-OUT genes
    # Create a map from gene name to its index in the full var_names list
    gene_to_idx = {name: i for i, name in enumerate(adata_train.var_names)}
    
    test_perts = sorted({
        p for p in adata_test.obs[args.target_label].unique() if p != args.control_label
    })
    
    pred_deltas = []
    for p_label in test_perts:
        p_idx = torch.tensor([gene_to_idx[p_label]], device=args.device)
        p_emb = model.pert_embedding(p_idx)
        
        # 4. Generate prediction by feeding the DECODER the neutral latent vector
        #    and the specific embedding of the held-out gene.
        decoder_input = h_neutral # In a more complex model, you might cat p_emb here too
        pred_delta = model.decoder(decoder_input)
        pred_deltas.append(pred_delta.squeeze().cpu().numpy())

    pred_delta_mat = np.array(pred_deltas)
    
    # 5. The model predicts DELTAS. Add the control mean back to get final expression.
    pred_mat = pred_delta_mat + ctrl_mean
    
    # 6. Prepare the final "pred_bundle" for the evaluation function
    test_pert_mask = adata_test.obs[args.target_label].isin(test_perts)
    true_mat = to_numpy(adata_test[test_pert_mask].X)
    pert_names = adata_test[test_pert_mask].obs[args.target_label].tolist()
    
    print("   ...Done. Predicted effects for {} perturbations.".format(len(test_perts)))
    return (pred_mat, true_mat, pert_names, ctrl_mean)

def train_dual_head_model(
    delta_vectors: torch.Tensor,
    pert_indices: torch.Tensor,
    model,
    args):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    
    dataset = TensorDataset(delta_vectors, pert_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print("  Training on {} perturbation response vectors...".format(len(dataset)))
    for epoch in range(1, args.epochs + 1):
        epoch_loss_recon, epoch_loss_pred = 0, 0
        for delta_batch, p_idx_batch in loader:
            delta_batch = delta_batch.to(args.device)
            p_idx_batch = p_idx_batch.to(args.device)
            
            mask = (torch.rand(delta_batch.shape, device=args.device) > 0.15)
            delta_masked = delta_batch * mask
            
            # Get outputs from both heads
            recon_output, pred_output = model(delta_masked, p_idx_batch)
            
            # Calculate loss for each task
            loss_recon = loss_fn(recon_output[~mask], delta_batch[~mask])
            loss_pred = loss_fn(pred_output, delta_batch)
            
            # Combine losses with weights
            total_loss = (args.weight_recon * loss_recon) + (args.weight_pred * loss_pred)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss_recon += loss_recon.item()
            epoch_loss_pred += loss_pred.item()
        
        if epoch % 5 == 0 or epoch == args.epochs:
            avg_loss_r = epoch_loss_recon / len(loader)
            avg_loss_p = epoch_loss_pred / len(loader)
            print(f"  [Epoch {epoch:02d}] Recon Loss: {avg_loss_r:.6f} | Pred Loss: {avg_loss_p:.6f}")
            
    return model.eval()

@torch.no_grad()
def predict_with_dual_head_model(
    model,
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    args,
) -> tuple:
    print("\n=== Predicting responses for held-out perturbations using Dual-Head Model ===")
    model.eval()
    
    ctrl_mean = to_numpy(adata_train[adata_train.obs[args.target_label] == args.control_label].X).mean(axis=0)
    gene_to_idx = {name: i for i, name in enumerate(adata_train.var_names)}
    test_perts = sorted({p for p in adata_test.obs[args.target_label].unique() if p != args.control_label})
    
    pred_deltas = []
    for p_label in test_perts:
        p_idx = torch.tensor([gene_to_idx[p_label]], device=args.device)
        
        # Use the model's dedicated prediction method
        pred_delta = model.predict(p_idx)
        pred_deltas.append(pred_delta.squeeze().cpu().numpy())
        
    pred_delta_mat = np.array(pred_deltas)
    pred_mat = pred_delta_mat + ctrl_mean
    
    test_pert_mask = adata_test.obs[args.target_label].isin(test_perts)
    true_mat = to_numpy(adata_test[test_pert_mask].X)
    pert_names = adata_test[test_pert_mask].obs[args.target_label].tolist()
    
    print("   ...Done. Predicted effects for {} perturbations.".format(len(test_perts)))
    return (pred_mat, true_mat, pert_names, ctrl_mean)

def run_dual_head_flow(args, adata_train: ad.AnnData, adata_test: ad.AnnData):
    """Orchestrates the training and prediction for the dual-head model."""
    if not args.use_pseudobulk:
        raise ValueError("--model_type 'dual_head_mt' requires the --use_pseudobulk flag.")

    # --- Stage 1: Train the multi-task model ---
    print("\n=== Training Dual-Head Multi-Task Model ===")
    labels = adata_train.obs[args.target_label].astype(str)
    control_vec = to_numpy(adata_train[labels == args.control_label].X).mean(axis=0)
    
    gene_to_idx = {name: i for i, name in enumerate(adata_train.var_names)}
    train_perts = sorted({p for p in labels if p != args.control_label and p in gene_to_idx})
    
    delta_vectors = [to_numpy(adata_train[labels == p].X).mean(axis=0) - control_vec for p in train_perts]
    pert_indices = [gene_to_idx[p] for p in train_perts]
    
    delta_vectors = torch.from_numpy(np.array(delta_vectors, dtype=np.float32))
    pert_indices = torch.from_numpy(np.array(pert_indices, dtype=np.int64))
    
    model = DualHeadAutoencoder(
        num_genes=adata_train.n_vars,
        pert_embed_dim=args.pert_embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    
    model = train_dual_head_model(delta_vectors, pert_indices, model, args)
    
    # --- Stage 2: Predict and Evaluate on held-out data ---
    if adata_test is not None:
        pred_bundle = predict_with_dual_head_model(model, adata_train, adata_test, args)
        
        print("\n=== Final Evaluation on Held-Out Perturbations ===")
        evaluate_model(adata=adata_test, args=args, pred_bundle=pred_bundle)

        if args.out_pred_h5ad:
            print(f"\n💾 Writing prediction and true AnnData files to {args.out_pred_h5ad}...")
            write_pred_true_h5ads(
                eval_adata=adata_test,
                pred_bundle=pred_bundle,
                out_pred_h5ad=args.out_pred_h5ad,
                target_label=args.target_label,
                control_label=args.control_label,
            )
    
    if args.eval_on_train:
        print("\n=== Evaluation on Training Set ===")
        pred_bundle_train = predict_with_dual_head_model(model, adata_train, adata_train, args)
        evaluate_model(adata=adata_train, args=args, pred_bundle=pred_bundle_train)

def evaluate_model(
    adata: ad.AnnData,
    args,
    pred_bundle: tuple[np.ndarray, np.ndarray, list[str], np.ndarray],
):
    """
    Computes:
      - per-perturbation MAE
      - knockdown efficiency (abs & %) for true vs predicted at the target gene
      - perturbation similarity: mean & min pairwise Pearson corr between predicted mean effect vectors
      - PDS (Perturbation Discrimination Score): mean over perturbations
    Prints a concise report and returns a dict with all metrics.
    """
    pred_mat, true_mat, pert_names, ctrl_mean = pred_bundle
    G = adata.n_vars
    df_obs = adata.obs
    labels = df_obs[args.target_label].astype(str).values

    # group indices by perturbation (excluding control)
    perts = sorted(set(pert_names))
    # target mapping
    t2gi = build_target_to_gene_index(adata, args.target_label)

    # per-pert pseudobulks (pred & true) and MAE
    pred_bulk = {}
    true_bulk = {}
    mae_per_pert = {}
    bulk_mae_per_pert = {}

    # map pert_names (length Np) to row indices for quick grouping
    rows_by_pert = defaultdict(list)
    for i, p in enumerate(pert_names):
        rows_by_pert[p].append(i)

    for p in perts:
        rows = rows_by_pert[p]
        yhat_p = pred_mat[rows]  # (n_p, G)
        ytrue_p = true_mat[rows] # (n_p, G)
        pred_bulk[p] = yhat_p.mean(axis=0)
        true_bulk[p] = ytrue_p.mean(axis=0)
        # per-cell MAE (cells+genes)
        mae_per_pert[p] = np.mean(np.abs(yhat_p - ytrue_p))
        # pseudobulk MAE (genes only)
        bulk_mae_per_pert[p] = float(np.mean(np.abs(pred_bulk[p] - true_bulk[p])))

    # knockdown efficiency at target gene (abs & %), true vs predicted
    # uses GLOBAL control pseudobulk as the "control" reference
    eps = 1e-8
    kd_eff = {}  # p -> dict
    for p in perts:
        t = t2gi.get(p, -1)
        if t < 0:
            kd_eff[p] = {"target_gene": None,
                         "true_abs": np.nan, "true_pct": np.nan,
                         "pred_abs": np.nan, "pred_pct": np.nan}
            continue
        ctrl_t = float(ctrl_mean[t])
        true_t = float(true_bulk[p][t])
        pred_t = float(pred_bulk[p][t])

        # absolute "knockdown" (positive if below control)
        true_abs = ctrl_t - true_t
        pred_abs = ctrl_t - pred_t
        # percentage relative to control level
        true_pct = true_abs / (ctrl_t + eps)
        pred_pct = pred_abs / (ctrl_t + eps)

        kd_eff[p] = {"target_gene": adata.var_names[t],
                     "true_abs": true_abs, "true_pct": true_pct,
                     "pred_abs": pred_abs, "pred_pct": pred_pct}

    # perturbation similarity (correlations between predicted mean effect vectors)
    # use predicted (pred_bulk[p] - ctrl_mean) as effect vector
    effect_vecs = []
    for p in perts:
        effect_vecs.append(pred_bulk[p] - ctrl_mean)
    effect_mat = np.stack(effect_vecs, axis=0)  # (K,G)
    # pairwise Pearson correlation matrix
    K = effect_mat.shape[0]
    # normalize
    em = effect_mat - effect_mat.mean(axis=1, keepdims=True)
    denom = np.sqrt((em ** 2).sum(axis=1, keepdims=True)) + 1e-8
    emn = em / denom
    corr_mat = emn @ emn.T  # (K,K)
    # take upper triangle excluding diagonal
    iu = np.triu_indices(K, k=1)
    mean_corr = float(corr_mat[iu].mean()) if iu[0].size > 0 else np.nan
    min_corr = float(corr_mat[iu].min()) if iu[0].size > 0 else np.nan

    # PDS (Perturbation Discrimination Score)
    # - use absolute deltas vs control
    # - exclude only the TRUE target gene for expression data, by name
    # - zero-based rank normalized by N (not N-1): PDS_p = 1 - rank/N
    # absolute deltas vs global control mean
    true_bulk_mat = np.stack([np.abs(true_bulk[p] - ctrl_mean) for p in perts], axis=0)  # (K,G)
    pred_bulk_mat = np.stack([np.abs(pred_bulk[p] - ctrl_mean) for p in perts], axis=0)  # (K,G)
    t_idx_per_pert = {p: t2gi.get(p, -1) for p in perts}

    # precompute masks per pair to exclude targets
    Kp = len(perts)
    PDS_scores = []
    for i, p in enumerate(perts):
        # build include mask: exclude target gene IF its name equals the perturbation label
        mask = np.ones(G, dtype=bool)
        tj = t_idx_per_pert[p]
        if tj >= 0:
            mask[tj] = False
        # distances from ALL real effects to this predicted effect
        dists = pairwise_distances(
            true_bulk_mat[:, mask],    # (K, G')
            pred_bulk_mat[i, mask][None, :],  # (1, G')
            metric="manhattan",
        ).ravel()
        order = np.argsort(dists)          # ascending
        # rank of the correct perturbation (zero-based)
        p_index = i  # same ordering
        rank0 = int(np.flatnonzero(order == p_index)[0])
        # normalize by K (not K-1), then invert
        PDS_scores.append(1.0 - rank0 / Kp)

    PDS_mean = float(np.mean(PDS_scores)) if len(PDS_scores) > 0 else np.nan

    # ---- Print concise report ----
    print("\n=== Evaluation ===")
    # print(f"Per-cell MAE (mean ± sd over perts): {np.mean(list(mae_per_pert.values())):.5f} ± {np.std(list(mae_per_pert.values())):.5f}")
    print(f"Pseudobulk MAE (mean over perts):   {np.mean(list(bulk_mae_per_pert.values())):.5f}")
    print(f"Perturbation similarity (pred mean effects): mean corr={mean_corr:.4f}, min corr={min_corr:.4f}")
    print(f"PDS (mean over perts): {PDS_mean:.4f}")
    print("\nKnockdown efficiency per perturbation (target gene, true_abs, true_pct, pred_abs, pred_pct):")
    # show a few lines sorted by true_abs descending
    preview = sorted(kd_eff.items(), key=lambda kv: (np.nan_to_num(kv[1]['true_abs'], nan=-1e9)), reverse=True)
    for p, d in preview[: min(10, len(preview))]:
        tg = d['target_gene'] or "N/A"
        print(f"  {p:20s}  tg={tg:12s}  true_abs={d['true_abs']:.4f}  true_pct={d['true_pct']:.2%}  "
              f"pred_abs={d['pred_abs']:.4f}  pred_pct={d['pred_pct']:.2%}")

    return {
        "mae_per_pert": mae_per_pert,
        "bulk_mae_per_pert": bulk_mae_per_pert,
        "kd_eff": kd_eff,
        "mean_corr_pred_effects": mean_corr,
        "min_corr_pred_effects": min_corr,
        "PDS_mean": PDS_mean,
        "PDS_scores": dict(zip(perts, PDS_scores)),
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    args = parse_arguments()

    # ---------------------------
    # Read input data
    # ---------------------------
    adata = ad.read_h5ad(args.in_h5ad)
    adata.obs['dataset_id'] = "target_all"
    adata.obs['cell_type'] = "UNK"
    pb_target = None  # pseudobulked target data for Stage-1 pretraining
    if args.include_target_pseudobulk:
        pb_target = make_pretrain_pseudobulk_from_adata(adata, args.target_label, args.control_label, dataset_id="target_all")
        sc.pp.normalize_total(pb_target, inplace=True)
        sc.pp.log1p(pb_target)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if args.use_pseudobulk:  # stage 2 pseudobulk
        args.batch_size = 1  # enforce single-row batches
        adata = collapse_to_pseudobulk(adata, args.target_label)
        adata.obs['dataset_id'] = "target_all"
        adata.obs['cell_type'] = "UNK"
    if sparse.isspmatrix(adata.X) and not sparse.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()  # nicer slicing, though we load to numpy anyway
    # train/test split
    adata_train, adata_test, pb_target = train_test_split(args, adata, pb_target)
    adata_train.obs['dataset_id'] = "target_all"
    adata_train.obs['cell_type'] = "UNK"
    if adata_test is not None:
        adata_test.obs['dataset_id'] = "target_all"
        adata_test.obs['cell_type'] = "UNK"

    if args.model_type == 'linear_mgm':
        # --- Run the original observational MGM flow ---
        print("\n=== Running Observational Model (linear_mgm) ===")
        linear_model = train_linear_model(
            adata=adata_train, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
        )
        analyze_asymmetry_vs_effects(
            model=linear_model, adata=adata_train,
            target_label=args.target_label, control_label=args.control_label,
        )
        if args.out_influence_csv:
            # Extract the weight matrix from the linear model
            influence_matrix = linear_model.reconstruct.weight.data.cpu().numpy()
            write_influence_scores_csv(
                influence_matrix=influence_matrix, # Pass the matrix
                adata=adata_train,
                output_path=args.out_influence_csv
            )
            
    elif args.model_type == 'mlp_ae':
        # --- Run the new interventional autoencoder flow ---
        print("\n=== Running Interventional Model (mlp_ae) ===")
        model = run_mlp_autoencoder_flow(args, adata_train)
        if adata_test is not None:
            # Perform Stage 2 prediction
            pred_bundle = predict_held_out_perts(
                model=model,
                adata_train=adata_train,
                adata_test=adata_test,
                args=args,
            )
            
            # Pass the predictions to the user's evaluation function
            print("\n=== Final Evaluation on Held-Out Perturbations ===")
            evaluate_model(
                adata=adata_test, # Evaluate on the test set AnnData
                args=args,
                pred_bundle=pred_bundle
            )

            if args.eval_on_train:
                print("\n=== Evaluation on TRAIN set (fit on TRAIN) ===")
                pred_bundle_tr = predict_held_out_perts(
                    model=model,
                    adata_train=adata_train,
                    adata_test=adata_train,
                    args=args,
                )
                
                # Pass the predictions to the user's evaluation function
                print("\n=== Final Evaluation on Held-Out Perturbations ===")
                evaluate_model(
                    adata=adata_train, # Evaluate on the test set AnnData
                    args=args,
                    pred_bundle=pred_bundle_tr
                )
    elif args.model_type == 'dual_head_mt':
        # --- Run the dual-head multi-task model flow ---
        print("\n=== Running Dual-Head Multi-Task Model (dual_head_mt) ===")
        run_dual_head_flow(args, adata_train, adata_test)
    elif args.model_type == 'causal_gnn':
            run_causal_gnn_flow(args, adata_train, adata_test)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # model = None
    # pred_bundle = None
    # # ---------------------------
    # # Evaluate: external test if provided, else held-out split, else train split
    # # ---------------------------
    # eval_adata = adata_test if adata_test is not None else adata_train
    # # 3) Evaluate with your existing metrics
    # print("\n=== Evaluation on {} set ===".format(
    #     "TEST (held-out perts)" if adata_test is not None else "TRAIN (no holdout)")
    # )
    # _ = evaluate_model(adata=eval_adata, args=args, pred_bundle=pred_bundle)

    # # 5) (Optional) Evaluate on TRAIN split as well (fit on TRAIN, eval on TRAIN)
    # if args.eval_on_train and (adata_test is not None):
    #     print("\n=== Evaluation on TRAIN set (fit on TRAIN) ===")
    #     train_labels = adata_train.obs[args.target_label].astype(str).values
    #     train_perts  = sorted({lab for lab in train_labels if lab != args.control_label})
    #     _ = evaluate_model(adata=adata_train, args=args, pred_bundle=None)


    # if args.out_pred_h5ad:
    #     if hasattr(eval_adata.X, "toarray"):
    #         eval_adata.X = eval_adata.X.toarray()
    #     write_pred_true_h5ads(
    #         eval_adata=eval_adata,
    #         pred_bundle=pred_bundle,
    #         out_pred_h5ad=args.out_pred_h5ad,
    #         target_label=args.target_label,
    #         control_label=args.control_label,
    #     )

if __name__ == "__main__":
    main()
