"""
sim_smoke.py

A self-contained script to test the core GRAFT model components in a
controlled simulation, bypassing the complexities of scVI and factor encoding.

This script does the following:
1.  Defines a `SimulatedGraftWorld` with a ground-truth generative process for
    cellular perturbations, including step-0, direct, and mediated effects.
2.  Creates a `SpoofedDataPipeline` to generate all the necessary temporary
    artifact files (configs, index files, etc.) that the training script expects.
3.  Implements a `MockGraftDataset` that "monkeypatches" the real data loader.
    Instead of loading real data, it generates perfect ground-truth batches
    from the simulation on the fly.
4.  Runs a simplified training loop for three distinct test cases to verify
    that each core model component (`StepZeroClamp`, `SparseDirectHead`,
    `StatePropagator`, `MediatedHead`) can learn the signal it was designed for.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Add project root to path to allow importing graft modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from the GRAFT pipeline ---
from graft.models.gnn_core import StatePropagator
from graft.models.step0 import StepZeroClamp
from graft.models.heads import MediatedHead, SparseDirectHead
from train_gnn import to_device, make_optimizer, pick_dist_fn, save_checkpoint


# --- 1. The Ground-Truth Simulation World ---
class SimulatedGraftWorld:
    """Creates a toy universe with known causal rules for perturbations."""

    def __init__(self, n_genes=500, z_dim=32, F_dim=50, n_pert_genes=100):
        self.n_genes = n_genes
        self.z_dim = z_dim
        self.F_dim = F_dim
        self.n_pert_genes = n_pert_genes  # Number of genes we will simulate perturbations for

        # Ground Truth Decoder: U_true (Factors -> Genes)
        self.U_true = torch.zeros(F_dim, n_genes)
        for i in range(F_dim):
            # Each factor affects ~30 random genes
            affected_genes = np.random.choice(n_genes, 30, replace=False)
            self.U_true[i, affected_genes] = torch.rand(30) * 2
        self.U_true = F.normalize(self.U_true, p=1, dim=1)

        # Ground Truth Direct Effects: A_true (Gene -> Gene)
        self.A_true = torch.zeros(n_genes, n_genes)
        for i in range(n_pert_genes):
            # Each perturbed gene directly affects 5 other genes
            affected_genes = np.random.choice(n_genes, 5, replace=False)
            self.A_true[affected_genes, i] = (torch.rand(5) - 0.5) * 4 # Positive and negative effects

        # Ground Truth State Shift Vectors (Perturbation -> z shift)
        self.delta_z_true = torch.randn(n_pert_genes, z_dim) * 0.5

        # Ground Truth State-to-Program Map (z -> m)
        self.MLP_z_to_m = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, F_dim),
            nn.Softplus() # Non-negative factor activations
        )

    def generate_batch(self, batch_size, alpha_step0, alpha_dir, alpha_med):
        """Generates a batch of ground-truth data based on the simulation knobs."""
        # 1. Generate control cells
        z_ctrl = torch.randn(batch_size, self.z_dim)
        m_ctrl = self.MLP_z_to_m(z_ctrl)
        x_bar_ctrl = m_ctrl @ self.U_true

        # 2. Pick targets and get ground truth effects
        target_idx = torch.randint(0, self.n_pert_genes, (batch_size,))
        delta_z = self.delta_z_true[target_idx]
        delta_x_dir_base = self.A_true[:, target_idx].T

        # 3. Apply knobs to get true effects
        true_delta_z = alpha_med * delta_z
        true_delta_x_dir = alpha_dir * delta_x_dir_base

        z_pert = z_ctrl + true_delta_z
        m_pert = self.MLP_z_to_m(z_pert)
        delta_x_med = (m_pert - m_ctrl) @ self.U_true
        
        x_bar_downstream = x_bar_ctrl + delta_x_med + true_delta_x_dir

        # 4. Apply Step-0 Clamp
        x_bar_pert = x_bar_downstream.clone()
        rows = torch.arange(batch_size)
        clamped_values = x_bar_ctrl[rows, target_idx] * (1.0 - alpha_step0)
        x_bar_pert[rows, target_idx] = clamped_values
        
        return {
            "z_q": z_pert.numpy(),
            "xbar_q": x_bar_pert.detach().numpy(),
            "xbar_ctrl": x_bar_ctrl.detach().numpy()[:, None, :], # Add k=1 dimension
            "target_idx": target_idx.numpy(),
            "env_code": np.zeros(batch_size, dtype=int),
            "dataset_id": "sim_dataset",
            # Ground truth for evaluation
            "delta_z_true": true_delta_z.detach().numpy(),
            "delta_x_dir_true": true_delta_x_dir.detach().numpy(),
            "eff_true": np.full(batch_size, alpha_step0, dtype=np.float32)
        }

# --- 2. The Spoofed Data Pipeline ---
class SpoofedDataPipeline:
    """Creates a temporary directory with all the fake files the model expects."""
    def __init__(self, sim_world: SimulatedGraftWorld):
        self.sim_world = sim_world
        self.temp_dir = Path(tempfile.mkdtemp())

    def setup(self):
        print(f"Creating spoofed data pipeline in: {self.temp_dir}")
        with open(self.temp_dir / "gene_list.tsv", "w") as f:
            for i in range(self.sim_world.n_genes): f.write(f"GENE_{i}\n")
        np.save(self.temp_dir / "factor_U.npy", self.sim_world.U_true.numpy())
        pd.DataFrame({"cell_id": []}).to_parquet(self.temp_dir / "cell_index.parquet")
        with open(self.temp_dir / "datasets.yaml", "w") as f:
            yaml.dump({"datasets": {"sim_dataset": {"raw_path": "dummy.h5ad"}}}, f)
        config = {
            "paths": {
                "datasets_yaml": str(self.temp_dir / "datasets.yaml"),
                "index_parquet": str(self.temp_dir / "cell_index.parquet"),
                "gene_list_tsv": str(self.temp_dir / "gene_list.tsv"),
                "factor_U": str(self.temp_dir / "factor_U.npy"),
                "output_dir": str(self.temp_dir / "output"),
                "scvi_model_dir": "dummy", "scvi_input_h5ad": "dummy",
                "control_index_dir": "dummy", "control_z_npz": "dummy", "control_xbar_npz": "dummy",
            },
            "training": { "batch_size": 128, "k_controls": 1, "seed": 42 },
            "loss": {
                "distribution": {"weight": 1.0}, "consistency": {"weight": 0.5},
                "rex": {"weight": 0.0}, "direct": {"l1": 0.01}
            },
            "model": {
                "propagator": {"hidden": 64, "layers": 2, "steps": 2, "dropout": 0.0, "use_env_film": False, "use_target_cond": True, "target_embed_dim": 16},
                "step0": {"hidden": 32, "init_eff": 0.9, "mode": "down"},
                "mediated": {"F": self.sim_world.F_dim, "hidden": 64},
                "direct": {"hidden": 64},
            },
             "optim": {"lr": 1e-3, "weight_decay": 1e-5}
        }
        with open(self.temp_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        return self.temp_dir / "config.yaml"

    def cleanup(self):
        print(f"Cleaning up spoofed data pipeline from: {self.temp_dir}")
        shutil.rmtree(self.temp_dir)

# --- 3. The Monkeypatch ---
class MockGraftDataset:
    """A mock dataset that generates simulated batches instead of loading data."""
    def __init__(self, sim_world, batch_size, test_case_knobs):
        self.sim_world = sim_world
        self.batch_size = batch_size
        self.knobs = test_case_knobs

    def __iter__(self):
        while True:
            yield self.sim_world.generate_batch(self.batch_size, **self.knobs)

# --- 4. Main Execution ---
def run_test_case(config_path, sim_world, test_case_knobs, n_steps=101):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    paths, train_cfg, loss_cfg, model_cfg, optim_cfg = \
        cfg["paths"], cfg["training"], cfg["loss"], cfg["model"], cfg["optim"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = torch.tensor(np.load(paths["factor_U"]), device=device)
    
    mock_dataset = MockGraftDataset(sim_world, train_cfg["batch_size"], test_case_knobs)
    data_iterator = iter(mock_dataset)
    
    first_batch = next(data_iterator)
    z_dim = first_batch["z_q"].shape[1]
    G = sim_world.n_genes
    
    prop = StatePropagator(z_dim, n_envs=1, n_genes=G, **model_cfg["propagator"]).to(device)
    step0 = StepZeroClamp(z_dim, n_labs=1, **model_cfg["step0"]).to(device)
    head_med = MediatedHead(z_dim, **model_cfg["mediated"]).to(device)
    head_dir = SparseDirectHead(z_dim, G, **model_cfg["direct"]).to(device)
    
    all_params = list(prop.parameters()) + list(step0.parameters()) + \
                 list(head_med.parameters()) + list(head_dir.parameters())
    opt = make_optimizer(all_params, optim_cfg)

    dist_fn = pick_dist_fn("swd")
    w_dist = loss_cfg["distribution"]["weight"]
    w_cons = loss_cfg["consistency"]["weight"]
    w_l1 = loss_cfg["direct"]["l1"]

    # Store final batch results for evaluation
    final_results = {}

    for step in range(n_steps):
        batch = next(data_iterator)
        tb = to_device(batch, device)
        y_true = tb["xbar_q"]
        x0 = tb["xbar_ctrl"].mean(dim=1)
        
        # --- CORRECTED Model Forward Pass (matches train_gnn.py) ---
        _, eff = step0(x0, tb["z_q"], tb["env_code"], tb["target_idx"])
        dx_dir = head_dir(tb["z_q"]) # In your logic, direct head uses pre-state z_q
        z_ref = prop(tb["z_q"], target_idx=tb["target_idx"], env_codes=tb["env_code"]) # Propagator also uses pre-state z_q
        m = head_med(z_ref)
        dx_med = m @ U
        y_pred_downstream = x0 + dx_med + dx_dir

        # Surgically intervene with authoritative clamp
        y_pred = y_pred_downstream.clone()
        mask = tb["target_idx"] >= 0
        if torch.any(mask):
            # 1. Get the indices of rows that have a valid target
            rows_to_update = mask.nonzero(as_tuple=False).view(-1)
            
            # 2. Get the corresponding target gene indices (the columns) for those rows
            cols_to_update = tb["target_idx"][mask]
            
            # 3. Calculate the clamped values using the predicted effectiveness for the affected rows
            clamped_vals = x0[rows_to_update, cols_to_update] * (1.0 - eff[rows_to_update])
            
            # 4. Apply the surgical intervention
            y_pred[rows_to_update, cols_to_update] = clamped_vals
        
        loss_dist = w_dist * dist_fn(y_pred, y_true)
        loss_cons = w_cons * F.l1_loss(y_pred[mask], y_true[mask])
        loss_l1 = w_l1 * dx_dir.abs().mean()
        total_loss = loss_dist + loss_cons + loss_l1
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"  Step {step:03d} | Loss: {total_loss.item():.4f}")

        if step == n_steps - 1:
            final_results = {
                'eff_pred': eff.detach().cpu().numpy(),
                'eff_true': batch['eff_true'],
                'dx_dir_pred': dx_dir.detach().cpu().numpy(),
                'dx_dir_true': batch['delta_x_dir_true']
            }

    # --- Final Evaluation Statistics ---
    avg_eff = np.mean(final_results['eff_pred'])
    print(f"\n  [Evaluation]")
    print(f"  > Average Predicted Target Effectiveness (eff): {avg_eff:.3f} (True: {np.mean(final_results['eff_true']):.3f})")

    # Direct effects stats
    pred_dir = final_results['dx_dir_pred']
    true_dir = final_results['dx_dir_true']
    
    # Avg number of predicted effects (using a small threshold)
    pred_sparsity_mask = np.abs(pred_dir) > 0.1
    avg_pred_effects = pred_sparsity_mask.sum(axis=1).mean()
    print(f"  > Avg Predicted # of Direct Effects (>0.1): {avg_pred_effects:.2f}")

    # Percentage of correct direct effects
    true_sparsity_mask = np.abs(true_dir) > 0.0
    correct_predictions = (pred_sparsity_mask & true_sparsity_mask).sum()
    total_true_effects = true_sparsity_mask.sum()
    if total_true_effects > 0:
        percent_correct = 100 * (correct_predictions / total_true_effects)
        print(f"  > Recall of True Direct Effects: {percent_correct:.2f}%")
    else:
        print("  > No true direct effects to measure recall against.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation smoke tests for GRAFT.")
    parser.add_argument("--test_case", choices=['a', 'b', 'c', 'all'], default='all', help="Test case")
    parser.add_argument("--n-steps", type=int, default=101, help="Number of training steps.")
    args = parser.parse_args()

    # Define test case knobs with small baseline step0 effect
    test_cases = {
        'a': ("Step0 Clamp", {"alpha_step0": 0.5, "alpha_dir": 0.0, "alpha_med": 0.0}),
        'b': ("Sparse Direct Head", {"alpha_step0": 0.1, "alpha_dir": 1.0, "alpha_med": 0.0}),
        'c': ("State Propagator & Mediated Head", {"alpha_step0": 0.1, "alpha_dir": 0.0, "alpha_med": 1.0}),
    }
    
    sim_world = SimulatedGraftWorld()
    pipeline = SpoofedDataPipeline(sim_world)
    
    try:
        config_path = pipeline.setup()
        cases_to_run = test_cases.keys() if args.test_case == 'all' else [args.test_case]
        
        for case in cases_to_run:
            name, knobs = test_cases[case]
            print(f"\n--- Running Test Case ({case}): {name} ---")
            print(f"Simulation Knobs: {knobs}")
            run_test_case(config_path, sim_world, knobs, n_steps=args.n_steps)

    finally:
        pipeline.cleanup()
