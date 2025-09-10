"""
basic_simulation.py

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
        delta_x_dir = self.A_true[:, target_idx].T

        # 3. Apply knobs
        z_pert = z_ctrl + alpha_med * delta_z
        m_pert = self.MLP_z_to_m(z_pert)
        delta_x_med = (m_pert - m_ctrl) @ self.U_true
        
        x_bar_downstream = x_bar_ctrl + delta_x_med + alpha_dir * delta_x_dir

        # 4. Apply Step-0 Clamp
        x_bar_pert = x_bar_downstream.clone()
        rows = torch.arange(batch_size)
        clamped_values = x_bar_ctrl[rows, target_idx] * (1.0 - alpha_step0)
        x_bar_pert[rows, target_idx] = clamped_values
        
        return {
            "z_q": z_pert.numpy(),
            "xbar_q": x_bar_pert.numpy(),
            "xbar_ctrl": x_bar_ctrl.numpy()[:, None, :], # Add k=1 dimension
            "target_idx": target_idx.numpy(),
            "env_code": np.zeros(batch_size, dtype=int),
            "dataset_id": "sim_dataset",
            # Ground truth for evaluation
            "delta_z_true": (alpha_med * delta_z).numpy(),
            "delta_x_dir_true": (alpha_dir * delta_x_dir).numpy(),
        }

# --- 2. The Spoofed Data Pipeline ---
class SpoofedDataPipeline:
    """Creates a temporary directory with all the fake files the model expects."""
    def __init__(self, sim_world: SimulatedGraftWorld):
        self.sim_world = sim_world
        self.temp_dir = Path(tempfile.mkdtemp())

    def setup(self):
        print(f"Creating spoofed data pipeline in: {self.temp_dir}")
        # Gene list
        with open(self.temp_dir / "gene_list.tsv", "w") as f:
            for i in range(self.sim_world.n_genes):
                f.write(f"GENE_{i}\n")
        # Factor U matrix
        np.save(self.temp_dir / "factor_U.npy", self.sim_world.U_true.numpy())
        # Index Parquet
        pd.DataFrame({"cell_id": []}).to_parquet(self.temp_dir / "cell_index.parquet")
        # Datasets YAML
        with open(self.temp_dir / "datasets.yaml", "w") as f:
            yaml.dump({"datasets": {"sim_dataset": {"raw_path": "dummy.h5ad"}}}, f)

        # Main config YAML
        config = {
            "paths": {
                "datasets_yaml": str(self.temp_dir / "datasets.yaml"),
                "index_parquet": str(self.temp_dir / "cell_index.parquet"),
                "gene_list_tsv": str(self.temp_dir / "gene_list.tsv"),
                "factor_U": str(self.temp_dir / "factor_U.npy"),
                "output_dir": str(self.temp_dir / "output"),
                # Dummy paths for things we monkeypatch
                "scvi_model_dir": "dummy", "scvi_input_h5ad": "dummy",
                "control_index_dir": "dummy", "control_z_npz": "dummy", "control_xbar_npz": "dummy",
            },
            "training": { "batch_size": 128, "k_controls": 1, "seed": 42 },
            "loss": {
                "distribution": {"weight": 1.0}, "consistency": {"weight": 1.0},
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
    
    # Setup mock dataset
    mock_dataset = MockGraftDataset(sim_world, train_cfg["batch_size"], test_case_knobs)
    data_iterator = iter(mock_dataset)
    
    # Lazy model init
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

    for step in range(n_steps):
        batch = next(data_iterator)
        tb = to_device(batch, device)
        y_true = tb["xbar_q"]
        x0 = tb["xbar_ctrl"].mean(dim=1)
        
        # --- Model Forward Pass ---
        z_ref = prop(tb["z_q"], target_idx=tb["target_idx"])
        x_clamp, eff = step0(x0, z_ref, tb["env_code"], tb["target_idx"])
        m = head_med(z_ref)
        dx_med = m @ U
        dx_dir = head_dir(z_ref)
        y_pred = x_clamp + dx_med + dx_dir

        # --- Loss Calculation ---
        loss_dist = w_dist * dist_fn(y_pred, y_true)
        loss_cons = w_cons * F.l1_loss(y_pred[torch.arange(len(y_pred)), tb["target_idx"]], 
                                       y_true[torch.arange(len(y_true)), tb["target_idx"]])
        loss_l1 = w_l1 * dx_dir.abs().mean()
        total_loss = loss_dist + loss_cons + loss_l1
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step % 50 == 0:
            # --- Evaluation ---
            pred_state_shift = z_ref - tb["z_q"]
            true_state_shift = tb["delta_z_true"]
            state_sim = F.cosine_similarity(pred_state_shift, true_state_shift).mean().item()
            
            true_direct_effect = tb["delta_x_dir_true"]
            dir_sim = F.cosine_similarity(dx_dir, true_direct_effect).mean().item()

            print(f"  Step {step:03d} | Loss: {total_loss.item():.4f} | "
                  f"Consist Loss: {loss_cons.item():.4f} | "
                  f"State Shift Sim: {state_sim:.3f} | "
                  f"Direct Effect Sim: {dir_sim:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation smoke tests for GRAFT.")
    parser.add_argument(
        "--test_case",
        choices=['a', 'b', 'c', 'all'],
        default='all',
        help="Which test case to run: (a) Step0 only, (b) Direct only, (c) Mediated only."
    )
    args = parser.parse_args()

    # Define test case knobs
    test_cases = {
        'a': ("Step0 Clamp", {"alpha_step0": 0.9, "alpha_dir": 0.0, "alpha_med": 0.0}),
        'b': ("Sparse Direct Head", {"alpha_step0": 0.0, "alpha_dir": 1.0, "alpha_med": 0.0}),
        'c': ("State Propagator & Mediated Head", {"alpha_step0": 0.0, "alpha_dir": 0.0, "alpha_med": 1.0}),
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
            run_test_case(config_path, sim_world, knobs)

    finally:
        pipeline.cleanup()
