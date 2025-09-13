"""
basic_simulation.py

A self-contained script to test the core GRAFT model components in a
controlled simulation, bypassing the complexities of scVI and factor encoding.

What it does:
1) Defines a *causal-by-construction* SimulatedGraftWorld:
   - step-0 on target -> propagate via stable SEM (I-B)^{-1}
   - decompose Δx_total into mediated (span(U)) and sparse direct residual
   - retains z (state), with baseline x0 = U m(z) and optional dz ~ d_tgt
2) Creates a SpoofedDataPipeline that writes the artifacts your training expects
3) Implements a MockGraftDataset that yields perfect ground-truth batches
4) Runs a training loop using your real forward utils & losses
"""

from __future__ import annotations
import argparse, tempfile, shutil, json, yaml, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Allow importing graft modules (assumes this file lives in repo_root/simulations/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Imports from your pipeline ---
from graft.models.gnn_core import StatePropagator
from graft.models.step0 import StepZeroClamp
from graft.models.heads import MediatedHead, TrueSparseDirectHead
from train_gnn import make_optimizer, pick_dist_fn, make_prediction, compute_losses, to_device


# === 1) Causal simulation world =================================================
class SimulatedGraftWorld:
    """
    Causal CRISPRi simulator.

    Mechanism
    ---------
    - Stable sparse gene->gene matrix B (spectral radius < 1)
    - On-target clamp: x_tgt' = x_tgt + d_tgt  (CRISPRi: d_tgt < 0)
    - Propagation: Δx_total = (I - B)^{-1} (d_tgt * e_tgt)
    - Decomposition: Δx_med = P_U Δx_total ; Δx_dir = Δx_total - Δx_med (sparsified)
    - Latent z matters: x0 = U m(z) with small RELU MLP; optional dz ∝ d_tgt
    """

    def __init__(self, n_genes=2000, z_dim=32, F_dim=64, n_pert_genes=400, seed=123):
        self.n_genes = int(n_genes)
        self.z_dim = int(z_dim)
        self.F_dim = int(F_dim)
        self.n_pert_genes = int(n_pert_genes)

        g = torch.Generator().manual_seed(seed)

        # Stable sparse B
        density = max(1, int(0.002 * self.n_genes * self.n_genes))
        rows = torch.randint(self.n_genes, (density,), generator=g)
        cols = torch.randint(self.n_genes, (density,), generator=g)
        vals = torch.randn(density, generator=g) * 0.05
        B = torch.zeros(self.n_genes, self.n_genes)
        B[rows, cols] = vals
        B.fill_diagonal_(0.0)
        with torch.no_grad():
            v = torch.randn(self.n_genes, generator=g); v /= v.norm() + 1e-8
            for _ in range(100):
                v = B @ v; v /= v.norm() + 1e-8
            spec = (B @ v).norm()
        scale = (0.9 / (spec + 1e-6)).clamp(max=1.0)
        self.B = B * scale
        self.I = torch.eye(self.n_genes)
        self.R = torch.linalg.inv(self.I - self.B) if self.n_genes <= 4000 else None

        # Program dictionary U (orthonormal columns) and projector P
        U = torch.randn(self.n_genes, self.F_dim, generator=g)
        U, _ = torch.linalg.qr(U)
        self.U = U
        self.U_true = U  # alias for pipeline export
        self.P = self.U @ self.U.T

        # Latent -> programs -> genes
        self.Wm = torch.randn(self.F_dim, self.z_dim, generator=g) * 0.3
        self.bm = torch.randn(self.F_dim, generator=g) * 0.1

        # Optional state shift from on-target shock
        self.gz = torch.randn(self.z_dim, generator=g) * 0.2

        # Targetable gene set
        self.perturbable = torch.arange(self.n_pert_genes)

    def _baseline_from_z(self, z):
        m = torch.nn.functional.relu((self.Wm @ z.T).T + self.bm)  # (N,F)
        x0 = (self.U @ m.T).T                                      # (N,G)
        return x0 + 0.1  # avoid exact zeros

    def generate_batch(self, batch_size=512, mode="down", k_direct=50):
        N, G, Z = batch_size, self.n_genes, self.z_dim

        # Latent state + baseline control
        z_ctrl = torch.randn(N, Z)
        x0 = self._baseline_from_z(z_ctrl)

        # Targets and heterogeneous effectiveness
        target_idx = self.perturbable[torch.randint(len(self.perturbable), (N,))]
        eff = torch.rand(N) * 0.6 + 0.2
        if mode == "down":
            d_tgt = -eff * x0[torch.arange(N), target_idx]
        else:
            d_tgt = +eff * (0.5 + x0[torch.arange(N), target_idx])

        # Authoritative step-0 clamp
        x_step0 = x0.clone()
        x_step0[torch.arange(N), target_idx] = x0[torch.arange(N), target_idx] + d_tgt

        # Propagate via SEM
        e_tgt = torch.nn.functional.one_hot(target_idx, num_classes=G).float()
        rhs = d_tgt[:, None] * e_tgt
        if self.R is not None:
            dx_total = rhs @ self.R.T
        else:
            dx_total = torch.linalg.solve(self.I - self.B, rhs.T).T

        # Decompose into mediated vs direct (then sparsify direct)
        dx_med = dx_total @ self.P
        dx_dir = dx_total - dx_med
        if k_direct is not None and 0 < k_direct < G:
            _, idxs = torch.topk(dx_dir.abs(), k=k_direct, dim=1)
            mask = torch.zeros_like(dx_dir).scatter_(1, idxs, 1.0)
            dx_dir = dx_dir * mask

        # Optional state shift tied to same shock
        dz = d_tgt[:, None] * self.gz[None, :]
        z_ref = z_ctrl + dz  # for diagnostics; model consumes z_ctrl

        # Final perturbed (overwrite target with clamp to keep semantics tight)
        x1 = x0 + dx_med + dx_dir
        x1[torch.arange(N), target_idx] = x_step0[torch.arange(N), target_idx]

        # Single environment for now (0)
        env_code = np.zeros(N, dtype=np.int64)

        return {
            "xbar_ctrl": x0[:, None, :].detach().numpy(),  # (N,1,G)
            "z_ctrl": z_ctrl.detach().numpy(),             # (N,Z)
            "xbar_q": x1.detach().numpy(),                 # (N,G)
            "target_idx": target_idx.detach().numpy(),     # (N,)
            "env_code": env_code,                          # (N,)
            "eff_true": eff.detach().numpy(),              # (N,)

            # diagnostics
            "U_true": self.U.detach().numpy(),             # (G,F)
            "dx_med_true": dx_med.detach().numpy(),        # (N,G)
            "dx_dir_true": dx_dir.detach().numpy(),        # (N,G)
            "x_clamped_authoritative": x_step0.detach().numpy(),  # (N,G)
            "mode": mode
        }


# === 2) Spoofed data pipeline ===================================================
class SpoofedDataPipeline:
    def __init__(self, sim_world: SimulatedGraftWorld):
        self.sim_world = sim_world
        self.temp_dir = Path(tempfile.mkdtemp())

    def setup(self) -> str:
        print(f"Creating spoofed data pipeline in: {self.temp_dir}")
        # genes
        with open(self.temp_dir / "gene_list.tsv", "w") as f:
            for i in range(self.sim_world.n_genes):
                f.write(f"GENE_{i}\n")
        # factors
        np.save(
            self.temp_dir / "factor_U.npy",
            self.sim_world.U_true.detach().cpu().numpy().T
        )
        # empty index (satisfy loader)
        pd.DataFrame({"cell_id": []}).to_parquet(self.temp_dir / "cell_index.parquet")
        # datasets yaml
        with open(self.temp_dir / "datasets.yaml", "w") as f:
            yaml.dump({"datasets": {"sim_dataset": {"raw_path": "dummy.h5ad"}}}, f)

        # config aligned with actual module signatures
        config = {
            "paths": {
                "gene_list": str(self.temp_dir / "gene_list.tsv"),
                "factor_U": str(self.temp_dir / "factor_U.npy"),
                "output_dir": str(self.temp_dir / "output"),
                "datasets_yaml": str(self.temp_dir / "datasets.yaml"),
                "index_parquet": str(self.temp_dir / "cell_index.parquet"),
                "scvi_model_dir": "dummy",
                "scvi_input_h5ad": "dummy",
                "control_index_dir": "dummy",
                "control_z_npz": "dummy",
                "control_xbar_npz": "dummy",
            },
            "training": {"batch_size": 128, "k_controls": 1, "seed": 42},
            "loss": {
                "distribution": {"weight": 1.0},
                "consistency": {"weight": 0.5},
                "rex": {"weight": 0.0},
                "direct": {"l1": 0.1},
                "orthogonality": {"weight": 0.1},
            },
            "model": {
                "propagator": {
                    "hidden": 64,
                    "layers": 2,
                    "steps": 2,
                    "dropout": 0.0,
                    "use_env_film": False,
                    "use_target_cond": True,
                    "target_embed_dim": 16,
                },
                "step0": {"hidden": 32, "init_eff": 0.9, "mode": "down"},
                "mediated": {"F": self.sim_world.F_dim, "hidden": 64},
                "direct": {"hidden": 64, "target_embed_dim": 32},
            },
            "optim": {"lr": 2e-4, "weight_decay": 1e-2},
        }
        (self.temp_dir / "output").mkdir(parents=True, exist_ok=True)
        with open(self.temp_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        return str(self.temp_dir / "config.yaml")

    def teardown(self):
        print(f"Cleaning up spoofed data pipeline from: {self.temp_dir}")
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# === 3) Mock dataset ============================================================
class MockGraftDataset:
    def __init__(self, sim_world: SimulatedGraftWorld, batch_size: int, test_case_knobs: dict):
        self.sim_world = sim_world
        self.batch_size = batch_size
        self.knobs = test_case_knobs

    def __iter__(self):
        while True:
            yield self.sim_world.generate_batch(self.batch_size, **self.knobs)


# === 4) Training loop ===========================================================
def run_test_case(config_path, sim_world, test_case_knobs, n_steps=101):
    print_every = max(50, n_steps // 20)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    paths, train_cfg, loss_cfg, model_cfg, optim_cfg = (
        cfg["paths"], cfg["training"], cfg["loss"], cfg["model"], cfg["optim"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U_np = np.load(paths["factor_U"])
    U_t = torch.tensor(U_np, dtype=torch.float32, device=device)  # (G,F)

    # Build components with correct signatures
    G = sim_world.n_genes
    z_dim = sim_world.z_dim

    propagator = StatePropagator(
        z_dim=z_dim,
        hidden=model_cfg["propagator"]["hidden"],
        layers=model_cfg["propagator"]["layers"],
        steps=model_cfg["propagator"]["steps"],
        dropout=model_cfg["propagator"]["dropout"],
        use_env_film=model_cfg["propagator"]["use_env_film"],
        use_target_cond=model_cfg["propagator"]["use_target_cond"],
        target_embed_dim=model_cfg["propagator"]["target_embed_dim"],
        n_envs=1,
        n_genes=G,
    ).to(device)

    step0 = StepZeroClamp(
        z_dim=z_dim,
        n_labs=1,
        hidden=model_cfg["step0"]["hidden"],
        init_eff=model_cfg["step0"]["init_eff"],
        mode=model_cfg["step0"]["mode"],
    ).to(device)

    mediated = MediatedHead(
        z_dim=z_dim,
        F=model_cfg["mediated"]["F"],
        hidden=model_cfg["mediated"]["hidden"],
    ).to(device)

    direct = TrueSparseDirectHead(
        z_dim=z_dim,
        n_genes=G,
        hidden=model_cfg["direct"]["hidden"],
        target_embed_dim=model_cfg["direct"]["target_embed_dim"],
    ).to(device)

    params = list(propagator.parameters()) + list(step0.parameters()) \
           + list(mediated.parameters()) + list(direct.parameters())
    opt = make_optimizer(params, optim_cfg)

    dist_fn = pick_dist_fn("swd")
    w_dist = loss_cfg["distribution"]["weight"]
    w_rex = loss_cfg.get("rex", {}).get("weight", 0.0)
    w_cons = loss_cfg["consistency"]["weight"]
    w_l1 = loss_cfg["direct"]["l1"]
    w_orth = loss_cfg["orthogonality"]["weight"]

    dataset = MockGraftDataset(sim_world, train_cfg["batch_size"], test_case_knobs)
    iterator = iter(dataset)

    loss_hist = {"total": [], "dist": [], "cons": [], "l1": [], "orth": []}
    final = {}

    for step in range(1, n_steps + 1):
        tb_np = next(iterator)
        # ensure env_code is in the batch the model expects
        tb_np.setdefault("env_code", np.zeros(tb_np["xbar_q"].shape[0], dtype=np.int64))
        tb = to_device(tb_np, device)

        # Forward pass with your actual helper
        y_pred, x_clamped_authoritative, dx_dir, dx_med, z_ref, eff = make_prediction(
            tb, step0, mediated, direct, propagator, U_t
        )

        # Losses (masked distribution, consistency, L1, orth)
        total, Ld, Lrex, Lc, Ll1, Lorth = compute_losses(
            dist_fn, y_pred, tb["xbar_q"], dx_dir, U_t, tb,
            w_dist, w_rex, w_cons, w_l1, w_orth, device
        )

        opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(params, max_norm=5.0)
        opt.step()

        loss_hist["total"].append(float(total.item()))
        loss_hist["dist"].append(float(Ld.item()))
        loss_hist["cons"].append(float(Lc.item()))
        loss_hist["l1"].append(float(Ll1.item()))
        loss_hist["orth"].append(float(Lorth.item()))

        if step % print_every == 0 or step == 1 or step == n_steps:
            print(f"[{step:05d}] total={total:.4f} dist={Ld:.4f} cons={Lc:.4f} l1={Ll1:.4f} orth={Lorth:.4f}")

        if step == n_steps:
            final = {
                "eff_pred": tb.get("eff_pred", eff.detach().cpu().numpy()),
                "eff_true": tb_np["eff_true"],
                "dx_dir_pred": dx_dir.detach().cpu().numpy(),
                "dx_dir_true": tb_np["dx_dir_true"],
            }

    # quick summary
    out = Path(paths["output_dir"]) / "sim_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"test_case": test_case_knobs, "n_steps": n_steps, "loss_avg": {k: float(np.mean(v)) for k, v in loss_hist.items()}}, f)
    print(f"Saved summary to {out}")

    # tiny eval readout
    avg_eff = float(np.mean(final["eff_pred"]))
    print("\n[Evaluation]")
    print(f"  eff_pred avg = {avg_eff:.3f} | eff_true avg = {np.mean(final['eff_true']):.3f}")
    # crude recall of sparse direct edges
    pred_mask = (np.abs(final["dx_dir_pred"]) > 0.1)
    true_mask = (np.abs(final["dx_dir_true"]) > 0.0)
    if true_mask.sum() > 0:
        recall = (pred_mask & true_mask).sum() / true_mask.sum()
        print(f"  direct-edge recall (>0.1): {100*float(recall):.2f}%")
    else:
        print("  (no true direct edges in this case)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_case", choices=["a", "b", "c"], default="b")
    ap.add_argument("--n-steps", type=int, default=501)
    ap.add_argument("--n_genes", type=int, default=2000)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--F_dim", type=int, default=64)
    ap.add_argument("--n_pert_genes", type=int, default=400)
    args = ap.parse_args()

    # knobs per case
    if args.test_case == "a":
        knobs = dict(mode="down", k_direct=0)    # step-0 only
    elif args.test_case == "b":
        knobs = dict(mode="down", k_direct=50)   # step-0 + sparse direct
    else:
        knobs = dict(mode="down", k_direct=50)   # step-0 + mediated + direct (decomp handled in sim)

    sim = SimulatedGraftWorld(
        n_genes=args.n_genes, z_dim=args.z_dim, F_dim=args.F_dim, n_pert_genes=args.n_pert_genes
    )
    pipeline = SpoofedDataPipeline(sim)
    config_path = pipeline.setup()

    try:
        run_test_case(config_path, sim, knobs, n_steps=args.n_steps)
    finally:
        pipeline.teardown()


if __name__ == "__main__":
    main()
