#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class AveragePredictor(nn.Module):
    """
    Baseline: compute the average delta (perturbed - control_mean) on the TRAIN pseudobulk
    and predict that same delta for every test perturbation.
    Final prediction = control_mean + avg_delta.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("ctrl_mean", None)   # (G,)
        self.register_buffer("avg_delta", None)   # (G,)

    @torch.no_grad()
    def fit(self, adata, target_label: str, control_label: str, device: str = "cpu"):
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X = X.astype("float32", copy=False)
        labels = adata.obs[target_label].astype(str).values
        ctrl_idx = (labels == control_label)
        pert_idx = ~ctrl_idx
        if not ctrl_idx.any() or not pert_idx.any():
            raise ValueError("AveragePredictor.fit: need both control and perturbed rows in pseudobulk.")
        ctrl_mean = X[ctrl_idx].mean(axis=0)                           # (G,)
        deltas = X[pert_idx] - ctrl_mean[None, :]                      # (P,G)
        avg_delta = deltas.mean(axis=0)                                # (G,)
        self.ctrl_mean = torch.tensor(ctrl_mean, device=device)
        self.avg_delta = torch.tensor(avg_delta, device=device)

    @torch.no_grad()
    def predict(self, n_rows: int) -> torch.Tensor:
        """
        Return a (n_rows, G) tensor of predictions = ctrl_mean + avg_delta, repeated.
        Assumes fit() was called.
        """
        if (self.ctrl_mean is None) or (self.avg_delta is None):
            raise RuntimeError("AveragePredictor: call fit() before predict().")
        y = self.ctrl_mean + self.avg_delta                             # (G,)
        return y.unsqueeze(0).repeat(n_rows, 1)                         # (n_rows, G)
