import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import warnings

# Import necessary functions from existing codebase
from utils import read_and_intersect, compute_deltas

from gp import (
    compute_global_kernel_weights, 
    aggregate_kernels,
    create_embedding_kernel_from_df,
    build_embedding_kernels_from_yaml # Now using this directly
)

class ActiveGPLearner:
    def __init__(self, all_genes, args):
        """
        Args:
            all_genes: List of the N genes in the simulation (the fixed "Universe").
            args: Namespace containing external_list, embeddings_yaml, agg_mode, etc.
        """
        self.genes = np.array(all_genes, dtype=str)
        self.n_genes = len(self.genes)
        self.args = args
        
        # Store kernels as (K_dense, perts_list) tuples
        self.kernels_and_perts = [] 
        self.kernel_names = []
        
        # --- ADAPTER: Dummy Target for Intersections ---
        # utils.read_and_intersect expects an AnnData object to define the "target" set of perturbations.
        # We create a lightweight container holding our master gene list in .obs[target_label].
        obs_perts = pd.DataFrame({args.target_label: self.genes})
        obs_perts.index = self.genes
        self.dummy_target_perts = ad.AnnData(
            X=sparse.csr_matrix((self.n_genes, 1), dtype=np.float32),
            obs=obs_perts
        )

        # State
        self.current_weights = None
        self.K_fused = None
        self.perts_fused = None 
        
        # Initialize
        self._load_all_kernels()
        self._fuse_kernels(initial=True)

    def _load_all_kernels(self):
        """
        Loads external expression datasets (list) and pathway databases (single yaml).
        Stores them as (Matrix, PertList) tuples.
        """
        print(f"[\u2699 ActiveGP] Loading kernels for universe of {self.n_genes} genes...")
        
        # ---------------------------------------------------------
        # 1. External Expression Kernels (from list of H5ADs)
        # ---------------------------------------------------------
        ext_files = []
        if self.args.external_list:
            if os.path.exists(self.args.external_list):
                with open(self.args.external_list, 'r') as f:
                    ext_files = [line.strip() for line in f if line.strip()]
        elif self.args.external_h5ad:
            ext_files = [self.args.external_h5ad]
            
        for fpath in ext_files:
            if not os.path.exists(fpath): 
                print(f"Warning: External file not found: {fpath}")
                continue
            try:
                # A. Read and Intersect (Using Dummy Target Adapter)
                # Filters external data to keep only perturbations present in our master list.
                adata_ext = read_and_intersect(
                    ext_path=fpath,
                    adata_target=self.dummy_target_perts,
                    target_label=self.args.target_label,
                    control_label=self.args.control_label,
                    already_logged=True # Assume externals are pre-processed
                )
                
                # B. Compute Deltas (Vectors)
                deltas, _ = compute_deltas(
                    adata_ext, 
                    self.args.target_label, 
                    self.args.control_label
                )
                
                # C. Build Correlation Kernel
                valid_perts = sorted(list(deltas.keys()))
                if len(valid_perts) < 5:
                    continue
                    
                M = np.stack([deltas[p] for p in valid_perts]) # (P, G_ext)
                
                # Row-wise Normalize (Cosine/Correlation)
                M = M - M.mean(axis=1, keepdims=True)
                norms = np.linalg.norm(M, axis=1, keepdims=True)
                M = np.divide(M, norms, where=norms > 1e-9)
                
                K_sub = M @ M.T # (P, P)
                np.fill_diagonal(K_sub, 1.0)
                
                self.kernels_and_perts.append((K_sub.astype(np.float32), valid_perts))
                self.kernel_names.append(os.path.basename(fpath))
                
            except Exception as e:
                print(f"    Error processing {fpath}: {e}")

        # ---------------------------------------------------------
        # 2. Pathway/Embedding Kernels (from single YAML)
        # ---------------------------------------------------------
        # Reuses gp.build_embedding_kernels_from_yaml exactly.
        # We pass our master gene list as both 'perts_O' and 'perts_U' so it builds 
        # kernels for the full universe of candidates.
        if getattr(self.args, 'embeddings_yaml', None) and os.path.exists(self.args.embeddings_yaml):
            try:
                # We use the master list as 'var_names_target' to align feature columns,
                # and as 'perts' to define the rows of the kernel.
                emb_kernels = build_embedding_kernels_from_yaml(
                    embeddings_yaml=self.args.embeddings_yaml,
                    var_names_target=list(self.genes),
                    perts_O=list(self.genes), # Pass all genes as candidates
                    perts_U=[],               # No split needed here
                    emb_metric=getattr(self.args, 'emb_metric', 'cosine'),
                    emb_pca_dim=getattr(self.args, 'emb_pca_dim', 0),
                    emb_rbf_gamma=getattr(self.args, 'emb_rbf_gamma', 0.0)
                )
                self.kernels_and_perts.extend(emb_kernels)
                # Generate generic names for these new kernels
                for i in range(len(emb_kernels)):
                    self.kernel_names.append(f"embedding_source_{i}")
            except Exception as e:
                print(f"    Error processing embeddings yaml: {e}")

        # ---------------------------------------------------------
        # 3. Fallback Identity
        # ---------------------------------------------------------
        if not self.kernels_and_perts:
            print("Warning: No kernels loaded. Using Identity kernel on all genes.")
            K_eye = np.eye(self.n_genes, dtype=np.float32)
            self.kernels_and_perts.append((K_eye, list(self.genes)))
            self.kernel_names.append("Identity")

    def _fuse_kernels(self, initial=False):
        """
        Aggregate kernels using gp.aggregate_kernels.
        """
        if initial:
            n = len(self.kernels_and_perts)
            self.current_weights = [1.0 / n] * n
            
        # 1. Call gp.aggregate_kernels
        # Returns a kernel over the UNION of all perts found in sources
        K_agg, perts_agg = aggregate_kernels(
            self.kernels_and_perts,
            method="wmean" if self.args.kernel_agg == "wmean" else "mean",
            weights=self.current_weights
        )
        
        # 2. Align to Master List
        # The aggregated kernel might be smaller/differently ordered than our Universe.
        # We align it to self.genes, filling missing spots with Identity behavior.
        if len(perts_agg) != self.n_genes or perts_agg != list(self.genes):
            K_final = np.zeros((self.n_genes, self.n_genes), dtype=np.float32)
            
            # Map indices
            pert_to_idx = {p: i for i, p in enumerate(perts_agg)}
            
            # Identify which master genes are in the aggregated kernel
            master_indices = []
            agg_indices = []
            
            for i, g in enumerate(self.genes):
                if g in pert_to_idx:
                    master_indices.append(i)
                    agg_indices.append(pert_to_idx[g])
            
            if master_indices:
                ix_grid = np.ix_(master_indices, master_indices)
                agg_grid = np.ix_(agg_indices, agg_indices)
                K_final[ix_grid] = K_agg[agg_grid]
            
            # Fill diagonal with 1.0 for everyone (crucial for invertibility)
            np.fill_diagonal(K_final, 1.0)
            
            self.K_fused = K_final
        else:
            self.K_fused = K_agg

    def update(self, obs_bool_mask, y_obs_vector):
        """
        Re-weights kernels based on alignment with the observed data.
        """
        if self.args.kernel_agg != "wmean":
            return 
            
        # --- ADAPTER: Dummy Training Data ---
        # gp.compute_global_kernel_weights requires an AnnData object to compute "deltas"
        # (Observed - Control). We only have a 1D vector of observed values.
        # We construct a dummy AnnData where X is that vector, so that 
        # compute_deltas(dummy) returns exactly y_obs_vector.
        
        obs_genes = self.genes[obs_bool_mask]
        n_obs = len(obs_genes)
        
        if n_obs < 10: return
        
        # Add a dummy "control" row with value 0.0.
        # This ensures that (Value - Control) = Value - 0 = Value.
        X_train = np.concatenate([y_obs_vector, [0.0]]).reshape(-1, 1) # (N_obs + 1, 1)
        obs_labels = list(obs_genes) + [self.args.control_label]
        
        adata_train_dummy = ad.AnnData(
            X=sparse.csr_matrix(X_train, dtype=np.float32),
            obs=pd.DataFrame({self.args.target_label: obs_labels}, index=range(n_obs+1)),
            var=pd.DataFrame(index=["DummyFeature"]) 
        )
        
        # 2. Call gp.compute_global_kernel_weights
        new_weights, _ = compute_global_kernel_weights(
            kernels_and_perts=self.kernels_and_perts,
            adata_train_target=adata_train_dummy,
            target_label=self.args.target_label,
            control_label=self.args.control_label,
            perts_O=list(obs_genes),
            gamma=getattr(self.args, 'kernel_weight_gamma', 1.0)
        )
        
        self.current_weights = new_weights
        self._fuse_kernels(initial=False)

    def predict(self, obs_bool_mask, y_obs_vector):
        """
        Predicts y for UNobserved genes given Observed genes.
        """
        obs_idx = np.where(obs_bool_mask)[0]
        unobs_idx = np.where(~obs_bool_mask)[0]
        
        # Center observed y
        y_mean = np.mean(y_obs_vector)
        y_centered = y_obs_vector - y_mean
        
        # Slice Fused Kernel
        K_OO = self.K_fused[np.ix_(obs_idx, obs_idx)]
        K_UO = self.K_fused[np.ix_(unobs_idx, obs_idx)]
        
        # Solve (Ridge)
        lambda_i = getattr(self.args, 'gp_noise_var', 0.01)
        K_reg = K_OO + lambda_i * np.eye(len(obs_idx), dtype=np.float32)
        
        try:
            # (N_obs, 1)
            alpha = np.linalg.solve(K_reg, y_centered)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K_reg, y_centered, rcond=None)[0]
            
        # Predict
        y_pred_centered = K_UO @ alpha
        y_pred = y_pred_centered + y_mean
        
        # Reconstruct full vector
        y_full = np.zeros(self.n_genes)
        y_full[obs_idx] = y_obs_vector
        y_full[unobs_idx] = y_pred
        
        return y_full