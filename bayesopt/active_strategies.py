import numpy as np
from scipy.sparse.linalg import eigsh

class BaseStrategy:
    """
    Abstract base class for all gene selection strategies.
    The simulation loop calls 'select_next_batch', reveals the data, 
    and then calls 'update' to let the strategy learn.
    """
    def __init__(self, total_genes, args):
        self.n_genes = total_genes
        self.args = args
        self.name = "BaseStrategy"

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        """
        Decides which genes to reveal next.
        Returns: np.array of indices (integers 0..N).
        """
        raise NotImplementedError

    def update(self, new_indices, new_values):
        """
        Hook for the strategy to digest new data after it has been revealed.
        """
        pass
    
    def predict(self, currently_known_mask, y_obs_vector):
        """
        Optional: Returns the imputed vector for ALL genes (known + unknown).
        If not implemented, the simulation runner defaults to Mean Imputation.
        """
        return None


class StaticStrategy(BaseStrategy):
    """
    Wrapper for passive strategies (Random, Magnitude, Covariance).
    The order is determined EXTERNALLY (in the simulation script) and passed here.
    Supports 'random_samp_pct' to mix top-ranked picks with random exploration.
    """
    def __init__(self, total_genes, args, sorted_indices, name="Static"):
        super().__init__(total_genes, args)
        self.name = name
        # Ensure indices are integers
        self.queue = np.array(sorted_indices, dtype=int)
        self.pointer = 0
        self.rng = np.random.default_rng(getattr(args, 'seed', 42))

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # 1. Determine Split (Static vs Random)
        pct = getattr(self.args, 'random_samp_pct', 0.0)
        n_rnd = int(n_to_select * pct)
        n_det = n_to_select - n_rnd
        
        selected_indices = []

        # 2. Deterministic Selection (Top of the Queue)
        # We must scan forward to find 'n_det' genes that haven't been revealed yet
        # (They might have been picked randomly in a previous batch)
        while len(selected_indices) < n_det and self.pointer < len(self.queue):
            candidate = self.queue[self.pointer]
            self.pointer += 1
            if not currently_known_mask[candidate]:
                selected_indices.append(candidate)
        
        # 3. Random Selection (Background/Exploration)
        if n_rnd > 0:
            # Candidates are: ~Known AND ~Selected_Just_Now
            candidates_mask = ~currently_known_mask
            if len(selected_indices) > 0:
                candidates_mask[selected_indices] = False
            
            valid_candidates = np.where(candidates_mask)[0]
            
            if len(valid_candidates) > 0:
                to_pick = min(n_rnd, len(valid_candidates))
                picked = self.rng.choice(valid_candidates, size=to_pick, replace=False)
                selected_indices.extend(picked)
        
        return np.array(selected_indices, dtype=int)


class StaticGPStrategy(StaticStrategy):
    """
    Same as StaticStrategy (fixed order), but uses a GP for prediction
    instead of Mean Imputation. Use this for the 'Random + GP' baseline.
    """
    def __init__(self, total_genes, args, sorted_indices, gp_learner, name="Static GP"):
        super().__init__(total_genes, args, sorted_indices, name)
        self.learner = gp_learner
        self._batch_count = 0

    def update(self, new_indices, new_values):
        # Static strategies do not change their queue based on data,
        # so we do not need to process updates here.
        pass

    def predict(self, currently_known_mask, y_obs_vector):
        """
        Uses the ActiveGPLearner to impute values.
        Retrains (updates weights) every 'gp_recompute_freq' batches.
        """
        self._batch_count += 1
        
        # Trigger GP re-weighting/training
        if self._batch_count % self.args.gp_recompute_freq == 0:
            self.learner.update(currently_known_mask, y_obs_vector)

        # Get Mean and Variance
        y_pred, y_var = self.learner.predict(currently_known_mask, y_obs_vector)
        
        mode = getattr(self.args, 'gp_imputation_mode', 'mean')
        
        if mode == 'sample':
            std = np.sqrt(np.maximum(y_var, 0))
            noise = np.random.randn(*y_pred.shape)
            return y_pred + (std * noise)
            
        return y_pred


class ActiveGPStrategy(BaseStrategy):
    """
    Base class for Active Learning.
    Holds the 'ActiveGPLearner' (from active_gp.py) to generate predictions.
    """
    def __init__(self, total_genes, args, gp_learner):
        super().__init__(total_genes, args)
        self.learner = gp_learner
        self.step_counter = 0

    def update(self, new_indices, new_values):
        """
        Triggers the GP Kernel update/re-weighting.
        """
        # We can implement the re-weighting frequency logic here
        pass

    def _get_scored_candidates(self, currently_known_mask, y_obs_vector):
        """
        Helper: Uses the learner to predict Mean and Variance for all UNKNOWN genes.
        Strategies will use these (Mean, Var) to calculate their own scores (Acquisition Functions).
        """
        # 1. Update Learner (if needed based on frequency)
        if self.step_counter % self.args.gp_recompute_freq == 0:
            self.learner.update(currently_known_mask, y_obs_vector)
        self.step_counter += 1

        # 2. Get Predictions (Mean AND Variance)
        preds_mean, preds_var = self.learner.predict_with_variance(currently_known_mask, y_obs_vector)

        return preds_mean, preds_var
    
    def predict(self, currently_known_mask, y_obs_vector):
        """
        Returns the imputed vector for ALL genes.
        Modes:
        - 'mean': Posterior Mean (Standard)
        - 'sample': Sample from Posterior Gaussian (Mean + std * N(0,1))
        """
        # Learner returns (mean, var) for unobserved
        y_pred, y_var = self.learner.predict(currently_known_mask, y_obs_vector)
        
        mode = getattr(self.args, 'gp_imputation_mode', 'mean')
        
        if mode == 'sample':
            # Sample: y ~ N(mean, var)
            # std = sqrt(var)
            # We assume y_var is variance vector (diagonal)
            std = np.sqrt(np.maximum(y_var, 0))
            noise = np.random.randn(*y_pred.shape)
            return y_pred + (std * noise)
            
        return y_pred


class HighLeverageStrategy(ActiveGPStrategy):
    """
    Strategy: "Anchor Sampling"
    Selects unknown genes with highest predicted MAGNITUDE (positive or negative).
    Goal: Find the strong regulators to anchor the correlation estimate.
    Score = |Mean| + Beta * StdDev
    """
    def __init__(self, total_genes, args, gp_learner, prior_indices=None):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active HighLeverage"
        self.beta = getattr(args, 'acq_beta', 1.0) # Allow CLI override, default 1.0
        self.prior_indices = prior_indices # Indices to use for the first batch (e.g. Covariance)

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # 0. Cold Start: If we know nothing...
        n_known = np.sum(currently_known_mask)
        if n_known == 0:
            # OPTION A: Use Prior (e.g. Control Covariance) if available
            if self.prior_indices is not None:
                # Return the top N genes from the prior list
                return self.prior_indices[:n_to_select]
            
            # OPTION B: Fallback to Random to span the space
            rng = np.random.default_rng(getattr(self.args, 'seed', 42))
            all_indices = np.arange(self.n_genes)
            return rng.choice(all_indices, size=n_to_select, replace=False)

        # 1. Get predictions for all genes
        means, vars = self._get_scored_candidates(currently_known_mask, y_obs_vector)
        stds = np.sqrt(vars)

        # 2. Calculate Acquisition Score
        # Score = |Mean| + Beta * Std
        # Prioritize high magnitude predictions (Anchors) with some exploration (Std)
        scores = np.abs(means) + (self.beta * stds)
        
        # 3. Mask out known genes
        # Set their score to -infinity so they are pushed to the bottom
        scores[currently_known_mask] = -np.inf
        
        # 4. Pick top N
        # argpartition moves the top N elements to the end of the array
        top_indices = np.argpartition(scores, -n_to_select)[-n_to_select:]
        
        return top_indices


class UncertaintyStrategy(ActiveGPStrategy):
    """
    Strategy: "Uncertainty Sampling"
    Selects genes where the GP is most unsure (highest Variance).
    Goal: Explore the unknown space.
    """
    def __init__(self, total_genes, args, gp_learner, prior_indices=None):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active Uncertainty"
        self.prior_indices = prior_indices

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # 0. Cold Start
        n_known = np.sum(currently_known_mask)
        if n_known == 0:
            if self.prior_indices is not None:
                return self.prior_indices[:n_to_select]
            
            rng = np.random.default_rng(getattr(self.args, 'seed', 42))
            all_indices = np.arange(self.n_genes)
            return rng.choice(all_indices, size=n_to_select, replace=False)

        # 1. Get predictions (Variance only needed)
        _, vars = self._get_scored_candidates(currently_known_mask, y_obs_vector)
        
        # 2. Score = Variance
        scores = vars.copy()
        # Mask known genes
        scores[currently_known_mask] = -np.inf
        
        # 3. Pick top N
        top_indices = np.argpartition(scores, -n_to_select)[-n_to_select:]
        
        return top_indices


class DiversityStrategy(ActiveGPStrategy):
    """
    Strategy: "Kernel Diversity" (Greedy Farthest Point Sampling)
    Selects genes that are least similar (in Kernel space) to the current known set.
    Goal: Span the biological space efficiently.
    """
    def __init__(self, total_genes, args, gp_learner, prior_indices=None):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active Diversity"
        self.prior_indices = prior_indices

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # 1. Access the Kernel Matrix (N, N)
        if self.learner.K_fused is None:
            # Fallback if kernel not ready
            rng = np.random.default_rng(getattr(self.args, 'seed', 42))
            unobs_indices = np.where(~currently_known_mask)[0]
            return rng.choice(unobs_indices, size=n_to_select, replace=False)
            
        K = self.learner.K_fused
        N = self.n_genes
        
        # 2. Initialize "Max Similarity" vector
        # max_sim[i] = similarity of gene i to its closest known neighbor
        # We want to pick i that MINIMIZES this max_sim (farthest from known set)
        
        known_indices = np.where(currently_known_mask)[0]
        
        # Cold Start Handling
        if len(known_indices) == 0:
            # Pick first point
            if self.prior_indices is not None:
                first_idx = self.prior_indices[0]
            else:
                rng = np.random.default_rng(getattr(self.args, 'seed', 42))
                first_idx = rng.integers(0, N)
            
            selected_indices = [first_idx]
            # Initialize max_sim with this first point's similarities
            current_max_sim = K[first_idx, :].copy() 
            # Mark first point as effectively "already picked" (max sim = infinity or 1.0)
            current_max_sim[first_idx] = 1.0
        else:
            selected_indices = []
            # Calculate initial max_sim against all currently known
            # shape (N,)
            current_max_sim = np.max(K[:, known_indices], axis=1)

        # 3. Greedy Selection Loop
        # We need to pick (n_to_select - len(selected_indices)) more points
        n_needed = n_to_select - len(selected_indices)
        
        # Mask out known genes from selection by setting their sim to 1.0 (max possible)
        # So argmin will never pick them (assuming unselected genes have sim < 1.0)
        # To be safe, we set them to infinity
        current_max_sim[currently_known_mask] = np.inf
        for idx in selected_indices:
             current_max_sim[idx] = np.inf
        
        for _ in range(n_needed):
            # Pick the "loneliest" gene
            next_idx = np.argmin(current_max_sim)
            selected_indices.append(next_idx)
            
            # Update max_sim: newly picked gene might be the new closest neighbor for some points
            new_sims = K[next_idx, :]
            current_max_sim = np.maximum(current_max_sim, new_sims)
            
            # Ensure we don't pick it again
            current_max_sim[next_idx] = np.inf
            
        return np.array(selected_indices, dtype=int)

class PCUncertaintyStrategy(ActiveGPStrategy):
    """
    Strategy: "Principal Component Uncertainty" (Eigengene Sampling)
    1. Computes Posterior Covariance.
    2. Decomposes into Top K Eigenvectors (Principal Components of Uncertainty).
    3. Scores genes by their weighted loading on these components.
    
    Goal: Target "Modules" of uncertainty rather than "Orphans".
    """
    def __init__(self, total_genes, args, gp_learner, prior_indices=None):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active PC-Uncertainty"
        self.prior_indices = prior_indices
        self.recompute_freq = getattr(args, 'pca_recompute_freq', 1)
        self.top_k = getattr(args, 'pca_top_k', 50)
        
        # Cache
        self.cached_scores = None
        self.batches_since_compute = 9999

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # 0. Cold Start
        n_known = np.sum(currently_known_mask)
        if n_known == 0:
            if self.prior_indices is not None:
                return self.prior_indices[:n_to_select]
            
            rng = np.random.default_rng(getattr(self.args, 'seed', 42))
            all_indices = np.arange(self.n_genes)
            return rng.choice(all_indices, size=n_to_select, replace=False)
            
        # 1. Check if we need to recompute the PCA
        if self.batches_since_compute >= self.recompute_freq:
            print(f"   [PC-Uncertainty] Recomputing Covariance & SVD (Top {self.top_k})...")
            
            # Update learner weights if needed
            self.learner.update(currently_known_mask, y_obs_vector)
            
            # Get Full Posterior Covariance (N_un x N_un)
            # Note: predict_with_covariance returns (mean, cov)
            _, cov_matrix = self.learner.predict_with_covariance(currently_known_mask, y_obs_vector)
            
            # Decompose (Truncated SVD / Eigsh)
            # cov_matrix is symmetric positive definite (or semi-definite)
            # Use 'LM' (Largest Magnitude)
            k = min(self.top_k, cov_matrix.shape[0] - 1)
            vals, vecs = eigsh(cov_matrix, k=k, which='LM')
            
            # vals: (k,) eigenvalues
            # vecs: (N_un, k) eigenvectors
            
            # Calculate Scores: Sum of squared loadings weighted by eigenvalue
            # Score_i = Sum_k ( lambda_k * (v_ik)^2 )
            # We take abs(vals) just in case of slight numerical noise (though Cov is PSD)
            weighted_sq_loadings = (vecs ** 2) @ np.abs(vals)
            
            # Map back to full genome size
            unobs_indices = np.where(~currently_known_mask)[0]
            full_scores = np.zeros(self.n_genes)
            full_scores[unobs_indices] = weighted_sq_loadings
            full_scores[currently_known_mask] = -np.inf
            
            self.cached_scores = full_scores
            self.batches_since_compute = 0
            
        else:
            # Use cached scores, but mask out newly revealed genes
            self.cached_scores[currently_known_mask] = -np.inf
            self.batches_since_compute += 1
            
        # 2. Select Top N
        top_indices = np.argpartition(self.cached_scores, -n_to_select)[-n_to_select:]
        
        return top_indices


class VarianceReductionStrategy(ActiveGPStrategy):
    """
    Strategy: "Stepwise Variance Reduction" (Kriging Believer / A-Optimality).
    
    Logic:
    1. Compute Full Posterior Covariance Matrix.
    2. Stepwise greedy selection (Batch Size times):
       a. Score(k) = ||Cov(:, k)||^2 / Var(k)  (Reduction in total trace)
       b. Pick Best Gene k.
       c. Update Covariance Matrix assuming k is known (Schur Complement).
       d. Repeat.
    
    Goal: Minimizes total system uncertainty while handling redundancy perfectly.
    """
    def __init__(self, total_genes, args, gp_learner, prior_indices=None):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active Var-Reduction (Stepwise)"
        self.prior_indices = prior_indices
        self.subset_size = getattr(args, 'stepwise_subset_size', 400)

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # 0. Cold Start
        n_known = np.sum(currently_known_mask)
        if n_known == 0:
            if self.prior_indices is not None:
                return self.prior_indices[:n_to_select]
            
            rng = np.random.default_rng(getattr(self.args, 'seed', 42))
            all_indices = np.arange(self.n_genes)
            return rng.choice(all_indices, size=n_to_select, replace=False)
            
        # 1. Initial Covariance Calculation
        self.learner.update(currently_known_mask, y_obs_vector)
        _, cov_matrix = self.learner.predict_with_covariance(currently_known_mask, y_obs_vector)
        
        unobs_indices = np.where(~currently_known_mask)[0]
        n_unobs = len(unobs_indices)
        
        # --- SPEED OPTIMIZATION: Candidate Subset Selection ---
        # Instead of updating the full 9400x9400 matrix, we identify the top M candidates
        # based on their *initial* reduction score, and then run the greedy loop only on them.
        
        # A. Calculate Initial Scores (Global)
        diags = np.diag(cov_matrix)
        diags = np.maximum(diags, 1e-9)
        col_norms_sq = np.sum(cov_matrix**2, axis=0)
        initial_scores = col_norms_sq / diags
        
        # B. Select Working Set (Top M)
        # We need at least n_to_select candidates
        M = max(self.subset_size, n_to_select)
        M = min(M, n_unobs) # Don't exceed available genes
        
        # Get indices of top M scores
        # argpartition puts top M at the end
        top_M_local_indices = np.argpartition(initial_scores, -M)[-M:]
        
        # Slice the covariance matrix to just these M candidates
        # current_cov is now (M, M)
        current_cov = cov_matrix[np.ix_(top_M_local_indices, top_M_local_indices)]
        
        # Local mask for the Working Set (size M)
        available_mask_subset = np.ones(M, dtype=bool)
        selected_indices_in_subset = []
        
        # 2. Greedy Stepwise Loop (on Subset)
        for _ in range(n_to_select):
            # Calculate Scores on the (shrinking) covariance of the subset
            diags_sub = np.diag(current_cov)
            diags_sub = np.maximum(diags_sub, 1e-9)
            col_norms_sq_sub = np.sum(current_cov**2, axis=0)
            
            scores_sub = col_norms_sq_sub / diags_sub
            scores_sub[~available_mask_subset] = -np.inf
            
            # Pick Winner (Index relative to the subset M)
            best_idx_in_subset = np.argmax(scores_sub)
            selected_indices_in_subset.append(best_idx_in_subset)
            available_mask_subset[best_idx_in_subset] = False
            
            # Update Covariance (M x M update)
            v = current_cov[:, best_idx_in_subset]
            var = current_cov[best_idx_in_subset, best_idx_in_subset]
            
            update_term = np.outer(v, v) / (var + 1e-9)
            current_cov = current_cov - update_term

        # 3. Map back to Global Indices
        # top_M_local_indices maps Subset(0..M) -> Unobserved(0..N_un)
        # selected_indices_in_subset is indices into top_M_local_indices
        
        chosen_local_indices = top_M_local_indices[selected_indices_in_subset]
        selected_global_indices = unobs_indices[chosen_local_indices]
        
        return selected_global_indices