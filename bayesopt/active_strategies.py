import numpy as np

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
    """
    def __init__(self, total_genes, args, sorted_indices, name="Static"):
        super().__init__(total_genes, args)
        self.name = name
        self.queue = np.array(sorted_indices, dtype=int)
        self.pointer = 0

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # Simply take the next N indices from the pre-calculated queue
        start = self.pointer
        end = min(self.pointer + n_to_select, len(self.queue))
        
        if start >= len(self.queue):
            return np.array([], dtype=int)
            
        batch_indices = self.queue[start:end]
        self.pointer = end
        
        return batch_indices


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

        # Return full genome prediction
        return self.learner.predict(currently_known_mask, y_obs_vector)


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
    def __init__(self, total_genes, args, gp_learner):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active_Uncertainty"

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # 1. Get predictions
        _, vars = self._get_scored_candidates(currently_known_mask, y_obs_vector)
        
        # 2. Score = Variance
        scores = vars.copy()
        scores[currently_known_mask] = -np.inf
        
        # 3. Pick top N
        top_indices = np.argpartition(scores, -n_to_select)[-n_to_select:]
        
        return top_indices


class DiversityStrategy(ActiveGPStrategy):
    """
    Strategy: "Kernel Diversity"
    Selects genes that are least similar (in Kernel space) to the current known set.
    Goal: Span the biological space (e.g. don't pick 100 ribosomal genes).
    """
    def __init__(self, total_genes, args, gp_learner):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active_Diversity"

    def select_next_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # This will require access to the Kernel Matrix from the learner.
        # Logic: Greedy selection of point with max min-distance to known set.
        pass
