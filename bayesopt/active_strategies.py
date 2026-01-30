import numpy as np
import pandas as pd

class BaseStrategy:
    """
    Abstract base class for all gene selection strategies.
    """
    def __init__(self, total_genes, args):
        self.n_genes = total_genes
        self.args = args
        self.name = "BaseStrategy"

    def select_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        """
        Selects the next batch of gene indices to reveal.
        
        Args:
            n_to_select (int): Number of genes to pick.
            currently_known_mask (np.array): Boolean mask of genes already revealed.
            y_obs_vector (np.array): (Optional) The observed values for the known genes.
                                     Used by active strategies to update their models.
        
        Returns:
            np.array: Indices of the selected genes (in the original 0..N order).
        """
        raise NotImplementedError

    def update(self, new_indices, new_values):
        """
        Optional hook to update internal models (e.g. GP) with newly revealed data.
        """
        pass


class StaticStrategy(BaseStrategy):
    """
    Wrapper for passive/static ordering (Random, Magnitude, Covariance).
    The order is determined at initialization and never changes.
    """
    def __init__(self, sorted_df_indices, name="Static"):
        """
        Args:
            sorted_df_indices (np.array): Array of integer indices representing the 
                                          fixed order of genes to reveal.
            name (str): Display name for the strategy.
        """
        # We don't need total_genes or args for the simple static case
        self.queue = sorted_df_indices
        self.name = name
        self.pointer = 0

    def select_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # Simply slice the next N items from the pre-calculated queue
        start = self.pointer
        end = start + n_to_select
        batch_indices = self.queue[start:end]
        
        self.pointer = end
        return batch_indices


class ActiveGPStrategy(BaseStrategy):
    """
    Base class for strategies that use the ActiveGPLearner.
    Handles the model updates and prediction calls common to all GP strategies.
    """
    def __init__(self, total_genes, args, gp_learner):
        super().__init__(total_genes, args)
        self.learner = gp_learner
        # We start with no genes known
        self.step_counter = 0

    def update(self, new_indices, new_values):
        """
        Updates the internal GP model with the new batch of data.
        """
        # Create full mask of what we know NOW (after this batch)
        # In the simulation runner, we typically pass the *current* mask.
        # This update hook allows the strategy to trigger the learner's update().
        
        # NOTE: The simulation runner handles the global mask. 
        # Here we just pass the indices and values to the learner if needed.
        # But ActiveGPLearner.update() takes a full mask. 
        # So we might rely on the 'select_batch' to trigger updates or 
        # construct the mask here.
        pass

    def _get_predictions(self, known_mask, y_obs):
        """
        Helper to get Mean and Variance for all unknown genes.
        """
        # Trigger learner update logic (e.g. re-weighting every N batches)
        if self.step_counter % self.args.gp_recompute_freq == 0:
            self.learner.update(known_mask, y_obs)
            
        self.step_counter += 1
        
        # Get full predictions vector (imputed)
        y_pred_full = self.learner.predict(known_mask, y_obs)
        
        # Get posterior variance (if implemented in learner, otherwise placeholder)
        # Note: We will need to update ActiveGPLearner to return variance too.
        y_var_full = getattr(self.learner, "last_prediction_variance", np.ones_like(y_pred_full))
        
        return y_pred_full, y_var_full


class HighLeverageStrategy(ActiveGPStrategy):
    """
    Active Learning: Selects genes likely to have high absolute effect (Anchors).
    Score = |Mean| + Beta * Std
    """
    def __init__(self, total_genes, args, gp_learner):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active_HighLeverage"
        self.beta = getattr(args, 'acq_beta', 1.0) # Trade-off parameter

    def select_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # Implementation:
        # 1. Get predictions (Mean, Var) for unknown genes
        # 2. Compute Score = Abs(Mean) + Beta * Sqrt(Var)
        # 3. Argpartition to get top N indices
        # 4. Return indices
        pass


class UncertaintyStrategy(ActiveGPStrategy):
    """
    Active Learning: Selects genes with highest posterior uncertainty.
    Score = Variance
    """
    def __init__(self, total_genes, args, gp_learner):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active_Uncertainty"

    def select_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # Implementation:
        # 1. Get predictions (Var only needed)
        # 2. Score = Var
        # 3. Return top N indices
        pass


class DiversityStrategy(ActiveGPStrategy):
    """
    Active Learning: Selects genes least similar to the current set (Kernel Distance).
    """
    def __init__(self, total_genes, args, gp_learner):
        super().__init__(total_genes, args, gp_learner)
        self.name = "Active_Diversity"

    def select_batch(self, n_to_select, currently_known_mask, y_obs_vector=None):
        # Implementation:
        # 1. Look at Kernel K_UU (Unknown vs Unknown) and K_UO (Unknown vs Known)
        # 2. Greedy selection (farthest point) or Kernel K-Means center selection
        # 3. Return N indices
        pass