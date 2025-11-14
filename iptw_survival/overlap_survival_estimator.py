from .iptw_survival_estimator import IPTWSurvivalEstimator
import numpy as np

class OverlapWeightSurvivalEstimator(IPTWSurvivalEstimator):
    """
    Overlap-weighted survival estimator using existing propensity score model and 
    survival machinery; only the weight construction changes.
    """

    def __init__(self, normalize: str = None):
        """
        Parameters
        ----------
        normalize : {'by_group', 'overall', None}, default = None
            If 'by_group': scale weights within each treatment arm to sum to that arm's sample size.
            If 'overall': scale all weights to sum to the total sample size.
            If None: leave raw overlap weights.
        """
        super().__init__()
        self.weight_col = 'overlap_weight'
        self.normalize = normalize
        self.stabilized = False

    def transform(self):
        """
        Calculate overlap weights based on fitted propensity scores.

        Returns
        -------
        pd.DataFrame
        A copy of the original DataFrame with the following columns:
            'propensity_score' : float
                calculated propensity scores 
            'overlap_weight' : float
                calculated overlap weight

        Notes
        -----
        Must call `.fit()` before calling `.transform()`.
        Formula for overlap weight: 
            - For treated patients: weight = 1 - propensity score
            - For control patients: weight = propensity score
        """
        if self.propensity_scores_ is None:
            raise ValueError("Propensity scores were not calculated. Did you forget to run .fit() first?")
        
        df = self.propensity_scores_.copy()
        
        ps = df[self.propensity_score_col].values
        z  = df[self.treatment_col].values
        w = np.where(z == 1, 1.0 - ps, ps)

        # Optional normalization
        if self.normalize is not None:
            if self.normalize == 'by_group':
                m_t = (z == 1)
                m_c = (z == 0)
                for m in (m_t, m_c):
                    s, n = w[m].sum(), m.sum()
                    if s > 0: w[m] = w[m] * (n / s)
            elif self.normalize == 'overall':
                s, n = w.sum(), len(w)
                if s > 0: w = w * (n / s)
            else:
                raise ValueError("normalize must be {'by_group','overall', None}")

        df[self.weight_col] = w
        self.weights_ = df
        return df
    
    def fit_transform(self, *args, **kwargs):
        """
        Fit the propensity score model and compute overlap weights in one step.

        Returns
        -------
        pd.DataFrame
            A DataFrame with 'propensity_score' and 'overlap_weight' columns added.
        """
        self.fit(*args, **kwargs)
        return self.transform()
