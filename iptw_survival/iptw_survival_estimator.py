import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, Tuple, List

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time, median_survival_times
from lifelines.exceptions import StatisticalWarning

import warnings

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class IPTWSurvivalEstimator:
    """
    This class estimates propensity scores using logistic regression, computes inverse
    probability of treatment weights (IPTWs), and applies those weights to
    inverse-weighted Kaplan–Meier and restricted mean survival time (RMST) analyses.

    It is designed for treatment–outcome comparisons in observational data,
    emulating the structure of a randomized trial. 

    The implementation includes:
        - Automated preprocessing pipelines (imputation, scaling, one-hot encoding)
        - Propensity score estimation via logistic regression
        - Optional stabilized weights and score clipping
        - Support for mirrored propensity score plots and Love plots
        - IPTW-adjusted survival analyses with bootstrap confidence intervals

    Methods
    -------
    fit(...)
        Fit the propensity score model and store predicted probabilities.
    transform()
        Compute IPTW weights based on fitted propensity scores.
    fit_transform(...)
        Convenience wrapper for `.fit()` followed by `.transform()`.
    propensity_score_plot(...)
        Plot mirrored histograms of propensity score distributions by treatment arm.
    standardized_mean_differences(...)
        Calculate and optionally plot covariate balance (SMDs) before and after weighting.
    survival_metrics(...)
        Estimate IPTW-adjusted survival probabilities, RMST, and median survival
        with bootstrapped 95% confidence intervals.
    km_confidence_intervals(...)
        Compute full Kaplan–Meier survival curves with bootstrapped confidence bands.

    Notes
    -----
    - For improved stability when overlap is poor, consider using
      `OverlapWeightSurvivalEstimator`.
    """
    
    def __init__(self):
        self.treatment_col = None
        self.cat_var = []
        self.cont_var = []
        self.binary_var = []            
        self.user_binary_var_ = []      
        self.missing_flag_var_ = []   
        self.all_var = []
        self.stabilized = False
        self.lr_kwargs = {}
        self.clip_bounds = None
        self.use_missing_flags = True 

        self.propensity_score_col = 'propensity_score'
        self.propensity_scores_ = None
        self.weight_col = 'iptw'
        self.weights_ = None
        self.smd_results_ = None
        self.survival_metrics_ = None
        self.km_confidence_intervals_ = None
        self.treat_km_ = None
        self.control_km_ = None

    def fit(self, 
            df: pd.DataFrame, 
            treatment_col: str, 
            cat_var: Optional[List[str]] = None, 
            cont_var: Optional[List[str]] = None, 
            binary_var: Optional[List[str]] = None,
            lr_kwargs: Optional[dict] = None,
            clip_bounds: Optional[Union[Tuple[float, float], List[float]]] = None,
            stabilized: bool = False,
            use_missing_flags: bool = True) -> None:
        """
        Fit logistic regression model to calculate propensity scores for receipt of treatment. 

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing treatment assignment and variables of interest for calculating weights.
        treatment_col : str
            Name of the binary treatment column (0 = control, 1 = treated). Must be of integer type. 
        cat_var : list of str, optional
            Categorical variables to be one-hot encoded. Must be of dtype 'category' and contain no missing values.
        cont_var : list of str, optional 
            Continuous variables to be imputed (median) and scaled. Must be numeric (int or float).
        binary_var : list of str, optional 
            Binary variables to be passed through without transformation. These must contain no missing values and should 
            have only two unique values (e.g., 0/1 or True/False).
        lr_kwargs : dict, optional
            Additional keyword arguments passed to sklearn's LogisticRegression. To ensure reproducibility when using 
            bootstrapped survival methods (e.g., .survival_metrics() or .km_confidence_intervals()), consider setting 
            random_state in lr_kwargs.
        clip_bounds : tuple of float or list of float, optional
            If provided, clip propensity scores to this (min, max) range. 
            Common choice is (0.01, 0.99) to reduce the influence of extreme values.
        stabilized : bool, default = False
            If True, enables stabilized weights in the transform step.
        use_missing_flags : bool, default = True
            If True, for every continuous variable in `cont_var` wiht a missing value, a binary missingness flag named `<col>_missing` is 
            generated. 

        Returns
        -------
        None
            Updates internal state with propensity scores. Use .transform() to calculate weights

        Notes
        -----
        This method only estimates propensity scores. Call `.transform()` to compute IPTW.
        At least one of cat_var, cont_var, or binary_var must be provided.
        If user does not specify clipping, safety clipping with 1e-6 is ued to avoid extreme weights.
        """

        # Input validation 
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        
        if treatment_col not in df.columns:
            raise ValueError('treatment_col not found in df.')
        if df[treatment_col].isnull().any():
            raise ValueError('treatment_col has missing values.')
        if not set(df[treatment_col].unique()).issubset({0, 1}):
            raise ValueError('treatment_col must contain only binary values (0 and 1).')
        if not pd.api.types.is_integer_dtype(df[treatment_col]):
            raise ValueError('treatment_col must be of integer type.')
        
        if all(var is None for var in [cat_var, cont_var, binary_var]):
            raise ValueError('at least one of cat_var, cont_var, or binary_var must be provided')
        
        if cat_var is not None:
            # Check that columns in cat_var are present in the df
            missing = [col for col in cat_var if col not in df.columns]
            if missing:
                raise ValueError(f"The following columns in cat_var are missing from the DataFrame: {missing}")
            
            # Check that columns in cat_var are categorical
            non_categorical = [col for col in cat_var if not pd.api.types.is_categorical_dtype(df[col])]
            if non_categorical:
                raise ValueError(f"The following columns in cat_var are not categorical dtype: {non_categorical}")

            # Check that columns in cat_var have no missing values 
            cat_missing = [col for col in cat_var if df[col].isnull().any()]
            if cat_missing:
                raise ValueError(f"The following columns in cat_var have missing values: {cat_missing}")

        if cont_var is not None:
            # Check that columns in cont_var are present in the df
            missing = [col for col in cont_var if col not in df.columns]
            if missing:
                raise ValueError(f"The following columns in cont_var are missing from the DataFrame: {missing}")
            
            # Check that columns in cont_var are numerical type 
            non_numeric = [col for col in cont_var if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                raise ValueError(f"The following columns in cont_var are not numeric: {non_numeric}")

        if binary_var is not None:
            # Check that columns in binary_var are present in the df
            missing = [col for col in binary_var if col not in df.columns]
            if missing:
                raise ValueError(f"The following columns in binary_var are missing from the DataFrame: {missing}")
            
            # Check that columns in binary_var have no missing values
            pt_missing = [col for col in binary_var if df[col].isnull().any()]
            if pt_missing:
                raise ValueError(f"The following columns in binary_var have missing values: {pt_missing}")
            
            # Check that all binary_var are binary (only 2 unique values)
            not_binary = [
                col for col in binary_var
                if df[col].nunique() > 2
            ]
            if not_binary:
                raise ValueError(f"The following columns in binary_var are not binary: {not_binary}")
            
            # Convert True/False to 1/0 for consistency
            for col in binary_var:
                if df[col].dtype == 'bool':
                    df[col] = df[col].astype(int)
        
        if not isinstance(stabilized, bool):
            raise ValueError("stabilized must be a boolean (True or False).")

        if clip_bounds is not None:
            if (not isinstance(clip_bounds, (tuple, list)) or
                len(clip_bounds) != 2):
                raise ValueError("clip_bounds must be a tuple or list of two float values (min, max).")

            lower, upper = clip_bounds

            if not (isinstance(lower, (int, float)) and isinstance(upper, (int, float))):
                raise ValueError("Both values in clip_bounds must be numeric.")

            if not (0 < lower < upper < 1):
                raise ValueError("clip_bounds values must be between 0 and 1 and satisfy 0 < lower < upper.")

        # Save config
        self.cat_var = cat_var or []
        self.cont_var = cont_var or []
        self.user_binary_var_ = binary_var or []
        self.use_missing_flags = bool(use_missing_flags) 
        
        df = df.copy()

        self.missing_flag_var_ = []
        if self.use_missing_flags: 
            for col in self.cont_var:
                if df[col].isna().any(): 
                    flag = f"{col}_missing"
                    df[flag] = df[col].isna().astype(int)
                    self.missing_flag_var_.append(flag)

        # Combined binary list = user binaries + generated flags
        self.binary_var = self.user_binary_var_ + self.missing_flag_var_

        # all_var now includes flags so SMDs will consider them
        self.all_var = self.cat_var + self.cont_var + self.binary_var

        self.treatment_col = treatment_col
        self.stabilized = stabilized
        self.lr_kwargs = lr_kwargs or {}
        self.clip_bounds = clip_bounds

        # Build pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers = [
                ('num', numeric_pipeline, self.cont_var),
                ('cat', categorical_pipeline, self.cat_var),
                ('pass', 'passthrough', self.binary_var)],
                remainder = 'drop'
        )

        # Fit and transform
        X_preprocessed = preprocessor.fit_transform(df)

        # Calculating propensity scores using logistic regression 
        if lr_kwargs is None:
            lr_kwargs = {}
        lr_model = LogisticRegression(**lr_kwargs)
        lr_model.fit(X_preprocessed, df[treatment_col])
        propensity_score = lr_model.predict_proba(X_preprocessed)[:, 1] # Select second column for probability of receiving treatment 
        
        # Apply clipping 
        if self.clip_bounds is not None:
            lower, upper = self.clip_bounds
            propensity_score = np.clip(propensity_score, lower, upper)
        else: 
            eps = 1e-6 # Small buffer to avoid division by zero when calculating IPTW in .transform()
            propensity_score = np.clip(propensity_score, eps, 1 - eps)
        
        df[self.propensity_score_col] = propensity_score
        self.propensity_scores_ = df
    
    def transform(self) -> pd.DataFrame:
        """
        Calculate IPTW based on fitted propensity scores.

        Returns
        -------
        pd.DataFrame
        A copy of the original DataFrame with the following columns:
            'propensity_score' : float
                calculated propensity scores 
            'iptw' : float
                calculated IPTW 

        Notes
        -----
        Must call `.fit()` before calling `.transform()`.
        Formula for IPTW: 
            - For treated patients: weight = 1 / propensity score
            - For control patients: weight = 1 / (1 - propensity score)
            - If stabilized = True, weights are multiplied by the marginal probability of treatment or control.
        """
        if self.propensity_scores_ is None:
            raise ValueError("Propensity scores were not calculated. Did you forget to run .fit() first?")

        df = self.propensity_scores_.copy()

        if self.stabilized:
            p_treated = df[self.treatment_col].mean()
            df[self.weight_col] = np.where(
                df[self.treatment_col] == 1,
                p_treated / df[self.propensity_score_col],
                (1 - p_treated) / (1 - df[self.propensity_score_col])
            )

        else:
            df[self.weight_col] = np.where(
                df[self.treatment_col] == 1,
                1 / df[self.propensity_score_col],
                1 / (1 - df[self.propensity_score_col])
            )

        self.weights_ = df
        return df
    
    def fit_transform(self, 
                      *args, 
                      **kwargs) -> pd.DataFrame:
        """
        Fit the propensity score model and compute IPTW weights in one step.

        Returns
        -------
        pd.DataFrame
            A DataFrame with 'propensity_score' and 'iptw' columns added.
        """
        self.fit(*args, **kwargs)
        return self.transform()
    
    def propensity_score_plot(self,
                              bins: int = 20): 
        """
        Generates a propensity score overlap plot for treatment vs control. 

        Parameters
        ----------
        bins : int, default = 20
            Number of bins for the histogram

        This method uses internal attributes set during the .fit() or fit_transform() calls:
            - self.propensity_scores_ : the DataFrame with variables, treatment, and propensity scores 

        Returns
        -------
        matplotlib.figure.Figure 
            A histogram plot showing raw propensity scores by treatment group. 

        Notes
        -----
        This method requires that .fit() or .fit_transform() has been run prior to use.
        """
        # Input validation 
        if self.propensity_scores_ is None:
            raise ValueError("propensity_scores_ is None. Please call `.fit()` or `.fit_transform()` first.")
        
        if not isinstance(bins, int) or bins <= 0:
            raise ValueError("bins must be a positive integer.")
        
        df = self.propensity_scores_
        treatment_col = self.treatment_col

        fig, ax = plt.subplots(figsize=(8, 5))

        # Histogram for treated patients
        ax.hist(df[df[treatment_col] == 1][self.propensity_score_col], 
                 bins = bins, 
                 alpha = 0.3, 
                 label = 'Treatment', 
                 color = 'blue',
                 edgecolor='black')
        
        # Histogram for untreated patients (horizontal, with negative counts to "flip" it)
        ax.hist(df[df[treatment_col] == 0][self.propensity_score_col], 
                 bins = bins, 
                 weights= -np.ones_like(df[df[treatment_col] == 0][self.propensity_score_col]),
                 alpha = 0.3, 
                 label = 'Control', 
                 color = 'green', 
                 edgecolor = 'black')

        # Adding titles and labels
        ax.set_title('Propensity Score Distribution by Treatment Group', pad = 25, size = 18, weight = 'bold')
        ax.set_xlabel('Propensity Score', labelpad = 15, size = 12, weight = 'bold')
        ax.set_ylabel('Count', labelpad = 15, size = 12, weight = 'bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        yticks = ax.get_yticks()
        ax.set_yticks(yticks)  # explicitly set the tick locations
        ax.set_yticklabels([f'{abs(int(tick))}' for tick in yticks])

        ax.legend(prop = {'size': 10})

        fig.tight_layout()
        plt.close(fig)
        return fig
        
    def standardized_mean_differences(self,
                                      return_fig: bool = False):
        """
        Compute and plot standardized mean differences (SMDs) before and after weighting for all variables used in the IPTW model.

        Parameters
        ----------
        return_fig : bool, default = False
            If True, returns a Love plot of the SMDs, which is a dot plot with variable names on the y-axis and standardized mean 
            differences on the x-axis.

        This method uses internal attributes set during the .fit() and .transform() or fit_transform() calls.

        Returns
        -------
        pd.DataFrame
            Contains:
                - variable : variable name
                - smd_unweighted : SMD with no weights
                - smd_weighted : SMD using IPTW 

        matplotlib.figure.Figure (optional)
            Returned only if return_fig = True.
            A plot showing unweighted and weighted SMDs for all included variables.

        Notes
        -----
        SMDs quantify the difference in variable distributions between treated and control groups.

        For continuous variables:
            SMD = (mean_treated - mean_control) / sqrt[(sd_treated² + sd_control²)/2]

        For categorical and binary variables:
            SMD = (p_treated - p_control) / sqrt[(p_treated * (1 - p_treated) + p_control * (1 - p_control)) / 2]

        Where:
            - mean_treated / mean_control = means of the variable in each group
            - sd_treated / sd_control = standard deviations
            - p_treated / p_control = proportion of group members in a given category or with value == 1
        
        Processing of variables: 
            - Median is imputed for missing continuous variables. 
            - Categorical variables are one-hot-encoded.
        """
        # Input validation ensures .fit() or .fit_transform() has been run
        if not self.all_var:
            raise ValueError("Please run .fit() or .fit_transform() before calling standardized_mean_differences().")
        if self.weights_ is None:
            raise ValueError("Please run .fit() or .fit_transform() before calling standardized_mean_differences().")

        # Input validation for return_fig
        if not isinstance(return_fig, bool):
            raise ValueError("return_fig must be a boolean (True or False).")

        smd_df = self.weights_[self.all_var + [self.treatment_col] + [self.weight_col]].copy()

        # Calculate SMD for continuous variables 
        smd_cont = []
        for var in self.cont_var:
            # Imput median for missing 
            smd_df[var] = pd.to_numeric(smd_df[var], errors='coerce').astype('Float64')
            smd_df[var] = smd_df[var].fillna(smd_df[var].median())

            treat_mask = smd_df[self.treatment_col] == 1
            control_mask = smd_df[self.treatment_col] == 0

            # Unweighted
            m1 = smd_df.loc[treat_mask, var].mean()
            m0 = smd_df.loc[control_mask, var].mean()
            s1 = smd_df.loc[treat_mask, var].std()
            s0 = smd_df.loc[control_mask, var].std()

            pooled_sd = np.sqrt(0.5 * (s1**2 + s0**2))
            smd_unweighted = (m1 - m0) / pooled_sd if pooled_sd > 0 else 0.0

            # Weighted
            m1_w = np.average(smd_df.loc[treat_mask, var], weights=smd_df.loc[treat_mask, self.weight_col])
            m0_w = np.average(smd_df.loc[control_mask, var], weights=smd_df.loc[control_mask, self.weight_col])
            s1_w = np.sqrt(np.average((smd_df.loc[treat_mask, var] - m1_w) ** 2, weights=smd_df.loc[treat_mask, self.weight_col]))
            s0_w = np.sqrt(np.average((smd_df.loc[control_mask, var] - m0_w) ** 2, weights=smd_df.loc[control_mask, self.weight_col]))

            pooled_sd_w = np.sqrt(0.5 * (s1_w**2 + s0_w**2))
            smd_weighted = (m1_w - m0_w) / pooled_sd_w if pooled_sd_w > 0 else 0.0

            smd_cont.append({
                'variable': var,
                'smd_unweighted': smd_unweighted,
                'smd_weighted': smd_weighted
            })

        # Calculate SMD for categorical variables 
        smd_cat = []
        for var in self.cat_var: 
            # One-hot encode categories
            categories = smd_df[var].dropna().unique()
            for cat in categories:
                var_cat = f"{var}__{cat}"
                treat_mask = (smd_df[self.treatment_col] == 1)
                control_mask = (smd_df[self.treatment_col] == 0)
                smd_df[var_cat] = (smd_df[var] == cat).astype(int)

                # Unweighted
                p1 = smd_df.loc[treat_mask, var_cat].mean()
                p0 = smd_df.loc[control_mask, var_cat].mean()
                denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
                smd_unweighted = (p1 - p0) / denom if denom > 0 else 0.0

                # Weighted
                p1_w = np.average(smd_df.loc[treat_mask, var_cat], weights=smd_df.loc[treat_mask, self.weight_col])
                p0_w = np.average(smd_df.loc[control_mask, var_cat], weights=smd_df.loc[control_mask, self.weight_col])
                denom_w = np.sqrt((p1_w * (1 - p1_w) + p0_w * (1 - p0_w)) / 2)
                smd_weighted = (p1_w - p0_w) / denom_w if denom_w > 0 else 0.0

                smd_cat.append({
                    'variable': var_cat,
                    'smd_unweighted': smd_unweighted,
                    'smd_weighted': smd_weighted
                })
        
        # Calculate SMD for binary variables 
        smd_bin = []
        for var in self.binary_var:
            treat_mask = smd_df[self.treatment_col] == 1
            control_mask = smd_df[self.treatment_col] == 0

            # Unweighted
            p1 = smd_df.loc[treat_mask, var].mean()
            p0 = smd_df.loc[control_mask, var].mean()
            denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
            smd_unweighted = (p1 - p0) / denom if denom > 0 else 0.0

            # Weighted
            p1_w = np.average(smd_df.loc[treat_mask, var], weights=smd_df.loc[treat_mask, self.weight_col])
            p0_w = np.average(smd_df.loc[control_mask, var], weights=smd_df.loc[control_mask, self.weight_col])
            denom_w = np.sqrt((p1_w * (1 - p1_w) + p0_w * (1 - p0_w)) / 2)
            smd_weighted = (p1_w - p0_w) / denom_w if denom_w > 0 else 0.0

            smd_bin.append({
                'variable': var,
                'smd_unweighted': smd_unweighted,
                'smd_weighted': smd_weighted
            })

        smd_results_df = pd.DataFrame(smd_cont + smd_cat + smd_bin)
        smd_results_df['smd_unweighted'] = smd_results_df['smd_unweighted'].abs()
        smd_results_df['smd_weighted'] = smd_results_df['smd_weighted'].abs()
        smd_results_df = smd_results_df.sort_values(by = 'smd_unweighted', ascending = True).reset_index(drop = True)

        self.smd_results_ = smd_results_df
        
        if return_fig:
            fig, ax = plt.subplots(figsize=(8, 0.4 * len(smd_results_df) + 2))
            
            # Plot points
            ax.scatter(smd_results_df['smd_unweighted'], smd_results_df['variable'], label = 'Unweighted', color = 'red')
            ax.scatter(smd_results_df['smd_weighted'], smd_results_df['variable'], label = 'Weighted', color = 'skyblue')

            # Reference lines
            ax.axvline(x = 0, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.5) 
            ax.axvline(x = 0.1, color = 'black', linestyle = '--', linewidth = 2, alpha = 0.5) 

            # Axis labels and limits
            ax.set_xlabel('Absolute Standardized Mean Difference', labelpad = 15, size = 12, weight = 'bold')
            ax.set_xlim(-0.02)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Title legend
            ax.set_title('Standardized Mean Difference Plot: Covariate Balance', pad = 20, size = 18, weight = 'bold')
            ax.legend(prop = {'size': 10})

            fig.tight_layout()
            plt.close(fig)
            return smd_results_df, fig
        
        else:
            return smd_results_df
        
    def survival_metrics(self,
                         df: pd.DataFrame,
                         duration_col: str,
                         event_col: str,
                         weight_col: str = 'iptw', 
                         psurv_time_points: Optional[List[float]] = None, 
                         rmst_time_points: Optional[List[float]] = None,
                         median_time: Optional[bool] = False,
                         n_bootstrap: int = 1000,
                         random_state: Optional[int] = None) -> dict:

        """
        Estimate survival metrics at discrete time points for treatment group, control group, 
        and their difference using IPTW-adjusted Kaplan-Meier analysis with bootstrapped 95% 
        confidence intervals.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing survival data, treatment assignment, and all variables used 
            in the IPTW model. This should match the structure of the dataframe used in the original 
            fit()/fit_transform() call.
        duration_col : str
            Name of the column containing the survival duration (e.g., time to event or censoring).
        event_col : str
            Name of the column indicating event occurrence (1 = event, 0 = censored). Must be of integer type. 
        weight_col : str 
            Name of the column in `df` containing the IPTW weights. Defaults to the column name 
            used in `.fit_transform()` ('iptw'). 
        psurv_time_points : list of float, optional
            Specific time points (in same units as duration_col) at which to estimate survival probability.
            If None, no survival probabilities will be calculated.
        rmst_time_points : list of float, optional
            Specific time horizons (in same units as duration_col) for restricted mean survival time (RMST) estimation.
            If None, RMST will not be calculated.
        median_time : bool, optional
            Whether to calculate median survival time (e.g., median overall survival) with bootstrapped confidence intervals.
        n_bootstrap : int, default=1000
            Number of bootstrap iterations used to estimate confidence intervals.
        random_state : int, optional
            Seed for reproducibility of bootstrap resampling. If you want .survival_metrics() and .km_confidence_intervals() to 
            use identical resamples and produce aligned results, pass the same integer to both methods.  
            To ensure complete reproducibility, also consider passing random_state to the logistic regression model via lr_kwargs 
            in .fit() or fit_transform() to fix propensity score estimation.

        Returns
        -------
        results : dict
            Dictionary containing survival metrics and 95% confidence intervals for treatment and control groups,
            as well as the difference between them. Example format:
            {
                'treatment': {
                    'survival_prob': {6: (est, lci, uci), 12: (est, lci, uci)},
                    'rmst': {24: (est, lci, uci)},
                    'median': (est, lower_ci, upper_ci),
                },
                'control': {
                    ...
                },
                'difference': {
                    ...
                }
            }

        Notes
        -----
        This method requires that .fit() or .fit_transform() has been run prior to use. During each bootstrap iteration, weights 
        are recalculated using the variables provided in the original .fit() or .fit_transform() call. Recalculated weights respect 
        the `stabilized` and `clip_bounds` parameters from the initial .fit() or .fit_transform() call.
        """
        # Input validation for df 
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if self.weights_ is None:
            raise ValueError("No model state found. Please run .fit() or .fit_transform() before calling survival_metrics().")
        if df.shape[0] != self.weights_.shape[0]:
            raise ValueError("df appears to be a different size than the one submitted in .fit() or .fit_transform().")
        missing_vars = [col for col in self.all_var if col not in df.columns]
        if missing_vars:
            raise ValueError(f"The following variables used in the IPTW model are missing from df: {missing_vars}.")

        # Input validation for weight_col and duration_col
        for col in [weight_col, duration_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}'contains missing values.")
            if (df[col] < 0).any():
                raise ValueError(f"Column '{col}' must contain non-negative values only.")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric (int or float).")
            
        # Input validation for treatment_col and event_col
        for col in [self.treatment_col, event_col]: 
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}' has missing values.")
            if not set(df[col].unique()).issubset({0, 1}):
                raise ValueError(f"Column '{col}' must contain only binary values (0 and 1).")
            if not pd.api.types.is_integer_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be of integer type.")

        # Ensure the calculation of at least one survival metric is requested
        if (
            psurv_time_points is None 
            and rmst_time_points is None 
            and not median_time
        ):
            raise ValueError("At least one of psurv_time_points, rmst_time_points, or median_time must be specified.")

        # Input validation for psurv_time_points and rmst_time_points
        for name, arg in [("psurv_time_points", psurv_time_points), 
                          ("rmst_time_points", rmst_time_points)]:
            if arg is not None:
                if not isinstance(arg, list):
                    raise ValueError(f"{name} must be a list of floats or ints.")
                if not all(isinstance(t, (float, int)) for t in arg):
                    raise ValueError(f"All values in {name} must be numeric (float or int).")
                if any(t <= 0 for t in arg):
                    raise ValueError(f"All values in {name} must be positive.")

        # Input validation for median_time
        if not isinstance(median_time, bool):
            raise ValueError("median_time must be a boolean (True or False).")

        # Input validation for n_bootstrap
        if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be a positive integer.")

        # Input validation for random_state
        if random_state is not None:
            if not isinstance(random_state, int):
                raise ValueError("random_state must be an integer or None.")
        
        # Estimate survival times
        # Kaplan-Meier models 
        treat_km = KaplanMeierFitter()
        control_km = KaplanMeierFitter()

        treat_mask = df[self.treatment_col] == 1
        control_mask = df[self.treatment_col] == 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category = StatisticalWarning)
            treat_km.fit(
                durations = df.loc[treat_mask, duration_col],
                event_observed = df.loc[treat_mask, event_col],
                weights = df.loc[treat_mask, weight_col]
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category = StatisticalWarning)
            control_km.fit(
                durations = df.loc[control_mask, duration_col],
                event_observed = df.loc[control_mask, event_col],
                weights = df.loc[control_mask, weight_col]
            )
        
        # Initialize empty dictionaries 
        estimate = {
            'treatment': {'survival_prob': {}, 'rmst': {}, 'median': []},
            'control': {'survival_prob': {}, 'rmst': {}, 'median': []},
            'difference': {'survival_prob': {}, 'rmst': {}, 'median': []}
        }

        # Probability of survival at selected time points
        if psurv_time_points is not None:
            for t in psurv_time_points:
                treat_surv = treat_km.predict(t)
                control_surv = control_km.predict(t)
                estimate['treatment']['survival_prob'][t] = treat_surv
                estimate['control']['survival_prob'][t] = control_surv
                estimate['difference']['survival_prob'][t] = treat_surv - control_surv

        # RMST calculation at selected time points
        if rmst_time_points is not None: 
            for t in rmst_time_points:
                treat_rmst = restricted_mean_survival_time(treat_km, t=t)
                control_rmst = restricted_mean_survival_time(control_km, t=t)
                estimate['treatment']['rmst'][t] = treat_rmst
                estimate['control']['rmst'][t] = control_rmst
                estimate['difference']['rmst'][t] = treat_rmst - control_rmst

        # Median survival time calculations
        if median_time: 
            treat_med = treat_km.median_survival_time_
            control_med = control_km.median_survival_time_
            estimate['treatment']['median'] = treat_med
            estimate['control']['median'] = control_med
            estimate['difference']['median'] = treat_med - control_med

        # Calculate bootstrapped 95% CIs
        # Initialize empty disctionaries for bootstrapped results 
        boot_results = {
            'treatment': {'survival_prob': {}, 'rmst': {}, 'median': []},
            'control': {'survival_prob': {}, 'rmst': {}, 'median': []},
            'difference': {'survival_prob': {}, 'rmst': {}, 'median': []}
        }

        # Loop over n_bootstrap
        rng = np.random.default_rng(seed = random_state)
        for i in range(n_bootstrap):
            
            # Sample with replacement using random indices
            indices = rng.choice(df.index, size = len(df), replace = True)
            df_boot = df.loc[indices].reset_index(drop = True)

            # Recalculate weights using saved model spec
            df_boot_weighted = self.fit_transform(
                df_boot,
                treatment_col = self.treatment_col,
                cat_var = self.cat_var,
                cont_var = self.cont_var,
                binary_var = self.user_binary_var_,
                stabilized = self.stabilized,
                lr_kwargs = self.lr_kwargs,
                clip_bounds = self.clip_bounds,
                use_missing_flags = self.use_missing_flags
            )

            # Kaplan-Meier models 
            treat_km = KaplanMeierFitter()
            control_km = KaplanMeierFitter()

            treat_mask = df_boot_weighted[self.treatment_col] == 1
            control_mask = df_boot_weighted[self.treatment_col] == 0

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category = StatisticalWarning)
                treat_km.fit(
                    durations = df_boot_weighted.loc[treat_mask, duration_col],
                    event_observed = df_boot_weighted.loc[treat_mask, event_col],
                    weights = df_boot_weighted.loc[treat_mask, weight_col]
                )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category = StatisticalWarning)
                control_km.fit(
                    durations = df_boot_weighted.loc[control_mask, duration_col],
                    event_observed = df_boot_weighted.loc[control_mask, event_col],
                    weights = df_boot_weighted.loc[control_mask, weight_col]
                )

            # Calculate survival metrics
            if psurv_time_points is not None:
                for t in psurv_time_points:
                    boot_results['treatment']['survival_prob'].setdefault(t, []).append(treat_km.predict(t))
                    boot_results['control']['survival_prob'].setdefault(t, []).append(control_km.predict(t))
                    boot_results['difference']['survival_prob'].setdefault(t, []).append(
                        treat_km.predict(t) - control_km.predict(t)
                    )

            if rmst_time_points is not None:
                for t in rmst_time_points:
                    treat_rmst = restricted_mean_survival_time(treat_km, t=t)
                    control_rmst = restricted_mean_survival_time(control_km, t=t)
                    boot_results['treatment']['rmst'].setdefault(t, []).append(treat_rmst)
                    boot_results['control']['rmst'].setdefault(t, []).append(control_rmst)
                    boot_results['difference']['rmst'].setdefault(t, []).append(treat_rmst - control_rmst)

            if median_time:
                treat_med = treat_km.median_survival_time_
                control_med = control_km.median_survival_time_
                boot_results['treatment']['median'].append(treat_med)
                boot_results['control']['median'].append(control_med)
                boot_results['difference']['median'].append(treat_med - control_med)

        # Get estimate plus lower 2.5%, and upper 97.5% of boot_results
        final_results = {
            'treatment': {'survival_prob': {}, 'rmst': {}, 'median': None},
            'control':   {'survival_prob': {}, 'rmst': {}, 'median': None},
            'difference':{'survival_prob': {}, 'rmst': {}, 'median': None}
        }

        for group in ['treatment', 'control', 'difference']:
            if psurv_time_points is not None:
                for t in psurv_time_points:
                    est = estimate[group]['survival_prob'][t]
                    lci = np.percentile(boot_results[group]['survival_prob'][t], 2.5)
                    uci = np.percentile(boot_results[group]['survival_prob'][t], 97.5)
                    final_results[group]['survival_prob'][t] = (float(est), float(lci), float(uci))

            if rmst_time_points is not None:
                for t in rmst_time_points:
                    est = estimate[group]['rmst'][t]
                    lci = np.percentile(boot_results[group]['rmst'][t], 2.5)
                    uci = np.percentile(boot_results[group]['rmst'][t], 97.5)
                    final_results[group]['rmst'][t] = (float(est), float(lci), float(uci))

            if median_time:
                est = estimate[group]['median']
                lci = np.percentile(boot_results[group]['median'], 2.5)
                uci = np.percentile(boot_results[group]['median'], 97.5)
                final_results[group]['median'] = (float(est), float(lci), float(uci))
        
        self.survival_metrics_ = final_results
        return final_results

    def km_confidence_intervals(self,
                                df: pd.DataFrame,
                                duration_col: str,
                                event_col: str, 
                                weight_col: str = 'iptw',
                                n_bootstrap: int = 1000,
                                random_state: Optional[int] = None) -> pd.DataFrame:

        """
        Estimate IPTW-adjusted Kaplan-Meier survival curves with 95% confidence intervals at each 
        time point using bootstrap resampling.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing survival data, treatment assignment, and all variables used 
            in the IPTW model. This should match the structure of the dataframe used in the original 
            fit()/fit_transform() call.
        duration_col : str
            Name of the column containing the survival duration (e.g., time to event or censoring).
        event_col : str
            Name of the column indicating event occurrence (1 = event, 0 = censored). Must be of integer type. 
        weight_col : str 
            Name of the column in `df` containing the IPTW weights. Defaults to the column name 
            used in `.fit_transform()` ('iptw'). 
        n_bootstrap : int, default=1000
            Number of bootstrap iterations used to estimate confidence intervals.
        random_state : int, optional
            Seed for reproducibility of bootstrap resampling. If you want .survival_metrics() and .km_confidence_intervals() to 
            use identical resamples and produce aligned results, pass the same integer to both methods.  
            To ensure complete reproducibility, also consider passing random_state to the logistic regression model via lr_kwargs 
            in .fit() or fit_transform() to fix propensity score estimation. 

        Returns
        -------
        pd.DataFrame
            Contains:
                - 'time': common time grid
                - 'treatment_estimate': point estimate of treatment group
                - 'treatment_lower_ci': 2.5 percentile for treatment group
                - 'treatment_upper_ci': 97.5 percentile for treatment group
                - 'control_estimate': point estimate of control group
                - 'control_lower_ci': 2.5 percentile for control group
                - 'control_upper_ci': 97.5 percentile for control group 

        Notes
        -----
        This method requires that .fit() or .fit_transform() has been run prior to use. During each bootstrap iteration, weights 
        are recalculated using the variables provided in the original .fit() or .fit_transform() call. Recalculated weights respect 
        the `stabilized` and `clip_bounds` parameters from the initial .fit() or .fit_transform() call.

        This method also stores the fitted treatment and control Kaplan-Meier models as `self.treat_km_` and `self.control_km_`, 
        which can be accessed for plotting purposes (e.g., with lifelines' add_at_risk_counts() function).
        """
        # Input validation for df 
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if self.weights_ is None:
            raise ValueError("No model state found. Please run .fit() or .fit_transform() before calling survival_metrics().")
        if df.shape[0] != self.weights_.shape[0]:
            raise ValueError("df appears to be a different size than the one submitted in .fit() or .fit_transform().")
        missing_vars = [col for col in self.all_var if col not in df.columns]
        if missing_vars:
            raise ValueError(f"The following variables used in the IPTW model are missing from df: {missing_vars}.")

        # Input validation for weight_col and duration_col
        for col in [weight_col, duration_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}'contains missing values.")
            if (df[col] < 0).any():
                raise ValueError(f"Column '{col}' must contain non-negative values only.")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric (int or float).")
            
        # Input validation for treatment_col and event_col
        for col in [self.treatment_col, event_col]: 
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}' has missing values.")
            if not set(df[col].unique()).issubset({0, 1}):
                raise ValueError(f"Column '{col}' must contain only binary values (0 and 1).")
            if not pd.api.types.is_integer_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be of integer type.")

        # Input validation for n_bootstrap
        if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be a positive integer.")

        # Input validation for random_state
        if random_state is not None:
            if not isinstance(random_state, int):
                raise ValueError("random_state must be an integer or None.")

        # Estimate survival times
        # Kaplan-Meier models 
        treat_km = KaplanMeierFitter()
        control_km = KaplanMeierFitter()

        treat_mask = df[self.treatment_col] == 1
        control_mask = df[self.treatment_col] == 0

        # Initialize time 
        time = np.unique(df[duration_col])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category = StatisticalWarning)
            treat_km.fit(
                durations = df.loc[treat_mask, duration_col],
                event_observed = df.loc[treat_mask, event_col],
                timeline = time,
                weights = df.loc[treat_mask, weight_col]
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category = StatisticalWarning)
            control_km.fit(
                durations = df.loc[control_mask, duration_col],
                event_observed = df.loc[control_mask, event_col],
                timeline = time,
                weights = df.loc[control_mask, weight_col]
            )

        # Store the original models
        self.treat_km_ = treat_km
        self.control_km_ = control_km
        
        # Extract survival probabilities directly
        treatment_est = treat_km.survival_function_['KM_estimate'].values
        control_est = control_km.survival_function_['KM_estimate'].values

        # Build DataFrame
        estimate_df = pd.DataFrame({
            'time': time,
            'treatment_estimate': treatment_est,
            'control_estimate': control_est
        })

        # Calculate bootstrapped 95% CIs
        # Arrays to store survival probabilities for each bootstrap sample
        treatment_boot = np.zeros((n_bootstrap, len(time)))
        control_boot = np.zeros((n_bootstrap, len(time)))

        # Loop over n_bootstrap
        rng = np.random.default_rng(seed = random_state)
        for i in range(n_bootstrap):
            
            # Sample with replacement using random indices
            indices = rng.choice(df.index, size = len(df), replace = True)
            df_boot = df.loc[indices].reset_index(drop = True)

            # Recalculate weights using saved model spec
            df_boot_weighted = self.fit_transform(
                df_boot,
                treatment_col = self.treatment_col,
                cat_var = self.cat_var,
                cont_var = self.cont_var,
                binary_var = self.user_binary_var_,
                stabilized = self.stabilized,
                lr_kwargs = self.lr_kwargs,
                clip_bounds = self.clip_bounds, 
                use_missing_flags = self.use_missing_flags
            )

            # Kaplan-Meier models 
            treat_km = KaplanMeierFitter()
            control_km = KaplanMeierFitter()

            treat_mask = df_boot_weighted[self.treatment_col] == 1
            control_mask = df_boot_weighted[self.treatment_col] == 0

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category = StatisticalWarning)
                treat_km.fit(
                    durations = df_boot_weighted.loc[treat_mask, duration_col],
                    event_observed = df_boot_weighted.loc[treat_mask, event_col],
                    timeline = time,
                    weights = df_boot_weighted.loc[treat_mask, weight_col]
                )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category = StatisticalWarning)
                control_km.fit(
                    durations = df_boot_weighted.loc[control_mask, duration_col],
                    event_observed = df_boot_weighted.loc[control_mask, event_col],
                    timeline = time,
                    weights = df_boot_weighted.loc[control_mask, weight_col]
                )
            
            treatment_boot[i, :] = treat_km.survival_function_['KM_estimate'].values
            control_boot[i, :] = control_km.survival_function_['KM_estimate'].values

        boot_df = pd.DataFrame({
            'time': time,
            'treatment_lower_ci': np.percentile(treatment_boot, 2.5, axis = 0),
            'treatment_upper_ci': np.percentile(treatment_boot, 97.5, axis = 0),
            'control_lower_ci': np.percentile(control_boot, 2.5, axis = 0),
            'control_upper_ci': np.percentile(control_boot, 97.5, axis = 0)
        })

        final_df = pd.merge(estimate_df, boot_df, on = 'time')
        self.km_confidence_intervals_ = final_df
        return final_df





                    





                


            