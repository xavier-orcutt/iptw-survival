# iptw-survival

`iptw-survival` is a Python package for performing weight-based causal survival analysis. It includes functionality for estimating inverse probability of treatment weights (IPTW) and overlap weights (OW), visualizing propensity score overlap, assessing covariate balance, and computing bootstrapped survival metrics.

## Features

- Calculate IPTW and overlap weights using logistic regression
- Visualize propensity score distributions for treatment groups
- Assess covariate balance using standardized mean differences (SMD), including a Love plot
- Generate bootstrapped Kaplan-Meier survival summaries with point estimates and 95% CIs
- Compute survival metrics with bootstrapped 95% confidence intervals:
    - Probability of survival at fixed timepoints
    - Restricted mean survival time (RMST)
    - Median survival

Standard variance estimates for survival curves can be biased when using non-integer or extreme weights. To address this, `iptw-survival` uses bootstrap resampling to provide robust confidence intervals for all survival metrics.

## Installation 

```python
pip install iptw-survival

```

## Quick Start

```python
from iptw_survival import IPTWSurvivalEstimator

# Instantiate and fit model
estimator = IPTWSurvivalEstimator()
iptw_df = estimator.fit_transform(df,
                                  treatment_col = 'treatment',
                                  cat_var = ['stage', 'ecog'],
                                  cont_var = ['age', 'creatinine'],
                                  binary_var = ['surgery', 'medicaid'],
                                  lr_kwargs = {
                                    'class_weight': 'balanced',
                                    'random_state': 42},
                                  clip_bounds = (0.01, 0.99),
                                  stabilized = True)

# Plot propensity score distribution
estimator.propensity_score_plot()

# Assess covariate balance
smd_df, fig = estimator.standardized_mean_differences(return_fig = True)

# Kaplan-Meier plot with bootstrapped CIs
km_df = estimator.km_confidence_interval(df = iptw_df
                                         event_col='event', 
                                         duration_col = 'duration',
                                         n_bootstrap = 500
                                         random_state = 42)

# Use km_df to plot survival curves 
import matplotlib.pyplot as plt

plt.plot(km_df['time'], km_df['treatment_estimate'], label = 'Treatment')
plt.fill_between(km_df['time'], km_df['treatment_lower_ci'], km_df['treatment_upper_ci'], alpha = 0.1)
plt.xlabel('Time (months)')
plt.ylabel('Survival probability')
plt.legend()
plt.title('IPTW-adjusted Kaplan-Meier Curve')
plt.show()

# Calculate survival metrics
results = estimator.survival_metrics(df = iptw_df
                                     event_col='event', 
                                     duration_col = 'duration',
                                     psurv_time_points = [24, 36],
                                     rmst_time_points = [12],
                                     median_time = True
                                     n_bootstrap = 500
                                     random_state = 42)
```

## Example Tutorial

A full walkthrough using the Flatiron Health advanced urothelial cancer dataset is available in `tutorial/tutorial.ipynb`.

## Requirements

Built and tested in python 3.13

Core dependencies: 
- pandas
- numpy
- scikit-learn
- lifelines
- matplotlib

## Contact

Contributions and feedback are welcome. Contact: xavierorcutt@gmail.com