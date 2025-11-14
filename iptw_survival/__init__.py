"""
IPTWSurvivalEstimator

A Python tool for IPTW-based survival analysis.
"""

__version__ = '0.1.8'

# Make key classes available at package level
from .iptw_survival_estimator import IPTWSurvivalEstimator
from .overlap_survival_estimator import OverlapWeightSurvivalEstimator

__all__ = ["IPTWSurvivalEstimator", "OverlapWeightSurvivalEstimator"]
