"""
Conditional Intervention Analysis (CIA) Framework

A practical approach to causal inference from observational data using
conditional sampling and counterfactual evaluation. This method approximates
interventional effects (do-calculus) by generating biologically plausible
counterfactuals through learned conditional distributions.

Uses consensus of three ML techniques (matching FIBE):
  - Regression: Linear SVR + Gaussian SVR + Random Forest Regressor
  - Classification: Linear SVC + Gaussian SVC + Random Forest Classifier

Reference:
    "We estimated feature-level interventional effects by sampling from the
    conditional distribution P(F_i | F_{-i}), thereby generating biologically
    plausible counterfactuals while preserving the joint feature structure."
"""

from .conditional_intervention import ConditionalInterventionAnalysis
from .utils import generate_synthetic_data, format_results_table
from .outcome_model import (
    get_models,
    train_outcome_models,
    predict_outcome,
    evaluate_outcome_models,
)

__version__ = "2.0.0"

__all__ = [
    "ConditionalInterventionAnalysis",
    "generate_synthetic_data",
    "format_results_table",
    "get_models",
    "train_outcome_models",
    "predict_outcome",
    "evaluate_outcome_models",
]
