"""
Conditional Intervention Analysis (CIA)

Implements the do-calculus approximation via conditional sampling.
For each feature, estimates the interventional effect on the outcome by:
1. Learning P(F_i | F_{-i}) via regression
2. Sampling plausible counterfactual values of F_i
3. Evaluating the outcome model under counterfactual inputs
4. Aggregating causal effect estimates across subjects and CV folds

Uses the same consensus of three ML techniques as FIBE:
  - Regression: Linear SVR + Gaussian SVR + Random Forest (averaged)
  - Classification: Linear SVC + Gaussian SVC + Random Forest (majority vote)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from typing import Optional

from .outcome_model import get_models, train_outcome_models, predict_outcome
from .utils import compute_confidence_interval


class ConditionalInterventionAnalysis:
    """
    Estimates feature-level interventional (causal) effects on an outcome
    using conditional sampling to generate biologically plausible counterfactuals.

    Uses consensus of three ML techniques (matching FIBE) for both the
    conditional distribution model and the outcome model.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'. Determines models and metrics.
    n_samples : int
        Number of counterfactual samples per subject per feature (K).
    n_folds : int
        Number of cross-validation folds for stability estimation.
    model_name : str
        Model selection for outcome prediction.
        For regression: 'linearSVR', 'gaussianSVR', 'RegressionForest', or 'consensus'.
        For classification: 'linearSVC', 'gaussianSVC', 'RandomForest', or 'consensus'.
        Default is 'consensus' (uses all three models).
    conditional_model : str
        Model for learning P(F_i | F_{-i}). Always regression regardless of task_type.
        Options: 'linearSVR', 'gaussianSVR', 'RegressionForest', or 'consensus'.
        Default is 'consensus'.
    confidence_level : float
        Confidence level for effect intervals (default 0.95).
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        task_type: str = "regression",
        n_samples: int = 50,
        n_folds: int = 5,
        model_name: str = "consensus",
        conditional_model: str = "consensus",
        confidence_level: float = 0.95,
        random_state: Optional[int] = 42,
    ):
        if task_type not in ("regression", "classification"):
            raise ValueError(
                f"task_type must be 'regression' or 'classification', got '{task_type}'."
            )
        self.task_type = task_type
        self.n_samples = n_samples
        self.n_folds = n_folds
        self.model_name = model_name
        self.conditional_model = conditional_model
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.results_ = None
        self.feature_names_ = None

    def _learn_conditional(self, X_train, feature_idx):
        """
        Learn P(F_i | F_{-i}) from training data using consensus of 3 regressors.

        The conditional distribution is always a regression problem (predicting
        the continuous/discrete feature value from others), regardless of
        whether the main task is classification or regression.

        Returns the fitted models dict and the residual standard deviation.
        """
        mask = np.ones(X_train.shape[1], dtype=bool)
        mask[feature_idx] = False

        X_other = X_train[:, mask]
        y_target = X_train[:, feature_idx]

        cond_models = get_models("regression", self.conditional_model, self.random_state)
        fitted_cond = {}
        all_residuals = []

        for name, model in cond_models.items():
            model.fit(X_other, y_target)
            fitted_cond[name] = model
            preds = model.predict(X_other)
            all_residuals.append(y_target - preds)

        avg_residuals = np.mean(all_residuals, axis=0)
        sigma = np.std(avg_residuals)

        return fitted_cond, sigma

    def _predict_conditional(self, fitted_cond_models, X_other_single):
        """Consensus prediction for conditional mean (always regression)."""
        preds = []
        for name, model in fitted_cond_models.items():
            preds.append(model.predict(X_other_single.reshape(1, -1))[0])
        return np.mean(preds)

    def _sample_counterfactuals(self, fitted_cond_models, sigma, X_subject, feature_idx):
        """
        Sample K plausible counterfactual values for feature_idx.

        Returns array of shape (n_samples,).
        """
        mask = np.ones(len(X_subject), dtype=bool)
        mask[feature_idx] = False

        X_other = X_subject[mask]
        mu = self._predict_conditional(fitted_cond_models, X_other)

        rng = np.random.default_rng(self.random_state)
        counterfactual_values = rng.normal(loc=mu, scale=sigma, size=self.n_samples)

        return counterfactual_values

    def _compute_feature_effect_regression(self, X_test, outcome_models, cond_models, sigma, feature_idx):
        """
        Compute interventional effect for regression task.

        Measures normalized delta: (ΔO / ΔF) for each counterfactual.
        """
        all_deltas = []

        for i in range(X_test.shape[0]):
            X_subject = X_test[i].copy()
            original_value = X_subject[feature_idx]

            O_orig, _ = predict_outcome(outcome_models, X_subject.reshape(1, -1), "regression")
            O_orig = O_orig[0]

            cf_values = self._sample_counterfactuals(
                cond_models, sigma, X_subject, feature_idx
            )

            for cf_val in cf_values:
                X_cf = X_subject.copy()
                X_cf[feature_idx] = cf_val

                O_cf, _ = predict_outcome(outcome_models, X_cf.reshape(1, -1), "regression")
                O_cf = O_cf[0]

                delta_O = O_cf - O_orig
                delta_F = cf_val - original_value

                if abs(delta_F) > 1e-10:
                    all_deltas.append(delta_O / delta_F)

        return np.array(all_deltas)

    def _compute_feature_effect_classification(self, X_test, outcome_models, cond_models, sigma, feature_idx):
        """
        Compute interventional effect for classification task.

        Uses change in predicted probability (ΔP / ΔF) as the effect measure.
        """
        all_deltas = []

        for i in range(X_test.shape[0]):
            X_subject = X_test[i].copy()
            original_value = X_subject[feature_idx]

            _, P_orig = predict_outcome(outcome_models, X_subject.reshape(1, -1), "classification")
            P_orig = P_orig[0]

            cf_values = self._sample_counterfactuals(
                cond_models, sigma, X_subject, feature_idx
            )

            for cf_val in cf_values:
                X_cf = X_subject.copy()
                X_cf[feature_idx] = cf_val

                _, P_cf = predict_outcome(outcome_models, X_cf.reshape(1, -1), "classification")
                P_cf = P_cf[0]

                delta_P = P_cf - P_orig
                delta_F = cf_val - original_value

                if abs(delta_F) > 1e-10:
                    all_deltas.append(delta_P / delta_F)

        return np.array(all_deltas)

    def fit(self, X, y, feature_names=None):
        """
        Run the full conditional intervention analysis.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix of shape (n_subjects, n_features).
        y : np.ndarray or pd.Series
            Outcome vector of shape (n_subjects,).
            For regression: continuous values.
            For classification: binary class labels (0/1).
        feature_names : list of str, optional
            Names for the features. If None and X is a DataFrame,
            column names are used.

        Returns
        -------
        self : ConditionalInterventionAnalysis
            Fitted instance with results in self.results_.
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = X.astype(np.float64)
        y = y.astype(np.float64)

        n_subjects, n_features = X.shape

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        self.feature_names_ = feature_names

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_effects = {i: [] for i in range(n_features)}

        model_desc = self.model_name
        if self.model_name == "consensus":
            if self.task_type == "regression":
                model_desc = "consensus (Linear SVR + Gaussian SVR + RF)"
            else:
                model_desc = "consensus (Linear SVC + Gaussian SVC + RF)"

        print(f"Running Conditional Intervention Analysis...")
        print(f"  Task type: {self.task_type}")
        print(f"  Subjects: {n_subjects}, Features: {n_features}")
        print(f"  CV Folds: {self.n_folds}, Counterfactual samples: {self.n_samples}")
        print(f"  Outcome model: {model_desc}")
        print(f"  Conditional model: {self.conditional_model}")
        print("-" * 60)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold_idx + 1}/{self.n_folds}...")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            outcome_fitted = train_outcome_models(
                X_train_scaled, y_train,
                task_type=self.task_type,
                model_name=self.model_name,
                random_state=self.random_state,
            )

            for feat_idx in range(n_features):
                cond_models, sigma = self._learn_conditional(X_train_scaled, feat_idx)

                if sigma < 1e-10:
                    fold_effects[feat_idx].append(0.0)
                    continue

                if self.task_type == "regression":
                    deltas = self._compute_feature_effect_regression(
                        X_test_scaled, outcome_fitted, cond_models, sigma, feat_idx
                    )
                else:
                    deltas = self._compute_feature_effect_classification(
                        X_test_scaled, outcome_fitted, cond_models, sigma, feat_idx
                    )

                if len(deltas) > 0:
                    fold_effects[feat_idx].append(np.mean(deltas))
                else:
                    fold_effects[feat_idx].append(0.0)

        results = []
        for feat_idx in range(n_features):
            effects = np.array(fold_effects[feat_idx])
            mean_effect = np.mean(effects)
            std_effect = np.std(effects)
            ci_low, ci_high = compute_confidence_interval(
                effects, self.confidence_level
            )

            t_stat, p_value = stats.ttest_1samp(effects, 0.0)

            results.append({
                "feature": feature_names[feat_idx],
                "causal_effect": mean_effect,
                "std": std_effect,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < (1 - self.confidence_level),
            })

        self.results_ = pd.DataFrame(results)
        self.results_ = self.results_.sort_values(
            "p_value", ascending=True
        ).reset_index(drop=True)

        print("-" * 60)
        print("Analysis complete.")
        return self

    def get_results(self):
        """Return results as a pandas DataFrame."""
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def get_causal_features(self, alpha=0.05):
        """
        Return features with statistically significant causal effects.

        Parameters
        ----------
        alpha : float
            Significance threshold.

        Returns
        -------
        pd.DataFrame
            Subset of results where p_value < alpha.
        """
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_[self.results_["p_value"] < alpha].reset_index(drop=True)

    def summary(self):
        """Print a formatted summary of results."""
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")

        effect_label = "Effect" if self.task_type == "regression" else "Prob. Effect"

        print("\n" + "=" * 70)
        print("CONDITIONAL INTERVENTION ANALYSIS - CAUSAL EFFECT SUMMARY")
        print(f"Task: {self.task_type} | Model: {self.model_name}")
        print("=" * 70)
        print(f"\n{'Feature':<25} {effect_label:>12} {'Std':>8} {'p-value':>10} {'Sig.':>6}")
        print("-" * 70)

        for _, row in self.results_.iterrows():
            sig_marker = "***" if row["p_value"] < 0.001 else (
                "**" if row["p_value"] < 0.01 else (
                    "*" if row["p_value"] < 0.05 else ""
                )
            )
            print(
                f"{row['feature']:<25} {row['causal_effect']:>12.4f} "
                f"{row['std']:>8.4f} {row['p_value']:>10.4f} {sig_marker:>6}"
            )

        print("-" * 70)
        n_sig = (self.results_["p_value"] < 0.05).sum()
        print(f"\nSignificant features (p < 0.05): {n_sig}/{len(self.results_)}")
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        if self.task_type == "classification":
            print("Effect = change in predicted probability per unit change in feature")
        else:
            print("Effect = change in predicted outcome per unit change in feature")
        print("=" * 70)
