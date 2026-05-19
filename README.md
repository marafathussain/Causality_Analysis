# Conditional Intervention Analysis (CIA)

A Python framework for estimating **causal effects** of features on an outcome variable using conditional sampling and counterfactual evaluation. This method approximates do-calculus interventions by generating biologically plausible counterfactuals through learned conditional distributions.

Uses the same **consensus of three ML techniques** as [FIBE](https://github.com/i3-research/fibe):
- **Regression**: Linear SVR + Gaussian SVR + Random Forest Regressor (averaged predictions)
- **Classification**: Linear SVC + Gaussian SVC + Random Forest Classifier (majority voting)

## Motivation

Feature selection methods (such as FIBE — Forward Inclusion Backward Elimination) identify features that are **correlated** with an outcome, but correlation does not imply causation. This framework takes the selected features one step further by estimating which features have a genuine **interventional (causal) effect** on the outcome.

## Method Overview

The approach is based on **Do-calculus via modeling** — a practical compromise between naive perturbation and full structural causal models:

1. **Train outcome model(s)**: Learn `O = f(F_1, F_2, ..., F_n)` using consensus of 3 ML models.

2. **Learn conditional distributions**: For each feature `F_i`, learn `P(F_i | F_{-i})` using consensus of 3 regression models.

3. **Generate counterfactuals via conditional sampling**: Instead of arbitrary perturbation (which breaks biological plausibility), sample:
   ```
   F_i' ~ P(F_i | F_{-i})
   ```
   This produces values that are consistent with the subject's other features.

4. **Evaluate interventional effect**:
   - Regression: `ΔO = f(F_i', F_{-i}) - f(F_i, F_{-i})` (change in predicted outcome)
   - Classification: `ΔP = P(class=1 | F_i', F_{-i}) - P(class=1 | F_i, F_{-i})` (change in probability)

5. **Aggregate across subjects and CV folds**: Report mean causal effects with confidence intervals and statistical significance.

### Why conditional sampling?

| Property               | Naive perturbation (F + δ) | Conditional sampling |
|------------------------|---------------------------|---------------------|
| Realistic values?      | No                        | Yes                 |
| Respects correlations? | No                        | Yes                 |
| Causal interpretation? | Weak                      | Stronger            |

### Why consensus of 3 models?

Using multiple diverse ML techniques (SVR/SVC + RF) and aggregating their outputs reduces model-specific bias. The effect estimate is more robust because it does not depend on any single model's assumptions.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Causality_Analysis.git
cd Causality_Analysis
pip install -r requirements.txt
```

## Quick Start

### Regression Task

```python
from causal_inference import ConditionalInterventionAnalysis

cia = ConditionalInterventionAnalysis(
    task_type="regression",       # continuous outcome
    n_samples=50,
    n_folds=5,
    model_name="consensus",       # Linear SVR + Gaussian SVR + RF
    conditional_model="consensus",
    random_state=42,
)
cia.fit(X, y)
cia.summary()
causal_features = cia.get_causal_features(alpha=0.05)
```

### Classification Task

```python
from causal_inference import ConditionalInterventionAnalysis

cia = ConditionalInterventionAnalysis(
    task_type="classification",   # binary outcome (0/1)
    n_samples=50,
    n_folds=5,
    model_name="consensus",       # Linear SVC + Gaussian SVC + RF
    conditional_model="consensus",
    random_state=42,
)
cia.fit(X, y_binary)
cia.summary()
```

### Using a Single Model (instead of consensus)

```python
cia = ConditionalInterventionAnalysis(
    task_type="regression",
    model_name="RegressionForest",      # only Random Forest for outcome
    conditional_model="gaussianSVR",    # only Gaussian SVR for conditional
)
```

## Running the Demo

```bash
python demo_run.py                     # runs both regression and classification
python demo_run.py --task regression   # regression only
python demo_run.py --task classification  # classification only
```

The demo uses synthetic data with known causal structure so you can verify the method correctly identifies truly causal features.

## Using with FIBE Output

```python
import pandas as pd
from causal_inference import ConditionalInterventionAnalysis

data = pd.read_csv("your_data.csv")

# Features selected by FIBE
selected_features = ["feature_A", "feature_B", "feature_C", ...]
X = data[selected_features]
y = data["outcome_score"]

# Run causal analysis on the FIBE-selected features
cia = ConditionalInterventionAnalysis(
    task_type="regression",    # or "classification"
    n_samples=50,
    n_folds=5,
    model_name="consensus",
)
cia.fit(X, y, feature_names=selected_features)
cia.summary()
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_type` | `"regression"` | `"regression"` for continuous outcomes, `"classification"` for binary (0/1) |
| `n_samples` | 50 | Number of counterfactual samples per subject per feature |
| `n_folds` | 5 | Number of cross-validation folds |
| `model_name` | `"consensus"` | Outcome model. Regression: `"linearSVR"`, `"gaussianSVR"`, `"RegressionForest"`, `"consensus"`. Classification: `"linearSVC"`, `"gaussianSVC"`, `"RandomForest"`, `"consensus"` |
| `conditional_model` | `"consensus"` | Model for P(F_i \| F_{-i}). Always regression: `"linearSVR"`, `"gaussianSVR"`, `"RegressionForest"`, `"consensus"` |
| `confidence_level` | 0.95 | Confidence level for effect intervals |
| `random_state` | 42 | Random seed for reproducibility |

## Output

The analysis returns a DataFrame with:

| Column | Description |
|--------|-------------|
| `feature` | Feature name |
| `causal_effect` | Estimated interventional effect (regression: ΔO/ΔF, classification: ΔP/ΔF) |
| `std` | Standard deviation across CV folds |
| `ci_lower` | Lower bound of confidence interval |
| `ci_upper` | Upper bound of confidence interval |
| `t_statistic` | One-sample t-test statistic (H0: effect = 0) |
| `p_value` | p-value for significance |
| `significant` | Boolean flag (p < alpha) |

## Repository Structure

```
Causality_Analysis/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── demo_run.py                        # Demo script (regression + classification)
├── causal_inference/                  # Main package
│   ├── __init__.py                    # Package initialization
│   ├── conditional_intervention.py    # Core CIA algorithm
│   ├── outcome_model.py              # 3 ML models + consensus (matching FIBE)
│   └── utils.py                      # Helper functions
└── Causality_discussion_with_LLM.txt  # Background discussion on methodology
```

## ML Models Used (Same as FIBE)

### Regression Task
| Model | Description |
|-------|-------------|
| Linear SVR | Support Vector Regression with linear kernel |
| Gaussian SVR | Support Vector Regression with RBF kernel |
| Regression Forest | Random Forest Regressor (100 trees, max_depth=5) |
| **Consensus** | Average of predictions from all three models |

### Classification Task
| Model | Description |
|-------|-------------|
| Linear SVC | Support Vector Classification with linear kernel |
| Gaussian SVC | Support Vector Classification with RBF kernel |
| Random Forest | Random Forest Classifier (100 trees, max_depth=5) |
| **Consensus** | Majority voting across all three models |

## Theoretical Note

This framework approximates the interventional distribution:

```
P(O | do(F_i))
```

by computing:

```
P(O | F_i' | F_{-i}),  where F_i' ~ P(F_i | F_{-i})
```

These are **not identical** unless certain assumptions hold (no hidden confounders, correct model specification). The method provides **causal evidence under assumptions** rather than definitive proof of causality. For a paper, appropriate language would be:

> "We estimated feature-level interventional effects by sampling from the conditional distribution P(F_i | F_{-i}), thereby generating biologically plausible counterfactuals while preserving the joint feature structure. Causal effects were aggregated using a consensus of three diverse ML models (SVR/SVC variants and Random Forest) to reduce model-specific bias."

## Dependencies

- Python >= 3.8
- NumPy >= 1.21
- Pandas >= 1.3
- Scikit-learn >= 1.0
- SciPy >= 1.7
- Matplotlib >= 3.4

## License

MIT License
