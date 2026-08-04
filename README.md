# Conditional Intervention Analysis (CIA)

A Python framework for estimating **causal effects** of features on an outcome using conditional sampling and counterfactual evaluation. CIA approximates do-calculus interventions by generating biologically plausible counterfactuals through learned conditional distributions.

It is designed to follow [FIBE](https://github.com/i3-research/fibe) feature selection: FIBE finds correlated features; CIA tests which of those features have a plausible **interventional effect** on the outcome.

---

## What problem does CIA solve?

| Step | Method | What it tells you |
|------|--------|-------------------|
| 1 | FIBE (or similar) | Features **correlated** with the outcome |
| 2 | **CIA (this repo)** | Which selected features show **causal evidence** under conditional intervention |

Correlation does not imply causation. CIA goes one step further by asking: *if we change feature F_i in a biologically plausible way, how much does the predicted outcome change?*

---

## How CIA works (step by step)

Suppose you have **n subjects**, **p features** (e.g., 10 features selected by FIBE), and one **outcome** (continuous score or binary label).

### Step 1: Train an outcome model

Learn a predictive mapping from features to outcome:

```
O = f(F_1, F_2, ..., F_p)
```

CIA uses a **consensus of 3 ML models** (same philosophy as FIBE):
- **Regression**: Linear SVR + Gaussian SVR + Random Forest (average predictions)
- **Classification**: Linear SVC + Gaussian SVC + Random Forest (majority vote / averaged probabilities)

For **large datasets** (n > 1000), use `consensus_fast` instead (Ridge/Logistic Regression + Gradient Boosting + Random Forest).

### Step 2: Learn conditional distributions

For each feature `F_i`, learn how it relates to the other features:

```
P(F_i | F_{-i})
```

This is done with regression models (consensus of 3 regressors). The model predicts the expected value of `F_i` given all other features for that subject.

### Step 3: Generate counterfactuals (conditional sampling)

For each test subject, CIA does **not** use naive perturbation like `F_i + δ` (which can create unrealistic values). Instead it samples:

```
F_i' ~ P(F_i | F_{-i})
```

So `F_i'` is a plausible alternative value of feature i that is consistent with that subject's other features.

**Feature types** (continuous, binary, categorical) are **auto-detected** from the data. Categorical and binary features are rounded to valid integer levels after sampling.

### Step 4: Measure interventional effect per subject

For each counterfactual sample:

- **Regression**: effect ∝ `(predicted outcome with F_i') - (predicted outcome with F_i)` per unit change in F_i → **ΔO / ΔF**
- **Classification**: effect ∝ change in **predicted probability of class 1** per unit change in F_i → **ΔP / ΔF**

Effects are averaged over counterfactual samples within each subject, then over test subjects within each CV fold.

### Step 5: Cross-validation and statistics

Data are split into **K folds** (default K=5). Each fold gives **one mean effect** per feature. Across folds:

| Output | Meaning |
|--------|---------|
| `causal_effect` | Mean effect across folds |
| `ci_lower`, `ci_upper` | 95% confidence interval (t-distribution) |
| `p_value` | One-sample t-test: H₀ = true effect is zero |
| `significant` | p < 0.05 |

**Interpretation:** A feature with a large |effect|, CI not crossing zero, and small p-value is strong **causal evidence under the method's assumptions** (not proof of causality without randomized trials).

### Why conditional sampling?

| Property | Naive perturbation (F + δ) | Conditional sampling |
|----------|---------------------------|----------------------|
| Realistic values? | No | Yes |
| Respects feature correlations? | No | Yes |
| Causal interpretation | Weak | Stronger |

---

## Installation

```bash
git clone https://github.com/marafathussain/Causality_Analysis.git
cd Causality_Analysis
pip install -r requirements.txt
```

---

## Quick start

### Small / medium datasets (n < 500)

```python
from causal_inference import ConditionalInterventionAnalysis

cia = ConditionalInterventionAnalysis(
    task_type="regression",       # or "classification"
    n_samples=30,
    n_folds=5,
    model_name="consensus",
    conditional_model="consensus",
    random_state=42,
)
cia.fit(X, y)                     # feature types auto-detected
cia.summary()
results = cia.get_results()
causal = cia.get_causal_features(alpha=0.05)
```

### Large datasets (n > 1000, e.g., ABCD-scale)

SVR/SVC training scales poorly (O(n²)–O(n³)). Use **`consensus_fast`** and **`max_subjects`**:

```python
cia = ConditionalInterventionAnalysis(
    task_type="regression",
    n_samples=20,
    n_folds=5,
    model_name="consensus_fast",        # Ridge + GBM + RF
    conditional_model="consensus_fast",
    max_subjects=300,                   # subsample test subjects per fold
    random_state=42,
)
cia.fit(X, y)
cia.summary()
```

**Recommended settings by dataset size:**

| Dataset size | Settings |
|--------------|----------|
| n < 500 | `model_name="consensus"`, defaults |
| 500 – 2,000 | `consensus_fast`, `n_samples=30` |
| 2,000 – 10,000 | `consensus_fast`, `max_subjects=500`, `n_samples=20` |
| n > 10,000 | `consensus_fast`, `max_subjects=300`, `n_samples=15`, `n_folds=3` |

`max_subjects` randomly subsamples test subjects per fold. Fold-level aggregation keeps estimates stable while cutting runtime sharply.

---

## Running the demo

```bash
python demo_run.py                          # regression + classification (small data)
python demo_run.py --task regression        # regression only
python demo_run.py --task classification    # classification only
python demo_run.py --task large             # large-data mode (consensus_fast + max_subjects)
```

The demo uses synthetic data with known causal features, prints a statistical summary, and saves bar plots with confidence intervals.

---

## Using with FIBE output

```python
import pandas as pd
from causal_inference import ConditionalInterventionAnalysis
from demo_run import plot_results

data = pd.read_csv("your_data.csv")
selected_features = ["feat_A", "feat_B", ...]  # from FIBE
X = data[selected_features]
y = data["outcome"]

# Use consensus_fast for large cohorts
cia = ConditionalInterventionAnalysis(
    task_type="regression",
    model_name="consensus_fast",
    conditional_model="consensus_fast",
    max_subjects=300,
    n_samples=20,
    n_folds=5,
)
cia.fit(X, y, feature_names=selected_features)
cia.summary()

# Save figure (colored by p-value)
plot_results(cia.get_results(), "causal_effects.png", "Regression")
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_type` | `"regression"` | `"regression"` or `"classification"` |
| `n_samples` | 50 | Counterfactual samples per subject per feature |
| `n_folds` | 5 | Cross-validation folds |
| `model_name` | `"consensus"` | Outcome model (see table below) |
| `conditional_model` | `"consensus"` | Model for P(F_i \| F_{-i}) |
| `max_subjects` | `None` | Max test subjects per fold; subsample if exceeded |
| `confidence_level` | 0.95 | CI level |
| `random_state` | 42 | Random seed |

**`fit()` arguments:**

| Argument | Description |
|----------|-------------|
| `X` | Feature matrix (DataFrame or ndarray) |
| `y` | Outcome vector |
| `feature_names` | Optional column names |
| `feature_types` | `None` or `"auto"` (default: auto-detect), dict, or list: `"continuous"`, `"binary"`, `"categorical"` |

### Model options

**Regression – `model_name` / `conditional_model`:**

| Value | Models |
|-------|--------|
| `consensus` | Linear SVR + Gaussian SVR + RF (best for small n) |
| `consensus_fast` | Ridge + Gradient Boosting + RF (best for large n) |
| `linearSVR`, `gaussianSVR`, `RegressionForest`, `Ridge`, `GradientBoosting` | Single model |

**Classification – `model_name`:**

| Value | Models |
|-------|--------|
| `consensus` | Linear SVC + Gaussian SVC + RF |
| `consensus_fast` | Logistic Regression + Gradient Boosting + RF |
| `linearSVC`, `gaussianSVC`, `RandomForest`, `LogisticRegression`, `GradientBoosting` | Single model |

---

## Output

### 1. DataFrame (`cia.get_results()`)

| Column | Description |
|--------|-------------|
| `feature` | Feature name |
| `causal_effect` | Mean interventional effect (ΔO/ΔF or ΔP/ΔF) |
| `std` | Std across CV folds |
| `ci_lower`, `ci_upper` | Confidence interval |
| `t_statistic` | t-test statistic |
| `p_value` | Significance |
| `significant` | p < alpha |

### 2. Console summary (`cia.summary()`)

Formatted table of effects, p-values, and significance markers.

### 3. Significant features (`cia.get_causal_features(alpha=0.05)`)

Subset of results with p < alpha.

### 4. Figure (`plot_results()` in `demo_run.py`)

Horizontal bar chart of causal effects with 95% CI error bars, colored by significance.

```python
from demo_run import plot_results
plot_results(cia.get_results(), "causal_effects.png", "Regression")
```

---

## Repository structure

```
Causality_Analysis/
├── README.md
├── requirements.txt
├── demo_run.py
├── causal_inference/
│   ├── __init__.py
│   ├── conditional_intervention.py   # Core CIA algorithm
│   ├── outcome_model.py              # Consensus / consensus_fast models
│   └── utils.py                      # Synthetic data, helpers
└── Causality_discussion_with_LLM.txt
```

---

## Theoretical note

CIA approximates:

```
P(O | do(F_i))
```

via:

```
P(O | F_i', F_{-i}),   where F_i' ~ P(F_i | F_{-i})
```

These are not identical without assumptions (no hidden confounders, correct models). Appropriate wording for a paper:

> "We estimated feature-level interventional effects by sampling from the conditional distribution P(F_i | F_{-i}), generating biologically plausible counterfactuals while preserving joint feature structure. Effects were aggregated using a consensus of three ML models to reduce model-specific bias."

---

## Dependencies

- Python >= 3.8
- NumPy, Pandas, scikit-learn, SciPy, Matplotlib

## Citation

If you use this code, please cite our paper:

```bibtex
@article{hussain2026conditional,
  title   = {Conditional Intervention Analysis: Toward Identifying Causal Features in Tabular Multimodal Data},
  author  = {Hussain, Mohammad Arafat and Du, Liyan and Grant, Ellen and Ou, Yangming},
  booktitle    = {International Workshop on Multimodal Learning with Medical Tabular Data},
  organization    = {Springer}
  year   = {2026}
}
```

## License

MIT License
