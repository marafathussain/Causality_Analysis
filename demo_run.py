"""
Demo: Conditional Intervention Analysis (CIA)

This script demonstrates how to use the CIA framework to estimate
causal effects of features on an outcome variable.

Supports both REGRESSION and CLASSIFICATION tasks, using the same
consensus of three ML techniques as FIBE:
  - Regression: Linear SVR + Gaussian SVR + Random Forest
  - Classification: Linear SVC + Gaussian SVC + Random Forest

Usage:
    python demo_run.py                    # runs both regression and classification demos
    python demo_run.py --task regression  # regression demo only
    python demo_run.py --task classification  # classification demo only
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from causal_inference import (
    ConditionalInterventionAnalysis,
    generate_synthetic_data,
    format_results_table,
)


def run_regression_demo():
    """Run regression task demonstration."""

    print("\n" + "=" * 70)
    print("  CIA DEMO — REGRESSION TASK")
    print("  Consensus: Linear SVR + Gaussian SVR + Random Forest Regressor")
    print("=" * 70)

    # Generate synthetic data
    print("\n[Step 1] Generating synthetic regression data...")
    print("  - 80 subjects, 8 features")
    print("  - First 3 features are truly CAUSAL")
    print("  - Remaining 5 are correlated but NON-CAUSAL")

    X, y, true_causal = generate_synthetic_data(
        n_subjects=80,
        n_features=8,
        n_causal=3,
        noise_level=0.5,
        task_type="regression",
        random_state=42,
    )

    print(f"\n  Ground truth causal features: {true_causal}")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Outcome (continuous) range: [{y.min():.2f}, {y.max():.2f}]")

    # Initialize CIA
    print("\n[Step 2] Initializing CIA with consensus models...")

    cia = ConditionalInterventionAnalysis(
        task_type="regression",
        n_samples=30,
        n_folds=5,
        model_name="consensus",
        conditional_model="consensus",
        confidence_level=0.95,
        random_state=42,
    )

    # Run analysis
    print("\n[Step 3] Running analysis...\n")
    cia.fit(X, y)

    # Summary
    cia.summary()

    # Significant features
    print("\n[Step 4] Significant causal features (p < 0.05):")
    causal_features = cia.get_causal_features(alpha=0.05)
    if len(causal_features) > 0:
        for _, row in causal_features.iterrows():
            print(f"    - {row['feature']}: effect = {row['causal_effect']:.4f}, p = {row['p_value']:.4f}")
    else:
        print("    None found at p < 0.05")

    # Validation
    print("\n[Step 5] Validation against ground truth:")
    _validate(causal_features, true_causal)

    # Plot
    plot_results(cia.get_results(), true_causal, "causal_effects_regression.png", "Regression")

    return cia


def run_classification_demo():
    """Run classification task demonstration."""

    print("\n" + "=" * 70)
    print("  CIA DEMO — CLASSIFICATION TASK")
    print("  Consensus: Linear SVC + Gaussian SVC + Random Forest Classifier")
    print("=" * 70)

    # Generate synthetic data
    print("\n[Step 1] Generating synthetic classification data...")
    print("  - 100 subjects, 8 features")
    print("  - First 3 features are truly CAUSAL")
    print("  - Remaining 5 are correlated but NON-CAUSAL")

    X, y, true_causal = generate_synthetic_data(
        n_subjects=100,
        n_features=8,
        n_causal=3,
        noise_level=0.3,
        task_type="classification",
        random_state=42,
    )

    print(f"\n  Ground truth causal features: {true_causal}")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Class distribution: {dict(pd.Series(y).value_counts())}")

    # Initialize CIA
    print("\n[Step 2] Initializing CIA with consensus models...")

    cia = ConditionalInterventionAnalysis(
        task_type="classification",
        n_samples=30,
        n_folds=5,
        model_name="consensus",
        conditional_model="consensus",
        confidence_level=0.95,
        random_state=42,
    )

    # Run analysis
    print("\n[Step 3] Running analysis...\n")
    cia.fit(X, y)

    # Summary
    cia.summary()

    # Significant features
    print("\n[Step 4] Significant causal features (p < 0.05):")
    causal_features = cia.get_causal_features(alpha=0.05)
    if len(causal_features) > 0:
        for _, row in causal_features.iterrows():
            print(f"    - {row['feature']}: prob. effect = {row['causal_effect']:.4f}, p = {row['p_value']:.4f}")
    else:
        print("    None found at p < 0.05")

    # Validation
    print("\n[Step 5] Validation against ground truth:")
    _validate(causal_features, true_causal)

    # Plot
    plot_results(cia.get_results(), true_causal, "causal_effects_classification.png", "Classification")

    return cia


def _validate(causal_features_df, true_causal):
    """Validate identified features against ground truth."""
    identified = set(causal_features_df["feature"].tolist())
    true_set = set(true_causal)

    true_positives = identified & true_set
    false_positives = identified - true_set
    false_negatives = true_set - identified

    print(f"    True positives:  {true_positives or 'None'}")
    print(f"    False positives: {false_positives or 'None'}")
    print(f"    False negatives: {false_negatives or 'None'}")

    if true_positives:
        precision = len(true_positives) / len(identified) if identified else 0
        recall = len(true_positives) / len(true_set)
        print(f"    Precision: {precision:.2f}, Recall: {recall:.2f}")


def plot_results(results_df, true_causal_features, filename, task_label):
    """Plot causal effects with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 6))

    features = results_df["feature"].tolist()
    effects = results_df["causal_effect"].values
    ci_low = results_df["ci_lower"].values
    ci_high = results_df["ci_upper"].values
    p_values = results_df["p_value"].values

    colors = []
    for i, feat in enumerate(features):
        if feat in true_causal_features:
            colors.append("darkred" if p_values[i] < 0.05 else "salmon")
        else:
            colors.append("steelblue" if p_values[i] < 0.05 else "lightsteelblue")

    y_pos = np.arange(len(features))

    ax.barh(y_pos, effects, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    for i in range(len(features)):
        ax.plot(
            [ci_low[i], ci_high[i]], [y_pos[i], y_pos[i]],
            color="black", linewidth=1.5, marker="|", markersize=8
        )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Estimated Causal Effect")
    ax.set_title(f"Conditional Intervention Analysis — {task_label} Task (Consensus of 3 ML Models)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="darkred", label="True causal (significant)"),
        Patch(facecolor="salmon", label="True causal (not significant)"),
        Patch(facecolor="steelblue", label="Non-causal (significant)"),
        Patch(facecolor="lightsteelblue", label="Non-causal (not significant)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {filename}")


def run_with_custom_data_example():
    """
    Example showing how to use CIA with your own data.
    Uncomment and modify for your actual dataset.
    """
    print("\n" + "=" * 70)
    print("  EXAMPLE: Using CIA with your own data")
    print("=" * 70)
    print("""
    # --- REGRESSION EXAMPLE ---
    from causal_inference import ConditionalInterventionAnalysis
    import pandas as pd

    X = pd.read_csv("your_features.csv")
    y = pd.read_csv("your_outcome.csv")["score"]

    cia = ConditionalInterventionAnalysis(
        task_type="regression",       # <-- choose 'regression' or 'classification'
        n_samples=50,
        n_folds=5,
        model_name="consensus",       # <-- 'consensus' uses 3 models from FIBE
        conditional_model="consensus",
        random_state=42,
    )
    cia.fit(X, y)
    cia.summary()
    causal = cia.get_causal_features(alpha=0.05)

    # --- CLASSIFICATION EXAMPLE ---
    cia_cls = ConditionalInterventionAnalysis(
        task_type="classification",
        n_samples=50,
        n_folds=5,
        model_name="consensus",
        random_state=42,
    )
    cia_cls.fit(X, y_binary)
    cia_cls.summary()

    # --- SINGLE MODEL (instead of consensus) ---
    cia_single = ConditionalInterventionAnalysis(
        task_type="regression",
        model_name="RegressionForest",      # just RF
        conditional_model="gaussianSVR",    # just Gaussian SVR for conditional
    )
    cia_single.fit(X, y)
    """)


if __name__ == "__main__":
    task_arg = None
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--task" and i + 1 < len(sys.argv):
                task_arg = sys.argv[i + 1]

    if task_arg == "regression":
        run_regression_demo()
    elif task_arg == "classification":
        run_classification_demo()
    else:
        run_regression_demo()
        run_classification_demo()

    run_with_custom_data_example()

    print("\n" + "=" * 70)
    print("  All demos complete!")
    print("=" * 70)
