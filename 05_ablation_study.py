"""Feature-group ablation study for the Paper 2 Gradient Boosting model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from common import (
    PROJECT_DIR,
    evaluate_predictions,
    load_engineered_data,
    load_gradient_boosting_model,
    make_preprocessor,
    split_engineered_data,
)

OUTPUT_DIR = PROJECT_DIR / "results_ablation"


def group_features(columns):
    """Group engineered columns by semantic name patterns."""
    groups = {
        "Financial_Features": [],
        "Academic_Features": [],
        "Phase_Temporal_Features": [],
        "Destination_Features": [],
        "Behavioral_Features": [],
        "Other_Features": [],
    }

    for column in columns:
        normalized = column.lower()
        if "budget" in normalized or "financial" in normalized:
            group = "Financial_Features"
        elif any(token in normalized for token in ("result", "gpa", "academic")):
            group = "Academic_Features"
        elif any(
            token in normalized
            for token in ("phase", "season", "month", "year", "days_since")
        ):
            group = "Phase_Temporal_Features"
        elif any(
            token in normalized for token in ("course", "country", "institution")
        ):
            group = "Destination_Features"
        elif "research" in normalized or "diligence" in normalized:
            group = "Behavioral_Features"
        else:
            group = "Other_Features"
        groups[group].append(column)

    return {name: features for name, features in groups.items() if features}


def run_ablation_study(output_dir=OUTPUT_DIR):
    """Evaluate the saved baseline and freshly retrained group ablations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_engineered_data()
    X_train, X_test, y_train, y_test = split_engineered_data(df)
    saved_pipeline = load_gradient_boosting_model()

    baseline_predictions = saved_pipeline.predict(X_test)
    baseline_metrics = evaluate_predictions(y_test, baseline_predictions)
    baseline_f1 = baseline_metrics["F1"]

    rows = [
        {
            "Experiment": "Baseline (all features)",
            "Removed_Group": "None",
            "Removed_Features": "",
            **baseline_metrics,
            "F1_Drop": 0.0,
        }
    ]

    feature_groups = group_features(X_train.columns)
    estimator_template = saved_pipeline.named_steps["model"]

    for group_name, removed_features in feature_groups.items():
        retained_features = [
            feature for feature in X_train.columns if feature not in removed_features
        ]
        X_train_reduced = X_train.loc[:, retained_features]
        X_test_reduced = X_test.loc[:, retained_features]

        preprocessor, _, _ = make_preprocessor(
            X_train_reduced, scale_numeric=True
        )
        ablated_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", clone(estimator_template)),
            ]
        )
        ablated_pipeline.fit(X_train_reduced, y_train)
        predictions = ablated_pipeline.predict(X_test_reduced)
        metrics = evaluate_predictions(y_test, predictions)

        rows.append(
            {
                "Experiment": f"Without {group_name}",
                "Removed_Group": group_name,
                "Removed_Features": ", ".join(removed_features),
                **metrics,
                "F1_Drop": baseline_f1 - metrics["F1"],
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "ablation_results.csv", index=False)
    save_f1_drop_chart(results, output_dir / "ablation_results.png")
    return results


def save_f1_drop_chart(results, output_path):
    """Save a horizontal bar chart of F1 loss relative to the baseline."""
    plot_data = results.loc[results["Removed_Group"] != "None"].copy()
    plot_data = plot_data.sort_values("F1_Drop", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        "#c62828" if value >= 0 else "#2e7d32" for value in plot_data["F1_Drop"]
    ]
    ax.barh(plot_data["Removed_Group"], plot_data["F1_Drop"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("F1-score drop from baseline")
    ax.set_ylabel("Removed feature group")
    ax.set_title("Gradient Boosting Feature-Group Ablation Study")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    results = run_ablation_study()
    display_columns = [
        "Experiment",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "F1_Drop",
    ]
    print(results.loc[:, display_columns].to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
