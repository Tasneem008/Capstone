"""Global and local SHAP explanations for the Paper 2 Gradient Boosting model."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import shap

from common import (
    PROJECT_DIR,
    load_engineered_data,
    load_gradient_boosting_model,
    split_engineered_data,
)

OUTPUT_DIR = PROJECT_DIR / "results_xai"


def transform_for_shap(model, features):
    """Apply the fitted pipeline preprocessor and preserve output feature names."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a Pandas DataFrame.")

    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(features)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(
        transformed,
        columns=feature_names,
        index=features.index,
    )


def make_explainer(model):
    """Create a TreeExplainer for the pipeline's fitted tree estimator."""
    estimator = model.named_steps["model"]
    return shap.TreeExplainer(estimator)


def _select_positive_class(explanation):
    """Normalize binary SHAP outputs across SHAP versions."""
    if explanation.values.ndim != 3:
        return explanation

    class_index = 1
    base_values = explanation.base_values
    if getattr(base_values, "ndim", 0) == 2:
        base_values = base_values[:, class_index]

    return shap.Explanation(
        values=explanation.values[:, :, class_index],
        base_values=base_values,
        data=explanation.data,
        feature_names=explanation.feature_names,
    )


def explain_student(
    model,
    explainer,
    student_features,
    student_id,
    output_dir=OUTPUT_DIR,
):
    """Generate a SHAP waterfall plot for one raw student feature row."""
    if len(student_features) != 1:
        raise ValueError("student_features must contain exactly one row.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed = transform_for_shap(model, student_features)
    explanation = _select_positive_class(explainer(transformed))

    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(student_id)).strip("_")
    safe_id = safe_id or "student"
    output_path = output_dir / f"shap_waterfall_{safe_id}.png"

    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_xai_analysis(output_dir=OUTPUT_DIR):
    """Create the global summary and one test-student local explanation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_engineered_data()
    _, X_test, _, _ = split_engineered_data(df)
    model = load_gradient_boosting_model()
    explainer = make_explainer(model)

    transformed_test = transform_for_shap(model, X_test)
    explanation = _select_positive_class(explainer(transformed_test))

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        explanation.values,
        transformed_test,
        max_display=20,
        show=False,
    )
    summary_path = output_dir / "shap_summary.png"
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    student_id = f"test_{X_test.index[0]}"
    waterfall_path = explain_student(
        model,
        explainer,
        X_test.iloc[[0]],
        student_id,
        output_dir,
    )
    return summary_path, waterfall_path


def main():
    summary_path, waterfall_path = run_xai_analysis()
    print(f"Global SHAP summary saved to: {summary_path}")
    print(f"Example local explanation saved to: {waterfall_path}")


if __name__ == "__main__":
    main()
