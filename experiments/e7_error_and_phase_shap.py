"""E5/E7. Error profiles and phase-stratified SHAP on the production GB model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

from common import MODEL_FEATURE_COLUMNS, load_gradient_boosting_model, split_engineered_data
from experiments.lib import PHASE_ORDER, RESULTS, load_frame
from risk_analyzer import load_operating_threshold

DUE_ORDER = ["No", "Yes Low", "Yes Medium", "Yes High"]


def raw_column(transformed_name: str) -> str:
    name = transformed_name
    if name.startswith("num__"):
        name = name[len("num__") :]
    elif name.startswith("cat__"):
        name = name[len("cat__") :]
    for column in sorted(MODEL_FEATURE_COLUMNS, key=len, reverse=True):
        if name == column or name.startswith(column + "_"):
            return column
    return name


def error_table(frame: pd.DataFrame, column: str, order=None) -> pd.DataFrame:
    grouped = frame.groupby(column).agg(
        n=("error_type", "size"),
        false_negatives=("error_type", lambda s: int((s == "FN").sum())),
        false_positives=("error_type", lambda s: int((s == "FP").sum())),
    )
    if order is not None:
        grouped = grouped.reindex(order)
    grouped["fn_rate"] = grouped["false_negatives"] / grouped["n"]
    grouped["fp_rate"] = grouped["false_positives"] / grouped["n"]
    return grouped


def main() -> None:
    frame = load_frame()
    _, x_test, _, y_test = split_engineered_data(frame)
    model = load_gradient_boosting_model()
    cutoff = load_operating_threshold()
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= cutoff).astype(int)
    y_test = y_test.to_numpy()

    detail = x_test.copy()
    detail["y"] = y_test
    detail["pred"] = pred
    detail["prob"] = prob
    detail["error_type"] = np.where(
        (detail["y"] == 1) & (detail["pred"] == 0),
        "FN",
        np.where((detail["y"] == 0) & (detail["pred"] == 1), "FP", "OK"),
    )
    detail.to_csv(RESULTS / "e7_test_predictions.csv", index=False)

    lines = ["# Error analysis and phase-stratified SHAP (E5, E7)", ""]
    lines.append(
        f"Predictions use the production Gradient Boosting model at the "
        f"current low-risk cutoff {cutoff:.2f}. A false negative is a student "
        f"who continued but was scored below that cutoff."
    )
    lines.append("")
    counts = detail["error_type"].value_counts()
    lines.append(counts.to_string())
    lines.append("")

    for column, order, filename in (
        ("Political_Phase", PHASE_ORDER, "e7_errors_by_phase.csv"),
        ("Researched", DUE_ORDER, "e7_errors_by_due_diligence.csv"),
        ("Budget_Level", ["Low", "Medium", "High"], "e7_errors_by_budget.csv"),
    ):
        table = error_table(detail, column, order)
        table.to_csv(RESULTS / filename)
        lines.append(f"## {column}")
        lines.append(table.round(3).to_string())
        lines.append("")

    preprocessor = model.named_steps["preprocessor"]
    transformed = pd.DataFrame(
        preprocessor.transform(x_test),
        columns=preprocessor.get_feature_names_out(),
        index=x_test.index,
    )
    explainer = shap.TreeExplainer(model.named_steps["model"])
    explanation = explainer(transformed)
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, 1]
    abs_values = np.abs(values)
    grouped = {}
    for idx, name in enumerate(transformed.columns):
        column = raw_column(name)
        grouped.setdefault(column, np.zeros(len(transformed)))
        grouped[column] += abs_values[:, idx]
    importance = pd.DataFrame(grouped, index=x_test.index)
    importance["Political_Phase"] = x_test["Political_Phase"].to_numpy()

    phase_means = (
        importance.groupby("Political_Phase")
        .mean()
        .reindex(PHASE_ORDER)
    )
    phase_means.to_csv(RESULTS / "e5_shap_mean_abs_by_phase.csv")

    top = (
        phase_means.mean()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    phase_means[top].plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Top drivers by political phase")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "e5_phase_shap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    lines.append("## Mean absolute SHAP by phase (top drivers)")
    lines.append(phase_means[top].round(4).to_string())
    lines.append("")
    lines.append(
        "Due diligence is stored three ways in the production model "
        "(Researched, Researched_Level, Researched_Bin), so its importance is split "
        "across those columns. Read them as one factor. Budget is split the same way. "
        "E8 drops the extra encodings and the F1 does not move."
    )
    text = "\n".join(lines)
    (RESULTS / "E7_ERROR_AND_SHAP.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
