"""
Generate confusion matrices and related evaluation plots for the
6,000-record engineered dataset using the saved production models.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from common import (
    PAPER1_XGB_MODEL_PATH,
    load_engineered_data,
    load_gradient_boosting_model,
    split_engineered_data,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "project_figures"
OUT.mkdir(parents=True, exist_ok=True)


def save_confusion(y_true, y_pred, title: str, filename: str, cmap: str = "Blues") -> dict:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(ax=ax, cmap=cmap, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    return {
        "filename": filename,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": float((tn + tp) / cm.sum()),
    }


def save_roc(y_true, y_prob, title: str, filename: str, color: str = "#1565c0") -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return float(auc)


def main() -> None:
    df = load_engineered_data()
    X_train, X_test, y_train, y_test = split_engineered_data(df)

    rows = []

    # --- Gradient Boosting (saved production model on 6K engineered features) ---
    gb = load_gradient_boosting_model()
    gb_prob = gb.predict_proba(X_test)[:, 1]
    gb_pred = (gb_prob >= 0.5).astype(int)

    rows.append(
        save_confusion(
            y_test,
            gb_pred,
            "Confusion Matrix: Gradient Boosting\n(6,000-record dataset, 20% held-out test)",
            "35_confusion_matrix_gradient_boosting_6000.png",
            cmap="Greens",
        )
    )
    rows[-1]["Model"] = "Gradient Boosting"
    rows[-1]["Threshold"] = 0.5
    rows[-1]["ROC_AUC"] = save_roc(
        y_test,
        gb_prob,
        "ROC Curve: Gradient Boosting (6,000-record test set)",
        "36_roc_curve_gradient_boosting_6000.png",
        color="#2e7d32",
    )

    # --- XGBoost (saved tuned model on same 6K engineered features) ---
    xgb = joblib.load(PAPER1_XGB_MODEL_PATH)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    xgb_pred_05 = (xgb_prob >= 0.5).astype(int)
    rows.append(
        save_confusion(
            y_test,
            xgb_pred_05,
            "Confusion Matrix: XGBoost\n(6,000-record dataset, threshold=0.50)",
            "37_confusion_matrix_xgboost_6000_threshold_0.50.png",
            cmap="Blues",
        )
    )
    rows[-1]["Model"] = "XGBoost"
    rows[-1]["Threshold"] = 0.5
    rows[-1]["ROC_AUC"] = save_roc(
        y_test,
        xgb_prob,
        "ROC Curve: XGBoost (6,000-record test set)",
        "38_roc_curve_xgboost_6000.png",
        color="#1565c0",
    )

    paper1 = pd.read_csv(ROOT / "results_paper1" / "paper1_model_comparison.csv")
    xgb_threshold = float(
        paper1.loc[paper1["Model"].str.contains("optimized"), "Threshold"].iloc[0]
    )
    xgb_pred_opt = (xgb_prob >= xgb_threshold).astype(int)
    rows.append(
        save_confusion(
            y_test,
            xgb_pred_opt,
            f"Confusion Matrix: XGBoost\n(6,000-record dataset, threshold={xgb_threshold:.2f})",
            "39_confusion_matrix_xgboost_6000_optimized_threshold.png",
            cmap="Purples",
        )
    )
    rows[-1]["Model"] = "XGBoost"
    rows[-1]["Threshold"] = xgb_threshold
    rows[-1]["ROC_AUC"] = rows[-2]["ROC_AUC"]

    stale = OUT / "39_confusion_matrix_xgboost_6000_threshold_0.73.png"
    if stale.exists():
        stale.unlink()

    base_probs = []
    for name in ("random_forest", "xgboost", "gradient_boosting"):
        model = joblib.load(ROOT / "results_paper2" / f"{name}.joblib")
        base_probs.append(model.predict_proba(X_test)[:, 1])
    meta = joblib.load(ROOT / "results_paper2" / "stacking_fnn_meta_model.joblib")
    stack_prob = meta.predict_proba(np.column_stack(base_probs))[:, 1]
    stack_pred = (stack_prob >= 0.5).astype(int)

    # Side-by-side comparison of the chosen model and the highest single-split stack
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pred, title, cmap in [
        (
            axes[0],
            gb_pred,
            "Gradient Boosting @ 0.50 (chosen)",
            "Greens",
        ),
        (
            axes[1],
            stack_pred,
            "Stack @ 0.50",
            "Purples",
        ),
    ]:
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(
            ax=ax, cmap=cmap, colorbar=False, values_format="d"
        )
        ax.set_title(title)
    fig.suptitle(
        "6,000-Record Dataset: Held-out Test Confusion Matrices",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        OUT / "40_confusion_matrix_gb_vs_xgboost_6000_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Dual ROC overlay
    fig, ax = plt.subplots(figsize=(6, 5))
    for prob, label, color in [
        (gb_prob, "Gradient Boosting", "#2e7d32"),
        (stack_prob, "Stack", "#6a1b9a"),
        (xgb_prob, "XGBoost", "#1565c0"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison on 6,000-Record Held-out Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "41_roc_comparison_gb_vs_xgboost_6000.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "00_6000_model_confusion_summary.csv", index=False)

    # Refresh index of all PNGs
    files = sorted(OUT.glob("*.png"))
    index_rows = [{"filename": p.name, "size_kb": round(p.stat().st_size / 1024, 1)} for p in files]
    pd.DataFrame(index_rows).to_csv(OUT / "00_figure_index.csv", index=False)
    lines = [
        "Capstone Project Figures Index",
        f"Folder: {OUT}",
        f"Total PNG files: {len(files)}",
        "",
    ]
    lines.extend(f"- {row['filename']}  ({row['size_kb']} KB)" for row in index_rows)
    (OUT / "00_figure_index.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Generated on 6,000-record held-out test set:")
    print(summary.to_string(index=False))
    print("\nClassification report — Gradient Boosting:")
    print(classification_report(y_test, gb_pred, target_names=["No", "Yes"]))
    print(f"Classification report — XGBoost @ {xgb_threshold:.2f}:")
    print(classification_report(y_test, xgb_pred_opt, target_names=["No", "Yes"]))
    print(f"\nSaved under: {OUT}")


if __name__ == "__main__":
    main()
