"""
Build a slim, paper-ready figure pack (~8 multi-panel figures).

Leaves project_figures/ untouched. Writes only to project_figures_paper/.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from common import (
    DUE_DILIGENCE_LEVEL,
    PAPER1_XGB_MODEL_PATH,
    load_engineered_data,
    load_gradient_boosting_model,
    split_engineered_data,
)
from risk_analyzer import load_operating_threshold

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "project_figures_paper"
DATA = ROOT / "Datasets" / "master_6000_engineered.csv"

PHASE_ORDER = [
    "Stable",
    "Unstable",
    "Transitional phase 1st 5 months",
    "Transitional phase Last 5 months",
    "Newly Elected",
]
PHASE_SHORT = {
    "Stable": "Stable",
    "Unstable": "Unstable",
    "Transitional phase 1st 5 months": "TP1",
    "Transitional phase Last 5 months": "TP2",
    "Newly Elected": "Newly Elected",
}

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {name}")


def fig01_dataset_overview(df: pd.DataFrame) -> None:
    """Class balance + continuation by phase + researched (core motivation)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    counts = df["Continuation"].value_counts().reindex(["No", "Yes"])
    axes[0].bar(counts.index.astype(str), counts.values, color=["#c62828", "#2e7d32"])
    axes[0].set_title("(a) Class balance")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, f"{v}\n({v/len(df)*100:.0f}%)", ha="center", fontsize=8)

    rates = df.groupby("Political_Phase")["Continuation_Bin"].mean().reindex(PHASE_ORDER) * 100
    axes[1].bar([PHASE_SHORT[p] for p in PHASE_ORDER], rates.values, color="#1565c0")
    axes[1].set_title("(b) Continuation by political phase")
    axes[1].set_ylabel("Rate (%)")
    axes[1].tick_params(axis="x", rotation=15)

    due_order = list(DUE_DILIGENCE_LEVEL)
    due_colors = ["#ef6c00", "#f9a825", "#66bb6a", "#1b5e20"]
    r = df.groupby("Researched")["Continuation_Bin"].mean().reindex(due_order) * 100
    axes[2].bar(due_order, r.values, color=due_colors)
    axes[2].set_title("(c) Continuation by Due_Diligence")
    axes[2].set_xlabel("Due_Diligence")
    axes[2].set_ylabel("Rate (%)")
    axes[2].tick_params(axis="x", rotation=20)
    for i, v in enumerate(r.values):
        axes[2].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=7)

    fig.suptitle("Dataset overview (n=6,000)", y=1.03, fontsize=12)
    save(fig, "fig01_dataset_overview.png")


def fig02_key_drivers(df: pd.DataFrame) -> None:
    """Budget + phase×researched heatmap + budget-academic score."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))

    order = ["Below 10k", "10-15K", "16-20K"]
    br = df.groupby("Budget")["Continuation_Bin"].mean().reindex(order) * 100
    axes[0].bar(order, br.values, color="#6a1b9a")
    axes[0].set_title("(a) Continuation by budget")
    axes[0].set_ylabel("Rate (%)")
    axes[0].tick_params(axis="x", rotation=15)

    pivot = (
        df.pivot_table(
            index="Political_Phase",
            columns="Researched",
            values="Continuation_Bin",
            aggfunc="mean",
        )
        .reindex(index=PHASE_ORDER, columns=list(DUE_DILIGENCE_LEVEL))
        * 100
    )
    pivot.index = [PHASE_SHORT[i] for i in pivot.index]
    im = axes[1].imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    axes[1].set_xticks(range(pivot.shape[1]))
    axes[1].set_xticklabels(list(DUE_DILIGENCE_LEVEL), rotation=20, ha="right")
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels(pivot.index)
    axes[1].set_title("(b) Phase × Due_Diligence")
    axes[1].set_xlabel("Due_Diligence")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            axes[1].text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    score_rate = df.groupby("Budget_Academic_Score")["Continuation_Bin"].mean() * 100
    axes[2].plot(score_rate.index, score_rate.values, marker="o", color="#4e342e")
    axes[2].set_title("(c) Budget-academic score")
    axes[2].set_xlabel("Score")
    axes[2].set_ylabel("Continuation rate (%)")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Key behavioral and financial drivers", y=1.03, fontsize=12)
    save(fig, "fig02_key_predictive_drivers.png")


def fig03_model_metrics(y_test, gb_prob, stack_prob, xgb_prob, xgb_f1_threshold) -> None:
    configs = [
        ("GB @0.50\n(chosen)", gb_prob, 0.50),
        ("Stack @0.50", stack_prob, 0.50),
        ("XGB @0.50", xgb_prob, 0.50),
        (f"XGB @{xgb_f1_threshold:.2f}", xgb_prob, xgb_f1_threshold),
    ]
    rows = []
    for label, prob, thr in configs:
        pred = (prob >= thr).astype(int)
        rows.append(
            {
                "Model": label,
                "Accuracy": (pred == y_test).mean(),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, prob),
            }
        )
    metrics = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(metrics))
    width = 0.16
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        axes[0].bar(x + (i - 2) * width, metrics[metric], width=width, label=metric)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics["Model"])
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("(a) Test metrics on 6,000-record holdout")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(axis="y", alpha=0.3)

    for prob, label, color in [
        (gb_prob, "Gradient Boosting", "#2e7d32"),
        (stack_prob, "Stack", "#6a1b9a"),
        (xgb_prob, "XGBoost", "#1565c0"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        axes[1].plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("(b) ROC curves")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Model performance on the 6,000-record dataset", y=1.03, fontsize=12)
    save(fig, "fig03_model_performance_metrics_roc.png")
    metrics.to_csv(OUT / "table_model_metrics.csv", index=False)


def fig04_confusion_matrices(y_test, gb_pred, stack_pred) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, pred, title, cmap in [
        (axes[0], gb_pred, "(a) Gradient Boosting @ 0.50 (chosen)", "Greens"),
        (axes[1], stack_pred, "(b) Stack @ 0.50", "Purples"),
    ]:
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(
            ax=ax, cmap=cmap, colorbar=False, values_format="d"
        )
        ax.set_title(title)
    fig.suptitle("Confusion matrices (held-out test, n=1,200)", y=1.03, fontsize=12)
    save(fig, "fig04_confusion_matrices_gb_stack.png")


def fig05_ablation() -> None:
    path = ROOT / "results_ablation" / "ablation_results.csv"
    ab = pd.read_csv(path)
    labels, f1s, drops = [], [], []
    for _, row in ab.iterrows():
        removed = row.get("Removed_Group")
        if pd.isna(removed) or str(removed) in {"None", "nan"}:
            labels.append("Baseline")
        else:
            labels.append(str(removed).replace("_Features", ""))
        f1s.append(row["F1"])
        drops.append(row["F1_Drop"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#2e7d32" if lab == "Baseline" else "#546e7a" for lab in labels]
    axes[0].bar(labels, f1s, color=colors)
    axes[0].set_title("(a) Absolute F1")
    axes[0].set_ylabel("F1")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylim(0, 1)

    # F1 drop excluding baseline
    drop_labels = [l for l, d in zip(labels, drops) if l != "Baseline"]
    drop_vals = [d for l, d in zip(labels, drops) if l != "Baseline"]
    order = np.argsort(drop_vals)
    drop_labels = [drop_labels[i] for i in order]
    drop_vals = [drop_vals[i] for i in order]
    axes[1].barh(drop_labels, drop_vals, color="#c62828")
    axes[1].set_title("(b) F1 drop when group removed")
    axes[1].set_xlabel("F1 drop")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("Feature-group ablation (Gradient Boosting)", y=1.03, fontsize=12)
    save(fig, "fig05_ablation_study.png")


def fig06_shap() -> None:
    """Copy existing SHAP plots into a 2-panel paper figure if available."""
    summary = ROOT / "results_xai" / "shap_summary.png"
    waterfalls = sorted((ROOT / "results_xai").glob("shap_waterfall_*.png"), key=lambda path: path.stat().st_mtime)
    if not summary.exists() or not waterfalls:
        print("SKIP fig06: SHAP source plots missing")
        return
    waterfall = waterfalls[-1]

    # Compose by embedding images
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].imshow(mpimg.imread(summary))
    axes[0].axis("off")
    axes[0].set_title("(a) Global SHAP summary", fontsize=11)
    axes[1].imshow(mpimg.imread(waterfall))
    axes[1].axis("off")
    axes[1].set_title("(b) Local SHAP waterfall (example student)", fontsize=11)
    fig.suptitle("Explainability (SHAP) for Gradient Boosting", y=1.02, fontsize=12)
    save(fig, "fig06_shap_global_and_local.png")


def fig07_threshold_operating_point() -> None:
    path = ROOT / "experiments" / "results" / "e6_threshold_scan.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    cutoff = load_operating_threshold()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(t["threshold"], t["precision"], label="Precision", color="#2e7d32")
    ax.plot(t["threshold"], t["recall"], label="Recall", color="#ef6c00")
    ax.axhline(0.90, color="#90a4ae", linestyle=":", label="Precision 0.90")
    ax.axvline(cutoff, color="#c62828", linestyle="--", label=f"Low-risk cutoff {cutoff:.2f}")
    ax.set_xlabel("Calibrated continuation probability")
    ax.set_ylabel("Score")
    ax.set_title("Chosen Gradient Boosting low-risk cutoff")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "fig07_operating_threshold.png")


def fig08_correlation_compact(df: pd.DataFrame) -> None:
    cols = [
        "Result",
        "Budget_Midpoint",
        "Phase_Number",
        "Budget_Level_Num",
        "Academic_Level_Num",
        "Researched_Level",
        "Budget_Academic_Score",
        "Days_Since_First_Intake",
        "Continuation_Bin",
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    short = [
        "GPA",
        "BudgetMid",
        "Phase#",
        "BudgetLvl",
        "AcadLvl",
        "Due_Diligence",
        "BudgAcad",
        "DaysIntake",
        "Continue",
    ]
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    ax.set_title("Correlation among key numeric features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "fig08_feature_correlation.png")


def write_readme() -> None:
    text = """Paper-ready figures (use these in the manuscript)

Folder: project_figures_paper/
Full exploratory set remains in: project_figures/ (archive / appendix only)

Recommended manuscript figures:
1. fig01_dataset_overview.png
2. fig02_key_predictive_drivers.png
3. fig03_model_performance_metrics_roc.png
4. fig04_confusion_matrices_gb_stack.png
5. fig05_ablation_study.png
6. fig06_shap_global_and_local.png
7. fig07_operating_threshold.png
8. fig08_feature_correlation.png  (optional / appendix)

Removed from the paper set (duplicates or low value for the narrative):
- Per-course / per-institution volume charts
- Duplicate ablation F1-drop redraws
- Separate legacy LR/RF metric/CM/ROC panels
- Duplicate Paper1/Paper2 overview bars already covered by fig03
- Many single-variable EDA plots (merged into fig01/fig02)
- Separate GB/XGB CM and ROC files (merged into fig03/fig04)
"""
    (OUT / "README_paper_figures.txt").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # clear previous paper pack only
    for old in OUT.glob("*.png"):
        old.unlink()

    df = load_engineered_data()
    _, X_test, _, y_test = split_engineered_data(df)
    gb = load_gradient_boosting_model()
    xgb = joblib.load(PAPER1_XGB_MODEL_PATH)
    gb_prob = gb.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    base_names = ["random_forest", "xgboost", "gradient_boosting"]
    base_probs = [
        joblib.load(ROOT / "results_paper2" / f"{name}.joblib").predict_proba(X_test)[:, 1]
        for name in base_names
    ]
    meta = joblib.load(ROOT / "results_paper2" / "stacking_fnn_meta_model.joblib")
    stack_prob = meta.predict_proba(np.column_stack(base_probs))[:, 1]
    paper1 = pd.read_csv(ROOT / "results_paper1" / "paper1_model_comparison.csv")
    xgb_f1_threshold = float(
        paper1.loc[paper1["Model"].str.contains("optimized"), "Threshold"].iloc[0]
    )
    gb_pred = (gb_prob >= 0.5).astype(int)
    stack_pred = (stack_prob >= 0.5).astype(int)

    fig01_dataset_overview(df)
    fig02_key_drivers(df)
    fig03_model_metrics(y_test, gb_prob, stack_prob, xgb_prob, xgb_f1_threshold)
    fig04_confusion_matrices(y_test, gb_pred, stack_pred)
    fig05_ablation()
    fig06_shap()
    fig07_threshold_operating_point()
    fig08_correlation_compact(df)
    write_readme()

    print(f"\nPaper figure pack: {OUT}")
    print(f"Count: {len(list(OUT.glob('*.png')))} PNGs")
    print("Full set kept in project_figures/ for appendix if needed.")


if __name__ == "__main__":
    main()
