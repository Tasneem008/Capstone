"""
Convert remaining Capstone result tables into charts.

Skips anything already covered in project_figures/ (ablation F1-drop,
Paper 1/2 overview bars, threshold curves, CMs, ROCs, RF importance).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from common import load_engineered_data, load_gradient_boosting_model, split_engineered_data
import joblib
from common import PAPER1_XGB_MODEL_PATH

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "project_figures"
OUT.mkdir(parents=True, exist_ok=True)

EXISTING = {p.name for p in OUT.glob("*.png")}


def save_if_new(fig: plt.Figure, name: str) -> bool:
    if name in EXISTING:
        plt.close(fig)
        print(f"SKIP (exists): {name}")
        return False
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    EXISTING.add(name)
    print(f"WROTE: {name}")
    return True


def chart_legacy_model_comparison_table() -> None:
    """Datasets/model_comparison_results.csv -> multi-metric bars."""
    name = "42_legacy_lr_rf_full_metrics_from_table.png"
    path = ROOT / "Datasets" / "model_comparison_results.csv"
    if not path.exists() or name in EXISTING:
        if name in EXISTING:
            print(f"SKIP (exists): {name}")
        return

    df = pd.read_csv(path)
    metric_cols = [
        c
        for c in [
            "Validation Accuracy",
            "Test Accuracy",
            "5-Fold CV Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
        ]
        if c in df.columns
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(df))
    width = 0.11
    for i, metric in enumerate(metric_cols):
        ax.bar(x + (i - (len(metric_cols) - 1) / 2) * width, df[metric], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"])
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Legacy LR vs RF Metrics (6,000-record pipeline table)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    save_if_new(fig, name)


def chart_ablation_full_metrics() -> None:
    """Ablation CSV Acc/Prec/Rec/F1 (not F1_Drop, which already exists)."""
    name = "43_ablation_accuracy_precision_recall_f1_by_group.png"
    path = ROOT / "results_ablation" / "ablation_results.csv"
    if not path.exists() or name in EXISTING:
        if name in EXISTING:
            print(f"SKIP (exists): {name}")
        return

    ab = pd.read_csv(path).copy()
    labels = []
    for _, row in ab.iterrows():
        if pd.isna(row.get("Removed_Group")) or str(row.get("Removed_Group")) in {
            "None",
            "nan",
        }:
            labels.append("Baseline")
        else:
            labels.append(str(row["Removed_Group"]).replace("_Features", ""))

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ab))
    width = 0.2
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(x + (i - 1.5) * width, ab[metric], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Ablation Study: Accuracy / Precision / Recall / F1 by Feature Group")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_if_new(fig, name)


def chart_ablation_absolute_f1() -> None:
    """Absolute F1 bars (distinct from existing F1_Drop charts)."""
    name = "44_ablation_absolute_f1_scores.png"
    path = ROOT / "results_ablation" / "ablation_results.csv"
    if not path.exists() or name in EXISTING:
        if name in EXISTING:
            print(f"SKIP (exists): {name}")
        return

    ab = pd.read_csv(path).copy()
    labels = []
    for _, row in ab.iterrows():
        if pd.isna(row.get("Removed_Group")) or str(row.get("Removed_Group")) in {
            "None",
            "nan",
        }:
            labels.append("Baseline")
        else:
            labels.append("w/o " + str(row["Removed_Group"]).replace("_Features", ""))

    colors = ["#2e7d32" if lab == "Baseline" else "#c62828" for lab in labels]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, ab["F1"], color=colors)
    ax.set_ylabel("F1 Score")
    ax.set_title("Ablation Study: Absolute F1 Score by Experiment")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, ab["F1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )
    ax.grid(axis="y", alpha=0.3)
    save_if_new(fig, name)


def chart_6000_production_model_metrics() -> None:
    """
    Metrics table for GB / XGB on the 6K held-out set.
    Skips pure Acc/AUC-only duplication by including Precision/Recall/F1 too.
    """
    name = "45_6000_gb_xgboost_precision_recall_f1_accuracy.png"
    if name in EXISTING:
        print(f"SKIP (exists): {name}")
        return

    df = load_engineered_data()
    _, X_test, _, y_test = split_engineered_data(df)
    gb = load_gradient_boosting_model()
    xgb = joblib.load(PAPER1_XGB_MODEL_PATH)

    configs = [
        ("Gradient Boosting\n@0.50", gb.predict_proba(X_test)[:, 1], 0.50),
        ("XGBoost\n@0.50", xgb.predict_proba(X_test)[:, 1], 0.50),
        ("XGBoost\n@0.73", xgb.predict_proba(X_test)[:, 1], 0.73),
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
            }
        )
    metrics = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(metrics))
    width = 0.18
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(x + (i - 1.5) * width, metrics[metric], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["Model"])
    ax.set_ylim(0.5, 1.0)
    ax.set_title("6,000-Record Test Metrics: Gradient Boosting vs XGBoost")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_if_new(fig, name)
    metrics.to_csv(OUT / "00_6000_gb_xgboost_metrics_table.csv", index=False)


def chart_threshold_precision_recall_tradeoff_table() -> None:
    """
    Compact view of threshold search best region.
    Full curves already exist as 30_*; this is a top-N bar from the table.
    """
    name = "46_xgb_threshold_top10_f1_from_search_table.png"
    path = ROOT / "results_paper1" / "xgb_threshold_search.csv"
    if not path.exists() or name in EXISTING:
        if name in EXISTING:
            print(f"SKIP (exists): {name}")
        return

    t = pd.read_csv(path).sort_values("F1", ascending=False).head(10).sort_values("Threshold")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        [f"{v:.2f}" for v in t["Threshold"]],
        t["F1"],
        color="#6a1b9a",
        label="F1",
    )
    ax.plot(
        [f"{v:.2f}" for v in t["Threshold"]],
        t["Precision"],
        marker="o",
        color="#2e7d32",
        label="Precision",
    )
    ax.plot(
        [f"{v:.2f}" for v in t["Threshold"]],
        t["Recall"],
        marker="s",
        color="#ef6c00",
        label="Recall",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Top-10 F1 Thresholds from XGBoost Search Table")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_if_new(fig, name)


def refresh_index() -> None:
    files = sorted(OUT.glob("*.png"))
    rows = [{"filename": p.name, "size_kb": round(p.stat().st_size / 1024, 1)} for p in files]
    pd.DataFrame(rows).to_csv(OUT / "00_figure_index.csv", index=False)
    lines = [
        "Capstone Project Figures Index",
        f"Folder: {OUT}",
        f"Total PNG files: {len(files)}",
        "",
    ]
    lines.extend(f"- {r['filename']}  ({r['size_kb']} KB)" for r in rows)
    (OUT / "00_figure_index.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    chart_legacy_model_comparison_table()
    chart_ablation_full_metrics()
    chart_ablation_absolute_f1()
    chart_6000_production_model_metrics()
    chart_threshold_precision_recall_tradeoff_table()
    refresh_index()
    print(f"\nDone. Total PNGs now: {len(list(OUT.glob('*.png')))}")


if __name__ == "__main__":
    main()
