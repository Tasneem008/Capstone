"""
Generate a full figure pack for the Capstone 6,000-record study.
All outputs land in project_figures/ with descriptive filenames.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from common import DUE_DILIGENCE_LEVEL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "project_figures"
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
    "Transitional phase 1st 5 months": "Trans. Phase 1",
    "Transitional phase Last 5 months": "Trans. Phase 2",
    "Newly Elected": "Newly Elected",
}

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def rate_by(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(col)["Continuation_Bin"].mean().sort_values(ascending=False)


def copy_existing() -> list[str]:
    mapping = {
        ROOT / "results_ablation" / "ablation_results.png": "19_ablation_f1_drop.png",
        ROOT / "results_xai" / "shap_summary.png": "20_shap_global_summary.png",
        ROOT
        / "Datasets"
        / "report_figures"
        / "class_balance.png": "01_class_balance_continuation.png",
        ROOT
        / "Datasets"
        / "report_figures"
        / "continuation_by_phase.png": "02_continuation_rate_by_political_phase.png",
        ROOT
        / "Datasets"
        / "report_figures"
        / "paper1_threshold_effect.png": "14_paper1_xgb_threshold_effect.png",
        ROOT
        / "Datasets"
        / "report_figures"
        / "paper2_metrics.png": "15_paper2_model_metrics_comparison.png",
    }
    copied = []
    for src, dst_name in mapping.items():
        if src.exists():
            shutil.copy2(src, OUT / dst_name)
            copied.append(dst_name)
    waterfalls = sorted(
        (ROOT / "results_xai").glob("shap_waterfall_*.png"),
        key=lambda path: path.stat().st_mtime,
    )
    if waterfalls:
        shutil.copy2(waterfalls[-1], OUT / "21_shap_local_waterfall_example.png")
        copied.append("21_shap_local_waterfall_example.png")
    return copied


def plot_class_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Continuation"].value_counts().reindex(["No", "Yes"])
    colors = ["#c62828", "#2e7d32"]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_title("Overall Continuation Class Balance (n=6000)")
    ax.set_ylabel("Count")
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 40,
            f"{value}\n({value / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save(fig, "01_class_balance_continuation.png")


def plot_continuation_by_phase(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rates = (
        df.groupby("Political_Phase")["Continuation_Bin"].mean().reindex(PHASE_ORDER)
        * 100
    )
    labels = [PHASE_SHORT[p] for p in PHASE_ORDER]
    bars = ax.bar(labels, rates.values, color="#1565c0")
    ax.set_ylabel("Continuation Rate (%)")
    ax.set_title("Continuation Rate by Political Phase")
    ax.set_ylim(0, max(rates.values) * 1.3)
    for bar, value in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontsize=9,
        )
    save(fig, "02_continuation_rate_by_political_phase.png")


def plot_counts_by_phase(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    counts = df["Political_Phase"].value_counts().reindex(PHASE_ORDER)
    labels = [PHASE_SHORT[p] for p in PHASE_ORDER]
    ax.bar(labels, counts.values, color="#455a64")
    ax.set_title("Sample Size by Political Phase (Balanced 1,200 each)")
    ax.set_ylabel("Records")
    save(fig, "03_sample_size_by_political_phase.png")


def plot_continuation_by_country(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates = rate_by(df, "Country") * 100
    ax.barh(rates.index[::-1], rates.values[::-1], color="#00838f")
    ax.set_xlabel("Continuation Rate (%)")
    ax.set_title("Continuation Rate by Destination Country")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    save(fig, "04_continuation_rate_by_country.png")


def plot_continuation_by_budget(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    order_budget = ["Below 10k", "10-15K", "16-20K"]
    order_level = ["Low", "Medium", "High"]
    r1 = df.groupby("Budget")["Continuation_Bin"].mean().reindex(order_budget) * 100
    r2 = (
        df.groupby("Budget_Level")["Continuation_Bin"].mean().reindex(order_level) * 100
    )
    axes[0].bar(order_budget, r1.values, color="#6a1b9a")
    axes[0].set_title("By Budget Band")
    axes[0].set_ylabel("Continuation Rate (%)")
    axes[1].bar(order_level, r2.values, color="#8e24aa")
    axes[1].set_title("By Budget Level")
    fig.suptitle("Continuation Rate by Financial Capacity", y=1.02)
    save(fig, "05_continuation_rate_by_budget.png")


def plot_continuation_by_researched(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    due_order = list(DUE_DILIGENCE_LEVEL)
    rates = df.groupby("Researched")["Continuation_Bin"].mean().reindex(due_order) * 100
    colors = ["#ef6c00", "#f9a825", "#66bb6a", "#1b5e20"]
    bars = ax.bar(due_order, rates.values, color=colors)
    ax.set_title("Continuation Rate by Due_Diligence")
    ax.set_ylabel("Continuation Rate (%)")
    ax.set_xlabel("Due_Diligence")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
        )
    save(fig, "06_continuation_rate_by_researched.png")


def plot_continuation_by_academic(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["Low", "Medium", "High"]
    rates = (
        df.groupby("Academic_Level")["Continuation_Bin"].mean().reindex(order) * 100
    )
    ax.bar(order, rates.values, color="#3949ab")
    ax.set_title("Continuation Rate by Academic Level")
    ax.set_ylabel("Continuation Rate (%)")
    save(fig, "07_continuation_rate_by_academic_level.png")


def plot_gpa_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(df["Result"], bins=30, color="#546e7a", edgecolor="white")
    axes[0].set_title("GPA / Result Distribution")
    axes[0].set_xlabel("Result")
    axes[0].set_ylabel("Count")
    for label, color in [("No", "#c62828"), ("Yes", "#2e7d32")]:
        subset = df.loc[df["Continuation"] == label, "Result"]
        axes[1].hist(subset, bins=25, alpha=0.55, label=label, color=color)
    axes[1].set_title("GPA by Continuation Outcome")
    axes[1].set_xlabel("Result")
    axes[1].legend(title="Continuation")
    save(fig, "08_gpa_distribution_overall_and_by_outcome.png")


def plot_result_band(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    order = ["Low", "Medium", "High"]
    counts = df["Result_Band"].astype(str).value_counts().reindex(order)
    rates = (
        df.groupby(df["Result_Band"].astype(str))["Continuation_Bin"]
        .mean()
        .reindex(order)
        * 100
    )
    axes[0].bar(order, counts.values, color="#00897b")
    axes[0].set_title("Count by Result Band")
    axes[1].bar(order, rates.values, color="#00695c")
    axes[1].set_title("Continuation Rate by Result Band")
    axes[1].set_ylabel("Continuation Rate (%)")
    save(fig, "09_result_band_counts_and_continuation.png")


def plot_top_courses(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    course_counts = df["Course"].value_counts().head(12)
    axes[0].barh(course_counts.index[::-1], course_counts.values[::-1], color="#5c6bc0")
    axes[0].set_title("Top 12 Courses by Volume")
    course_rate = (
        df.groupby("Course")
        .agg(n=("Continuation_Bin", "size"), rate=("Continuation_Bin", "mean"))
        .query("n >= 80")
        .sort_values("rate", ascending=False)
        .head(12)
    )
    axes[1].barh(
        course_rate.index[::-1],
        (course_rate["rate"] * 100).values[::-1],
        color="#7986cb",
    )
    axes[1].set_title("Highest Continuation Courses (n>=80)")
    axes[1].set_xlabel("Continuation Rate (%)")
    save(fig, "10_course_volume_and_continuation.png")


def plot_top_institutions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    inst_counts = df["Institution"].value_counts().head(12)
    axes[0].barh(inst_counts.index[::-1], inst_counts.values[::-1], color="#8d6e63")
    axes[0].set_title("Top 12 Institutions by Volume")
    inst_rate = (
        df.groupby("Institution")
        .agg(n=("Continuation_Bin", "size"), rate=("Continuation_Bin", "mean"))
        .query("n >= 80")
        .sort_values("rate", ascending=False)
        .head(12)
    )
    axes[1].barh(
        inst_rate.index[::-1],
        (inst_rate["rate"] * 100).values[::-1],
        color="#a1887f",
    )
    axes[1].set_title("Highest Continuation Institutions (n>=80)")
    axes[1].set_xlabel("Continuation Rate (%)")
    save(fig, "11_institution_volume_and_continuation.png")


def plot_season_and_month(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    season_rate = df.groupby("Season")["Continuation_Bin"].mean() * 100
    month_rate = df.groupby("Intake_Month")["Continuation_Bin"].mean() * 100
    axes[0].bar(season_rate.index.astype(str), season_rate.values, color="#0288d1")
    axes[0].set_title("Continuation by Season (1=Dec-Feb ... 4=Sep-Nov)")
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Continuation Rate (%)")
    axes[1].plot(month_rate.index, month_rate.values, marker="o", color="#01579b")
    axes[1].set_title("Continuation by Intake Month")
    axes[1].set_xlabel("Month")
    axes[1].set_xticks(range(1, 13))
    save(fig, "12_continuation_by_season_and_month.png")


def plot_heatmap_phase_budget(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot = (
        df.pivot_table(
            index="Political_Phase",
            columns="Budget_Level",
            values="Continuation_Bin",
            aggfunc="mean",
        )
        .reindex(index=PHASE_ORDER, columns=["Low", "Medium", "High"])
        * 100
    )
    pivot.index = [PHASE_SHORT[i] for i in pivot.index]
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Continuation Rate (%) by Phase x Budget Level")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "13_heatmap_continuation_phase_by_budget_level.png")


def plot_heatmap_researched_phase(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
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
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Continuation Rate (%) by Phase x Due_Diligence")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "22_heatmap_continuation_phase_by_researched.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = [
        "Result",
        "Intake_Month",
        "Year",
        "Season",
        "Days_Since_First_Intake",
        "Budget_Midpoint",
        "Phase_Number",
        "Budget_Level_Num",
        "Academic_Level_Num",
        "Researched_Level",
        "Budget_Academic_Score",
        "GPA_Academic_Match",
        "Continuation_Bin",
    ]
    corr = df[cols].corr()
    display_cols = [c.replace("Researched_Level", "Due_Diligence") for c in cols]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(display_cols, rotation=75, ha="right")
    ax.set_yticklabels(display_cols)
    ax.set_title("Numeric Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "23_correlation_heatmap_numeric_features.png")


def plot_budget_academic_score(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    counts = df["Budget_Academic_Score"].value_counts().sort_index()
    rates = df.groupby("Budget_Academic_Score")["Continuation_Bin"].mean() * 100
    axes[0].bar(counts.index.astype(str), counts.values, color="#6d4c41")
    axes[0].set_title("Budget-Academic Score Counts")
    axes[0].set_xlabel("Budget_Academic_Score")
    axes[1].plot(rates.index, rates.values, marker="o", color="#4e342e")
    axes[1].set_title("Continuation vs Budget-Academic Score")
    axes[1].set_xlabel("Budget_Academic_Score")
    axes[1].set_ylabel("Continuation Rate (%)")
    save(fig, "24_budget_academic_score_vs_continuation.png")


def plot_gpa_match(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    rates = (
        df.groupby("GPA_Academic_Match")["Continuation_Bin"].mean().reindex([0, 1])
        * 100
    )
    ax.bar(["Mismatch (0)", "Match (1)"], rates.values, color=["#f57c00", "#43a047"])
    ax.set_title("Continuation by GPA-Academic Match")
    ax.set_ylabel("Continuation Rate (%)")
    save(fig, "25_continuation_by_gpa_academic_match.png")


def plot_days_since_intake(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, color in [("No", "#c62828"), ("Yes", "#2e7d32")]:
        subset = df.loc[df["Continuation"] == label, "Days_Since_First_Intake"]
        ax.hist(subset, bins=30, alpha=0.5, label=label, color=color)
    ax.set_title("Days Since First Intake by Continuation")
    ax.set_xlabel("Days_Since_First_Intake")
    ax.set_ylabel("Count")
    ax.legend(title="Continuation")
    save(fig, "26_days_since_first_intake_by_continuation.png")


def plot_stacked_phase_outcome(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ct = (
        pd.crosstab(df["Political_Phase"], df["Continuation"])
        .reindex(PHASE_ORDER)[["No", "Yes"]]
    )
    ct.index = [PHASE_SHORT[i] for i in ct.index]
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#c62828", "#2e7d32"])
    ax.set_title("Continuation Counts Stacked by Political Phase")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend(title="Continuation")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "27_stacked_counts_phase_by_continuation.png")


def plot_box_gpa_by_phase(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [df.loc[df["Political_Phase"] == p, "Result"].values for p in PHASE_ORDER]
    ax.boxplot(data, tick_labels=[PHASE_SHORT[p] for p in PHASE_ORDER], showfliers=False)
    ax.set_title("GPA Distribution by Political Phase")
    ax.set_ylabel("Result (GPA)")
    ax.tick_params(axis="x", rotation=15)
    save(fig, "28_boxplot_gpa_by_political_phase.png")


def plot_paper1_metrics() -> None:
    path = ROOT / "results_paper1" / "paper1_model_comparison.csv"
    if not path.exists():
        return
    p1 = pd.read_csv(path)
    # Keep one row per model family for clarity
    keep = [
        "Logistic Regression",
        "Random Forest",
        "LightGBM",
        "XGBoost (Tuned)",
        "XGBoost (Tuned, optimized threshold)",
    ]
    subset = p1[p1["Model"].isin(keep)].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(subset))
    width = 0.18
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        ax.bar(x + (i - 1.5) * width, subset[metric], width=width, label=metric)
    labels = []
    for _, row in subset.iterrows():
        if "optimized" in str(row["Model"]):
            labels.append(f"XGB@{row['Threshold']:.2f}")
        elif row["Model"] == "XGBoost (Tuned)":
            labels.append("XGB@0.50")
        elif row["Model"] == "Logistic Regression":
            labels.append("LR")
        elif row["Model"] == "Random Forest":
            labels.append("RF")
        elif row["Model"] == "LightGBM":
            labels.append("LGBM")
        else:
            labels.append(str(row["Model"])[:12])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Paper 1 Methodology Model Comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "29_paper1_model_metrics_comparison.png")


def plot_threshold_search() -> None:
    path = ROOT / "results_paper1" / "xgb_threshold_search.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t["Threshold"], t["F1"], label="F1", color="#1565c0")
    ax.plot(t["Threshold"], t["Precision"], label="Precision", color="#2e7d32")
    ax.plot(t["Threshold"], t["Recall"], label="Recall", color="#ef6c00")
    best = t.loc[t["F1"].idxmax()]
    ax.axvline(best["Threshold"], color="red", linestyle="--", label=f"Best={best['Threshold']:.2f}")
    ax.set_title("Paper 1 XGBoost threshold search (training out-of-fold)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "30_paper1_xgb_threshold_search_curves.png")
    # Replace the stale copy of the old 0.73 chart.
    src = OUT / "30_paper1_xgb_threshold_search_curves.png"
    if src.exists():
        shutil.copy2(src, OUT / "14_paper1_xgb_threshold_effect.png")


def plot_ablation_from_csv() -> None:
    path = ROOT / "results_ablation" / "ablation_results.csv"
    if not path.exists():
        return
    ab = pd.read_csv(path)
    plot_data = ab.loc[ab["Removed_Group"].notna()].copy()
    if plot_data.empty:
        plot_data = ab.loc[ab["Experiment"] != "Baseline (all features)"].copy()
    plot_data = plot_data.sort_values("F1_Drop", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_data["Removed_Group"], plot_data["F1_Drop"], color="#c62828")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("F1-score drop from baseline")
    ax.set_title("Feature-Group Ablation: F1 Impact")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "31_ablation_f1_drop_from_csv.png")


def plot_feature_importance_legacy() -> None:
    model_path = ROOT / "results_paper2" / "random_forest.joblib"
    if not model_path.exists():
        return
    model = joblib.load(model_path)
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = model.named_steps["model"].feature_importances_
    imp = (
        pd.DataFrame({"Feature": names, "Importance": values})
        .sort_values("Importance", ascending=True)
        .tail(15)
    )
    imp["Feature"] = (
        imp["Feature"]
        .astype(str)
        .str.replace(r"^num__|^cat__", "", regex=True)
        .str.replace("Researched", "Due_Diligence")
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(imp["Feature"], imp["Importance"], color="#37474f")
    ax.set_title("Random Forest Feature Importance (Top 15, revised 6,000-record model)")
    ax.set_xlabel("Importance")
    save(fig, "32_legacy_rf_feature_importance.png")


def plot_pairwise_key_drivers(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    due_order = list(DUE_DILIGENCE_LEVEL)
    due_colors = ["#ef6c00", "#f9a825", "#66bb6a", "#1b5e20"]
    pivot = (
        df.pivot_table(
            index="Budget_Level",
            columns="Researched",
            values="Continuation_Bin",
            aggfunc="mean",
        )
        .reindex(index=["Low", "Medium", "High"], columns=due_order)
        * 100
    )
    x = np.arange(len(pivot.index))
    width = 0.18
    for i, level in enumerate(due_order):
        axes[0].bar(
            x + (i - 1.5) * width,
            pivot[level],
            width=width,
            label=level,
            color=due_colors[i],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pivot.index)
    axes[0].set_ylabel("Continuation Rate (%)")
    axes[0].set_title("Due_Diligence x Budget Level")
    axes[0].legend(fontsize=7)

    phase_rates = (
        df.groupby(["Political_Phase", "Researched"])["Continuation_Bin"]
        .mean()
        .unstack()
        .reindex(PHASE_ORDER)[due_order]
        * 100
    )
    x = np.arange(len(PHASE_ORDER))
    for i, level in enumerate(due_order):
        axes[1].bar(
            x + (i - 1.5) * width,
            phase_rates[level],
            width=width,
            label=level,
            color=due_colors[i],
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([PHASE_SHORT[p] for p in PHASE_ORDER], rotation=20)
    axes[1].set_title("Due_Diligence x Political Phase")
    axes[1].legend(fontsize=7)
    save(fig, "33_interaction_researched_with_budget_and_phase.png")


def plot_missingness_and_quality(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    nulls = df.isna().sum()
    axes[0].bar(range(len(nulls)), nulls.values, color="#607d8b")
    axes[0].set_title("Missing Values by Column (Should be ~0)")
    axes[0].set_ylabel("Null count")
    axes[0].set_xticks([])
    dupes = pd.Series(
        {
            "Exact duplicate rows": int(df.duplicated().sum()),
            "Duplicate Name": int(df["Name"].duplicated().sum()),
            "Total rows": len(df),
        }
    )
    axes[1].bar(dupes.index.astype(str), dupes.values, color="#455a64")
    axes[1].set_title("Data Quality Snapshot")
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(dupes.values):
        axes[1].text(i, v + max(dupes.values) * 0.01, str(v), ha="center", fontsize=8)
    save(fig, "34_data_quality_missing_and_duplicates.png")


def write_index(copied: list[str]) -> None:
    files = sorted(OUT.glob("*.png"))
    rows = []
    for path in files:
        rows.append(
            {
                "filename": path.name,
                "size_kb": round(path.stat().st_size / 1024, 1),
                "source": "copied" if path.name in copied else "generated",
            }
        )
    index = pd.DataFrame(rows)
    index.to_csv(OUT / "00_figure_index.csv", index=False)

    lines = [
        "Capstone Project Figures Index",
        f"Folder: {OUT}",
        f"Total PNG files: {len(files)}",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['filename']}  ({row['size_kb']} KB, {row['source']})")
    (OUT / "00_figure_index.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)

    copied = copy_existing()

    plot_class_balance(df)
    plot_continuation_by_phase(df)
    plot_counts_by_phase(df)
    plot_continuation_by_country(df)
    plot_continuation_by_budget(df)
    plot_continuation_by_researched(df)
    plot_continuation_by_academic(df)
    plot_gpa_distribution(df)
    plot_result_band(df)
    plot_top_courses(df)
    plot_top_institutions(df)
    plot_season_and_month(df)
    plot_heatmap_phase_budget(df)
    plot_heatmap_researched_phase(df)
    plot_correlation_heatmap(df)
    plot_budget_academic_score(df)
    plot_gpa_match(df)
    plot_days_since_intake(df)
    plot_stacked_phase_outcome(df)
    plot_box_gpa_by_phase(df)
    plot_paper1_metrics()
    plot_threshold_search()
    plot_ablation_from_csv()
    plot_feature_importance_legacy()
    plot_pairwise_key_drivers(df)
    plot_missingness_and_quality(df)

    p2_path = ROOT / "results_paper2" / "paper2_model_comparison.csv"
    if p2_path.exists():
        p2 = pd.read_csv(p2_path)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels = ["RF", "XGBoost", "GB", "Stack"][: len(p2)]
        x = np.arange(len(p2))
        w = 0.2
        for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
            ax.bar(x + (i - 1.5) * w, p2[metric], width=w, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.6, 1.0)
        ax.set_title("Paper 2 model comparison (held-out test)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        save(fig, "15_paper2_model_metrics_comparison.png")

    write_index(copied)
    print(f"Wrote {len(list(OUT.glob('*.png')))} PNG files to {OUT}")
    print(f"Index: {OUT / '00_figure_index.txt'}")


if __name__ == "__main__":
    main()
