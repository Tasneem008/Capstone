"""
Build a unified comparison of Paper 1, Paper 2, and this project's models.

Outputs:
  - results_comparison/final_methodology_comparison.csv
  - results_comparison/methodology_summary.csv
  - results_comparison/chart01_best_model_head_to_head.png
  - results_comparison/chart02_paper1_all_models.png
  - results_comparison/chart03_paper2_all_models.png
  - results_comparison/chart04_project_vs_papers.png
  - results_comparison/chart05_f1_leaderboard.png
  - results_comparison/chart06_precision_recall_tradeoff.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# Consistent palette
COLOR_PAPER1 = "#1565c0"
COLOR_PAPER2 = "#ef6c00"
COLOR_PROJECT = "#2e7d32"
COLOR_STACK = "#6a1b9a"
METRIC_COLORS = ["#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

METRIC_COLS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "MCC",
    "Average_Precision",
    "TN",
    "FP",
    "FN",
    "TP",
]


def round_metrics(df: pd.DataFrame) -> pd.DataFrame:
  out = df.copy()
  for col in METRIC_COLS:
    if col in out.columns:
      out[col] = out[col].astype(float).round(4)
  if "Threshold" in out.columns:
    out["Threshold"] = out["Threshold"].astype(float).round(2)
  return out


def load_paper1() -> pd.DataFrame:
  path = ROOT / "results_paper1" / "paper1_model_comparison.csv"
  df = pd.read_csv(path)
  df = df.rename(columns={"Model": "Method"})
  df["Methodology"] = "Paper 1 (Carballo-Mendívil et al., 2025)"
  df["Role"] = "Comparator"
  df.loc[df["Method"].str.contains("optimized", case=False), "Role"] = (
      "Paper 1 best (threshold optimized)"
  )
  return df


def load_paper2() -> pd.DataFrame:
  path = ROOT / "results_paper2" / "paper2_model_comparison.csv"
  df = pd.read_csv(path)
  df = df.rename(columns={"Model": "Method"})
  df["Methodology"] = "Paper 2 (Niyogisubizo et al., 2022)"
  df["Role"] = "Comparator"
  df.loc[df["Method"] == "Gradient Boosting", "Role"] = "Paper 2 best (base learner)"
  df.loc[df["Method"].str.contains("Stacking", case=False), "Role"] = (
      "Paper 2 ensemble (stacking)"
  )
  if "Threshold" not in df.columns:
    df["Threshold"] = 0.50
  return df


def load_legacy() -> pd.DataFrame:
  path = ROOT / "Datasets" / "model_comparison_results.csv"
  if not path.exists():
    return pd.DataFrame()
  df = pd.read_csv(path)
  return pd.DataFrame(
      {
          "Methodology": "Legacy pipeline (label-encoded CSV)",
          "Method": df["Model"],
          "Threshold": 0.50,
          "Accuracy": df["Test Accuracy"],
          "Precision": df["Precision"],
          "Recall": df["Recall"],
          "F1": df["F1 Score"],
          "ROC_AUC": df["ROC-AUC"],
          "Role": "Early baseline",
      }
  )


def load_project_selection() -> pd.DataFrame:
  """Rows for models adopted in the DSS / capstone deliverable."""
  gb = pd.read_csv(ROOT / "results_paper2" / "paper2_model_comparison.csv")
  gb_row = gb[gb["Model"] == "Gradient Boosting"].iloc[0]

  p1 = pd.read_csv(ROOT / "results_paper1" / "paper1_model_comparison.csv")
  xgb_opt = p1[
      p1["Model"].str.contains("optimized threshold", case=False)
  ].iloc[0]

  rows = [
      {
          "Methodology": "This project (SARP-Net DSS)",
          "Method": "Gradient Boosting (Paper 2 pipeline)",
          "Threshold": 0.50,
          "Accuracy": gb_row["Accuracy"],
          "Precision": gb_row["Precision"],
          "Recall": gb_row["Recall"],
          "F1": gb_row["F1"],
          "ROC_AUC": gb_row["ROC_AUC"],
          "MCC": gb_row["MCC"],
          "TN": gb_row["TN"],
          "FP": gb_row["FP"],
          "FN": gb_row["FN"],
          "TP": gb_row["TP"],
          "Role": "Primary inference model (probability + SHAP)",
      },
      {
          "Methodology": "This project (SARP-Net DSS)",
          "Method": "XGBoost tuned (Paper 1 pipeline)",
          "Threshold": xgb_opt["Threshold"],
          "Accuracy": xgb_opt["Accuracy"],
          "Precision": xgb_opt["Precision"],
          "Recall": xgb_opt["Recall"],
          "F1": xgb_opt["F1"],
          "ROC_AUC": xgb_opt["ROC_AUC"],
          "MCC": xgb_opt["MCC"],
          "TN": xgb_opt["TN"],
          "FP": xgb_opt["FP"],
          "FN": xgb_opt["FN"],
          "TP": xgb_opt["TP"],
          "Role": "Paper 1 XGBoost at its F1-optimal threshold",
      },
  ]
  return pd.DataFrame(rows)


def build_full_table() -> pd.DataFrame:
  parts = [load_paper1(), load_paper2(), load_legacy(), load_project_selection()]
  combined = pd.concat([p for p in parts if not p.empty], ignore_index=True)

  display_cols = [
      "Methodology",
      "Method",
      "Role",
      "Threshold",
      "Accuracy",
      "Precision",
      "Recall",
      "F1",
      "ROC_AUC",
      "MCC",
      "Average_Precision",
      "TN",
      "FP",
      "FN",
      "TP",
  ]
  for col in display_cols:
    if col not in combined.columns:
      combined[col] = np.nan

  combined = combined[display_cols]
  combined = round_metrics(combined)
  combined = combined.sort_values(["F1", "ROC_AUC"], ascending=False, na_position="last")
  return combined


def build_summary(full: pd.DataFrame) -> pd.DataFrame:
  """One highlighted row per methodology (best F1 within each)."""
  summaries = []

  paper1 = full[full["Methodology"].str.startswith("Paper 1")]
  if not paper1.empty:
    best = paper1.sort_values("F1", ascending=False).iloc[0]
    summaries.append(best)

  paper2 = full[full["Methodology"].str.startswith("Paper 2")]
  if not paper2.empty:
    best = paper2.sort_values("F1", ascending=False).iloc[0]
    summaries.append(best)

  legacy = full[full["Methodology"].str.startswith("Legacy")]
  if not legacy.empty:
    best = legacy.sort_values("F1", ascending=False).iloc[0]
    summaries.append(best)

  project = full[full["Methodology"].str.startswith("This project")]
  for role in project["Role"].unique():
    row = project[project["Role"] == role].iloc[0]
    summaries.append(row)

  return pd.DataFrame(summaries).reset_index(drop=True)


def get_head_to_head_rows(full: pd.DataFrame) -> pd.DataFrame:
  """Three representative rows: Paper 1 best, Paper 2 best, Our project primary."""
  p1 = full[full["Methodology"].str.startswith("Paper 1")].sort_values("F1", ascending=False).iloc[0]
  p2 = full[full["Methodology"].str.startswith("Paper 2")].sort_values("F1", ascending=False).iloc[0]
  proj = full[
      (full["Methodology"].str.startswith("This project"))
      & (full["Role"].str.contains("Primary", na=False))
  ].iloc[0]

  return pd.DataFrame(
      [
          {
              **p1.to_dict(),
              "ShortLabel": f"Paper 1\n(XGB @ {p1['Threshold']:.2f})",
              "Group": "Paper 1",
          },
          {
              **p2.to_dict(),
              "ShortLabel": "Paper 2\n(Stack)",
              "Group": "Paper 2",
          },
          {
              **proj.to_dict(),
              "ShortLabel": "Our Project\n(SARP-Net DSS)",
              "Group": "Our Project",
          },
      ]
  )


def _save(fig: plt.Figure, name: str) -> None:
  fig.tight_layout()
  fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"Wrote {OUT / name}")


def chart_best_model_head_to_head(head: pd.DataFrame) -> None:
  """Grouped bars: Paper 1 vs Paper 2 vs Our Project across all metrics."""
  fig, ax = plt.subplots(figsize=(10, 5.5))
  x = np.arange(len(METRICS))
  width = 0.25
  group_colors = [COLOR_PAPER1, COLOR_PAPER2, COLOR_PROJECT]

  for i, (_, row) in enumerate(head.iterrows()):
    values = [row[m] for m in METRICS]
    bars = ax.bar(x + (i - 1) * width, values, width=width, label=row["Group"], color=group_colors[i])
    for bar, val in zip(bars, values):
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + 0.008,
          f"{val:.3f}",
          ha="center",
          va="bottom",
          fontsize=7,
          rotation=90,
      )

  ax.set_xticks(x)
  ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
  ax.set_ylim(0.68, 1.02)
  ax.set_ylabel("Score (held-out test set)")
  ax.set_title("Best Model per Methodology — Metric Comparison")
  ax.legend(loc="lower right")
  ax.grid(axis="y", alpha=0.3)
  _save(fig, "chart01_best_model_head_to_head.png")


def chart_paper_models(paper_df: pd.DataFrame, title: str, filename: str, highlight: str | None = None) -> None:
  """Multi-metric bars for every model in one paper's results."""
  df = paper_df.copy()
  if "XGBoost (Tuned, threshold=0.50)" in df["Method"].values:
    df = df[df["Method"] != "XGBoost (Tuned, threshold=0.50)"]

  labels = []
  for method in df["Method"]:
    short = (
        method.replace("Logistic Regression", "LR")
        .replace("Random Forest", "RF")
        .replace("LightGBM", "LGBM")
        .replace("XGBoost (Tuned, optimized threshold)", "XGB@F1")
        .replace("XGBoost (Tuned)", "XGB@0.50")
        .replace("Gradient Boosting", "GB")
        .replace("Paper 2 Stacking (RF + XGBoost + GB -> FNN)", "Stacking")
    )
    labels.append(short)

  fig, ax = plt.subplots(figsize=(11, 5))
  x = np.arange(len(df))
  width = 0.18

  for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
    ax.bar(x + (i - 1.5) * width, df[metric], width=width, label=metric, color=METRIC_COLORS[i])

  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation=15, ha="right")
  ax.set_ylim(0.5, 1.0)
  ax.set_ylabel("Score")
  ax.set_title(title)
  ax.legend(ncol=4, fontsize=8, loc="upper left")
  ax.grid(axis="y", alpha=0.3)

  if highlight:
    idx = list(df["Method"]).index(highlight) if highlight in df["Method"].values else None
    if idx is not None:
      ax.axvspan(idx - 0.45, idx + 0.45, alpha=0.12, color=COLOR_PROJECT, zorder=0)

  _save(fig, filename)


def chart_project_vs_papers(head: pd.DataFrame) -> None:
  """Two-panel: radar profile + Accuracy vs F1 head-to-head."""
  fig = plt.figure(figsize=(12, 5))
  ax_radar = fig.add_subplot(1, 2, 1, projection="polar")
  ax_bar = fig.add_subplot(1, 2, 2)

  angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
  angles += angles[:1]
  tick_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

  for i, (_, row) in enumerate(head.iterrows()):
    vals = [row[m] for m in METRICS]
    vals += vals[:1]
    color = [COLOR_PAPER1, COLOR_PAPER2, COLOR_PROJECT][i]
    ax_radar.plot(angles, vals, "o-", linewidth=2, label=row["Group"], color=color)
    ax_radar.fill(angles, vals, alpha=0.08, color=color)

  ax_radar.set_xticks(angles[:-1])
  ax_radar.set_xticklabels(tick_labels, fontsize=9)
  ax_radar.set_ylim(0.68, 1.0)
  ax_radar.set_title("Metric Profile (Best per Methodology)", pad=16)
  ax_radar.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=8)

  labels = ["Paper 1", "Paper 2", "Our Project"]
  x = np.arange(3)
  w = 0.35
  acc = head["Accuracy"].values
  f1 = head["F1"].values
  b1 = ax_bar.bar(x - w / 2, acc, w, label="Accuracy", color="#1565c0", alpha=0.85)
  b2 = ax_bar.bar(x + w / 2, f1, w, label="F1", color="#2e7d32", alpha=0.85)
  ax_bar.set_xticks(x)
  ax_bar.set_xticklabels(labels)
  ax_bar.set_ylim(0.75, 0.95)
  ax_bar.set_ylabel("Score")
  ax_bar.set_title("Accuracy vs F1 — Head-to-Head")
  ax_bar.legend()
  ax_bar.grid(axis="y", alpha=0.3)
  for bars in (b1, b2):
    for bar in bars:
      ax_bar.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + 0.004,
          f"{bar.get_height():.3f}",
          ha="center",
          va="bottom",
          fontsize=8,
      )

  fig.suptitle("Our Project vs Paper 1 & Paper 2 (6,000-record test set)", y=1.02, fontsize=12)
  _save(fig, "chart04_project_vs_papers.png")


def chart_f1_leaderboard(full: pd.DataFrame) -> None:
  """Horizontal F1 bar chart, all models, colored by methodology."""
  plot_df = full.copy()
  plot_df = plot_df.sort_values("F1", ascending=True)

  def group_color(row):
    if row["Methodology"].startswith("This project"):
      return COLOR_PROJECT
    if row["Methodology"].startswith("Paper 1"):
      return COLOR_PAPER1
    if row["Methodology"].startswith("Paper 2"):
      return COLOR_PAPER2
    return "#757575"

  colors = [group_color(r) for _, r in plot_df.iterrows()]
  labels = [
      f"{r['Method'][:28]}{'…' if len(str(r['Method'])) > 28 else ''}"
      for _, r in plot_df.iterrows()
  ]

  fig, ax = plt.subplots(figsize=(10, 7))
  bars = ax.barh(labels, plot_df["F1"], color=colors, edgecolor="white", linewidth=0.5)
  for bar, val in zip(bars, plot_df["F1"]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)

  ax.set_xlim(0.65, 0.85)
  ax.set_xlabel("F1 Score")
  ax.set_title("F1 Score Leaderboard — All Models")
  ax.legend(
      handles=[
          Patch(color=COLOR_PAPER1, label="Paper 1 methodology"),
          Patch(color=COLOR_PAPER2, label="Paper 2 methodology"),
          Patch(color=COLOR_PROJECT, label="Our project (SARP-Net)"),
          Patch(color="#757575", label="Legacy baseline"),
      ],
      loc="lower right",
      fontsize=8,
  )
  ax.grid(axis="x", alpha=0.3)
  _save(fig, "chart05_f1_leaderboard.png")


def chart_precision_recall_tradeoff(full: pd.DataFrame) -> None:
  """Scatter: precision vs recall, sized by F1, colored by source."""
  fig, ax = plt.subplots(figsize=(8, 6))

  groups = [
      ("Paper 1", COLOR_PAPER1, full[full["Methodology"].str.startswith("Paper 1")]),
      ("Paper 2", COLOR_PAPER2, full[full["Methodology"].str.startswith("Paper 2")]),
      ("Our Project", COLOR_PROJECT, full[full["Methodology"].str.startswith("This project")]),
  ]

  for name, color, subset in groups:
    ax.scatter(
        subset["Recall"],
        subset["Precision"],
        s=subset["F1"] * 400,
        c=color,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.8,
        label=name,
        zorder=3,
    )
    for _, row in subset.iterrows():
      short = (
          str(row["Method"])
          .replace("XGBoost (Tuned, optimized threshold)", "XGB@F1")
          .replace("Gradient Boosting (Paper 2 pipeline)", "Our GB")
          .replace("Gradient Boosting", "GB")
          .replace("Paper 2 Stacking (RF + XGBoost + GB -> FNN)", "Stack")
      )
      if len(short) > 12:
        short = short[:10] + "…"
      ax.annotate(
          short,
          (row["Recall"], row["Precision"]),
          textcoords="offset points",
          xytext=(4, 4),
          fontsize=7,
          color=color,
      )

  ax.set_xlabel("Recall")
  ax.set_ylabel("Precision")
  ax.set_xlim(0.68, 0.88)
  ax.set_ylim(0.55, 0.92)
  ax.set_title("Precision–Recall Trade-off (bubble size = F1)")
  ax.legend(loc="lower left")
  ax.grid(alpha=0.3)
  _save(fig, "chart06_precision_recall_tradeoff.png")


def plot_all_charts(full: pd.DataFrame, summary: pd.DataFrame) -> None:
  head = get_head_to_head_rows(full)
  chart_best_model_head_to_head(head)

  p1 = full[full["Methodology"].str.startswith("Paper 1")].copy()
  chart_paper_models(
      p1,
      "Paper 1 Methodology — All Models (Carballo-Mendívil et al., 2025)",
      "chart02_paper1_all_models.png",
      highlight="XGBoost (Tuned, optimized threshold)",
  )

  p2 = full[full["Methodology"].str.startswith("Paper 2")].copy()
  chart_paper_models(
      p2,
      "Paper 2 Methodology — All Models (Niyogisubizo et al., 2022)",
      "chart03_paper2_all_models.png",
      highlight="Paper 2 Stacking (RF + XGBoost + GB -> FNN)",
  )

  chart_project_vs_papers(head)
  chart_f1_leaderboard(full)
  chart_precision_recall_tradeoff(full)


def write_markdown(full: pd.DataFrame, summary: pd.DataFrame) -> None:
  p1_best = summary[summary["Methodology"].str.startswith("Paper 1")].iloc[0]
  p2_best = summary[summary["Methodology"].str.startswith("Paper 2")].iloc[0]
  gb = summary[summary["Role"].str.contains("Primary", na=False)].iloc[0]
  xgb = summary[summary["Method"].str.contains("XGBoost tuned", na=False)].iloc[0]

  lines = [
      "# Methodology and Model Comparison (6,000-Record Dataset)",
      "",
      "These rows re-implement the methods from Paper 1 (Carballo-Mendívil et al., 2025) "
      "and Paper 2 (Niyogisubizo et al., 2022) on this project's 6,000-record dataset. "
      "They are not a claim that this thesis beats the published papers on their data.",
      "",
      "All scores below are on the **same held-out test set** "
      "(20% stratified split, `random_state=42`, n=1,200) unless noted.",
      "",
      "## 1. Source methodologies",
      "",
      "### Paper 1 — Carballo-Mendívil et al. (2025)",
      "",
      "Early-warning dropout prediction using pre-enrollment data.",
      "",
      "**Adapted pipeline:**",
      "- 80/20 stratified train/test split",
      "- Random undersampling of majority class **inside training folds only**",
      "- StandardScaler (numeric) + One-Hot Encoding (categorical)",
      "- Models: Logistic Regression, Random Forest, LightGBM, tuned XGBoost",
      "- 5-fold stratified CV; XGBoost hyperparameter grid search (F1 scoring)",
      "- Decision threshold chosen on **out-of-fold training probabilities** (max F1)",
      "",
      f"**Best Paper 1 result:** {p1_best['Method']} @ threshold "
      f"{p1_best['Threshold']:.2f}",
      f"- Accuracy: {p1_best['Accuracy']:.4f}",
      f"- Precision: {p1_best['Precision']:.4f}",
      f"- Recall: {p1_best['Recall']:.4f}",
      f"- F1: {p1_best['F1']:.4f}",
      f"- ROC-AUC: {p1_best['ROC_AUC']:.4f}",
      f"- MCC: {p1_best['MCC']:.4f}",
      "",
      "### Paper 2 — Niyogisubizo et al. (2022)",
      "",
      "Two-layer stacked ensemble for university dropout prediction.",
      "",
      "**Adapted pipeline:**",
      "- 80/20 stratified train/test split",
      "- Layer 1: Random Forest, XGBoost, Gradient Boosting (individual pipelines)",
      "- 10-fold CV to produce out-of-fold probabilities for meta-training",
      "- Layer 2: feed-forward neural network (MLP) on Layer-1 OOF predictions",
      "- No undersampling; default threshold 0.50",
      "",
      f"**Best Paper 2 base learner:** {p2_best['Method']}",
      f"- Accuracy: {p2_best['Accuracy']:.4f} ({p2_best['Accuracy']*100:.2f}%)",
      f"- Precision: {p2_best['Precision']:.4f}",
      f"- Recall: {p2_best['Recall']:.4f}",
      f"- F1: {p2_best['F1']:.4f}",
      f"- ROC-AUC: {p2_best['ROC_AUC']:.4f}",
      f"- MCC: {p2_best['MCC']:.4f}",
      "",
      "On the revised file the Paper 2 feedforward stack is the highest single-split "
      "accuracy. A 5-fold check of the same base models with a logistic meta-learner "
      "does not beat Gradient Boosting (see experiments/results/MODEL_SELECTION.md). "
      "The project keeps Gradient Boosting.",
      "",
      "## 2. This project's adopted model (SARP-Net DSS)",
      "",
      "| Component | Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |",
      "|---|---|---:|---:|---:|---:|---:|---:|",
      f"| Continuation probability, SHAP, and risk tiers | Gradient Boosting | "
      f"0.50 | {gb['Accuracy']:.4f} | {gb['Precision']:.4f} | "
      f"{gb['Recall']:.4f} | {gb['F1']:.4f} | {gb['ROC_AUC']:.4f} |",
      "",
      "Gradient Boosting is the project model. A fresh training-set search "
      "(logistic, random forest, LightGBM, HistGradientBoosting, several XGBoost "
      "and Gradient Boosting settings, an undersampled XGBoost, and a stack) "
      "did not beat it on the locked test. The low-risk cutoff is the calibrated "
      "0.54 threshold in `results_paper1/operating_point.json`, not an XGBoost cutoff. "
      f"Tuned XGBoost at {xgb['Threshold']:.2f} remains a Paper 1 comparator "
      f"(accuracy {xgb['Accuracy']:.4f}, F1 {xgb['F1']:.4f}).",
      "",
      "## 3. Head-to-head (best model per methodology)",
      "",
      "| Methodology | Best model | Thr. | Accuracy | Precision | Recall | F1 | ROC-AUC | MCC |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]

  for _, row in summary.iterrows():
    thr = f"{row['Threshold']:.2f}" if pd.notna(row["Threshold"]) else "—"
    mcc = f"{row['MCC']:.4f}" if pd.notna(row["MCC"]) else "—"
    lines.append(
        f"| {row['Methodology'].split('(')[0].strip()} | {row['Method']} | "
        f"{thr} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | "
        f"{row['Recall']:.4f} | {row['F1']:.4f} | {row['ROC_AUC']:.4f} | {mcc} |"
    )

  lines.extend(
      [
          "",
          "## 4. Full model leaderboard (all runs)",
          "",
          "See `final_methodology_comparison.csv` for every model evaluated.",
          "",
          "## 5. Confusion matrix (project primary model)",
          "",
          f"Gradient Boosting @ 0.50: TN={int(gb['TN'])}, FP={int(gb['FP'])}, "
          f"FN={int(gb['FN'])}, TP={int(gb['TP'])}",
          "",
          f"XGBoost tuned @ {xgb['Threshold']:.2f}: TN={int(xgb['TN'])}, FP={int(xgb['FP'])}, "
          f"FN={int(xgb['FN'])}, TP={int(xgb['TP'])}",
          "",
          "## 6. Feature contract (shared across Paper 1 and Paper 2 adaptations)",
          "",
          "- 22 engineered features from `master_6000_engineered.csv`, including the four-level due diligence score",
          "- Target: `Continuation_Bin` (52.2% positive class on the revised file)",
          "- Preprocessing: StandardScaler + OneHotEncoder in sklearn pipelines",
          "",
      ]
  )

  (OUT / "METHODOLOGY_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
  full = build_full_table()
  summary = build_summary(full)

  full.to_csv(OUT / "final_methodology_comparison.csv", index=False)
  summary.to_csv(OUT / "methodology_summary.csv", index=False)
  write_markdown(full, summary)
  plot_all_charts(full, summary)

  print(f"\nWrote CSVs to {OUT}")
  print("Charts:")
  for name in sorted(OUT.glob("chart*.png")):
    print(f"  - {name.name}")


if __name__ == "__main__":
  main()
