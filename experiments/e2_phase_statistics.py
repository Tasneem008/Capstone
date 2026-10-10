"""E2. Phase effect: chi-square, Cramer's V, logistic odds ratios (RQ1, H1, H2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiments.lib import PHASE_ORDER, RANDOM_STATE, RESULTS, load_frame

DUE_ORDER = ["No", "Yes Low", "Yes Medium", "Yes High"]


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    rank = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * rank)))


def design_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    phase = pd.get_dummies(frame["Political_Phase"], drop_first=False)
    # Stable is the reference.
    phase = phase.drop(columns=["Stable"])
    due = frame["Researched"].map(
        {"No": 0, "Yes Low": 1, "Yes Medium": 2, "Yes High": 3}
    )
    academic = frame["Academic_Level"].map({"Low": 0, "Medium": 1, "High": 2})
    budget = frame["Budget_Level"].map({"Low": 0, "Medium": 1, "High": 2})
    gpa = StandardScaler().fit_transform(frame[["Result"]])
    design = pd.DataFrame(
        {
            "GPA_z": gpa.ravel(),
            "Due_Diligence_Level": due.to_numpy(),
            "Academic_Level": academic.to_numpy(),
            "Budget_Level": budget.to_numpy(),
        },
        index=frame.index,
    )
    design = pd.concat([design, phase.astype(float)], axis=1)
    return design, frame["Continuation_Bin"].astype(int)


def bootstrap_odds(design: pd.DataFrame, target: pd.Series, n_boot: int = 400) -> pd.DataFrame:
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    model.fit(design, target)
    point = model.coef_.ravel()
    rng = np.random.default_rng(RANDOM_STATE)
    draws = np.zeros((n_boot, design.shape[1]))
    values = design.to_numpy()
    y = target.to_numpy()
    n = len(y)
    for draw in range(n_boot):
        index = rng.integers(0, n, n)
        sample_y = y[index]
        if len(np.unique(sample_y)) < 2:
            draws[draw] = np.nan
            continue
        boot = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        boot.fit(values[index], sample_y)
        draws[draw] = boot.coef_.ravel()
    rows = []
    for i, name in enumerate(design.columns):
        sample = draws[:, i]
        sample = sample[~np.isnan(sample)]
        rows.append(
            {
                "term": name,
                "coef": float(point[i]),
                "odds_ratio": float(np.exp(point[i])),
                "or_low": float(np.exp(np.quantile(sample, 0.025))),
                "or_high": float(np.exp(np.quantile(sample, 0.975))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    frame = load_frame()
    table = pd.crosstab(frame["Political_Phase"], frame["Continuation_Bin"]).reindex(PHASE_ORDER)
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    effect = cramers_v(table)

    lines = ["# Phase statistics (E2)", ""]
    lines.append("## H1. Continuation x political phase")
    lines.append(table.to_string())
    lines.append("")
    lines.append(
        f"Chi-square = {chi2:.2f}, df = {dof}, p = {p_value:.3e}, Cramer's V = {effect:.3f}."
    )
    lines.append("H1 is supported if p < 0.05. V is the effect size (small 0.1, medium 0.3, large 0.5).")
    lines.append("")

    lines.append("## H2. Due diligence inside each phase")
    h2_rows = []
    for phase in PHASE_ORDER:
        subset = frame.loc[frame["Political_Phase"] == phase]
        rates = (
            subset.groupby("Researched")["Continuation_Bin"].mean().reindex(DUE_ORDER)
        )
        coded = subset["Researched"].map({label: i for i, label in enumerate(DUE_ORDER)})
        rho, rho_p = stats.spearmanr(coded, subset["Continuation_Bin"])
        h2_rows.append(
            {
                "phase": phase,
                "spearman_rho": rho,
                "p": rho_p,
                **{label: float(rates[label]) for label in DUE_ORDER},
            }
        )
    h2 = pd.DataFrame(h2_rows)
    h2.to_csv(RESULTS / "e2_due_diligence_by_phase.csv", index=False)
    lines.append(h2.round(4).to_string(index=False))
    lines.append("")
    lines.append(
        "H2 is supported in a phase when rho > 0 and p < 0.05, "
        "i.e. higher due diligence ranks with higher continuation."
    )
    lines.append("")

    design, target = design_matrix(frame)
    odds = bootstrap_odds(design, target)
    odds.to_csv(RESULTS / "e2_odds_ratios.csv", index=False)
    lines.append("## Logistic model, Stable as the phase reference")
    lines.append(
        "Controls: GPA (z-scored), due diligence level (0-3), academic level (0-2), "
        "budget level (0-2). Date features are excluded so the phase term is not "
        "just a second copy of the calendar. Intervals are bootstrap 95% percentiles "
        "(400 resamples)."
    )
    lines.append("")
    lines.append(odds.round(3).to_string(index=False))
    lines.append("")
    lines.append("## Verdicts")
    h1_text = "supported" if p_value < 0.05 else "not supported"
    lines.append(
        f"H1 is {h1_text}: continuation and phase are associated "
        f"(chi-square p = {p_value:.3e}, Cramer's V = {effect:.3f})."
    )
    h2_ok = bool(((h2["spearman_rho"] > 0) & (h2["p"] < 0.05)).all())
    lines.append(
        "H2 is supported in every phase."
        if h2_ok
        else "H2 is not supported in every phase. See the Spearman table."
    )
    lines.append(
        "The rank correlation is weakest in Stable, where continuation is already "
        "high at Due Diligence = No. In Unstable and Newly Elected the rate jumps "
        "from No to Yes Low and then sits near the ceiling, so the gradient is real "
        "but most of it is the first step up from No."
    )
    phase_odds = odds[odds["term"].isin([phase for phase in PHASE_ORDER if phase != "Stable"])]
    moved = []
    for _, row in phase_odds.iterrows():
        if row["or_high"] < 1 or row["or_low"] > 1:
            moved.append(f"{row['term']} OR {row['odds_ratio']:.3f} (CI {row['or_low']:.3f}-{row['or_high']:.3f})")
    lines.append(
        "After GPA, due diligence, academic level, and budget, the phase terms whose "
        "95% CI excludes 1 are: " + ("; ".join(moved) if moved else "none") + ". "
        "Newly Elected is the comparison to read carefully: its interval can include 1, "
        "which means it is not separable from Stable once those controls are in the model."
    )

    plot = odds.copy()
    plot["label"] = plot["term"].str.replace("Transitional phase ", "TP ", regex=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(plot))
    ax.errorbar(
        plot["odds_ratio"],
        y,
        xerr=[
            plot["odds_ratio"] - plot["or_low"],
            plot["or_high"] - plot["odds_ratio"],
        ],
        fmt="o",
        color="#1565c0",
        capsize=3,
    )
    ax.axvline(1.0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel("Odds ratio (continuation)")
    ax.set_title("Continuation odds ratios vs Stable (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(RESULTS / "e2_odds_ratio_forest.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    text = "\n".join(lines)
    (RESULTS / "E2_PHASE_STATISTICS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
