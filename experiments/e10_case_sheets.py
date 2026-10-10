"""E10 materials. Ten anonymized held-out cases for the counselor protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import load_gradient_boosting_model, split_engineered_data
from experiments.lib import RESULTS, load_frame
from risk_analyzer import MEDIUM_RISK_THRESHOLD, load_operating_threshold

SHOW = [
    "Result",
    "Academic_Level",
    "Budget_Level",
    "Researched",
    "Course",
    "Country",
    "Institution",
    "Political_Phase",
    "Date Intake",
]


def short_tier(probability: float, cutoff: float) -> str:
    if probability >= cutoff:
        return "Low"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "High"


def main() -> None:
    frame = load_frame()
    _, x_test, _, y_test = split_engineered_data(frame)
    model = load_gradient_boosting_model()
    cutoff = load_operating_threshold()
    prob = model.predict_proba(x_test)[:, 1]

    detail = x_test.copy()
    detail["Date Intake"] = frame.loc[x_test.index, "Date Intake"]
    detail = detail.loc[:, SHOW].copy()
    detail["continuation"] = y_test.to_numpy()
    detail["probability"] = prob
    detail["tier"] = [short_tier(float(value), cutoff) for value in prob]

    rng = np.random.default_rng(42)
    chosen = []
    for tier in ("High", "Medium", "Low"):
        pool = detail.index[detail["tier"] == tier].to_numpy()
        if len(pool) < 2:
            raise RuntimeError(f"Need at least 2 {tier} cases, found {len(pool)}.")
        chosen.extend(rng.choice(pool, size=2, replace=False).tolist())
    remaining = detail.index.difference(chosen).to_numpy()
    chosen.extend(rng.choice(remaining, size=10 - len(chosen), replace=False).tolist())

    sheet = detail.loc[chosen].copy()
    sheet.insert(0, "case_id", [f"Case {i:02d}" for i in range(1, 11)])
    sheet = sheet.rename(columns={"Researched": "Due_Diligence", "Result": "GPA"})
    sheet["Date Intake"] = pd.to_datetime(sheet["Date Intake"]).dt.strftime("%Y-%m-%d")

    key_columns = ["case_id", "tier", "probability", "continuation"]
    sheet[key_columns].to_csv(RESULTS / "e10_researcher_key.csv", index=False)

    counselor = sheet.drop(columns=["tier", "probability", "continuation"])
    lines = [
        "# Counselor case sheet",
        "",
        "Names are removed. Do not show continuation, probability, or tier until the counselor has given a follow-up flag.",
        f"Tiers on the researcher key use low-risk cutoff {cutoff:.2f} and medium risk from {MEDIUM_RISK_THRESHOLD:.2f}.",
        "",
    ]
    for _, row in counselor.iterrows():
        lines.append(f"## {row['case_id']}")
        for column in counselor.columns:
            if column == "case_id":
                continue
            lines.append(f"- {column}: {row[column]}")
        lines.append("")
        lines.append("Follow-up flag before the model (Yes/No):")
        lines.append("")
        lines.append("Main reason:")
        lines.append("")
        lines.append("Changed the flag after seeing the model (Yes/No):")
        lines.append("")
    text = "\n".join(lines)
    (RESULTS / "e10_counselor_cases.md").write_text(text, encoding="utf-8")
    print(sheet[key_columns].to_string(index=False))
    print(f"Wrote {RESULTS / 'e10_counselor_cases.md'}")


if __name__ == "__main__":
    main()
