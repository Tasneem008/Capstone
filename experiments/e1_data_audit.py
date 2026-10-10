"""E1. Data audit of the revised 6,000-record study file."""

from __future__ import annotations

import re

import pandas as pd

from common import DATA_FILES
from experiments.lib import (
    FINGERPRINT,
    PHASE_MAP,
    PHASE_ORDER,
    RESULTS,
    ROOT,
    deduplicate,
    load_frame,
)

# Same fields as FINGERPRINT, without academic level, so a row that matches
# on identity and outcome but not on academic band is still counted.
LOOSE_FINGERPRINT = [column for column in FINGERPRINT if column != "Academic_Level"]

EXPECTED_WINDOWS = {
    "Stable": ("2023-11-01", "2024-03-31"),
    "Unstable": ("2024-04-01", "2024-08-07"),
    "Transitional phase 1st 5 months": ("2024-08-08", "2025-06-30"),
    "Transitional phase Last 5 months": ("2025-09-01", "2026-01-31"),
    "Newly Elected": ("2026-02-15", "2026-07-15"),
}


def main() -> None:
    frame = load_frame()
    lines = ["# Data audit (E1)", ""]
    lines.append(f"Rows: {len(frame)}. Columns checked: model features plus name, date, and continuation.")
    lines.append("")

    missing = frame.isna().sum()
    missing = missing[missing > 0]
    lines.append("## Missing values")
    if missing.empty:
        lines.append("No missing values in the engineered table.")
    else:
        lines.append(missing.to_string())
    lines.append("")

    lines.append("## Continuation by phase")
    rates = (
        frame.groupby("Political_Phase")["Continuation_Bin"]
        .agg(["mean", "size"])
        .reindex(PHASE_ORDER)
    )
    rates["mean"] = (rates["mean"] * 100).round(2)
    lines.append(rates.rename(columns={"mean": "continuation_pct", "size": "n"}).to_string())
    lines.append("")
    lines.append(
        f"Overall continuation: {frame['Continuation_Bin'].mean() * 100:.2f}% "
        f"({int(frame['Continuation_Bin'].sum())} Yes / "
        f"{int((1 - frame['Continuation_Bin']).sum())} No)."
    )
    lines.append("")

    lines.append("## Date windows")
    for phase in PHASE_ORDER:
        dates = frame.loc[frame["Political_Phase"] == phase, "Date Intake"]
        start, end = dates.min().date(), dates.max().date()
        expected = EXPECTED_WINDOWS[phase]
        flag = "OK" if (str(start) == expected[0] and str(end) == expected[1]) else "CHECK"
        lines.append(f"- {phase}: {start} to {end} (expected {expected[0]} to {expected[1]}) [{flag}]")
    lines.append("")

    lines.append("## Due diligence")
    due = (
        frame.groupby("Researched")["Continuation_Bin"].agg(["mean", "size"])
    )
    due["mean"] = (due["mean"] * 100).round(2)
    lines.append(due.rename(columns={"mean": "continuation_pct", "size": "n"}).to_string())
    lines.append("")

    suffix = frame["Name"].astype(str).str.contains(r"\s\d+$", regex=True)
    lines.append("## Names")
    lines.append(
        f"Names ending in a space and a number: {int(suffix.sum())} of {len(frame)}. "
        "Individual names are not listed in this report."
    )
    lines.append(f"Unique names: {frame['Name'].nunique()}.")
    lines.append("")

    lines.append("## In-file Political_Phase labels")
    lines.append(
        "Modeling uses the workbook period, not the cell. "
        "These are the labels stored in each file before that correction."
    )
    data_dir = ROOT / "Datasets"
    for phase, filename in DATA_FILES.items():
        raw = pd.read_excel(data_dir / filename, usecols=["Political_Phase"])
        counts = raw["Political_Phase"].astype(str).value_counts()
        stored = "; ".join(f"{label} n={int(count)}" for label, count in counts.items())
        status = "matches the file period" if list(counts.index) == [phase] else "DIFFERS from the file period"
        lines.append(f"- {phase} ({filename}): {stored}. {status}.")
    lines.append("")

    lines.append("## Duplicates")
    exact = int(frame.duplicated().sum())
    lines.append(f"Exact duplicate rows: {exact}.")
    fp = frame.duplicated(subset=FINGERPRINT, keep=False)
    lines.append(
        f"Rows sharing a cross-field fingerprint (name, academics, destination, "
        f"budget, due diligence, continuation): {int(fp.sum())}."
    )
    pair_rows = []
    ordered = frame.copy()
    ordered["_ord"] = ordered["Political_Phase"].map(PHASE_MAP)
    for left_i, left in enumerate(PHASE_ORDER):
        for right in PHASE_ORDER[left_i + 1 :]:
            a = set(
                map(
                    tuple,
                    ordered.loc[ordered["Political_Phase"] == left, FINGERPRINT].itertuples(
                        index=False, name=None
                    ),
                )
            )
            b = set(
                map(
                    tuple,
                    ordered.loc[ordered["Political_Phase"] == right, FINGERPRINT].itertuples(
                        index=False, name=None
                    ),
                )
            )
            pair_rows.append({"phase_a": left, "phase_b": right, "shared_fingerprints": len(a & b)})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(RESULTS / "e1_cross_phase_fingerprints.csv", index=False)
    lines.append("")
    lines.append(pairs.to_string(index=False))
    lines.append("")
    loose_pairs = []
    for left_i, left in enumerate(PHASE_ORDER):
        for right in PHASE_ORDER[left_i + 1 :]:
            a = set(map(tuple, ordered.loc[ordered["Political_Phase"] == left, LOOSE_FINGERPRINT].itertuples(index=False, name=None)))
            b = set(map(tuple, ordered.loc[ordered["Political_Phase"] == right, LOOSE_FINGERPRINT].itertuples(index=False, name=None)))
            loose_pairs.append(len(a & b))
    lines.append(
        "A looser fingerprint that ignores academic level "
        f"(name, result, course, country, institution, budget, due diligence, continuation) "
        f"shares {max(loose_pairs)} fingerprints on its largest phase pair. "
        "E9 deduplicates on the stricter fingerprint, which also requires the academic level to match."
    )
    lines.append("")

    kept, summary = deduplicate(frame)
    lines.append("## Dedup rule used later in E9")
    lines.append(
        "Keep the earliest row when the fingerprint repeats. "
        f"Removed {summary['cross_phase_copies_removed']} later copies "
        f"and {summary['within_phase_copies_removed']} residual copies. "
        f"Rows after: {summary['rows_after']} (from {summary['rows_before']})."
    )
    lines.append("")
    lines.append("## Outliers on GPA")
    result = frame["Result"]
    lines.append(
        f"Result min {result.min():.2f}, max {result.max():.2f}, "
        f"mean {result.mean():.3f}. Values outside 2.0-4.0: "
        f"{int(((result < 2) | (result > 4)).sum())}."
    )
    lines.append("")
    lines.append("## What this audit does not prove")
    lines.append(
        "- It does not prove the records are independent students. Shared fingerprints "
        "across Unstable and Newly Elected need a written explanation from the agency."
    )
    lines.append(
        "- It does not prove Due_Diligence was recorded before Continuation. "
        "That timing has to come from the agency, not from the spreadsheet."
    )
    lines.append(
        "- Modeling still uses the full 6,000-row file. The deduplicated table is "
        "only the E9 sensitivity set."
    )

    text = "\n".join(lines)
    (RESULTS / "DATA_AUDIT.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {RESULTS / 'DATA_AUDIT.md'}")


if __name__ == "__main__":
    main()
