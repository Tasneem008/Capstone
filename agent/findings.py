"""The project's recorded findings, loaded from the repository's result files.

The agent cites these numbers so its advice matches the paper. Nothing is hard-coded: if the team
re-runs the experiments, the agent picks up the new figures on the next start.

Sources (all relative to the repository root):
    results_ablation/ablation_results.csv           F1 drop when each feature group is removed
    experiments/results/e2_due_diligence_by_phase.csv  continuation rate per due-diligence level, per phase
    experiments/results/e2_odds_ratios.csv           logistic-model odds ratios with 95% CIs
    results_paper2/paper2_model_comparison.csv       held-out test metrics of the deployed model
    Datasets/master_6000_engineered.csv (passed in)  the 6,000 agency records, for rates by group
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ABLATION_FILE = Path("results_ablation/ablation_results.csv")
DUE_DILIGENCE_FILE = Path("experiments/results/e2_due_diligence_by_phase.csv")
ODDS_RATIO_FILE = Path("experiments/results/e2_odds_ratios.csv")
MODEL_METRICS_FILE = Path("results_paper2/paper2_model_comparison.csv")
DEPLOYED_MODEL = "Gradient Boosting"
LEVELS = ["No", "Yes Low", "Yes Medium", "Yes High"]


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


@dataclass
class Findings:
    ablation_f1_drop: dict[str, float]                 # group -> F1 drop
    due_diligence_by_phase: dict[str, dict[str, float]]  # phase -> level -> continuation rate
    odds_ratios: dict[str, tuple[float, float, float]]   # term -> (OR, low, high)
    model_metrics: dict[str, float]
    records: pd.DataFrame = field(repr=False)
    sources: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ record-based rates
    def rate(self, phase: str, column: str, value) -> tuple[float, int] | None:
        """Continuation rate and count in the agency records for one phase and one value of a column."""
        rows = self.records[(self.records["Political_Phase"] == phase) & (self.records[column] == value)]
        return (float(rows["Continuation_Bin"].mean()), len(rows)) if len(rows) else None

    def phase_rate(self, phase: str) -> float:
        rows = self.records[self.records["Political_Phase"] == phase]
        return float(rows["Continuation_Bin"].mean())

    def ablation_rank(self, group: str) -> int:
        order = sorted(self.ablation_f1_drop, key=self.ablation_f1_drop.get, reverse=True)
        return order.index(group) + 1

    # ------------------------------------------------------------------ evidence text
    def evidence(self, group: str, profile: dict) -> list[str]:
        """Counsellor-facing evidence for one feature group, for this student's phase and choices."""
        phase = profile["Political_Phase"]
        lines = []
        drop = self.ablation_f1_drop.get(group)
        if drop is not None:
            lines.append(f"Ablation: removing this group lowers model F1 by {drop:.3f} "
                         f"(rank {self.ablation_rank(group)} of {len(self.ablation_f1_drop)}).")
        if group == "Behavioral":
            rates = self.due_diligence_by_phase.get(phase)
            if rates:
                lines.append(f"Records, {phase} phase: continuation " + ", ".join(
                    f"{lvl} {_pct(rates[lvl])}" for lvl in LEVELS) + ".")
            orr = self.odds_ratios.get("Due_Diligence_Level")
            if orr:
                lines.append(f"Each step up in due diligence multiplies the odds of continuing by {orr[0]:.2f} "
                             f"(95% CI {orr[1]:.2f}-{orr[2]:.2f}), controlling for GPA, academic level, "
                             "budget and phase.")
        elif group == "Financial":
            hit = self.rate(phase, "Budget", profile["Budget"])
            if hit:
                lines.append(f"Records, {phase} phase: students with budget {profile['Budget']} "
                             f"continued at {_pct(hit[0])} (n={hit[1]}).")
            orr = self.odds_ratios.get("Budget_Level")
            if orr:
                lines.append(f"Each budget level up multiplies the odds of continuing by {orr[0]:.2f} "
                             f"(95% CI {orr[1]:.2f}-{orr[2]:.2f}).")
        elif group == "Academic":
            hit = self.rate(phase, "Academic_Level", profile["Academic_Level"])
            if hit:
                lines.append(f"Records, {phase} phase: academic level {profile['Academic_Level']} "
                             f"continued at {_pct(hit[0])} (n={hit[1]}).")
            orr = self.odds_ratios.get("Academic_Level")
            if orr:
                lines.append(f"Each academic level up multiplies the odds of continuing by {orr[0]:.2f} "
                             f"(95% CI {orr[1]:.2f}-{orr[2]:.2f}); GPA itself is not significant "
                             f"(OR {self.odds_ratios.get('GPA_z', (float('nan'),))[0]:.2f}).")
        elif group == "Phase_Temporal":
            lines.append(f"Records: overall continuation in the {phase} phase is {_pct(self.phase_rate(phase))}.")
            orr = self.odds_ratios.get(phase)
            if orr:
                lines.append(f"Odds of continuing vs the Stable phase: OR {orr[0]:.3f} (95% CI {orr[1]:.3f}-{orr[2]:.3f}).")
        elif group == "Destination":
            hit = self.rate(phase, "Country", profile["Country"])
            if hit:
                lines.append(f"Records, {phase} phase: students choosing {profile['Country']} "
                             f"continued at {_pct(hit[0])} (n={hit[1]}).")
        return lines

    def student_facts(self, groups: list[str], profile: dict) -> list[str]:
        """Aggregate findings that are safe to share with a student (no personal risk, no politics)."""
        facts = []
        if "Behavioral" in groups:
            rates = self.due_diligence_by_phase.get(profile["Political_Phase"])
            if rates:
                researched = max(rates[lvl] for lvl in LEVELS[1:])
                facts.append("In our agency's recent records, students who had researched their course and "
                             f"institution went ahead with their study plans far more often (up to {_pct(researched)}) "
                             f"than students who had not ({_pct(rates['No'])}).")
        return facts

    def model_card(self) -> dict:
        m = self.model_metrics
        return {"model": DEPLOYED_MODEL, "records": len(self.records),
                "test_accuracy": round(m["Accuracy"], 4), "test_f1": round(m["F1"], 4),
                "test_roc_auc": round(m["ROC_AUC"], 4), "sources": self.sources}


def load_findings(repo_dir: Path, records: pd.DataFrame) -> Findings:
    repo_dir = Path(repo_dir)
    ablation = {}
    for row in _rows(repo_dir / ABLATION_FILE):
        group = row["Removed_Group"].replace("_Features", "")
        if group != "None":
            ablation[group] = float(row["F1_Drop"])
    dd = {row["phase"]: {lvl: float(row[lvl]) for lvl in LEVELS} for row in _rows(repo_dir / DUE_DILIGENCE_FILE)}
    ors = {row["term"]: (float(row["odds_ratio"]), float(row["or_low"]), float(row["or_high"]))
           for row in _rows(repo_dir / ODDS_RATIO_FILE)}
    deployed = next(r for r in _rows(repo_dir / MODEL_METRICS_FILE) if r["Model"] == DEPLOYED_MODEL)
    metrics = {k: float(deployed[k]) for k in ("Accuracy", "Precision", "Recall", "F1", "ROC_AUC")}
    sources = [str(p).replace("\\", "/") for p in (ABLATION_FILE, DUE_DILIGENCE_FILE, ODDS_RATIO_FILE,
                                                   MODEL_METRICS_FILE)]
    return Findings(ablation, dd, ors, metrics, records, sources)
