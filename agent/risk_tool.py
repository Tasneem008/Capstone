"""SARP-Agent risk tool: score a student profile gathered in chat with the SARP-Net model.

The agent's planner calls ``assess(profile)`` after each student message. It returns either

* ``status: "incomplete"`` with the details still missing, ordered by how much they matter to the
  model (ablation study), so the agent knows what to ask next; or
* ``status: "ok"`` with the continuation probability, risk tier, SHAP drivers grouped like the
  dashboard, a due-diligence what-if, and student-safe guidance (topics plus aggregate facts from the
  agency records; never the student's own probability or risk label).

It calls the dashboard's own code (``common.build_student_features``, the saved Gradient Boosting
pipeline and ``risk_analyzer`` tiers), so a profile gets exactly the probability the dashboard shows.
Each driver carries evidence read from the project's result files (ablation F1 drops, continuation
rates by due diligence and phase, odds ratios; see agent/findings.py), so the agent's advice matches
the paper's findings.

Chat values are normalised first: "CSE" -> "Computer Science", "Toronto University" ->
"University of Toronto", "20 lakh taka" -> the "16-20K" USD band, HSC GPA 5.00 -> 4.0 scale.

Demo (from the repository root):
    .venv/Scripts/python.exe -m agent.risk_tool --demo
    .venv/Scripts/python.exe -m agent.risk_tool --profile "{\"gpa\": 3.1, \"country\": \"UK\"}"

Settings (environment variables, or constructor arguments):
    SARP_POLITICAL_PHASE  current political phase used for every prediction (default "Newly Elected")
    SARP_BDT_PER_USD      exchange rate for budgets given in taka (default 122)
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from common import (  # noqa: E402
    DUE_DILIGENCE_LEVEL,
    PHASE_MAP,
    build_student_features,
    load_engineered_data,
    load_gradient_boosting_model,
    parse_intake_dates,
)
from risk_analyzer import MEDIUM_RISK_THRESHOLD, calculate_risk_tier, load_operating_threshold  # noqa: E402

from agent.findings import load_findings  # noqa: E402

DEFAULT_PHASE = os.environ.get("SARP_POLITICAL_PHASE", "Newly Elected")
DEFAULT_BDT_PER_USD = float(os.environ.get("SARP_BDT_PER_USD", "122"))

BUDGET_TO_LEVEL = {"Below 10k": "Low", "10-15K": "Medium", "16-20K": "High"}  # same as the dashboard
DUE_DILIGENCE_ORDER = list(DUE_DILIGENCE_LEVEL)  # No, Yes Low, Yes Medium, Yes High

# Same groups as FEATURE_GROUPS in 07_dss_app.py (test_risk_tool checks they stay identical).
FEATURE_GROUPS = {
    "Financial": ["Budget", "Budget_Level", "Budget_Midpoint", "Budget_Level_Num", "Budget_Academic_Score"],
    "Academic": ["Result", "Academic_Level", "Academic_Level_Num", "Result_Band", "GPA_Academic_Match"],
    "Phase_Temporal": ["Political_Phase", "Intake_Month", "Year", "Season", "Days_Since_First_Intake",
                       "Phase_Number"],
    "Destination": ["Course", "Country", "Institution"],
    "Behavioral": ["Researched", "Researched_Bin", "Researched_Level"],
}
GROUP_OF = {col: group for group, cols in FEATURE_GROUPS.items() for col in cols}

# What the agent should talk about when a group pulls the prediction down. Student topics never mention
# risk, probabilities or politics; the political phase is context the student cannot change.
GROUP_GUIDANCE = {
    "Behavioral": {
        "label": "Due diligence", "focus": "help_research", "actionable": True,
        "topic": "Help the student research their course content, entry requirements, total cost and "
                 "visa steps for their chosen institution.",
    },
    "Financial": {
        "label": "Budget", "focus": "funding_plan", "actionable": True,
        "topic": "Work out the full cost of the programme and the funds the visa requires; discuss "
                 "scholarships or lower-cost options if the budget falls short.",
    },
    "Academic": {
        "label": "Academic profile", "focus": "academic_fit", "actionable": True,
        "topic": "Check the programme's entry requirements against the student's results; mention "
                 "foundation, pathway or alternative programmes if needed.",
    },
    "Phase_Temporal": {
        "label": "Political phase / timing", "focus": "timing", "actionable": False,
        "topic": "Give up-to-date intake, deadline and visa timing information so the student can "
                 "plan a realistic intake.",
    },
    "Destination": {
        "label": "Destination fit", "focus": "destination_fit", "actionable": True,
        "topic": "Compare the chosen course, institution and country with suitable alternatives.",
    },
}

FRIENDLY_NAMES = {
    "Result": "GPA", "Course": "Course", "Country": "Destination country", "Institution": "Institution",
    "Budget": "Budget band", "Researched": "Due diligence", "Political_Phase": "Political phase",
    "Budget_Level": "Budget level", "Academic_Level": "Academic level", "Intake_Month": "Intake month",
    "Year": "Intake year", "Season": "Intake season", "Days_Since_First_Intake": "Days since first intake",
    "Budget_Midpoint": "Budget midpoint", "Phase_Number": "Political phase (numeric)",
    "Budget_Level_Num": "Budget level (numeric)", "Academic_Level_Num": "Academic level (numeric)",
    "Researched_Bin": "Due diligence (any)", "Researched_Level": "Due diligence level",
    "Result_Band": "GPA band", "Budget_Academic_Score": "Budget + academic score",
    "GPA_Academic_Match": "GPA / academic level match",
}

# Details the model needs, most important first (ablation F1 drop: due diligence 0.132,
# phase/timing 0.097, budget 0.027, academic 0.011, destination 0.006). Course and institution
# are optional: the model's encoder ignores values it has not seen.
REQUIRED_FIELDS = ["due_diligence", "intake", "budget", "gpa", "country"]
ASK_HINTS = {
    "due_diligence": "how much the student already knows about their course, institution, costs and requirements",
    "intake": "which intake (month and year) the student is aiming for",
    "budget": "the student's total yearly budget for tuition and living costs (taka or dollars)",
    "gpa": "the student's latest GPA or CGPA, and its scale",
    "country": "which country the student wants to study in",
}

# JSON schema for the chat-extraction step (e.g. Ollama `format=`): the LLM fills what the student said.
PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "gpa": {"type": ["number", "null"], "description": "Latest GPA/CGPA exactly as stated"},
        "gpa_scale": {"type": ["number", "null"], "description": "4 for university CGPA, 5 for SSC/HSC GPA"},
        "course": {"type": ["string", "null"], "description": "Intended subject, e.g. Computer Science"},
        "country": {"type": ["string", "null"], "description": "Intended destination country"},
        "institution": {"type": ["string", "null"], "description": "Intended university, if named"},
        "budget_amount": {"type": ["number", "null"], "description": "Total yearly budget as a number"},
        "budget_unit": {"type": ["string", "null"], "enum": ["lakh_taka", "taka", "usd", None],
                        "description": "Unit of budget_amount; '20 lakh' -> 20 with lakh_taka"},
        "due_diligence": {"type": ["string", "null"], "enum": DUE_DILIGENCE_ORDER + [None]},
        "intake": {"type": ["string", "null"], "description": "Target intake as YYYY-MM"},
    },
}
# Working definitions for the four due-diligence levels, to be confirmed with agency counsellors.
DUE_DILIGENCE_GUIDE = {
    "No": "Has not researched institutions or courses; unsure what or where to study.",
    "Yes Low": "Knows the destination and roughly the subject, but little about specific courses, "
               "institutions, costs or requirements.",
    "Yes Medium": "Has shortlisted specific courses or institutions and knows some requirements or costs.",
    "Yes High": "Knows the course content, entry requirements, total costs and visa/financial steps "
                "for their choice.",
}

COUNTRY_ALIASES = {
    "uk": "UK", "united kingdom": "UK", "england": "UK", "britain": "UK", "great britain": "UK",
    "scotland": "UK", "wales": "UK", "london": "UK", "usa": "USA", "us": "USA", "america": "USA",
    "united states": "USA", "united states of america": "USA", "nz": "New Zealand",
}
COURSE_ALIASES = {
    "cse": "Computer Science", "computer science and engineering": "Computer Science",
    "artificial intelligence": "AI", "information technology": "IT", "cybersecurity": "Cyber Security",
    "mba": "Business Administration", "bba": "Business Administration", "business": "Business Administration",
    "hospitality": "Hospitality Management", "hospitality and tourism": "Hospitality Management",
    "tourism": "Hospitality Management", "supply chain": "Supply Chain Management",
    "eee": "Electrical Engineering", "electrical and electronic engineering": "Electrical Engineering",
    "llb": "Law", "llm": "Law", "mph": "Public Health", "bsn": "Nursing",
}
INSTITUTION_ALIASES = {
    "ubc": "University of British Columbia", "uoft": "University of Toronto", "u of t": "University of Toronto",
    "kcl": "King's College London", "asu": "Arizona State University", "tum": "Technical University of Munich",
    "tu munich": "Technical University of Munich", "ucd": "University College Dublin",
    "um": "University of Malaya", "rmit": "RMIT University",
}
GENERIC_WORDS = {"university", "of", "the", "college", "institute", "campus", "and", "in", "master", "masters",
                 "bachelor", "bachelors", "msc", "bsc", "ma", "ba", "meng", "beng", "phd", "degree",
                 "programme", "program", "hons", "honours", "diploma", "course"}
LEVEL_ALIASES = {"no": "No", "none": "No", "0": "No", "low": "Yes Low", "yes low": "Yes Low", "1": "Yes Low",
                 "medium": "Yes Medium", "yes medium": "Yes Medium", "2": "Yes Medium",
                 "high": "Yes High", "yes high": "Yes High", "3": "Yes High"}
PHASE_ALIASES = {"stable": "Stable", "unstable": "Unstable", "tp1": "Transitional phase 1st 5 months",
                 "transitional 1": "Transitional phase 1st 5 months",
                 "tp2": "Transitional phase Last 5 months",
                 "transitional 2": "Transitional phase Last 5 months", "newly elected": "Newly Elected"}


def _key(text: str) -> str:
    text = str(text).lower().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def match_name(text: str | None, options: list[str], aliases: dict[str, str]) -> str | None:
    """Map free text to one known option, or None if unknown/ambiguous."""
    if not text or not str(text).strip():
        return None
    key = _key(text)
    by_key = {_key(o): o for o in options}
    if key in by_key:
        return by_key[key]
    if key in aliases and aliases[key] in options:
        return aliases[key]
    tokens = set(key.split()) - GENERIC_WORDS
    if tokens:
        stripped = " ".join(t for t in key.split() if t not in GENERIC_WORDS)
        if stripped in by_key:
            return by_key[stripped]
        if stripped in aliases and aliases[stripped] in options:
            return aliases[stripped]
        hits = [o for o in options if tokens <= set(_key(o).split())]
        if len(hits) == 1:
            return hits[0]
    close = difflib.get_close_matches(key, list(by_key), n=1, cutoff=0.85)
    return by_key[close[0]] if close else None


def parse_intake(value) -> pd.Timestamp | None:
    """Accept a date, 'YYYY-MM-DD', 'YYYY-MM' or 'January 2027'; month-only values use the 1st."""
    if value is None or value == "":
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%B %Y", "%b %Y", "%m/%Y"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    text = re.sub(r"(?i)\bsept\b", "Sep", text)
    for fmt in ("%b %Y", "%B %Y"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return None


def budget_band(usd: float) -> str:
    if usd < 10_000:
        return "Below 10k"
    if usd < 16_000:
        return "10-15K"
    return "16-20K"


class RiskTool:
    """Loads the SARP-Net model once and scores chat-gathered profiles."""

    def __init__(self, political_phase: str | None = None, bdt_per_usd: float | None = None):
        self.political_phase = self._phase(political_phase or DEFAULT_PHASE)
        self.bdt_per_usd = float(bdt_per_usd or DEFAULT_BDT_PER_USD)
        self.model = load_gradient_boosting_model()
        self.preprocessor = self.model.named_steps["preprocessor"]
        self.explainer = shap.TreeExplainer(self.model.named_steps["model"])
        self.reference = load_engineered_data()
        self.low_cutoff = load_operating_threshold()
        ref = self.reference
        self.courses = sorted(ref["Course"].astype(str).unique())
        self.countries = sorted(ref["Country"].astype(str).unique())
        self.institutions = sorted(ref["Institution"].astype(str).unique())
        self.country_of_institution = (
            ref.groupby("Institution")["Country"].agg(lambda s: s.mode()[0]).to_dict())
        # Default academic level: the most common level among students in the same GPA band.
        self.academic_level_for_band = (
            ref.groupby("Result_Band")["Academic_Level"].agg(lambda s: s.mode()[0]).to_dict())
        dates = parse_intake_dates(ref["Date Intake"])
        self.first_intake, self.last_intake = dates.min(), dates.max()
        self.categorical = [c for name, _, cols in self.preprocessor.transformers_ if name == "cat" for c in cols]
        self.findings = load_findings(REPO_DIR, ref)

    # ------------------------------------------------------------------ normalisation
    @staticmethod
    def _phase(value: str) -> str:
        phase = value if value in PHASE_MAP else PHASE_ALIASES.get(_key(value))
        if phase is None:
            raise ValueError(f"Unknown political phase {value!r}; expected one of {list(PHASE_MAP)}")
        return phase

    def normalise(self, profile: dict) -> tuple[dict, list[str], list[dict]]:
        """Turn chat values into the dashboard's model fields. Returns (fields, missing, warnings)."""
        p = {k: v for k, v in (profile or {}).items() if v not in (None, "")}
        fields: dict = {}
        warnings: list[dict] = []

        def warn(code: str, message: str):
            warnings.append({"code": code, "message": message})

        # Due diligence
        if "due_diligence" in p:
            raw = str(p["due_diligence"]).strip()
            level = raw if raw in DUE_DILIGENCE_LEVEL else LEVEL_ALIASES.get(_key(raw))
            if level:
                fields["Researched"] = level
            else:
                warn("due_diligence_unrecognised", f"Due diligence {raw!r} is not one of {DUE_DILIGENCE_ORDER}.")

        # Intake date
        if "intake" in p:
            intake = parse_intake(p["intake"])
            if intake is None:
                warn("intake_unrecognised", f"Could not read intake {p['intake']!r}; use YYYY-MM.")
            else:
                fields["Date Intake"] = intake
                if intake > self.last_intake:
                    warn("intake_after_training_period",
                         f"Intake {intake.date()} is after the training data ends ({self.last_intake.date()}); "
                         "the model is extrapolating.")
                elif intake < self.first_intake:
                    warn("intake_before_training_period", f"Intake {intake.date()} is before the training data.")

        # Budget (yearly)
        if "budget_band" in p and p["budget_band"] in BUDGET_TO_LEVEL:
            fields["Budget"] = p["budget_band"]
        elif "budget_amount" in p:
            amount, unit = float(p["budget_amount"]), str(p.get("budget_unit") or "").lower()
            if unit not in ("lakh_taka", "taka", "usd"):
                unit = "lakh_taka" if amount < 1_000 else "taka" if amount >= 100_000 else "usd"
                warn("budget_unit_assumed", f"Budget unit not given; assumed {unit}.")
            usd = {"lakh_taka": amount * 100_000 / self.bdt_per_usd,
                   "taka": amount / self.bdt_per_usd, "usd": amount}[unit]
            fields["Budget"] = budget_band(usd)
            fields["budget_usd"] = round(usd)
            if usd > 20_000:
                warn("budget_above_training_range", f"Budget about ${usd:,.0f}/year is above the highest band "
                     "in the training data (16-20K); scored as 16-20K.")
        if "Budget" in fields:
            fields["Budget_Level"] = BUDGET_TO_LEVEL[fields["Budget"]]

        # GPA, converted to the 4.0 scale the model was trained on
        if "gpa" in p:
            gpa = float(p["gpa"])
            scale = float(p.get("gpa_scale") or (5.0 if gpa > 4.0 else 4.0))
            result = min(4.0, max(0.0, gpa * 4.0 / scale))
            if scale != 4.0:
                warn("gpa_converted", f"GPA {gpa:g} out of {scale:g} converted to {result:.2f} out of 4.")
            if result < 2.0:
                warn("gpa_below_training_range", "GPA is below the lowest value in the training data (2.0).")
            fields["Result"] = round(result, 2)

        # Academic level: counsellor-set if given, otherwise estimated from the GPA band.
        if p.get("academic_level") in ("Low", "Medium", "High"):
            fields["Academic_Level"] = p["academic_level"]
        elif "Result" in fields:
            r = fields["Result"]
            band = "Low" if r <= 2.99 else "Medium" if r <= 3.39 else "High"
            fields["Academic_Level"] = self.academic_level_for_band[band]
            warn("academic_level_estimated", f"Academic level not set; estimated as "
                 f"{fields['Academic_Level']} (most common for GPA band {band}).")

        # Destination
        if "country" in p:
            country = match_name(p["country"], self.countries, COUNTRY_ALIASES)
            if country:
                fields["Country"] = country
            else:
                fields["Country"] = str(p["country"]).strip()
                warn("country_not_in_model", f"{fields['Country']} is not one of the model's countries "
                     f"({', '.join(self.countries)}); prediction is less reliable.")
        course = match_name(p.get("course"), self.courses, COURSE_ALIASES)
        if course:
            fields["Course"] = course
        else:
            fields["Course"] = str(p["course"]).strip() if "course" in p else "Unknown"
            warn("course_not_in_model" if "course" in p else "course_not_given",
                 f"Course {p['course']!r} not matched to a known course." if "course" in p
                 else "Course not given yet.")
        institution = match_name(p.get("institution"), self.institutions, INSTITUTION_ALIASES)
        if institution:
            fields["Institution"] = institution
            home = self.country_of_institution[institution]
            if fields.get("Country") and fields["Country"] != home:
                warn("institution_country_mismatch", f"{institution} is in {home}, not {fields['Country']}.")
        else:
            fields["Institution"] = str(p["institution"]).strip() if "institution" in p else "Unknown"
            warn("institution_not_in_model" if "institution" in p else "institution_not_given",
                 f"Institution {p['institution']!r} is not one of the model's institutions." if "institution" in p
                 else "Institution not given yet.")

        fields["Political_Phase"] = self._phase(p["political_phase"]) if "political_phase" in p \
            else self.political_phase

        have = {"due_diligence": "Researched", "intake": "Date Intake", "budget": "Budget",
                "gpa": "Result", "country": "Country"}
        missing = [f for f in REQUIRED_FIELDS if have[f] not in fields]
        return fields, missing, warnings

    # ------------------------------------------------------------------ scoring
    def _model_profile(self, fields: dict) -> dict:
        keys = ["Result", "Course", "Country", "Institution", "Budget", "Researched", "Political_Phase",
                "Budget_Level", "Academic_Level", "Date Intake"]
        return {k: fields[k] for k in keys}

    def probability(self, model_profile: dict) -> float:
        X = build_student_features(model_profile, self.reference)
        return float(self.model.predict_proba(X)[0, 1])

    def explain(self, model_profile: dict) -> tuple[dict[str, float], list[dict], float]:
        """SHAP contributions (log-odds) per feature group and per original column, plus the base value."""
        X = build_student_features(model_profile, self.reference)
        transformed = self.preprocessor.transform(X)
        values = self.explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        per_column: dict[str, float] = {}
        for name, value in zip(self.preprocessor.get_feature_names_out(), values[0]):
            column = self._raw_column(name)
            per_column[column] = per_column.get(column, 0.0) + float(value)
        groups = {g: sum(v for c, v in per_column.items() if GROUP_OF.get(c) == g) for g in FEATURE_GROUPS}
        features = [{"feature": c, "label": FRIENDLY_NAMES.get(c, c), "group": GROUP_OF.get(c, "Other"),
                     "shap": round(v, 4)} for c, v in sorted(per_column.items(), key=lambda kv: -abs(kv[1]))]
        base = np.ravel(self.explainer.expected_value)[-1]
        return groups, features, float(base)

    def _raw_column(self, name: str) -> str:
        if name.startswith("num__"):
            return name[5:]
        rest = name[5:] if name.startswith("cat__") else name
        for col in sorted(self.categorical, key=len, reverse=True):
            if rest == col or rest.startswith(col + "_"):
                return col
        return rest

    def assess(self, profile: dict) -> dict:
        """Main entry point for the agent's planner. `profile` uses PROFILE_SCHEMA keys."""
        fields, missing, warnings = self.normalise(profile)
        known = {k: (str(v.date()) if isinstance(v, pd.Timestamp) else v) for k, v in fields.items()}
        if missing:
            return {"status": "incomplete", "missing": missing, "ask_next": missing[0],
                    "ask_hint": ASK_HINTS[missing[0]], "known": known, "warnings": warnings}

        model_profile = self._model_profile(fields)
        prob = self.probability(model_profile)
        tier = calculate_risk_tier(prob, self.low_cutoff)
        groups, features, base = self.explain(model_profile)

        current = model_profile["Researched"]
        what_if = {level: round(self.probability({**model_profile, "Researched": level}), 4)
                   for level in DUE_DILIGENCE_ORDER[DUE_DILIGENCE_ORDER.index(current) + 1:]}

        drivers = [{"group": g, "label": GROUP_GUIDANCE[g]["label"], "shap": round(v, 4),
                    "direction": "lowers continuation" if v < 0 else "raises continuation",
                    "actionable": GROUP_GUIDANCE[g]["actionable"],
                    "evidence": self.findings.evidence(g, model_profile)}
                   for g, v in sorted(groups.items(), key=lambda kv: -abs(kv[1]))]
        negatives = [g for g, v in sorted(groups.items(), key=lambda kv: kv[1]) if v < -0.05]
        return {
            "status": "ok",
            "counsellor_view": {
                "probability": round(prob, 4),
                "risk_tier": tier,
                "risk_level": tier.split()[0].lower(),
                "cutoffs": {"low_risk_from": self.low_cutoff, "medium_risk_from": MEDIUM_RISK_THRESHOLD},
                "drivers": drivers,
                "top_features": features[:6],
                "shap_base_value": round(base, 4),
                "due_diligence_what_if": {"current": current, "probability_if": what_if},
                "model_profile": known,
                "model_card": self.findings.model_card(),
            },
            # Safe to pass to the reply generator: topics and aggregate record facts only,
            # never this student's probability or risk label.
            "student_guidance": {
                "focus": [GROUP_GUIDANCE[g]["focus"] for g in negatives],
                "topics": [GROUP_GUIDANCE[g]["topic"] for g in negatives],
                "facts": self.findings.student_facts(negatives, model_profile),
            },
            "warnings": warnings,
        }


@lru_cache(maxsize=1)
def get_tool() -> RiskTool:
    """Shared instance for the backend (the model loads once per process)."""
    return RiskTool()


DEMO_PROFILE = {"gpa": 3.1, "course": "Business Analytics", "country": "United Kingdom",
                "institution": "Coventry", "budget_amount": 18, "budget_unit": "lakh_taka",
                "due_diligence": "No", "intake": "2027-01"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a chat-gathered student profile with SARP-Net.")
    ap.add_argument("--profile", help="JSON object using PROFILE_SCHEMA keys")
    ap.add_argument("--demo", action="store_true", help="score a built-in example profile")
    ap.add_argument("--phase", help="political phase override")
    args = ap.parse_args()
    if not args.profile and not args.demo:
        ap.error("give --profile JSON or --demo")
    profile = DEMO_PROFILE if args.demo else json.loads(args.profile)
    result = RiskTool(political_phase=args.phase).assess(profile)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
