"""Tests for the agent risk tool.

Run from the repository root:
    .venv/Scripts/python.exe -m unittest agent.test_risk_tool -v
"""
import importlib.util
import json
import unittest

import numpy as np

from agent.risk_tool import FEATURE_GROUPS, REPO_DIR, RiskTool, match_name, COURSE_ALIASES, INSTITUTION_ALIASES
from common import build_student_features

# The dashboard's default sidebar profile on 2026-11-12 ("Analyze Student" showed 28.1%, High Risk).
DASHBOARD_DEFAULT = {"gpa": 3.20, "academic_level": "Medium", "budget_band": "10-15K", "course": "AI",
                     "country": "Australia", "institution": "Deakin University", "due_diligence": "No",
                     "political_phase": "Stable", "intake": "2026-11-12"}
COMPLETE = {"gpa": 3.4, "course": "CSE", "country": "Canada", "institution": "Toronto University",
            "budget_amount": 20, "budget_unit": "lakh_taka", "due_diligence": "Yes Low", "intake": "2027-09"}


class RiskToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = RiskTool(political_phase="Newly Elected", bdt_per_usd=122)
        spec = importlib.util.spec_from_file_location("dss_app", REPO_DIR / "07_dss_app.py")
        cls.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.app)

    # --- same results as SARP-Net -------------------------------------------------------------
    def test_matches_dashboard_prediction(self):
        result = self.tool.assess(DASHBOARD_DEFAULT)
        view = result["counsellor_view"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(round(view["probability"] * 100, 1), 28.1)
        self.assertTrue(view["risk_tier"].startswith("High Risk"))
        direct_profile = {"Result": 3.20, "Course": "AI", "Country": "Australia", "Institution": "Deakin University",
                          "Budget": "10-15K", "Researched": "No", "Political_Phase": "Stable",
                          "Budget_Level": "Medium", "Academic_Level": "Medium", "Date Intake": "2026-11-12"}
        X = build_student_features(direct_profile, self.tool.reference)
        self.assertAlmostEqual(view["probability"], round(float(self.tool.model.predict_proba(X)[0, 1]), 4))

    def test_shap_adds_up_to_model_output(self):
        fields, missing, _ = self.tool.normalise(COMPLETE)
        self.assertEqual(missing, [])
        profile = self.tool._model_profile(fields)
        groups, _, base = self.tool.explain(profile)
        X = self.tool.preprocessor.transform(build_student_features(profile, self.tool.reference))
        log_odds = self.tool.model.named_steps["model"].decision_function(X)[0]
        self.assertAlmostEqual(base + sum(groups.values()), log_odds, places=5)

    def test_feature_groups_match_dashboard(self):
        self.assertEqual(FEATURE_GROUPS, self.app.FEATURE_GROUPS)

    def test_dashboard_advice_quotes_current_ablation_results(self):
        templates = self.app.RECOMMENDATION_TEMPLATES
        drops = self.tool.findings.ablation_f1_drop
        for group in ("Behavioral", "Financial", "Phase_Temporal"):
            self.assertIn(f"F1 drop of {drops[group]:.3f}", templates[group]["negative"], group)
        all_text = json.dumps(templates)
        for stale in ("0.391", "0.209", "0.142"):  # pre-revision status-report values
            self.assertNotIn(stale, all_text)

    # --- cites the project's recorded findings ----------------------------------------------
    def test_findings_match_paper(self):
        f = self.tool.findings
        self.assertEqual(round(f.ablation_f1_drop["Behavioral"], 3), 0.132)
        self.assertEqual(round(f.ablation_f1_drop["Phase_Temporal"], 3), 0.097)
        self.assertEqual(round(f.odds_ratios["Due_Diligence_Level"][0], 2), 4.60)
        card = f.model_card()
        self.assertEqual((card["test_accuracy"], card["test_f1"], card["test_roc_auc"]), (0.8642, 0.8682, 0.9513))
        self.assertEqual(card["records"], 6000)

    def test_drivers_carry_evidence(self):
        view = self.tool.assess(DASHBOARD_DEFAULT)["counsellor_view"]
        behavioral = next(d for d in view["drivers"] if d["group"] == "Behavioral")
        text = " ".join(behavioral["evidence"])
        self.assertIn("0.132", text)
        self.assertIn("4.60", text)
        self.assertIn("Stable phase", text)

    def test_student_guidance_hides_personal_risk(self):
        result = self.tool.assess(DASHBOARD_DEFAULT)
        guidance = json.dumps(result["student_guidance"]).lower()
        prob = result["counsellor_view"]["probability"]
        for banned in ("risk", "drop out", "political", "phase", f"{prob * 100:.1f}", f"{prob:.2f}"):
            self.assertNotIn(banned, guidance)
        self.assertIn("help_research", result["student_guidance"]["focus"])
        self.assertTrue(result["student_guidance"]["facts"])

    # --- conversation handling ----------------------------------------------------------------
    def test_incomplete_profile_asks_most_important_first(self):
        result = self.tool.assess({"gpa": 3.5, "country": "UK"})
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing"], ["due_diligence", "intake", "budget"])
        self.assertEqual(result["ask_next"], "due_diligence")

    def test_budget_in_lakh_taka(self):
        cases = {10: "Below 10k", 15: "10-15K", 20: "16-20K"}  # at 122 taka per dollar
        for lakh, band in cases.items():
            fields, _, _ = self.tool.normalise({"budget_amount": lakh, "budget_unit": "lakh_taka"})
            self.assertEqual(fields["Budget"], band, lakh)
        _, _, warnings = self.tool.normalise({"budget_amount": 40, "budget_unit": "lakh_taka"})
        self.assertIn("budget_above_training_range", [w["code"] for w in warnings])

    def test_gpa_on_five_point_scale(self):
        fields, _, warnings = self.tool.normalise({"gpa": 5.0})
        self.assertEqual(fields["Result"], 4.0)
        self.assertIn("gpa_converted", [w["code"] for w in warnings])

    def test_name_matching(self):
        t = self.tool
        self.assertEqual(match_name("CSE", t.courses, COURSE_ALIASES), "Computer Science")
        self.assertEqual(match_name("Master of IT", t.courses, COURSE_ALIASES), "IT")
        self.assertEqual(match_name("Engineering", t.courses, COURSE_ALIASES), None)  # ambiguous
        self.assertEqual(match_name("Toronto University", t.institutions, INSTITUTION_ALIASES), "University of Toronto")
        self.assertEqual(match_name("kings college", t.institutions, INSTITUTION_ALIASES), "King's College London")
        self.assertEqual(match_name("Monash", t.institutions, INSTITUTION_ALIASES), "Monash University")

    def test_country_outside_model_still_scored_with_warning(self):
        result = self.tool.assess({**COMPLETE, "country": "Italy", "institution": None})
        self.assertEqual(result["status"], "ok")
        self.assertIn("country_not_in_model", [w["code"] for w in result["warnings"]])

    def test_institution_country_mismatch_warned(self):
        _, _, warnings = self.tool.normalise({"country": "UK", "institution": "Monash"})
        self.assertIn("institution_country_mismatch", [w["code"] for w in warnings])

    def test_what_if_only_higher_due_diligence(self):
        view = self.tool.assess(COMPLETE)["counsellor_view"]
        self.assertEqual(list(view["due_diligence_what_if"]["probability_if"]), ["Yes Medium", "Yes High"])
        probs = list(view["due_diligence_what_if"]["probability_if"].values())
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertTrue(np.isfinite(probs).all())


if __name__ == "__main__":
    unittest.main()
