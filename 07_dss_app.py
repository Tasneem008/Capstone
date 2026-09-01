"""Streamlit decision support system for student continuation counseling.

Extended with:
  - Analytics Dashboard  (continuation trends by phase / budget / research / academic level)
  - What-If Scenario Simulator (compare a modified profile against a baseline)
  - Personalized Counseling Recommendations (SHAP-grounded, group-level advice)
  - Student Assessment Report (structured summary, downloadable as Markdown)
"""

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from common import (
    PHASE_MAP,
    build_student_features,
    load_engineered_data,
    load_gradient_boosting_model,
)
from risk_analyzer import calculate_risk_tier

APP_TITLE = "Student Continuation Decision Support System (SARP-Net)"

# ---------------------------------------------------------------------------
# Static reference data: mirrors the ablation feature groups (Table 8 of the
# status report) and the 21-column model contract from common.py.
# ---------------------------------------------------------------------------

CATEGORICAL_COLUMNS = [
    "Political_Phase",
    "Budget_Level",
    "Academic_Level",
    "Result_Band",
    "Institution",
    "Country",
    "Researched",
    "Course",
    "Budget",
]

FEATURE_GROUPS = {
    "Financial": [
        "Budget", "Budget_Level", "Budget_Midpoint",
        "Budget_Level_Num", "Budget_Academic_Score",
    ],
    "Academic": [
        "Result", "Academic_Level", "Academic_Level_Num",
        "Result_Band", "GPA_Academic_Match",
    ],
    "Phase_Temporal": [
        "Political_Phase", "Intake_Month", "Year", "Season",
        "Days_Since_First_Intake", "Phase_Number",
    ],
    "Destination": ["Course", "Country", "Institution"],
    "Behavioral": ["Researched", "Researched_Bin"],
}

GROUP_LABELS = {
    "Financial": "💰 Financial / Budget",
    "Academic": "🎓 Academic Profile",
    "Phase_Temporal": "🗓️ Political Phase / Timing",
    "Destination": "🌍 Destination Fit",
    "Behavioral": "🔎 Research Behavior",
}

FRIENDLY_NAMES = {
    "Result": "Academic GPA",
    "Course": "Chosen Course",
    "Country": "Destination Country",
    "Institution": "Destination Institution",
    "Budget": "Budget Band",
    "Researched": "Researched Options",
    "Political_Phase": "Political Phase",
    "Budget_Level": "Budget Level",
    "Academic_Level": "Academic Level",
    "Intake_Month": "Intake Month",
    "Year": "Intake Year",
    "Season": "Intake Season",
    "Days_Since_First_Intake": "Days Since First Intake",
    "Budget_Midpoint": "Budget Midpoint",
    "Phase_Number": "Political Phase (numeric)",
    "Budget_Level_Num": "Budget Level (numeric)",
    "Academic_Level_Num": "Academic Level (numeric)",
    "Researched_Bin": "Researched (binary)",
    "Result_Band": "GPA Band",
    "Budget_Academic_Score": "Budget + Academic Score",
    "GPA_Academic_Match": "GPA / Academic Level Match",
    "Date Intake": "Intake Date",
}

# Ablation F1-drop context (Table 9) baked into the recommendation copy so
# counselors see *why* a group matters, not just that it does.
RECOMMENDATION_TEMPLATES = {
    "Behavioral": {
        "negative": (
            "**Research behavior** is pulling this prediction toward "
            "non-continuation. This is historically the single strongest "
            "driver of dropout risk (ablation F1 drop of 0.391 when removed). "
            "Recommend a structured research session covering visa "
            "requirements, cost of living, and course accreditation before "
            "the student finalizes plans."
        ),
        "positive": (
            "The student has already researched their options, which is the "
            "strongest positive predictor available in this model. Reinforce "
            "it by connecting them with current students or alumni from the "
            "chosen institution."
        ),
    },
    "Financial": {
        "negative": (
            "**Budget constraints** are a significant risk factor here "
            "(ablation F1 drop of 0.209 when removed). Discuss scholarships, "
            "part-time work eligibility, cost-of-living differences between "
            "destination cities, or more affordable course/institution "
            "alternatives."
        ),
        "positive": (
            "Budget level is comfortably matched to the destination choice. "
            "No immediate financial counseling is needed, but confirm the "
            "budget still covers the full program duration."
        ),
    },
    "Academic": {
        "negative": (
            "The **academic profile** (GPA / academic level) is not well "
            "aligned with the chosen course or institution. Consider "
            "foundation or bridging programs, or discuss alternative courses "
            "and institutions with entry requirements closer to the "
            "student's current GPA."
        ),
        "positive": (
            "Academic standing is a strong fit for the intended course and "
            "institution. Encourage the student to leverage this in "
            "scholarship or competitive-program applications."
        ),
    },
    "Phase_Temporal": {
        "negative": (
            "The current **political phase / intake timing** is contributing "
            "meaningfully to uncertainty (ablation F1 drop of 0.142 when "
            "removed). Provide up-to-date guidance on visa and policy "
            "changes, and consider whether a different intake window would "
            "reduce exposure to this instability."
        ),
        "positive": (
            "Timing and political climate are currently favorable for this "
            "student's pathway. No additional action needed here, but keep "
            "monitoring for policy shifts before the intake date."
        ),
    },
    "Destination": {
        "negative": (
            "The **course / country / institution combination** is a "
            "comparatively weaker historical fit. Explore alternative "
            "institutions or programs with stronger continuation track "
            "records for similar student profiles."
        ),
        "positive": (
            "The chosen destination combination is historically associated "
            "with strong continuation. No change recommended on this front."
        ),
    },
}

RISK_TIER_ADVICE = {
    "Low Risk": (
        "Overall risk is low. A light-touch check-in before departure is "
        "likely sufficient."
    ),
    "Medium Risk": (
        "Overall risk is transitional/uncertain. Schedule at least one "
        "structured follow-up before the intake date to address the factors "
        "below."
    ),
    "High Risk": (
        "Overall risk is high. Prioritize this student for direct, "
        "in-person counseling before the intake date — the factors below "
        "should be addressed as early as possible."
    ),
}

PROFILE_DEFAULTS = {
    "Result": 3.20,
    "Academic_Level": "Medium",
    "Budget": "10-15K",
    "Researched": "No",
    "Political_Phase": "Stable",
}

BUDGET_TO_LEVEL = {"Below 10k": "Low", "10-15K": "Medium", "16-20K": "High"}


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------

@st.cache_data
def get_reference_data():
    """Load the model-compatible engineered records once per app process."""
    return load_engineered_data()


@st.cache_resource
def get_model_and_explainer():
    """Load the fitted pipeline and initialize its SHAP tree explainer."""
    model = load_gradient_boosting_model()
    explainer = shap.TreeExplainer(model.named_steps["model"])
    return model, explainer


def sorted_options(df, column):
    """Return clean, deterministic sidebar choices from the training data."""
    return sorted(df[column].dropna().astype(str).unique().tolist())


# ---------------------------------------------------------------------------
# Shared prediction / explanation helpers (used by every tab)
# ---------------------------------------------------------------------------

def raw_column_from_transformed(name):
    """Map a ColumnTransformer output name (e.g. 'cat__Budget_Level_High')
    back to its original engineered column name."""
    if name.startswith("num__"):
        return name[len("num__"):]
    if name.startswith("cat__"):
        remainder = name[len("cat__"):]
        for col in sorted(CATEGORICAL_COLUMNS, key=len, reverse=True):
            if remainder == col or remainder.startswith(col + "_"):
                return col
        return remainder
    return name


def get_feature_group(raw_column):
    for group, columns in FEATURE_GROUPS.items():
        if raw_column in columns:
            return group
    return "Other"


def friendly_name(raw_column):
    return FRIENDLY_NAMES.get(raw_column, raw_column)


def render_risk_tier(risk_tier):
    """Render the risk result using Streamlit's semantic color components."""
    if risk_tier.startswith("Low Risk"):
        st.success(risk_tier)
    elif risk_tier.startswith("Medium Risk"):
        st.warning(risk_tier)
    else:
        st.error(risk_tier)


def get_shap_explanation(explainer, model, student_features):
    """Return a binarized (positive-class) SHAP explanation for one student."""
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(student_features)
    transformed_df = pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
    )
    explanation = explainer(transformed_df)

    if explanation.values.ndim == 3:
        base_values = explanation.base_values
        if getattr(base_values, "ndim", 0) == 2:
            base_values = base_values[:, 1]
        explanation = shap.Explanation(
            values=explanation.values[:, :, 1],
            base_values=base_values,
            data=explanation.data,
            feature_names=explanation.feature_names,
        )
    return explanation


def render_waterfall(explanation):
    """Render a local SHAP waterfall for the positive continuation class."""
    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    figure = plt.gcf()
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True, width="stretch")
    plt.close(figure)


def compute_top_factors(explanation, top_n=8):
    """Return the top-N individual features by absolute SHAP magnitude."""
    values = explanation.values[0]
    names = explanation.feature_names
    pairs = sorted(zip(names, values), key=lambda p: abs(p[1]), reverse=True)

    top_factors = []
    for name, value in pairs[:top_n]:
        raw_column = raw_column_from_transformed(name)
        top_factors.append(
            {
                "feature": name,
                "raw_column": raw_column,
                "label": friendly_name(raw_column),
                "group": get_feature_group(raw_column),
                "shap_value": float(value),
            }
        )
    return top_factors


def compute_group_contributions(explanation):
    """Aggregate SHAP values across ALL transformed features, grouped by the
    ablation-study feature groups. Used to ground recommendations."""
    values = explanation.values[0]
    names = explanation.feature_names
    scores = {group: 0.0 for group in FEATURE_GROUPS}
    for name, value in zip(names, values):
        raw_column = raw_column_from_transformed(name)
        group = get_feature_group(raw_column)
        if group in scores:
            scores[group] += float(value)
    return scores


def generate_recommendations(group_contributions, risk_tier, max_recommendations=4):
    """Turn group-level SHAP contributions into counselor-facing advice."""
    ranked = sorted(
        group_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    recommendations = []
    for group, score in ranked:
        template = RECOMMENDATION_TEMPLATES.get(group)
        if not template or abs(score) < 1e-6:
            continue
        direction = "negative" if score < 0 else "positive"
        recommendations.append(
            {
                "group": group,
                "label": GROUP_LABELS.get(group, group),
                "direction": direction,
                "score": score,
                "text": template[direction],
            }
        )
        if len(recommendations) >= max_recommendations:
            break

    tier_key = risk_tier.split(" (")[0]
    overall_advice = RISK_TIER_ADVICE.get(tier_key, "")
    return recommendations, overall_advice


def run_prediction(profile, reference_df, model):
    """Score one student profile and return features + probability + tier."""
    student_features = build_student_features(profile, reference_df)
    probability = float(model.predict_proba(student_features)[0, 1])
    risk_tier = calculate_risk_tier(probability)
    return student_features, probability, risk_tier


def format_profile_table(profile):
    rows = [
        {"Field": friendly_name(key), "Value": str(value)}
        for key, value in profile.items()
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sidebar: student profile entry (feeds the Individual Assessment tab, and
# supplies default values for the What-If simulator)
# ---------------------------------------------------------------------------

def collect_profile(reference_df):
    """Build the counselor input form and return a raw student profile."""
    st.sidebar.header("Student Profile")

    result = st.sidebar.slider(
        "Academic result (GPA)",
        min_value=0.0,
        max_value=4.0,
        value=3.20,
        step=0.01,
    )
    academic_level = st.sidebar.selectbox(
        "Academic Level",
        ["Low", "Medium", "High"],
        index=1,
    )
    budget = st.sidebar.selectbox(
        "Budget",
        ["Below 10k", "10-15K", "16-20K"],
        index=1,
    )
    budget_level = BUDGET_TO_LEVEL[budget]
    st.sidebar.caption(f"Derived budget level: {budget_level}")

    course = st.sidebar.selectbox(
        "Course",
        sorted_options(reference_df, "Course"),
    )
    country = st.sidebar.selectbox(
        "Destination Country",
        sorted_options(reference_df, "Country"),
    )
    institutions = sorted_options(
        reference_df.loc[reference_df["Country"].astype(str) == country],
        "Institution",
    )
    institution = st.sidebar.selectbox("Institution", institutions)
    researched = st.sidebar.selectbox("Researched options?", ["No", "Yes"])
    political_phase = st.sidebar.selectbox(
        "Political Phase",
        list(PHASE_MAP.keys()),
    )

    intake_min = pd.to_datetime(
        reference_df["Date Intake"], dayfirst=True, format="mixed"
    ).min()
    intake_date = st.sidebar.date_input(
        "Intake Date",
        value=pd.Timestamp.today().date(),
        min_value=intake_min.date(),
    )
    season = ((intake_date.month % 12) // 3) + 1
    st.sidebar.caption(f"Derived intake season: {season}")

    return {
        "Result": result,
        "Course": course,
        "Country": country,
        "Institution": institution,
        "Budget": budget,
        "Researched": researched,
        "Political_Phase": political_phase,
        "Budget_Level": budget_level,
        "Academic_Level": academic_level,
        "Date Intake": intake_date,
    }


# ---------------------------------------------------------------------------
# Tab 1: Individual Assessment
# ---------------------------------------------------------------------------

def render_individual_assessment(profile, reference_df, model, explainer):
    st.subheader("Individual Student Assessment")

    analyze_clicked = st.sidebar.button(
        "Analyze Student", type="primary", width="stretch"
    )

    if analyze_clicked:
        student_features, probability, risk_tier = run_prediction(
            profile, reference_df, model
        )
        explanation = get_shap_explanation(explainer, model, student_features)
        top_factors = compute_top_factors(explanation)
        group_contributions = compute_group_contributions(explanation)

        st.session_state["last_analysis"] = {
            "profile": profile,
            "probability": probability,
            "risk_tier": risk_tier,
            "explanation": explanation,
            "top_factors": top_factors,
            "group_contributions": group_contributions,
            "timestamp": datetime.now(),
        }

    last_analysis = st.session_state.get("last_analysis")

    if not last_analysis:
        st.info("Enter a student profile in the sidebar, then select Analyze Student.")
        return

    probability = last_analysis["probability"]
    risk_tier = last_analysis["risk_tier"]

    probability_column, risk_column = st.columns([1, 2])
    with probability_column:
        st.subheader("Continuation Probability")
        st.metric("Probability Score", f"{probability:.1%}")
    with risk_column:
        st.subheader("Risk Level")
        render_risk_tier(risk_tier)

    intake_max = pd.to_datetime(
        reference_df["Date Intake"], dayfirst=True, format="mixed"
    ).max()
    if pd.Timestamp(last_analysis["profile"]["Date Intake"]) > intake_max:
        st.info(
            "The selected intake date is later than the training period "
            f"({intake_max.date()}); interpret this extrapolation carefully."
        )

    st.subheader("Important Factors")
    st.caption(
        "Positive SHAP values push the prediction toward continuation; "
        "negative values push it toward non-continuation."
    )
    render_waterfall(last_analysis["explanation"])


# ---------------------------------------------------------------------------
# Tab 2: Analytics Dashboard
# ---------------------------------------------------------------------------

def rate_by_category(df, column, order=None):
    grouped = df.groupby(column)["Continuation_Bin"].agg(["mean", "size"])
    grouped.columns = ["Continuation Rate (%)", "Count"]
    grouped["Continuation Rate (%)"] = (grouped["Continuation Rate (%)"] * 100).round(1)
    if order:
        grouped = grouped.reindex(order)
    return grouped


def render_dashboard(reference_df):
    st.subheader("📊 Analytics Dashboard")
    st.caption(
        "Continuation trends across the 6,000-record training corpus "
        "(5 political phases × budget × research behavior × academic level)."
    )

    total_students = len(reference_df)
    overall_rate = reference_df["Continuation_Bin"].mean() * 100
    top_row = st.columns(3)
    top_row[0].metric("Total Students", f"{total_students:,}")
    top_row[1].metric("Overall Continuation Rate", f"{overall_rate:.1f}%")
    top_row[2].metric(
        "Class Balance",
        f"{int(reference_df['Continuation_Bin'].sum())} Yes / "
        f"{int((1 - reference_df['Continuation_Bin']).sum())} No",
    )

    phase_order = list(PHASE_MAP.keys())
    level_order = ["Low", "Medium", "High"]
    researched_order = ["No", "Yes"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**By Political Phase**")
        phase_rates = rate_by_category(reference_df, "Political_Phase", phase_order)
        st.bar_chart(phase_rates["Continuation Rate (%)"])
    with col_b:
        st.markdown("**By Budget Level**")
        budget_rates = rate_by_category(reference_df, "Budget_Level", level_order)
        st.bar_chart(budget_rates["Continuation Rate (%)"])

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**By Research Behavior**")
        research_rates = rate_by_category(reference_df, "Researched", researched_order)
        st.bar_chart(research_rates["Continuation Rate (%)"])
    with col_d:
        st.markdown("**By Academic Level**")
        academic_rates = rate_by_category(reference_df, "Academic_Level", level_order)
        st.bar_chart(academic_rates["Continuation Rate (%)"])

    st.markdown("**Political Phase × Research Behavior (interaction view)**")
    interaction = (
        pd.crosstab(
            reference_df["Political_Phase"],
            reference_df["Researched"],
            values=reference_df["Continuation_Bin"],
            aggfunc="mean",
        )
        .reindex(phase_order)
        .mul(100)
        .round(1)
    )
    st.bar_chart(interaction)

    st.markdown("**Continuation Rate Over Intake Period**")
    period_df = reference_df.copy()
    period_df["Period"] = (
        period_df["Year"].astype(str) + "-S" + period_df["Season"].astype(str)
    )
    period_order = (
        period_df[["Period", "Year", "Season"]]
        .drop_duplicates()
        .sort_values(["Year", "Season"])["Period"]
    )
    period_rates = (
        period_df.groupby("Period")["Continuation_Bin"].mean().mul(100).round(1)
    )
    period_rates = period_rates.reindex(period_order)
    st.line_chart(period_rates)
    st.caption(
        "Season: 1 = Dec–Feb, 2 = Mar–May, 3 = Jun–Aug, 4 = Sep–Nov."
    )

    with st.expander("Show underlying counts"):
        st.write("Political Phase")
        st.dataframe(phase_rates, width="stretch")
        st.write("Budget Level")
        st.dataframe(budget_rates, width="stretch")
        st.write("Research Behavior")
        st.dataframe(research_rates, width="stretch")
        st.write("Academic Level")
        st.dataframe(academic_rates, width="stretch")


# ---------------------------------------------------------------------------
# Tab 3: What-If / Scenario Simulator
# ---------------------------------------------------------------------------

def get_baseline_profile(reference_df):
    last_analysis = st.session_state.get("last_analysis")
    if last_analysis:
        return dict(last_analysis["profile"])

    defaults = dict(PROFILE_DEFAULTS)
    defaults["Course"] = sorted_options(reference_df, "Course")[0]
    defaults["Country"] = sorted_options(reference_df, "Country")[0]
    defaults["Institution"] = sorted_options(
        reference_df.loc[reference_df["Country"].astype(str) == defaults["Country"]],
        "Institution",
    )[0]
    defaults["Budget_Level"] = BUDGET_TO_LEVEL[defaults["Budget"]]
    defaults["Date Intake"] = pd.Timestamp.today().date()
    return defaults


def render_whatif(reference_df, model, explainer):
    st.subheader("🔄 What-If / Scenario Simulator")
    st.caption(
        "Adjust one or more fields below and compare the resulting "
        "continuation probability against the baseline profile."
    )

    baseline_profile = get_baseline_profile(reference_df)
    if st.session_state.get("last_analysis"):
        st.info(
            "Baseline = the last profile analyzed in the Individual "
            "Assessment tab."
        )
    else:
        st.info(
            "No assessment has been run yet, so a default baseline profile "
            "is used. Run an assessment in the first tab to compare against "
            "a real student instead."
        )

    with st.expander("Baseline profile", expanded=False):
        st.dataframe(format_profile_table(baseline_profile), hide_index=True)

    st.markdown("**Modify scenario**")
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        scenario_result = st.slider(
            "Scenario GPA",
            0.0, 4.0, float(baseline_profile["Result"]), 0.01,
            key="whatif_result",
        )
        scenario_academic_level = st.selectbox(
            "Scenario Academic Level",
            ["Low", "Medium", "High"],
            index=["Low", "Medium", "High"].index(baseline_profile["Academic_Level"]),
            key="whatif_academic_level",
        )
        scenario_budget = st.selectbox(
            "Scenario Budget",
            ["Below 10k", "10-15K", "16-20K"],
            index=["Below 10k", "10-15K", "16-20K"].index(baseline_profile["Budget"]),
            key="whatif_budget",
        )
        scenario_researched = st.selectbox(
            "Scenario Researched Options?",
            ["No", "Yes"],
            index=["No", "Yes"].index(baseline_profile["Researched"]),
            key="whatif_researched",
        )
    with form_col2:
        scenario_phase = st.selectbox(
            "Scenario Political Phase",
            list(PHASE_MAP.keys()),
            index=list(PHASE_MAP.keys()).index(baseline_profile["Political_Phase"]),
            key="whatif_phase",
        )
        course_options = sorted_options(reference_df, "Course")
        scenario_course = st.selectbox(
            "Scenario Course",
            course_options,
            index=course_options.index(baseline_profile["Course"])
            if baseline_profile["Course"] in course_options else 0,
            key="whatif_course",
        )
        country_options = sorted_options(reference_df, "Country")
        scenario_country = st.selectbox(
            "Scenario Destination Country",
            country_options,
            index=country_options.index(baseline_profile["Country"])
            if baseline_profile["Country"] in country_options else 0,
            key="whatif_country",
        )
        institution_options = sorted_options(
            reference_df.loc[reference_df["Country"].astype(str) == scenario_country],
            "Institution",
        )
        scenario_institution = st.selectbox(
            "Scenario Institution",
            institution_options,
            key="whatif_institution",
        )

    compare_clicked = st.button("Compare Scenario", type="primary")

    if not compare_clicked:
        return

    scenario_profile = dict(baseline_profile)
    scenario_profile.update(
        {
            "Result": scenario_result,
            "Academic_Level": scenario_academic_level,
            "Budget": scenario_budget,
            "Budget_Level": BUDGET_TO_LEVEL[scenario_budget],
            "Researched": scenario_researched,
            "Political_Phase": scenario_phase,
            "Course": scenario_course,
            "Country": scenario_country,
            "Institution": scenario_institution,
        }
    )

    _, baseline_probability, baseline_tier = run_prediction(
        baseline_profile, reference_df, model
    )
    scenario_features, scenario_probability, scenario_tier = run_prediction(
        scenario_profile, reference_df, model
    )

    delta = scenario_probability - baseline_probability

    st.markdown("### Comparison")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Baseline Probability", f"{baseline_probability:.1%}")
    metric_cols[1].metric(
        "Scenario Probability",
        f"{scenario_probability:.1%}",
        delta=f"{delta:+.1%}",
    )
    metric_cols[2].metric(
        "Risk Tier",
        scenario_tier.split(" (")[0],
        delta=None if baseline_tier == scenario_tier else "changed",
    )

    st.bar_chart(
        pd.DataFrame(
            {
                "Probability": [baseline_probability, scenario_probability],
            },
            index=["Baseline", "Scenario"],
        )
    )

    render_risk_tier(scenario_tier)

    changed_fields = [
        key
        for key in scenario_profile
        if str(scenario_profile[key]) != str(baseline_profile.get(key))
    ]
    if changed_fields:
        st.markdown("**Fields changed from baseline:**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Field": [friendly_name(f) for f in changed_fields],
                    "Baseline": [str(baseline_profile.get(f)) for f in changed_fields],
                    "Scenario": [str(scenario_profile[f]) for f in changed_fields],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    with st.expander("See detailed factors for this scenario"):
        scenario_explanation = get_shap_explanation(
            explainer, model, scenario_features
        )
        render_waterfall(scenario_explanation)


# ---------------------------------------------------------------------------
# Tab 4: Personalized Counseling Recommendations
# ---------------------------------------------------------------------------

def render_recommendations():
    st.subheader("💡 Personalized Counseling Recommendations")

    last_analysis = st.session_state.get("last_analysis")
    if not last_analysis:
        st.info(
            "Run an assessment in the Individual Assessment tab first — "
            "recommendations are generated from that student's prediction "
            "and SHAP factors."
        )
        return

    risk_tier = last_analysis["risk_tier"]
    group_contributions = last_analysis["group_contributions"]
    recommendations, overall_advice = generate_recommendations(
        group_contributions, risk_tier
    )

    render_risk_tier(risk_tier)
    if overall_advice:
        st.markdown(f"**Overall guidance:** {overall_advice}")

    st.markdown("---")
    st.markdown("**Areas ranked by model impact:**")
    for rec in recommendations:
        icon = "⚠️" if rec["direction"] == "negative" else "✅"
        with st.container():
            st.markdown(f"{icon} **{rec['label']}**")
            st.write(rec["text"])
            st.caption(f"Aggregate SHAP contribution: {rec['score']:+.3f}")

    st.markdown("---")
    st.markdown("**Top individual factors (from SHAP):**")
    factors_df = pd.DataFrame(
        [
            {
                "Factor": f["label"],
                "Group": GROUP_LABELS.get(f["group"], f["group"]),
                "SHAP Value": round(f["shap_value"], 3),
                "Direction": "Toward continuation" if f["shap_value"] > 0
                else "Toward non-continuation",
            }
            for f in last_analysis["top_factors"]
        ]
    )
    st.dataframe(factors_df, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Tab 5: Student Assessment Report
# ---------------------------------------------------------------------------

def build_report_markdown(last_analysis):
    profile = last_analysis["profile"]
    probability = last_analysis["probability"]
    risk_tier = last_analysis["risk_tier"]
    top_factors = last_analysis["top_factors"]
    group_contributions = last_analysis["group_contributions"]
    timestamp = last_analysis["timestamp"]

    recommendations, overall_advice = generate_recommendations(
        group_contributions, risk_tier
    )

    lines = []
    lines.append("# Student Continuation Assessment Report")
    lines.append(f"_Generated: {timestamp.strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")
    lines.append("## 1. Student Profile")
    for key, value in profile.items():
        lines.append(f"- **{friendly_name(key)}**: {value}")
    lines.append("")
    lines.append("## 2. Prediction")
    lines.append(f"- **Continuation Probability**: {probability:.1%}")
    lines.append(f"- **Risk Tier**: {risk_tier}")
    lines.append("")
    lines.append("## 3. Top Contributing Factors")
    lines.append("| Factor | Group | SHAP Value | Direction |")
    lines.append("|---|---|---|---|")
    for factor in top_factors:
        direction = (
            "Toward continuation" if factor["shap_value"] > 0
            else "Toward non-continuation"
        )
        lines.append(
            f"| {factor['label']} | {GROUP_LABELS.get(factor['group'], factor['group'])} "
            f"| {factor['shap_value']:+.3f} | {direction} |"
        )
    lines.append("")
    lines.append("## 4. Counseling Recommendations")
    if overall_advice:
        lines.append(f"**Overall guidance:** {overall_advice}")
        lines.append("")
    for rec in recommendations:
        lines.append(f"- **{rec['label']}**: {rec['text']}")
    lines.append("")
    lines.append("---")
    lines.append(
        "_Generated by SARP-Net (Student Continuation Decision Support "
        "System). Probability from Paper 2 Gradient Boosting; risk tier "
        "from Paper 1 optimized threshold (0.73)._"
    )
    return "\n".join(lines)


def render_report():
    st.subheader("📄 Student Assessment Report")

    last_analysis = st.session_state.get("last_analysis")
    if not last_analysis:
        st.info(
            "Run an assessment in the Individual Assessment tab first — "
            "the report is compiled from that student's result."
        )
        return

    report_markdown = build_report_markdown(last_analysis)
    st.markdown(report_markdown)

    timestamp = last_analysis["timestamp"].strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="⬇️ Download Report (Markdown)",
        data=report_markdown,
        file_name=f"assessment_report_{timestamp}.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="SARP-Net",
        page_icon="🎓",
        layout="wide",
    )
    st.title(APP_TITLE)
    st.caption(
        "Paper 2 Gradient Boosting inference with Paper 1 probability risk tiers"
    )

    reference_df = get_reference_data()
    model, explainer = get_model_and_explainer()
    profile = collect_profile(reference_df)

    tab_assess, tab_dashboard, tab_whatif, tab_recs, tab_report = st.tabs(
        [
            " Individual Assessment",
            " Analytics Dashboard",
            " What-If Simulator",
            " Recommendations",
            " Assessment Report",
        ]
    )

    with tab_assess:
        render_individual_assessment(profile, reference_df, model, explainer)

    with tab_dashboard:
        render_dashboard(reference_df)

    with tab_whatif:
        render_whatif(reference_df, model, explainer)

    with tab_recs:
        render_recommendations()

    with tab_report:
        render_report()


if __name__ == "__main__":
    main()