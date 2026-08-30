"""Streamlit decision support system for student continuation counseling."""

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

APP_TITLE = (
    "Student Continuation Decision Support System (SARP-Net)"
)


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


def render_risk_tier(risk_tier):
    """Render the risk result using Streamlit's semantic color components."""
    if risk_tier.startswith("Low Risk"):
        st.success(risk_tier)
    elif risk_tier.startswith("Medium Risk"):
        st.warning(risk_tier)
    else:
        st.error(risk_tier)


def render_local_explanation(model, explainer, student_features):
    """Render a local SHAP waterfall for the positive continuation class."""
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

    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    figure = plt.gcf()
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True, use_container_width=True)
    plt.close(figure)


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
    budget_level = {
        "Below 10k": "Low",
        "10-15K": "Medium",
        "16-20K": "High",
    }[budget]
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

    if st.sidebar.button("Analyze Student", type="primary", use_container_width=True):
        student_features = build_student_features(profile, reference_df)
        probability = float(model.predict_proba(student_features)[0, 1])
        risk_tier = calculate_risk_tier(probability)

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
        if pd.Timestamp(profile["Date Intake"]) > intake_max:
            st.info(
                "The selected intake date is later than the training period "
                f"({intake_max.date()}); interpret this extrapolation carefully."
            )

        st.subheader("Important Factors")
        st.caption(
            "Positive SHAP values push the prediction toward continuation; "
            "negative values push it toward non-continuation."
        )
        render_local_explanation(model, explainer, student_features)
    else:
        st.info("Enter a student profile in the sidebar, then select Analyze Student.")


if __name__ == "__main__":
    main()
