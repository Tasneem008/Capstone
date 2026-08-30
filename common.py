
# common.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Datasets"
ENGINEERED_DATA_PATH = DATA_DIR / "master_6000_engineered.csv"
PAPER1_XGB_MODEL_PATH = PROJECT_DIR / "results_paper1" / "xgboost_tuned_final.joblib"
PAPER2_GB_MODEL_PATH = PROJECT_DIR / "results_paper2" / "gradient_boosting.joblib"

MODEL_FEATURE_COLUMNS = [
    "Result",
    "Course",
    "Country",
    "Institution",
    "Budget",
    "Researched",
    "Political_Phase",
    "Budget_Level",
    "Academic_Level",
    "Intake_Month",
    "Year",
    "Season",
    "Days_Since_First_Intake",
    "Budget_Midpoint",
    "Phase_Number",
    "Budget_Level_Num",
    "Academic_Level_Num",
    "Researched_Bin",
    "Result_Band",
    "Budget_Academic_Score",
    "GPA_Academic_Match",
]

DATA_FILES = {
    "Stable": "stable.xlsx",
    "Unstable": "unstable.xlsx",
    "Transitional phase 1st 5 months": "transitional phase 1.xlsx",
    "Transitional phase Last 5 months": "transitional phase 2.xlsx",
    "Newly Elected": "Transitional.xlsx",
}

REQUIRED_COLUMNS = [
    "Name", "Result", "Course", "Country", "Institution", "Date Intake",
    "Budget", "Researched", "Political_Phase", "Budget_Level",
    "Academic_Level", "Continuation"
]

PHASE_MAP = {
    "Stable": 1,
    "Unstable": 2,
    "Transitional phase 1st 5 months": 3,
    "Transitional phase Last 5 months": 4,
    "Newly Elected": 5,
}

BUDGET_MIDPOINT = {
    "Below 10k": 8000,
    "10-15K": 12500,
    "16-20K": 18000,
}

def load_6k(base_dir="."):
    """Load the five 1,200-row Excel files and create the 6,000-row master table."""
    base_dir = Path(base_dir)
    data_dir = base_dir / "Datasets"
    if not data_dir.exists():
        # Backward compatibility with the original methodology-code layout.
        data_dir = base_dir.parent / "data"

    frames = []

    for phase, filename in DATA_FILES.items():
        path = data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Could not find Excel file: {path}\n"
                f"Expected data folder: {data_dir}"
            )

        df = pd.read_excel(path)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")

        # Use the project period definition, including correcting the stable-file label.
        df["Political_Phase"] = phase
        df["Source_File"] = filename
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)

    if len(master) != 6000:
        raise ValueError(f"Expected 6,000 rows, found {len(master)}.")

    return master


def engineer_features(df):
    """
    Feature engineering based on the supplied Dataset Description Report,
    while retaining the categorical variables so the weekly-report
    One-Hot Encoding approach can be reproduced.
    """
    d = df.copy()

    # Date features
    dt = pd.to_datetime(
        d["Date Intake"], errors="coerce", dayfirst=True, format="mixed"
    )
    if dt.isna().any():
        raise ValueError("Some Date Intake values could not be parsed.")

    d["Intake_Month"] = dt.dt.month
    d["Year"] = dt.dt.year

    # 1=Dec-Feb, 2=Mar-May, 3=Jun-Aug, 4=Sep-Nov.
    # This is an explicit implementation choice because the report only
    # states that Season is coded 1-4 without defining the month mapping.
    d["Season"] = ((dt.dt.month % 12) // 3) + 1

    d["Days_Since_First_Intake"] = (dt - dt.min()).dt.days

    # Derived features documented in the dataset report
    d["Budget_Midpoint"] = d["Budget"].map(BUDGET_MIDPOINT)
    d["Phase_Number"] = d["Political_Phase"].map(PHASE_MAP)

    d["Budget_Level_Num"] = d["Budget_Level"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    )
    d["Academic_Level_Num"] = d["Academic_Level"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    )
    d["Researched_Bin"] = d["Researched"].map({"No": 0, "Yes": 1})

    d["Result_Band"] = pd.cut(
        d["Result"],
        bins=[0, 2.99, 3.39, 4.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    d["Budget_Academic_Score"] = (
        d["Budget_Level_Num"] + d["Academic_Level_Num"]
    )

    d["GPA_Academic_Match"] = (
        d["Result_Band"].astype(str) == d["Academic_Level"]
    ).astype(int)

    # Target
    d["Continuation_Bin"] = d["Continuation"].map({"No": 0, "Yes": 1})
    if d["Continuation_Bin"].isna().any():
        raise ValueError("Continuation contains values other than Yes/No.")

    return d


def get_model_matrix(engineered_df):
    """
    Return X/y. Name and Source_File are identifiers and are excluded.
    Date Intake is replaced by derived date features.
    """
    drop_cols = [
        "Name", "Source_File", "Date Intake",
        "Continuation", "Continuation_Bin"
    ]
    X = engineered_df.drop(columns=drop_cols)
    y = engineered_df["Continuation_Bin"].astype(int)

    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in X.columns]
    if missing:
        raise ValueError(f"Engineered data is missing model features: {missing}")

    X = X.loc[:, MODEL_FEATURE_COLUMNS]
    return X, y


def make_preprocessor(X, scale_numeric=True):
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Compatibility with older scikit-learn versions.
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    if scale_numeric:
        numeric_pipe = StandardScaler()
    else:
        numeric_pipe = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", ohe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def save_master_and_cleaned(base_dir="."):
    base_dir = Path(base_dir)
    master = load_6k(base_dir)
    engineered = engineer_features(master)

    master.to_csv(base_dir / "master_6000_raw.csv", index=False)
    engineered.to_csv(base_dir / "master_6000_engineered.csv", index=False)

    return master, engineered


def load_engineered_data(path=ENGINEERED_DATA_PATH):
    """Load and validate the model-compatible 6,000-row engineered dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Engineered dataset not found: {path}")

    df = pd.read_csv(path)
    required = set(MODEL_FEATURE_COLUMNS + ["Continuation_Bin"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Engineered dataset is missing columns: {missing}")
    if len(df) != 6000:
        raise ValueError(f"Expected 6,000 engineered rows, found {len(df)}.")
    return df


def split_engineered_data(engineered_df=None, test_size=0.20):
    """Return the canonical stratified train/test split used by Papers 1 and 2."""
    if engineered_df is None:
        engineered_df = load_engineered_data()
    X, y = get_model_matrix(engineered_df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def evaluate_predictions(y_true, predictions):
    """Calculate the four classification metrics used in the ablation report."""
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
    }


def load_gradient_boosting_model(path=PAPER2_GB_MODEL_PATH):
    """Load the fitted Paper 2 Gradient Boosting preprocessing/model pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gradient Boosting model not found: {path}")
    try:
        model = joblib.load(path)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The saved model requires Python 3.10-3.13 with "
            "scikit-learn 1.7.1. Recreate the environment from requirements.txt."
        ) from error
    expected_steps = {"preprocessor", "model"}
    if not expected_steps.issubset(model.named_steps):
        raise TypeError(
            "Expected a fitted Pipeline with 'preprocessor' and 'model' steps."
        )
    return model


def build_student_features(profile, reference_df=None):
    """
    Convert one counselor-entered profile into the exact 21-column model input.

    The saved model owns the fitted scaler and one-hot encoder, so this function
    deliberately retains raw categorical labels.
    """
    if reference_df is None:
        reference_df = load_engineered_data()

    required_inputs = {
        "Result",
        "Course",
        "Country",
        "Institution",
        "Budget",
        "Researched",
        "Political_Phase",
        "Budget_Level",
        "Academic_Level",
        "Date Intake",
    }
    missing = sorted(required_inputs.difference(profile))
    if missing:
        raise ValueError(f"Student profile is missing fields: {missing}")

    intake_date = pd.to_datetime(profile["Date Intake"])
    reference_dates = pd.to_datetime(
        reference_df["Date Intake"],
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )
    if reference_dates.isna().any():
        raise ValueError("Reference data contains invalid intake dates.")
    first_intake = reference_dates.min()

    result = float(profile["Result"])
    if not 0 <= result <= 4.0:
        raise ValueError("Result must be between 0.0 and 4.0.")

    budget_level_num = {"Low": 0, "Medium": 1, "High": 2}[
        profile["Budget_Level"]
    ]
    academic_level_num = {"Low": 0, "Medium": 1, "High": 2}[
        profile["Academic_Level"]
    ]
    result_band = (
        "Low" if result <= 2.99 else "Medium" if result <= 3.39 else "High"
    )

    row = {
        "Result": result,
        "Course": profile["Course"],
        "Country": profile["Country"],
        "Institution": profile["Institution"],
        "Budget": profile["Budget"],
        "Researched": profile["Researched"],
        "Political_Phase": profile["Political_Phase"],
        "Budget_Level": profile["Budget_Level"],
        "Academic_Level": profile["Academic_Level"],
        "Intake_Month": intake_date.month,
        "Year": intake_date.year,
        "Season": ((intake_date.month % 12) // 3) + 1,
        "Days_Since_First_Intake": (intake_date - first_intake).days,
        "Budget_Midpoint": BUDGET_MIDPOINT[profile["Budget"]],
        "Phase_Number": PHASE_MAP[profile["Political_Phase"]],
        "Budget_Level_Num": budget_level_num,
        "Academic_Level_Num": academic_level_num,
        "Researched_Bin": {"No": 0, "Yes": 1}[profile["Researched"]],
        "Result_Band": result_band,
        "Budget_Academic_Score": budget_level_num + academic_level_num,
        "GPA_Academic_Match": int(result_band == profile["Academic_Level"]),
    }
    return pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)
