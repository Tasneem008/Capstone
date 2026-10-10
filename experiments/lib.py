"""Shared helpers for the Capstone C experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    ENGINEERED_DATA_PATH,
    MODEL_FEATURE_COLUMNS,
    PHASE_MAP,
    RANDOM_STATE,
    get_model_matrix,
    load_engineered_data,
)

RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PHASE_ORDER = sorted(PHASE_MAP, key=PHASE_MAP.get)

# Encodings that repeat another column. Kept out of the reduced model.
REDUNDANT_FEATURES = [
    "Budget",
    "Budget_Midpoint",
    "Budget_Level_Num",
    "Budget_Academic_Score",
    "Researched_Bin",
    "Phase_Number",
    "Academic_Level_Num",
    "Result_Band",
    "GPA_Academic_Match",
]

REDUCED_FEATURES = [
    column for column in MODEL_FEATURE_COLUMNS if column not in REDUNDANT_FEATURES
]

# Columns that identify the political window. Removed in leave-one-phase-out
# so an unseen period is not one-hot encoded as an all-zero phase dummy,
# and so the model cannot read the window off the calendar.
PHASE_WINDOW_FEATURES = [
    "Political_Phase",
    "Phase_Number",
    "Year",
    "Days_Since_First_Intake",
]

FINGERPRINT = [
    "Name",
    "Result",
    "Course",
    "Country",
    "Institution",
    "Budget",
    "Researched",
    "Academic_Level",
    "Continuation",
]


def load_frame() -> pd.DataFrame:
    frame = load_engineered_data()
    frame["Date Intake"] = pd.to_datetime(frame["Date Intake"], dayfirst=True, format="mixed")
    return frame


def matrix(frame: pd.DataFrame, columns: list[str] | None = None):
    features, target = get_model_matrix(frame)
    if columns is not None:
        features = features.loc[:, columns]
    return features, target.astype(int)


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop later-phase copies of an earlier fingerprint, then within-phase copies.

    The earliest political window is kept. This is the sensitivity set, not a
    silent replacement of the modeling file.
    """
    ordered = frame.copy()
    ordered["_phase_ord"] = ordered["Political_Phase"].map(PHASE_MAP)
    ordered = ordered.sort_values(["_phase_ord", "Date Intake"]).reset_index(drop=True)
    before = len(ordered)
    cross = ordered.duplicated(subset=FINGERPRINT, keep="first")
    kept = ordered.loc[~cross].drop(columns="_phase_ord").reset_index(drop=True)
    within = kept.duplicated(subset=FINGERPRINT, keep="first")
    kept = kept.loc[~within].reset_index(drop=True)
    summary = {
        "rows_before": int(before),
        "cross_phase_copies_removed": int(cross.sum()),
        "within_phase_copies_removed": int(within.sum()),
        "rows_after": int(len(kept)),
    }
    return kept, summary


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", encoder, categorical))
    return ColumnTransformer(transformers=transformers)


def make_gb() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.10,
        max_depth=3,
        random_state=RANDOM_STATE,
    )


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.10,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def make_logit() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)


def as_pipeline(model, features: pd.DataFrame) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(features)),
            ("model", model),
        ]
    )


def score_predictions(y_true, y_pred, y_prob) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    roc = np.nan
    if len(np.unique(y_true)) > 1 and np.nanmax(y_prob) > np.nanmin(y_prob):
        roc = roc_auc_score(y_true, y_prob)
    elif len(np.unique(y_true)) > 1:
        roc = 0.5
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc,
        "Brier": brier_score_loss(y_true, np.clip(y_prob, 0, 1)),
    }


def mean_ci(values, alpha: float = 0.05) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    n = len(array)
    mean = float(array.mean()) if n else np.nan
    if n < 2:
        return {"mean": mean, "low": mean, "high": mean, "n": n}
    se = float(array.std(ddof=1) / np.sqrt(n))
    critical = float(stats.t.ppf(1 - alpha / 2, n - 1))
    return {
        "mean": mean,
        "low": mean - critical * se,
        "high": mean + critical * se,
        "n": n,
    }


def nadeau_bengio(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """Corrected resampled t-test (Nadeau and Bengio, 2003)."""
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    n = len(diff)
    mean = float(diff.mean())
    variance = float(diff.var(ddof=1))
    corrected = (1.0 / n + n_test / n_train) * variance
    t_stat = mean / np.sqrt(corrected) if corrected > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_stat), n - 1))
    return {"mean_diff": mean, "t": float(t_stat), "df": n - 1, "p": p_value}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on discordant test-set predictions."""
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    b_only = int((~correct_a & correct_b).sum())
    a_only = int((correct_a & ~correct_b).sum())
    n_discordant = a_only + b_only
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(a_only, n_discordant, 0.5).pvalue)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p": p_value}
