"""
Rebuild the 6,000-row study from Datasets/updated_datasheet and retrain
the Paper 1 and Paper 2 methodologies on that revision.

Due_Diligence is now four levels (No, Yes Low, Yes Medium, Yes High), so the
old Yes/No pipelines cannot be reused.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import (
    DATA_DIR,
    ENGINEERED_DATA_PATH,
    PROJECT_DIR,
    RANDOM_STATE,
    engineer_features,
    get_model_matrix,
    load_6k,
    make_preprocessor,
)

PAPER1_DIR = PROJECT_DIR / "results_paper1"
PAPER2_DIR = PROJECT_DIR / "results_paper2"


def metrics_row(name, y_true, pred, prob, threshold=0.5, with_ap=False):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    row = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, prob),
        "MCC": matthews_corrcoef(y_true, pred),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    if with_ap:
        row["Average_Precision"] = average_precision_score(y_true, prob)
    return row


def best_f1_threshold(y_true, probabilities):
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "Threshold": float(threshold),
                "F1": f1_score(y_true, pred, zero_division=0),
                "Precision": precision_score(y_true, pred, zero_division=0),
                "Recall": recall_score(y_true, pred, zero_division=0),
            }
        )
    table = pd.DataFrame(rows)
    best = float(table.loc[table["F1"].idxmax(), "Threshold"])
    return best, table


def rebuild_dataset():
    backup = DATA_DIR / "master_6000_engineered_pre_revision.csv"
    if ENGINEERED_DATA_PATH.exists() and not backup.exists():
        shutil.copy2(ENGINEERED_DATA_PATH, backup)
        print(f"Backed up previous engineered data to {backup}")

    master = load_6k(PROJECT_DIR)
    engineered = engineer_features(master)
    master.to_csv(DATA_DIR / "master_6000_raw_revised.csv", index=False)
    engineered.to_csv(ENGINEERED_DATA_PATH, index=False)

    print("\nRevised dataset")
    print(f"Rows: {len(engineered)}")
    print("Continuation rate by phase:")
    print(
        engineered.groupby("Political_Phase")["Continuation_Bin"]
        .mean()
        .mul(100)
        .round(2)
        .to_string()
    )
    print("Continuation rate by Due_Diligence:")
    print(
        engineered.groupby("Researched")["Continuation_Bin"]
        .mean()
        .mul(100)
        .round(2)
        .to_string()
    )
    print(
        "Overall continuation:",
        f"{engineered['Continuation_Bin'].mean() * 100:.2f}%",
    )
    print("Intake range:", engineered["Date Intake"].min().date(), "to", engineered["Date Intake"].max().date())
    return engineered


def train_paper1(X_train, X_test, y_train, y_test):
    PAPER1_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor, _, _ = make_preprocessor(X_train, scale_numeric=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def make_pipeline(model):
        return ImbPipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("undersample", RandomUnderSampler(random_state=RANDOM_STATE)),
                ("model", model),
            ]
        )

    models = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
            )
        ),
        "LightGBM": make_pipeline(
            LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
    }

    xgb_pipe = make_pipeline(
        XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    grid = GridSearchCV(
        estimator=xgb_pipe,
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 7],
            "model__learning_rate": [0.05, 0.10],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    print("\nTuning Paper 1 XGBoost...")
    grid.fit(X_train, y_train)
    models["XGBoost (Tuned)"] = grid.best_estimator_
    print("Best XGBoost parameters:", grid.best_params_)
    print("Best CV F1:", round(grid.best_score_, 4))

    results = []
    for name, pipe in models.items():
        if name != "XGBoost (Tuned)":
            pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results.append(metrics_row(name, y_test, pred, prob, 0.5, with_ap=True))
        print(f"Paper 1 {name}: F1={results[-1]['F1']:.4f} Acc={results[-1]['Accuracy']:.4f}")

    best_xgb = models["XGBoost (Tuned)"]
    oof_prob = cross_val_predict(
        best_xgb, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    best_threshold, threshold_df = best_f1_threshold(y_train, oof_prob)
    threshold_df.to_csv(PAPER1_DIR / "xgb_threshold_search.csv", index=False)
    print(f"Paper 1 OOF F1 threshold: {best_threshold:.2f}")

    best_xgb.fit(X_train, y_train)
    test_prob = best_xgb.predict_proba(X_test)[:, 1]
    optimized_pred = (test_prob >= best_threshold).astype(int)
    results.append(
        metrics_row(
            "XGBoost (Tuned, optimized threshold)",
            y_test,
            optimized_pred,
            test_prob,
            best_threshold,
            with_ap=True,
        )
    )
    joblib.dump(best_xgb, PAPER1_DIR / "xgboost_tuned_final.joblib")
    pd.DataFrame(results).to_csv(PAPER1_DIR / "paper1_model_comparison.csv", index=False)
    return best_threshold


def train_paper2(X_train, X_test, y_train, y_test):
    PAPER2_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor, _, _ = make_preprocessor(X_train, scale_numeric=True)

    base_models = {
        "Random Forest": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.10,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    oof_matrix = np.zeros((len(X_train), len(base_models)))
    test_matrix = np.zeros((len(X_test), len(base_models)))
    results = []

    for idx, (name, model) in enumerate(base_models.items()):
        print(f"\nPaper 2 OOF: {name}")
        oof_prob = cross_val_predict(
            model, X_train, y_train, cv=cv10, method="predict_proba", n_jobs=-1
        )[:, 1]
        model.fit(X_train, y_train)
        test_prob = model.predict_proba(X_test)[:, 1]
        oof_matrix[:, idx] = oof_prob
        test_matrix[:, idx] = test_prob
        pred = (test_prob >= 0.5).astype(int)
        results.append(metrics_row(name, y_test, pred, test_prob, 0.5))
        print(f"  F1={results[-1]['F1']:.4f} Acc={results[-1]['Accuracy']:.4f}")
        joblib.dump(model, PAPER2_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    meta_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "fnn",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    max_iter=1500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    print("\nTraining Paper 2 stacking FNN...")
    meta_model.fit(oof_matrix, y_train)
    stack_prob = meta_model.predict_proba(test_matrix)[:, 1]
    stack_pred = (stack_prob >= 0.5).astype(int)
    results.append(
        metrics_row(
            "Paper 2 Stacking (RF + XGBoost + GB -> FNN)",
            y_test,
            stack_pred,
            stack_prob,
            0.5,
        )
    )
    joblib.dump(meta_model, PAPER2_DIR / "stacking_fnn_meta_model.joblib")
    pd.DataFrame(results).to_csv(PAPER2_DIR / "paper2_model_comparison.csv", index=False)

    gb = base_models["Gradient Boosting"]
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gb_oof = cross_val_predict(
        gb, X_train, y_train, cv=cv5, method="predict_proba", n_jobs=-1
    )[:, 1]
    gb_threshold, gb_table = best_f1_threshold(y_train, gb_oof)
    gb_table.to_csv(PAPER2_DIR / "gb_threshold_search.csv", index=False)
    gb.fit(X_train, y_train)
    joblib.dump(gb, PAPER2_DIR / "gradient_boosting.joblib")

    payload = {
        "model": "Gradient Boosting",
        "threshold": gb_threshold,
        "selection": "max F1 on 5-fold out-of-fold training probabilities",
        "medium_risk_below": 0.40,
    }
    (PAPER1_DIR / "operating_point.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"DSS operating threshold (GB OOF F1): {gb_threshold:.2f}")
    return gb_threshold


def main():
    engineered = rebuild_dataset()
    X, y = get_model_matrix(engineered)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain {X_train.shape}, test {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}")
    train_paper1(X_train, X_test, y_train, y_test)
    train_paper2(X_train, X_test, y_train, y_test)
    print("\nRetraining complete.")


if __name__ == "__main__":
    main()
