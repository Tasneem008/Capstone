"""Train-set model selection, then one locked test evaluation.

Candidates are ranked by mean stratified CV accuracy on the training split.
The held-out test set is scored only after that ranking is fixed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import RANDOM_STATE, make_preprocessor, split_engineered_data
from experiments.lib import RESULTS, load_frame

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def pipe(model, features, undersample=False):
    preprocessor, _, _ = make_preprocessor(features, scale_numeric=True)
    steps = [("preprocessor", preprocessor)]
    if undersample:
        steps.append(("undersample", RandomUnderSampler(random_state=RANDOM_STATE)))
        return ImbPipeline(steps + [("model", model)])
    return Pipeline(steps + [("model", model)])


def xgb(**kwargs):
    params = dict(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    params.update(kwargs)
    return XGBClassifier(**params)


def gb(**kwargs):
    params = dict(n_estimators=100, learning_rate=0.10, max_depth=3, random_state=RANDOM_STATE)
    params.update(kwargs)
    return GradientBoostingClassifier(**params)


def candidates(features):
    models = {
        "Logistic regression": pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), features),
        "Random Forest 200": pipe(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
            features,
        ),
        "LightGBM 300": pipe(
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
            ),
            features,
        ),
        "HistGradientBoosting": pipe(
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=RANDOM_STATE,
            ),
            features,
        ),
        "GB current (100, depth 3, lr 0.10)": pipe(gb(), features),
        "GB 300 depth 3 lr 0.05": pipe(gb(n_estimators=300, learning_rate=0.05, max_depth=3), features),
        "GB 200 depth 4 lr 0.05": pipe(gb(n_estimators=200, learning_rate=0.05, max_depth=4), features),
        "XGB Paper 2 defaults": pipe(xgb(), features),
        "XGB 400 depth 4 lr 0.05": pipe(xgb(n_estimators=400, max_depth=4, learning_rate=0.05), features),
        "XGB 300 depth 6 lr 0.05": pipe(xgb(n_estimators=300, max_depth=6, learning_rate=0.05), features),
        "XGB undersampled (Paper 1 style)": pipe(
            xgb(n_estimators=200, max_depth=5, learning_rate=0.10),
            features,
            undersample=True,
        ),
    }
    models["Stack RF+XGB+GB logistic"] = StackingClassifier(
        estimators=[
            ("rf", clone(models["Random Forest 200"])),
            ("xgb", clone(models["XGB Paper 2 defaults"])),
            ("gb", clone(models["GB current (100, depth 3, lr 0.10)"])),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=3,
        n_jobs=1,
        passthrough=False,
    )
    return models


def cv_table(models, x_train, y_train) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        print(f"CV {name}", flush=True)
        scored = cross_validate(
            model,
            x_train,
            y_train,
            cv=CV,
            scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"},
            n_jobs=1,
        )
        rows.append(
            {
                "model": name,
                "cv_accuracy": float(scored["test_accuracy"].mean()),
                "cv_accuracy_std": float(scored["test_accuracy"].std(ddof=1)),
                "cv_f1": float(scored["test_f1"].mean()),
                "cv_auc": float(scored["test_roc_auc"].mean()),
            }
        )
        print(
            f"  acc {rows[-1]['cv_accuracy']:.4f}  f1 {rows[-1]['cv_f1']:.4f}  auc {rows[-1]['cv_auc']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values(
        ["cv_accuracy", "cv_auc", "cv_f1"], ascending=False
    ).reset_index(drop=True)


def test_row(name, model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "model": name,
        "threshold": 0.5,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob),
        "MCC": matthews_corrcoef(y_test, pred),
    }


def main() -> None:
    frame = load_frame()
    x_train, x_test, y_train, y_test = split_engineered_data(frame)
    models = candidates(x_train)
    ranking = cv_table(models, x_train, y_train)
    ranking.to_csv(RESULTS / "model_selection_cv.csv", index=False)

    # Locked test: current project model, CV winner, and the next two.
    chosen = ranking.iloc[0]["model"]
    report_names = []
    for name in [chosen, "GB current (100, depth 3, lr 0.10)", "XGB Paper 2 defaults", "Stack RF+XGB+GB logistic"]:
        if name in models and name not in report_names:
            report_names.append(name)
    # Also score the second-ranked model if it is not already included.
    runner = ranking.iloc[1]["model"]
    if runner not in report_names:
        report_names.append(runner)

    test_rows = []
    for name in report_names:
        print(f"Test {name}", flush=True)
        test_rows.append(test_row(name, clone(models[name]), x_train, y_train, x_test, y_test))
    tested = pd.DataFrame(test_rows)
    tested.to_csv(RESULTS / "model_selection_test.csv", index=False)

    gb_acc = float(ranking.loc[ranking["model"].str.startswith("GB current"), "cv_accuracy"].iloc[0])
    winner_acc = float(ranking.iloc[0]["cv_accuracy"])
    winner_is_tree = not str(chosen).startswith("Stack")
    # A stack has to beat the explainable GB by more than half a point of CV accuracy.
    adopt = chosen
    reason = "Highest training CV accuracy."
    if (not winner_is_tree) and (winner_acc - gb_acc) < 0.005:
        adopt = "GB current (100, depth 3, lr 0.10)"
        reason = (
            "The stack leads CV accuracy by under 0.5 points. "
            "Gradient Boosting stays the project model because the gap is inside fold noise "
            "and the tool needs a tree model for exact SHAP."
        )
    elif chosen != "GB current (100, depth 3, lr 0.10)" and winner_is_tree and (winner_acc - gb_acc) < 0.003:
        adopt = "GB current (100, depth 3, lr 0.10)"
        reason = (
            "A tuned tree is within 0.3 points of the current Gradient Boosting CV accuracy. "
            "The current model stays. The extra fit is not a real gain."
        )

    decision = {
        "cv_winner": chosen,
        "cv_winner_accuracy": winner_acc,
        "adopted_model": adopt,
        "reason": reason,
        "selection_rule": "Mean 5-fold accuracy on the training split. Test metrics are reported after the choice.",
    }
    (RESULTS / "model_selection_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    lines = [
        "# Model selection",
        "",
        "Ranking uses 5-fold stratified CV on the training split only (4,800 rows). "
        "Accuracy is the selection metric. Test numbers below were not used to pick the winner.",
        "",
        ranking.round(4).to_string(index=False),
        "",
        "## Locked test (threshold 0.50)",
        "",
        tested.round(4).to_string(index=False),
        "",
        "## Decision",
        "",
        f"CV winner: {chosen} ({winner_acc:.4f}).",
        f"Adopted project model: {adopt}.",
        reason,
    ]
    text = "\n".join(lines)
    (RESULTS / "MODEL_SELECTION.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
