"""E3. Baselines, 5x5 repeated stratified CV, and paired tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from experiments.lib import (
    RANDOM_STATE,
    RESULTS,
    as_pipeline,
    load_frame,
    make_gb,
    make_logit,
    make_rf,
    make_xgb,
    matrix,
    mcnemar,
    mean_ci,
    nadeau_bengio,
    score_predictions,
)

N_REPEATS = 5
N_SPLITS = 5


def _builders(columns):
    def stacking():
        return StackingClassifier(
            estimators=[
                ("rf", make_rf()),
                ("xgb", make_xgb()),
                ("gb", make_gb()),
            ],
            final_estimator=make_logit(),
            cv=2,
            n_jobs=1,
        )

    return {
        "Majority class": None,
        "Phase-only logistic": make_logit,
        "Logistic regression": make_logit,
        "Random Forest": make_rf,
        "Gradient Boosting": make_gb,
        "XGBoost": make_xgb,
        "Stacking (RF+XGB+GB, logistic meta)": stacking,
    }


def _xy_for(name, features, target):
    if name == "Phase-only logistic":
        return features.loc[:, ["Political_Phase"]], target
    return features, target


def _fit_predict(name, builder, x_train, y_train, x_test):
    if name == "Majority class":
        majority = int(y_train.mode().iloc[0]) if hasattr(y_train, "mode") else int(pd_mode(y_train))
        pred = np.full(len(x_test), majority)
        prob = np.full(len(x_test), float(np.mean(y_train)))
        return pred, prob
    model = builder()
    if name == "Stacking (RF+XGB+GB, logistic meta)":
        pipe = Pipeline(
            [
                ("preprocessor", as_pipeline(make_logit(), x_train).named_steps["preprocessor"]),
                ("model", model),
            ]
        )
    else:
        pipe = as_pipeline(model, x_train)
    pipe.fit(x_train, y_train)
    prob = pipe.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob


def pd_mode(y_train):
    values, counts = np.unique(y_train, return_counts=True)
    return values[np.argmax(counts)]


def run(frame, tag: str = "full") -> pd.DataFrame:
    features, target = matrix(frame)
    builders = _builders(features.columns)
    fold_rows = []
    heldout_preds = {}

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    for name, builder in builders.items():
        xt, yt = _xy_for(name, x_train, y_train)
        xs, _ = _xy_for(name, x_test, y_test)
        pred, prob = _fit_predict(name, builder, xt, yt, xs)
        heldout_preds[name] = pred
        row = score_predictions(y_test, pred, prob)
        row.update({"model": name, "protocol": "heldout_80_20", "tag": tag})
        fold_rows.append(row)
        print(f"[{tag}] held-out {name}: F1={row['F1']:.3f} Acc={row['Accuracy']:.3f}")

    comparisons = [
        ("Gradient Boosting", "Logistic regression"),
        ("Gradient Boosting", "Stacking (RF+XGB+GB, logistic meta)"),
        ("Stacking (RF+XGB+GB, logistic meta)", "Logistic regression"),
    ]
    mcnemar_rows = []
    for left, right in comparisons:
        result = mcnemar(y_test, heldout_preds[left], heldout_preds[right])
        result.update({"model_a": left, "model_b": right, "tag": tag})
        mcnemar_rows.append(result)

    score_bank = {name: [] for name in builders}
    for repeat in range(N_REPEATS):
        splitter = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(features, target)):
            print(f"[{tag}] repeat {repeat + 1}/{N_REPEATS} fold {fold + 1}/{N_SPLITS}")
            for name, builder in builders.items():
                x_all, y_all = _xy_for(name, features, target)
                pred, prob = _fit_predict(
                    name,
                    builder,
                    x_all.iloc[train_idx],
                    y_all.iloc[train_idx],
                    x_all.iloc[test_idx],
                )
                scored = score_predictions(y_all.iloc[test_idx], pred, prob)
                score_bank[name].append(scored["F1"])
                fold_rows.append(
                    {
                        "model": name,
                        "protocol": "repeated_cv",
                        "repeat": repeat,
                        "fold": fold,
                        "tag": tag,
                        **scored,
                    }
                )

    summary_rows = []
    for name, scores in score_bank.items():
        interval = mean_ci(scores)
        summary_rows.append(
            {
                "model": name,
                "tag": tag,
                "f1_mean": interval["mean"],
                "f1_low": interval["low"],
                "f1_high": interval["high"],
                "n_folds": interval["n"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("f1_mean", ascending=False)

    n_test = len(features) // N_SPLITS
    n_train = len(features) - n_test
    paired = []
    for left, right in comparisons:
        test = nadeau_bengio(score_bank[left], score_bank[right], n_train, n_test)
        test.update({"model_a": left, "model_b": right, "metric": "F1", "tag": tag})
        paired.append(test)

    pd.DataFrame(fold_rows).to_csv(RESULTS / f"e3_fold_scores_{tag}.csv", index=False)
    summary.to_csv(RESULTS / f"e3_cv_summary_{tag}.csv", index=False)
    pd.DataFrame(paired).to_csv(RESULTS / f"e3_nadeau_bengio_{tag}.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(RESULTS / f"e3_mcnemar_{tag}.csv", index=False)

    lines = [f"# Baselines and repeated CV ({tag})", ""]
    lines.append(
        "Repeated stratified CV: 5 repeats x 5 folds. "
        "Intervals are 95% t intervals on the 25 fold F1 scores. "
        "Stacking uses a logistic meta-learner and 2-fold inner CV so the "
        "repeated comparison finishes. The single-split FNN stack remains in "
        "`results_paper2/paper2_model_comparison.csv`."
    )
    lines.append("")
    lines.append(summary.round(4).to_string(index=False))
    lines.append("")
    lines.append("## Nadeau-Bengio corrected resampled t-test on F1")
    lines.append(pd.DataFrame(paired).round(4).to_string(index=False))
    lines.append("")
    lines.append("## McNemar on the random_state=42 holdout")
    lines.append(pd.DataFrame(mcnemar_rows).round(4).to_string(index=False))
    lines.append("")
    lines.append(
        "Read the majority and phase-only rows as the floor. "
        "The jump from phase-only to the full logistic is the student-feature lift "
        "on a random split. It does not replace the forward test in E4."
    )
    lines.append(
        "The stacking row here is RF + XGBoost + Gradient Boosting with a logistic "
        "meta-learner and 2-fold inner CV. It is not the Paper 2 feedforward stack "
        "in `results_paper2/paper2_model_comparison.csv`. That single-split stack "
        "is a different procedure and is not inside this repeated test."
    )
    text = "\n".join(lines)
    (RESULTS / f"E3_BASELINES_{tag}.md").write_text(text, encoding="utf-8")
    print(text)
    return summary


def main() -> None:
    run(load_frame(), tag="full")


if __name__ == "__main__":
    main()
