"""E8. Full 22-column model vs the reduced feature set."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiments.lib import (
    MODEL_FEATURE_COLUMNS,
    RANDOM_STATE,
    REDUCED_FEATURES,
    REDUNDANT_FEATURES,
    RESULTS,
    as_pipeline,
    load_frame,
    make_gb,
    matrix,
    mean_ci,
    score_predictions,
)


def evaluate(features, target, columns) -> dict:
    x = features.loc[:, columns]
    x_train, x_test, y_train, y_test = train_test_split(
        x, target, test_size=0.20, stratify=target, random_state=RANDOM_STATE
    )
    pipe = as_pipeline(make_gb(), x_train)
    pipe.fit(x_train, y_train)
    prob = pipe.predict_proba(x_test)[:, 1]
    heldout = score_predictions(y_test, (prob >= 0.5).astype(int), prob)

    fold_f1 = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in splitter.split(x, target):
        fold = as_pipeline(make_gb(), x.iloc[train_idx])
        fold.fit(x.iloc[train_idx], target.iloc[train_idx])
        fold_prob = fold.predict_proba(x.iloc[test_idx])[:, 1]
        fold_f1.append(
            score_predictions(
                target.iloc[test_idx],
                (fold_prob >= 0.5).astype(int),
                fold_prob,
            )["F1"]
        )
    interval = mean_ci(fold_f1)
    heldout["cv_f1_mean"] = interval["mean"]
    heldout["cv_f1_low"] = interval["low"]
    heldout["cv_f1_high"] = interval["high"]
    return heldout


def main() -> None:
    features, target = matrix(load_frame())
    full = evaluate(features, target, MODEL_FEATURE_COLUMNS)
    reduced = evaluate(features, target, REDUCED_FEATURES)
    table = pd.DataFrame(
        [
            {"feature_set": "full_22", "n_features": len(MODEL_FEATURE_COLUMNS), **full},
            {"feature_set": "reduced", "n_features": len(REDUCED_FEATURES), **reduced},
        ]
    )
    table.to_csv(RESULTS / "e8_reduced_vs_full.csv", index=False)
    lines = ["# Reduced feature set (E8)", ""]
    lines.append("Dropped duplicate encodings: " + ", ".join(REDUNDANT_FEATURES) + ".")
    lines.append("")
    lines.append("Kept: " + ", ".join(REDUCED_FEATURES) + ".")
    lines.append("")
    lines.append(table.round(4).to_string(index=False))
    lines.append("")
    delta = reduced["F1"] - full["F1"]
    lines.append(
        f"Held-out F1 change (reduced minus full): {delta:+.4f}. "
        "A change inside a few thousandths means the extra encodings were not "
        "buying a different decision."
    )
    text = "\n".join(lines)
    (RESULTS / "E8_REDUCED_FEATURES.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
