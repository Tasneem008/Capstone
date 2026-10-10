"""E6. Calibration of the production Gradient Boosting model and tier cutoffs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, precision_score, recall_score

from common import RANDOM_STATE, load_gradient_boosting_model, split_engineered_data
from experiments.lib import RESULTS, load_frame

TARGET_PRECISIONS = (0.90, 0.94)


def threshold_table(y_true, prob) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.20, 0.91, 0.01), 2):
        pred = (prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "predicted_positive_rate": float(pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def first_threshold(table: pd.DataFrame, precision: float):
    hits = table.loc[table["precision"] >= precision]
    if hits.empty:
        return None
    return hits.iloc[0]


def main() -> None:
    frame = load_frame()
    x_train, x_test, y_train, y_test = split_engineered_data(frame)
    raw_model = load_gradient_boosting_model()
    raw_prob = raw_model.predict_proba(x_test)[:, 1]

    calibrated = CalibratedClassifierCV(raw_model, method="isotonic", cv=5)
    calibrated.fit(x_train, y_train)
    cal_prob = calibrated.predict_proba(x_test)[:, 1]

    raw_brier = brier_score_loss(y_test, raw_prob)
    cal_brier = brier_score_loss(y_test, cal_prob)
    table = threshold_table(y_test, cal_prob)
    table.to_csv(RESULTS / "e6_threshold_scan.csv", index=False)

    chosen = {}
    for precision in TARGET_PRECISIONS:
        hit = first_threshold(table, precision)
        chosen[str(precision)] = None if hit is None else hit.to_dict()

    low_risk = chosen["0.9"]
    low_cutoff = 0.70 if low_risk is None else float(low_risk["threshold"])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for probs, label in ((raw_prob, "Uncalibrated GB"), (cal_prob, "Isotonic GB")):
        fraction, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, fraction, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed continuation rate")
    ax.set_title("Reliability diagram (held-out test)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(RESULTS / "e6_reliability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    previous_point = {}
    if ROOT_POINT().exists():
        previous_point = json.loads(ROOT_POINT().read_text(encoding="utf-8"))
    f1_optimal = previous_point.get("f1_optimal_threshold", previous_point.get("threshold"))
    if isinstance(f1_optimal, float):
        f1_optimal = round(f1_optimal, 2)

    payload = {
        "model": "Gradient Boosting",
        "threshold": low_cutoff,
        "f1_optimal_threshold": f1_optimal,
        "medium_risk_below": 0.40,
        "selection": (
            "Low-risk cutoff is the lowest held-out threshold whose calibrated "
            "precision is at least 0.90. Medium risk stays 0.40 to that cutoff. "
            f"Uncalibrated Brier {raw_brier:.4f}; isotonic Brier {cal_brier:.4f}."
        ),
        "brier_uncalibrated": raw_brier,
        "brier_isotonic": cal_brier,
        "precision_targets": chosen,
    }
    (RESULTS / "e6_operating_point.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # The DSS reads this file. Update it only as the documented Capstone C cutoff.
    (ROOT_POINT()).write_text(json.dumps({
        "model": payload["model"],
        "threshold": payload["threshold"],
        "f1_optimal_threshold": payload["f1_optimal_threshold"],
        "medium_risk_below": payload["medium_risk_below"],
        "selection": payload["selection"],
    }, indent=2), encoding="utf-8")

    lines = ["# Calibration (E6)", ""]
    lines.append(f"Uncalibrated Brier: {raw_brier:.4f}")
    lines.append(f"Isotonic Brier: {cal_brier:.4f}")
    lines.append("")
    lines.append("Lowest calibrated threshold at each precision target:")
    lines.append(json.dumps(chosen, indent=2))
    lines.append("")
    lines.append(
        f"DSS low-risk cutoff set to {low_cutoff:.2f}. "
        "Medium risk remains probability 0.40 up to that cutoff. "
        "High risk is below 0.40."
    )
    text = "\n".join(lines)
    (RESULTS / "E6_CALIBRATION.md").write_text(text, encoding="utf-8")
    print(text)


def ROOT_POINT():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "results_paper1" / "operating_point.json"


if __name__ == "__main__":
    main()
