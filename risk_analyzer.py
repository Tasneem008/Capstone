"""Translate continuation probabilities into counselor-facing risk tiers."""

import json
from numbers import Real
from pathlib import Path

MEDIUM_RISK_THRESHOLD = 0.40
DEFAULT_LOW_RISK_THRESHOLD = 0.73
OPERATING_POINT_PATH = (
    Path(__file__).resolve().parent / "results_paper1" / "operating_point.json"
)


def load_operating_threshold():
    """Calibrated low-risk cutoff written by experiments/e6_calibration.py."""
    if OPERATING_POINT_PATH.exists():
        payload = json.loads(OPERATING_POINT_PATH.read_text(encoding="utf-8"))
        return float(payload["threshold"])
    return DEFAULT_LOW_RISK_THRESHOLD


def calculate_risk_tier(probability_score, low_risk_threshold=None):
    """Return the risk tier for a continuation probability in [0, 1]."""
    if not isinstance(probability_score, Real):
        raise TypeError("probability_score must be numeric.")

    probability = float(probability_score)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability_score must be between 0.0 and 1.0.")

    cutoff = (
        float(low_risk_threshold)
        if low_risk_threshold is not None
        else load_operating_threshold()
    )
    if probability >= cutoff:
        return "Low Risk (High likelihood to continue pathway)"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk (Transitional/Uncertain)"
    return "High Risk (Likely to drop out/reverse decision)"


def main():
    for probability in (0.20, 0.55, 0.80):
        print(f"{probability:.2f}: {calculate_risk_tier(probability)}")


if __name__ == "__main__":
    main()
