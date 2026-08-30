"""Translate continuation probabilities into counselor-facing risk tiers."""

from numbers import Real


def calculate_risk_tier(probability_score):
    """Return the Paper 1 risk tier for a continuation probability in [0, 1]."""
    if not isinstance(probability_score, Real):
        raise TypeError("probability_score must be numeric.")

    probability = float(probability_score)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability_score must be between 0.0 and 1.0.")

    if probability >= 0.73:
        return "Low Risk (High likelihood to continue pathway)"
    if probability >= 0.40:
        return "Medium Risk (Transitional/Uncertain)"
    return "High Risk (Likely to drop out/reverse decision)"


def main():
    for probability in (0.20, 0.55, 0.80):
        print(f"{probability:.2f}: {calculate_risk_tier(probability)}")


if __name__ == "__main__":
    main()
