"""Interpret the model probability as a simple presentation score."""

def health_score(bankruptcy_probability: float) -> int:
    """Return a 0-100 model-derived score (higher means lower predicted risk).

    This is a project-specific presentation metric, not an audited credit score.
    """
    probability = min(max(float(bankruptcy_probability), 0.0), 1.0)
    return round((1.0 - probability) * 100)


def risk_label(bankruptcy_probability: float) -> str:
    probability = float(bankruptcy_probability)
    if probability >= 0.50:
        return "High"
    if probability >= 0.20:
        return "Medium"
    return "Low"
