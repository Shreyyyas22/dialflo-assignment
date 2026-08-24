from app.core.config import settings
from app.schemas.analysis import AgeBracket, AgeResult


def process_age_prediction(raw_age_scalar: float) -> AgeResult:
    """
    Processes model regression age scalar (0-1 range * 100 -> years)
    and maps it to AgeResult brackets with confidence calculation.
    """
    predicted_age = raw_age_scalar * 100.0

    if predicted_age < 31.0:
        bracket: AgeBracket = "18-30"
        midpoint = 24.0
    elif 31.0 <= predicted_age <= 45.0:
        bracket: AgeBracket = "31-45"
        midpoint = 38.0
    elif 45.0 < predicted_age <= 60.0:
        bracket: AgeBracket = "46-60"
        midpoint = 53.0
    else:
        bracket: AgeBracket = "60+"
        midpoint = 70.0

    # Calculate confidence based on standard deviation / boundary distance metric
    dist = abs(predicted_age - midpoint)
    confidence = max(0.0, min(1.0, 1.0 - (dist / 35.0)))

    if confidence < settings.AGE_CONFIDENCE_THRESHOLD:
        bracket = "unknown"

    return AgeResult(bracket=bracket, confidence=round(confidence, 3))
