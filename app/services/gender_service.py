import numpy as np

from app.core.config import settings
from app.schemas.analysis import GenderResult, GenderValue


def process_gender_prediction(gender_probs: np.ndarray) -> GenderResult:
    """
    Processes 3-class softmax probabilities (0: female, 1: male, 2: child)
    and maps them to API GenderResult ('male', 'female', 'unknown').
    Child class and low-confidence predictions map to 'unknown'.
    """
    max_idx = int(np.argmax(gender_probs))
    max_prob = float(gender_probs[max_idx])

    if max_idx == 2 or max_prob < settings.GENDER_CONFIDENCE_THRESHOLD:
        gender_val: GenderValue = "unknown"
    elif max_idx == 0:
        gender_val: GenderValue = "female"
    elif max_idx == 1:
        gender_val: GenderValue = "male"
    else:
        gender_val: GenderValue = "unknown"

    return GenderResult(
        value=gender_val, confidence=round(min(1.0, max(0.0, max_prob)), 3)
    )
