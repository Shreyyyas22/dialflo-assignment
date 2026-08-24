import logging

import numpy as np
import torch

from app.core.exceptions import ModelInferenceError
from app.ml.model_manager import get_model, get_processor

logger = logging.getLogger(__name__)


def run_inference(
    signal: np.ndarray, sample_rate: int = 16000
) -> tuple[float, np.ndarray]:
    model = get_model()
    processor = get_processor()

    if model is None or processor is None:
        raise ModelInferenceError("Model or processor is not loaded.")

    try:
        inputs = processor(signal, sampling_rate=sample_rate, return_tensors="pt")
        with torch.inference_mode():
            raw_age, raw_gender = model(inputs.input_values)

        gender_probs = torch.softmax(raw_gender, dim=-1).squeeze().cpu().numpy()
        age_scalar = float(raw_age.squeeze().cpu().item())

        return age_scalar, gender_probs
    except Exception as e:
        logger.exception("Error during model inference")
        raise ModelInferenceError(f"Model forward pass failed: {e!s}") from e
