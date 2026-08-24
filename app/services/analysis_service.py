import asyncio
import logging
import time
import uuid

from app.core.config import settings
from app.core.exceptions import AudioTooLargeError, InvalidUUIDError
from app.ml.inference import run_inference
from app.schemas.analysis import (
    AgeResult,
    AnalyzeResponse,
    GenderResult,
)
from app.services.age_service import process_age_prediction
from app.services.audio_service import decode_and_preprocess_audio
from app.services.gender_service import process_gender_prediction
from app.services.language_service import detect_language
from app.services.quality_service import analyze_audio_quality

logger = logging.getLogger(__name__)


async def analyze_audio_request(contact_id: str, audio_bytes: bytes) -> AnalyzeResponse:
    start_time = time.perf_counter()

    # 1. Validate contact_id UUID format
    try:
        uuid_obj = uuid.UUID(contact_id)
        formatted_contact_id = str(uuid_obj)
    except (ValueError, TypeError) as e:
        raise InvalidUUIDError(
            f"Invalid contact_id '{contact_id}'. Must be a valid UUID string."
        ) from e

    # 2. Validate max audio size
    max_bytes = int(settings.MAX_AUDIO_SIZE_MB * 1024 * 1024)
    if len(audio_bytes) > max_bytes:
        raise AudioTooLargeError(
            f"Audio file size exceeds limit of {settings.MAX_AUDIO_SIZE_MB}MB."
        )

    # 3. Decode & preprocess audio using FFmpeg
    signal, sample_rate = decode_and_preprocess_audio(audio_bytes)

    # 4. Analyze audio quality
    quality_result = analyze_audio_quality(signal, sample_rate)

    # 5. Short-circuit or inference path
    if quality_result.classification == "insufficient":
        logger.info(
            f"Audio quality insufficient for contact_id={formatted_contact_id}. Short-circuiting inference."
        )
        gender_res = GenderResult(value="unknown", confidence=0.0)
        age_res = AgeResult(bracket="unknown", confidence=0.0)
    else:
        # Run inference in a worker thread to prevent blocking event loop
        raw_age_scalar, gender_probs = await asyncio.to_thread(
            run_inference, signal, sample_rate
        )
        gender_res = process_gender_prediction(gender_probs)
        age_res = process_age_prediction(raw_age_scalar)

    # 6. Language detection (best-effort, never blocks core flow)
    try:
        language_res = detect_language(
            signal, sample_rate, quality_result.classification
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Language detection failed for {formatted_contact_id}: {e}")
        from app.schemas.analysis import LanguageResult

        language_res = LanguageResult(code="unknown", confidence=0.0)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Structured logging line (no raw audio logged)
    logger.info(
        f"Request Processed | contact_id={formatted_contact_id} | "
        f"quality={quality_result.classification} | gender={gender_res.value} ({gender_res.confidence}) | "
        f"age={age_res.bracket} ({age_res.confidence}) | language={language_res.code} ({language_res.confidence}) | time_ms={elapsed_ms:.2f}ms"
    )

    return AnalyzeResponse(
        contact_id=formatted_contact_id,
        gender=gender_res,
        age=age_res,
        language=language_res,
        quality=quality_result,
        processing_time_ms=round(elapsed_ms, 2),
    )
