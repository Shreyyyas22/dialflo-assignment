import asyncio
import base64
import json
import logging
import uuid
from binascii import Error as BinasciiError

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.ml.inference import run_inference
from app.schemas.analysis import (
    AgeResult,
    GenderResult,
    LanguageResult,
    StreamPredictionMessage,
)
from app.services.age_service import process_age_prediction
from app.services.audio_service import decode_and_preprocess_audio
from app.services.gender_service import process_gender_prediction
from app.services.language_service import detect_language
from app.services.quality_service import analyze_audio_quality

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Streaming"])


def _validate_contact_id(contact_id: str | None) -> str | None:
    if not contact_id:
        return None
    try:
        return str(uuid.UUID(contact_id))
    except (ValueError, AttributeError):
        return None


async def _process_accumulated_buffer(
    audio_bytes: bytes,
) -> StreamPredictionMessage | None:
    """
    Attempt to decode accumulated bytes and produce a prediction message.
    Returns None if decoding fails (not enough data yet).
    """
    if not audio_bytes or len(audio_bytes) < 1024:
        return None
    try:
        signal, sample_rate = decode_and_preprocess_audio(audio_bytes)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Streaming decode failed (buffer {len(audio_bytes)} bytes): {e}")
        return None

    quality_result = analyze_audio_quality(signal, sample_rate)
    accumulated_duration = quality_result.metrics.duration_seconds

    if quality_result.classification == "insufficient":
        gender_res = GenderResult(value="unknown", confidence=0.0)
        age_res = AgeResult(bracket="unknown", confidence=0.0)
    else:
        try:
            raw_age_scalar, gender_probs = await asyncio.to_thread(
                run_inference, signal, sample_rate
            )
            gender_res = process_gender_prediction(gender_probs)
            age_res = process_age_prediction(raw_age_scalar)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Streaming inference failed: {e}")
            gender_res = GenderResult(value="unknown", confidence=0.0)
            age_res = AgeResult(bracket="unknown", confidence=0.0)

    try:
        language_res = detect_language(signal, sample_rate, quality_result.classification)
    except Exception:  # noqa: BLE001
        language_res = LanguageResult(code="unknown", confidence=0.0)

    return StreamPredictionMessage(
        event="prediction_update",
        accumulated_duration_seconds=round(accumulated_duration, 3),
        gender=gender_res,
        age=age_res,
        language=language_res,
        quality_classification=quality_result.classification,
    )


@router.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    """
    WebSocket streaming endpoint for real-time audio analysis.

    Protocol:
    - Connect to `ws://host:8000/ws/analyze?contact_id=<uuid>` or send first text frame
      `{"contact_id":"<uuid>"}` after connecting.
    - Send audio as binary frames (any FFmpeg-decodable format). Chunks are accumulated;
      server re-decodes the full buffer on each chunk and emits progressive predictions.
      Alternatively send text frame `{"event":"chunk","audio_base64":"<b64>"}`
      for base64-encoded audio.
    - Send text frame `{"event":"end"}` to finalize. Server replies with `stream_completed`
      and closes.
    - Server emits JSON frames: `{"event":"prediction_update", ...}` or `{"event":"error", ...}`.
    """
    await websocket.accept()
    query_contact_id = websocket.query_params.get("contact_id")
    contact_id = _validate_contact_id(query_contact_id)
    accumulated_bytes = bytearray()

    # Handshake: if contact_id not in query, expect it in first text message
    if contact_id is None:
        try:
            init_msg = await asyncio.wait_for(websocket.receive(), timeout=10.0)
            text = init_msg.get("text")
            if text:
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "event": "error",
                            "message": "First message must be JSON with contact_id.",
                            "code": "INVALID_CONTACT_ID",
                        }
                    )
                    await websocket.close(code=1008)
                    return
                candidate = data.get("contact_id")
                validated = _validate_contact_id(candidate)
                if not validated:
                    await websocket.send_json(
                        {
                            "event": "error",
                            "message": "Invalid or missing contact_id. Must be valid UUID.",
                            "code": "INVALID_CONTACT_ID",
                        }
                    )
                    await websocket.close(code=1008)
                    return
                contact_id = validated
                await websocket.send_json({"event": "stream_started", "contact_id": contact_id})
                # If init message also carried a chunk, ingest it
                if data.get("event") == "chunk" and data.get("audio_base64"):
                    try:
                        b64_bytes = base64.b64decode(data["audio_base64"])
                        accumulated_bytes.extend(b64_bytes)
                        msg = await _process_accumulated_buffer(bytes(accumulated_bytes))
                        if msg:
                            await websocket.send_json(msg.model_dump())
                    except (BinasciiError, ValueError):
                        await websocket.send_json(
                            {
                                "event": "error",
                                "message": "Invalid base64 audio data.",
                                "code": "INVALID_AUDIO",
                            }
                        )
            else:
                await websocket.send_json(
                    {
                        "event": "error",
                        "message": "Missing contact_id. Provide ?contact_id=<uuid> or JSON {contact_id}.",
                        "code": "INVALID_CONTACT_ID",
                    }
                )
                await websocket.close(code=1008)
                return
        except asyncio.TimeoutError:
            await websocket.send_json(
                {
                    "event": "error",
                    "message": "Timeout waiting for contact_id.",
                    "code": "INVALID_CONTACT_ID",
                }
            )
            await websocket.close(code=1008)
            return
    else:
        await websocket.send_json({"event": "stream_started", "contact_id": contact_id})

    last_processed_len = 0
    min_chunk_growth = settings.WS_CHUNK_GROWTH_BYTES

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for contact_id={contact_id}")
                break

            if message.get("type") == "websocket.disconnect":
                break

            # Text handling
            text_data = message.get("text")
            if text_data is not None:
                try:
                    data = json.loads(text_data)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "event": "error",
                            "message": "Invalid JSON text frame.",
                            "code": "INVALID_REQUEST",
                        }
                    )
                    continue

                event = data.get("event")
                if event in ("end", "final", "close", "complete"):
                    if len(accumulated_bytes) == 0:
                        await websocket.send_json(
                            {
                                "event": "error",
                                "message": "No audio data received before end.",
                                "code": "INSUFFICIENT_AUDIO",
                            }
                        )
                        await websocket.close()
                        break
                    final_msg = await _process_accumulated_buffer(bytes(accumulated_bytes))
                    if final_msg:
                        final_msg.event = "stream_completed"  # type: ignore[attr-defined]
                        await websocket.send_json(final_msg.model_dump())
                    else:
                        await websocket.send_json(
                            {
                                "event": "error",
                                "message": "Could not decode accumulated audio.",
                                "code": "INVALID_AUDIO",
                            }
                        )
                    await websocket.close()
                    break

                if event == "chunk" and data.get("audio_base64"):
                    try:
                        chunk_bytes = base64.b64decode(data["audio_base64"])
                    except (BinasciiError, ValueError):
                        await websocket.send_json(
                            {
                                "event": "error",
                                "message": "Invalid base64 audio data.",
                                "code": "INVALID_AUDIO",
                            }
                        )
                        continue
                    accumulated_bytes.extend(chunk_bytes)
                elif data.get("contact_id") and not contact_id:
                    validated = _validate_contact_id(data.get("contact_id"))
                    if validated:
                        contact_id = validated
                        continue
                    await websocket.send_json(
                        {
                            "event": "error",
                            "message": "Invalid contact_id.",
                            "code": "INVALID_CONTACT_ID",
                        }
                    )
                    continue
                elif event is not None:
                    logger.debug(f"Unknown text event {event} for {contact_id}")
                    continue

            # Binary handling
            bytes_data = message.get("bytes")
            if bytes_data is not None:
                if not bytes_data:
                    continue
                accumulated_bytes.extend(bytes_data)

            # Throttle inference: require growth
            if len(accumulated_bytes) - last_processed_len < min_chunk_growth:
                if len(accumulated_bytes) < 16000:
                    continue
                if last_processed_len != 0:
                    continue

            msg = await _process_accumulated_buffer(bytes(accumulated_bytes))
            if msg:
                last_processed_len = len(accumulated_bytes)
                await websocket.send_json(msg.model_dump())
            else:
                logger.debug(f"Buffer not yet decodable ({len(accumulated_bytes)} bytes) for {contact_id}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected (outer) for contact_id={contact_id}")
    except Exception:
        logger.exception(f"WebSocket error for {contact_id}")
        try:
            await websocket.send_json(
                {
                    "event": "error",
                    "message": "Internal server error during streaming.",
                    "code": "INTERNAL_ERROR",
                }
            )
            await websocket.close(code=1011)
        except Exception:
            logger.debug("Failed to send WebSocket error close", exc_info=True)
