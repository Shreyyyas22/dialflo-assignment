import logging
import os
import shutil
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np

from app.core.config import settings
from app.core.exceptions import InvalidAudioError

logger = logging.getLogger(__name__)


def _get_ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (RuntimeError, ValueError) as e:
        logger.error(f"FFmpeg binary search failed: {e}")
        raise InvalidAudioError("FFmpeg binary unavailable on server.") from e


def decode_and_preprocess_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decodes arbitrary audio bytes using FFmpeg subprocess to 16kHz mono float32 PCM numpy array.
    Ensures temporary file cleanup in a try/finally block.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise InvalidAudioError("Empty audio content provided.")

    ffmpeg_bin = _get_ffmpeg_binary()

    temp_in_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_in:
            temp_in.write(audio_bytes)
            temp_in_path = temp_in.name

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            temp_in_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            stdout_data, stderr_data = process.communicate(
                timeout=settings.FFMPEG_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as e:
            process.kill()
            process.communicate()
            raise InvalidAudioError("Audio processing timed out.") from e

        if process.returncode != 0:
            err_msg = stderr_data.decode("utf-8", errors="ignore").strip()
            logger.warning(
                f"FFmpeg decoding error (code {process.returncode}): {err_msg}"
            )
            raise InvalidAudioError("Corrupted or unsupported audio format.")

        if not stdout_data:
            raise InvalidAudioError("Decoded audio payload is empty.")

        audio_array = np.frombuffer(stdout_data, dtype=np.float32)
        if audio_array.size == 0:
            raise InvalidAudioError("Decoded audio array is empty.")

        # Clip values to standard float range [-1.0, 1.0] if needed
        audio_array = np.clip(audio_array, -1.0, 1.0)
        return audio_array, 16000

    finally:
        if temp_in_path and os.path.exists(temp_in_path):
            try:
                os.remove(temp_in_path)
            except OSError as e:
                logger.warning(f"Failed to remove temp file {temp_in_path}: {e}")
