import io
import uuid
import wave
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_synthetic_wav(duration: float = 2.0, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Sine wave + slight noise
    audio = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


@pytest.fixture
def valid_wav_bytes():
    return _create_synthetic_wav(duration=2.0)


@patch("app.services.analysis_service.run_inference")
def test_analyze_success(mock_inference, valid_wav_bytes):
    # Mock inference returning age scalar ~0.25 (25 years) and gender probs (female=0.9, male=0.08, child=0.02)
    mock_inference.return_value = (0.25, np.array([0.9, 0.08, 0.02], dtype=np.float32))

    contact_id = str(uuid.uuid4())
    files = {"audio": ("test.wav", valid_wav_bytes, "audio/wav")}
    data = {"contact_id": contact_id}

    response = client.post("/analyze", data=data, files=files)
    assert response.status_code == 200
    res_json = response.json()

    assert res_json["contact_id"] == contact_id
    assert res_json["gender"]["value"] == "female"
    assert res_json["gender"]["confidence"] == 0.9
    assert res_json["age"]["bracket"] == "18-30"
    assert res_json["quality"]["classification"] in ["good", "degraded"]
    assert "processing_time_ms" in res_json


def test_analyze_invalid_contact_id(valid_wav_bytes):
    files = {"audio": ("test.wav", valid_wav_bytes, "audio/wav")}
    data = {"contact_id": "invalid-uuid-12345"}

    response = client.post("/analyze", data=data, files=files)
    assert response.status_code == 400
    res_json = response.json()
    assert res_json["error"]["code"] == "INVALID_CONTACT_ID"


def test_analyze_corrupted_audio():
    contact_id = str(uuid.uuid4())
    files = {"audio": ("test.wav", b"this is not audio data", "audio/wav")}
    data = {"contact_id": contact_id}

    response = client.post("/analyze", data=data, files=files)
    assert response.status_code == 400
    res_json = response.json()
    assert res_json["error"]["code"] == "INVALID_AUDIO"
