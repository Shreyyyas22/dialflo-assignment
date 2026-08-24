import io
import sys
import time
import uuid
import wave

import httpx
import numpy as np


def generate_sample_wav(duration: float = 3.0, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate clean speech-like signal (440Hz tone with volume modulation and silence pause)
    audio = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    audio[16000:20000] = 0  # small silence pause
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"--- Starting Smoke Test against {base_url} ---")

    with httpx.Client(timeout=30.0) as client:
        # 1. Health check
        print("\n1. Testing GET /health ...")
        resp = client.get(f"{base_url}/health")
        print(f"Status Code: {resp.status_code}, Payload: {resp.json()}")
        assert resp.status_code == 200, "Health check failed!"

        # 2. Readiness check
        print("\n2. Testing GET /ready ...")
        resp = client.get(f"{base_url}/ready")
        print(f"Status Code: {resp.status_code}, Payload: {resp.json()}")
        assert resp.status_code in [200, 503], "Readiness check unexpected response!"

        # 3. Analyze Endpoint
        print("\n3. Testing POST /analyze ...")
        audio_bytes = generate_sample_wav(duration=3.0)
        contact_id = str(uuid.uuid4())

        start_time = time.perf_counter()
        resp = client.post(
            f"{base_url}/analyze",
            data={"contact_id": contact_id},
            files={"audio": ("sample.wav", audio_bytes, "audio/wav")},
        )
        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        print(f"Status Code: {resp.status_code}")
        print(f"Total Client Roundtrip Time: {total_time_ms:.2f}ms")
        print(f"Response Payload:\n{resp.text}")

        assert resp.status_code == 200, (
            f"Analyze request failed with status {resp.status_code}"
        )
        payload = resp.json()
        assert payload["contact_id"] == contact_id
        assert "gender" in payload
        assert "age" in payload
        assert "quality" in payload

    print("\n--- Smoke Test PASSED Successfully! ---")


if __name__ == "__main__":
    main()
