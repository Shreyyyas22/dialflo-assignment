import io
import sys
import time
import uuid
import wave

import httpx
import numpy as np


def generate_wav(duration: float = 5.0, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.35 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    audio[16000:24000] = 0  # slight silence pause
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def run_benchmark(base_url: str = "http://localhost:8000", num_requests: int = 10):
    print("==================================================")
    print(f" Starting Latency Benchmark (N={num_requests} requests) ")
    print(f" Endpoint: {base_url}/analyze ")
    print(" Test Payload: 5.0s WAV audio (16kHz mono) ")
    print("==================================================\n")

    audio_bytes = generate_wav(duration=5.0)

    server_latencies = []
    roundtrip_latencies = []

    with httpx.Client(timeout=30.0) as client:
        # Warmup request
        print("Executing warmup request...")
        client.post(
            f"{base_url}/analyze",
            data={"contact_id": str(uuid.uuid4())},
            files={"audio": ("warmup.wav", audio_bytes, "audio/wav")},
        )

        print("Executing benchmark runs...")
        for i in range(num_requests):
            contact_id = str(uuid.uuid4())
            t0 = time.perf_counter()
            resp = client.post(
                f"{base_url}/analyze",
                data={"contact_id": contact_id},
                files={"audio": ("bench.wav", audio_bytes, "audio/wav")},
            )
            rtt = (time.perf_counter() - t0) * 1000.0

            if resp.status_code == 200:
                payload = resp.json()
                server_ms = payload.get("processing_time_ms", 0.0)
                server_latencies.append(server_ms)
                roundtrip_latencies.append(rtt)
                print(
                    f" Run [{i + 1:02d}/{num_requests:02d}]: Server={server_ms:.2f}ms | RTT={rtt:.2f}ms"
                )
            else:
                print(
                    f" Run [{i + 1:02d}/{num_requests:02d}]: FAILED (Status {resp.status_code})"
                )

    if server_latencies:
        p50_server = np.percentile(server_latencies, 50)
        p95_server = np.percentile(server_latencies, 95)
        avg_server = np.mean(server_latencies)

        p50_rtt = np.percentile(roundtrip_latencies, 50)
        p95_rtt = np.percentile(roundtrip_latencies, 95)
        avg_rtt = np.mean(roundtrip_latencies)

        print("\n================ BENCHMARK RESULTS ================")
        print(" Server Processing Latency (ms):")
        print(f"   - Average: {avg_server:.2f} ms")
        print(f"   - P50:     {p50_server:.2f} ms")
        print(f"   - P95:     {p95_server:.2f} ms")
        print(
            f"   - Min/Max: {min(server_latencies):.2f} ms / {max(server_latencies):.2f} ms"
        )
        print(" Total Client Round-Trip Latency (ms):")
        print(f"   - Average: {avg_rtt:.2f} ms")
        print(f"   - P50:     {p50_rtt:.2f} ms")
        print(f"   - P95:     {p95_rtt:.2f} ms")
        print("====================================================")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run_benchmark(url, n)
