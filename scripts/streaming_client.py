#!/usr/bin/env python3
"""
WebSocket streaming demo client for /ws/analyze.

Splits a local audio file into chunks and streams them to the server,
printing progressive predictions.

Usage:
    python scripts/streaming_client.py sample_audio/real_speech.wav
    python scripts/streaming_client.py sample_audio/real_speech.wav --url ws://localhost:8000 --chunk-ms 500
"""
import argparse
import asyncio
import sys
import uuid
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Missing dependency: websockets. Install with `pip install websockets`")
    sys.exit(1)


async def stream_file(file_path: Path, url: str, chunk_ms: int = 500):
    # Validate contact_id UUID
    contact_id = str(uuid.uuid4())
    uri = f"{url}/ws/analyze?contact_id={contact_id}"
    print(f"Connecting to {uri}")
    print(f"Streaming file: {file_path} (chunk {chunk_ms}ms)")

    # Read file bytes
    audio_bytes = file_path.read_bytes()
    # Simple chunking by raw bytes: split into N equal parts
    # For real PCM streaming, chunk size ~ sample_rate * bytes_per_sample * chunk_ms/1000
    # Here we just split payload into ~4 chunks for demo
    num_chunks = max(2, min(10, len(audio_bytes) // 8000 + 1))
    chunk_size = len(audio_bytes) // num_chunks
    chunks = [audio_bytes[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks - 1)]
    chunks.append(audio_bytes[(num_chunks - 1) * chunk_size :])

    async with websockets.connect(uri, max_size=20 * 1024 * 1024) as ws:
        hello = await ws.recv()
        print(f"Server hello: {hello}")

        for idx, chunk in enumerate(chunks):
            print(f" -> Sending chunk {idx+1}/{len(chunks)} ({len(chunk)} bytes)")
            await ws.send(chunk)
            # Await prediction_update with timeout (server throttles ~0.5s growth)
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print(f" <- Prediction: {resp}")
            except asyncio.TimeoutError:
                print(" <- (no prediction yet, buffering)")
            await asyncio.sleep(chunk_ms / 1000.0)

        print(" -> Sending end signal")
        await ws.send('{"event":"end"}')
        final = await asyncio.wait_for(ws.recv(), timeout=10.0)
        print(f" <- Final: {final}")
        print("Stream completed.")


def main():
    parser = argparse.ArgumentParser(description="WebSocket streaming demo client")
    parser.add_argument("audio_file", type=str, help="Path to audio file to stream")
    parser.add_argument("--url", type=str, default="ws://localhost:8000", help="WebSocket base URL (without /ws/analyze)")
    parser.add_argument("--chunk-ms", type=int, default=500, help="Delay between chunks in ms")
    args = parser.parse_args()

    fp = Path(args.audio_file)
    if not fp.exists():
        print(f"File not found: {fp}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(stream_file(fp, args.url.rstrip("/"), args.chunk_ms))


if __name__ == "__main__":
    main()
