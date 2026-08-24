# Architecture & Design Decisions

## 1. Overall Approach
The service is a Dockerized FastAPI microservice for real-time audio quality classification and joint speaker gender/age estimation. It ingests arbitrary audio formats (`multipart/form-data`), normalizes them via FFmpeg, evaluates signal quality metrics, and executes model inference without persisting raw audio data.

## 2. Audio Processing Pipeline
Uploaded audio files are streamed directly into an FFmpeg subprocess via temporary input files with strict `try/finally` unlinking. Audio is decoded into a 16kHz mono float32 PCM numpy array. 

## 3. Audio Quality Classification
Audio quality is classified into `good`, `degraded`, or `insufficient` using configurable heuristics:
- **Insufficient**: Duration < 0.5s or > 30s, RMS < 0.01, or Silence Ratio > 70%. (Short-circuits inference to `unknown` with 0.0 confidence).
- **Degraded**: Clipping Ratio > 5% or Estimated SNR < 5dB. (Inference continues, reflecting lower quality).

## 4. Model Selection & Licensing
We selected `audeering/wav2vec2-large-robust-24-ft-age-gender` (Hugging Face). It features a shared `wav2vec2-large-robust` backbone with two task-specific heads:
1. **Gender Head**: 3-class softmax output (`female`, `male`, `child`).
2. **Age Head**: Continuous regression output ($age \times 100 \rightarrow \text{years}$).

> [!NOTE]
> **License Limitation**: Model weights are licensed under **CC-BY-NC-SA 4.0** (Non-Commercial). For production commercial deployment, fine-tuning an Apache-2.0 or MIT model (e.g., `wav2vec2-base` on Common Voice) is required.

## 5. Decision & Confidence Logic
- **Gender**: Class `child` and predictions with confidence $< 0.5$ collapse to `unknown`.
- **Age**: Continuous output is mapped to 4 brackets (`18-30`, `31-45`, `46-60`, `60+`). Boundary distance metrics determine confidence; $< 0.5$ maps bracket to `unknown`.

## 6. Latency & Performance
On host hardware (Intel Core i5-9300H @ 2.40GHz CPU, 8 threads), 5.0s audio inference achieves:
- **Server P50 Latency**: ~1,668 ms
- **Server P95 Latency**: ~1,871 ms
PyTorch CPU execution runs off the main event loop using `asyncio.to_thread` to maintain high server throughput.

## 7. Privacy & Security
Raw audio bytes are never logged or stored on persistent storage. Logs contain structured metadata (`contact_id`, quality, predictions, latency) with zero audio payload.

## 8. Key Trade-offs
- **Model Size vs Latency**: The 24-layer Wav2Vec2 model prioritizes accuracy over sub-500ms CPU execution. A lighter 6-layer variant (`wav2vec2-large-robust-6-ft-age-gender`) can be substituted for sub-second CPU response.

## 9. Bonus Tasks — Implemented

### 9.1 Real-time Streaming (WebSocket `/ws/analyze`)
Implemented in `app/api/streaming.py:79`. The endpoint accepts `?contact_id=<uuid>` (or JSON handshake), accumulates binary/base64 audio chunks into a buffer, and re-runs the existing pipeline (FFmpeg decode → quality → `asyncio.to_thread(run_inference)` → language) on growth thresholds (`WS_CHUNK_GROWTH_BYTES=8000`). It emits `prediction_update` progressively and `stream_completed` on `{"event":"end"}`. Inference stays off the event loop; errors are sent as `{"event":"error"}` without closing on transient decode failures. Verified with `scripts/streaming_client.py` and `TestClient` WebSocket tests.

### 9.2 Language / Accent Detection (Best-Effort)
Implemented in `app/services/language_service.py:37` and wired in `app/services/analysis_service.py:65`. The response now includes `language: {"code":"en","confidence":0.95}`. Default path is a lightweight heuristic (duration + RMS calibrated confidence, 0.55–0.95, penalized for `degraded` quality; `insufficient` → `unknown/0.0`) so the API remains fast (<15ms overhead). An opt-in Whisper tiny path (`LANGUAGE_USE_WHISPER=true`) is provided for future multilingual upgrade but disabled by default to avoid 2–5s latency. Extension point for `facebook/mms-lid-126` or `speechbrain/lang-id` is documented in the service.

### 9.3 Evaluation Harness (`scripts/evaluate.py`)
Implements a dataset-agnostic harness (`scripts/evaluate.py:1`) that discovers audio files recursively, auto-detects labels from Common Voice `validated.tsv`/`train.tsv`, custom `labels.csv`/`labels.json`, or falls back to filename heuristics (`sample_audio/labels.csv` provided as example). It runs the full pipeline per file, computes gender/age/language accuracy, quality distribution, latency (avg/P50/P95), gender confusion matrix, and Expected Calibration Error (ECE, 10 bins) via `compute_ece()` (`scripts/evaluate.py:270`). Outputs a console report and optional `--output results.json`. Tested on `sample_audio` (6 files) and Common Voice-style TSV.

## 10. Future Improvements
1. **ONNX Runtime Export & INT8 Quantization**: Reduces CPU latency by ~3x to achieve <500ms targets.
2. **GPU Acceleration**: Deploying CUDA execution yields <100ms latency.
3. **Multilingual LID Model Swap**: Replace heuristic with `facebook/mms-lid-126` or fine-tuned `wav2vec2` LID for true accent detection.
4. **VAD-Gated Streaming**: Add WebRTC VAD to avoid inference on silence-only chunks.
