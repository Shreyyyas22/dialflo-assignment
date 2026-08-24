# Voice Gender/Age/Quality Analyzer

A production-ready Dockerized FastAPI microservice that ingests short caller audio, evaluates audio signal quality, and estimates speaker gender and age bracket using a pretrained Hugging Face model (`audeering/wav2vec2-large-robust-24-ft-age-gender`).

---

## 🌟 Key Features

- **Audio Signal Quality Classification**: Classifies uploaded audio into `good`, `degraded`, or `insufficient` using metrics (RMS, peak amplitude, silence ratio, clipping ratio, estimated SNR).
- **Joint Gender & Age Estimation**: Predicts gender (`male`, `female`, `unknown`) and age bracket (`18-30`, `31-45`, `46-60`, `60+`, `unknown`).
- **Single Model Multi-Task Architecture**: Uses a single pretrained Wav2Vec2 model with joint regression/classification heads for optimal memory efficiency.
- **Strict Privacy & Zero-Persistence**: Audio payloads are processed strictly in-memory and cleaned up immediately. Raw audio is never saved to disk or written to log files.
- **Dockerized & Self-Contained**: Multi-stage Docker setup with pre-baked model weights for instant container startup without runtime network dependencies.
- **Fully Configurable**: All threshold parameters, max audio sizes, and timeouts are configurable via environment variables (`pydantic-settings`).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    Client([Caller / API Client]) -->|POST /analyze multipart/form-data| API[FastAPI /analyze]
    API --> Middleware[X-Request-ID & Logging Middleware]
    Middleware --> Validation{Validate UUID & File Size}
    Validation -->|Invalid| Error400[Return 400 Bad Request]
    Validation -->|Valid| FFmpeg[FFmpeg Preprocessor 16kHz Mono Float32]
    FFmpeg --> Quality[Quality Service RMS, Silence, Clipping, SNR]
    Quality --> QualityCheck{Quality Insufficient?}
    QualityCheck -->|Yes| ShortCircuit[Return unknown gender/age with 0.0 confidence]
    QualityCheck -->|No| Inference[PyTorch Wav2Vec2 Multi-Head Inference]
    Inference --> PostProcess[Map Gender Softmax & Age Brackets]
    PostProcess --> Response[Return JSON Response + Processing Time]
    ShortCircuit --> Response
```

---

## 📊 Model Information & License

- **Selected Model**: [`audeering/wav2vec2-large-robust-24-ft-age-gender`](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)
- **License**: **CC-BY-NC-SA 4.0** (Non-Commercial). *Note: Suitable for evaluation/research. Commercial deployment would require fine-tuning an Apache-2.0 model.*
- **Gender Mapping**: Model 3-class output (`female`, `male`, `child`). `child` and low-confidence predictions map to `unknown`.
- **Age Mapping**: Continuous regression output ($0\text{--}1 \times 100 \rightarrow \text{years}$) mapped into discrete age brackets (`18-30`, `31-45`, `46-60`, `60+`).

---

## 🚀 Quick Start Guide

### Prerequisites
- Docker & Docker Compose **OR** Python 3.11+ and FFmpeg.

### Option A: Running with Docker Compose (Recommended)

1. **Build and start the service**:
   ```bash
   docker compose up --build
   ```

2. **Verify endpoints**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/ready
   ```

### Option B: Local Python Setup

1. **Create virtual environment & install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

---

## 🧪 Testing & Verification

### Running Unit & Integration Tests
```bash
PYTHONPATH=. pytest tests/
```

### Running Lint Checks
```bash
ruff check .
```

### Automated Smoke Test
Run the automated smoke test against a running service:
```bash
python scripts/smoke_test.py http://localhost:8000
```

### Running Latency Benchmark
Measure latency against a 5-second audio payload (N=10 requests):
```bash
python scripts/benchmark.py http://localhost:8000 10
```

---

## 📡 API Endpoints

### 1. `GET /health`
Liveness check.
- **Response `200 OK`**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. `GET /ready`
Readiness check verifying ML model initialization.
- **Response `200 OK`**:
  ```json
  {
    "status": "ready",
    "model_loaded": true
  }
  ```

### 3. `POST /analyze`
Analyzes uploaded audio file for caller quality, gender, and age.
- **Request**: `multipart/form-data`
  - `contact_id` (string, required): Valid UUID string.
  - `audio` (file, required): Audio file (WAV, MP3, OGG, FLAC, etc.).

- **Sample Request (`curl`)**:
  ```bash
  curl -X POST "http://localhost:8000/analyze" \
    -F "contact_id=123e4567-e89b-12d3-a456-426614174000" \
    -F "audio=@sample_audio/test_caller.wav"
  ```

- **Sample Response `200 OK`**:
  ```json
  {
    "contact_id": "123e4567-e89b-12d3-a456-426614174000",
    "gender": {
      "value": "female",
      "confidence": 0.892
    },
    "age": {
      "bracket": "31-45",
      "confidence": 0.814
    },
    "quality": {
      "classification": "good",
      "metrics": {
        "duration_seconds": 4.5,
        "rms": 0.0452,
        "peak_amplitude": 0.312,
        "silence_ratio": 0.12,
        "clipping_ratio": 0.0,
        "estimated_snr_db": 18.4
      }
    },
    "processing_time_ms": 1642.5
  }
  ```

- **Sample Error Response `400 Bad Request`**:
  ```json
  {
    "error": {
      "code": "INVALID_CONTACT_ID",
      "message": "Invalid contact_id '12345'. Must be a valid UUID string."
    }
  }
  ```

---

## ⚡ Latency & Benchmarks

Empirical performance measured on Intel Core i5-9300H CPU @ 2.40GHz (8 vCPUs) for a **5.0s audio payload**:

| Metric | Server Latency | Total Client RTT |
|---|---|---|
| **Average** | 1709 ms | 1712 ms |
| **P50 (Median)** | 1668 ms | 1672 ms |
| **P95** | 1871 ms | 1875 ms |

---

## ⚙️ Configuration Reference

All settings can be customized via `.env` or environment variables:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Host IP binding |
| `PORT` | `8000` | Port binding |
| `MODEL_NAME` | `audeering/wav2vec2-large-robust-24-ft-age-gender` | Hugging Face model repository |
| `MAX_AUDIO_SIZE_MB` | `10.0` | Max allowed audio file size in MB |
| `MIN_AUDIO_DURATION_SECONDS` | `0.5` | Min audio duration for analysis |
| `MAX_AUDIO_DURATION_SECONDS` | `30.0` | Max audio duration for analysis |
| `GENDER_CONFIDENCE_THRESHOLD` | `0.5` | Threshold below which gender is `unknown` |
| `AGE_CONFIDENCE_THRESHOLD` | `0.5` | Threshold below which age is `unknown` |
| `MIN_RMS` | `0.01` | Minimum RMS threshold for valid audio |
| `MAX_CLIPPING_RATIO` | `0.05` | Max allowed clipping ratio before `degraded` |
| `MAX_SILENCE_RATIO` | `0.70` | Max allowed silence ratio before `insufficient` |
| `MIN_ESTIMATED_SNR` | `5.0` | Minimum SNR in dB before `degraded` |
