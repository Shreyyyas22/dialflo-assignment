from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MODEL_NAME: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    MAX_AUDIO_SIZE_MB: float = 10.0
    MIN_AUDIO_DURATION_SECONDS: float = 0.5
    MAX_AUDIO_DURATION_SECONDS: float = 30.0
    GENDER_CONFIDENCE_THRESHOLD: float = 0.5
    AGE_CONFIDENCE_THRESHOLD: float = 0.5
    MIN_RMS: float = 0.01
    MAX_CLIPPING_RATIO: float = 0.05
    MAX_SILENCE_RATIO: float = 0.70
    MIN_ESTIMATED_SNR: float = 5.0
    FFMPEG_TIMEOUT_SECONDS: float = 10.0
    LANGUAGE_DETECTION_ENABLED: bool = True
    LANGUAGE_USE_WHISPER: bool = False
    LANGUAGE_FALLBACK_CODE: str = "en"
    WS_MAX_ACCUMULATED_MB: float = 10.0
    WS_CHUNK_GROWTH_BYTES: int = 8000


settings = Settings()
