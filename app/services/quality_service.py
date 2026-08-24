import numpy as np

from app.core.config import settings
from app.schemas.analysis import QualityClassification, QualityMetrics, QualityResult


def analyze_audio_quality(
    signal: np.ndarray, sample_rate: int = 16000
) -> QualityResult:
    """
    Analyzes audio signal metrics (RMS, peak, silence ratio, clipping ratio, SNR)
    and classifies quality as 'good', 'degraded', or 'insufficient'.
    """
    num_samples = len(signal)
    if num_samples == 0:
        return QualityResult(
            classification="insufficient",
            metrics=QualityMetrics(
                duration_seconds=0.0,
                rms=0.0,
                peak_amplitude=0.0,
                silence_ratio=1.0,
                clipping_ratio=0.0,
                estimated_snr_db=0.0,
            ),
        )

    duration_seconds = float(num_samples / sample_rate)
    rms = float(np.sqrt(np.mean(signal**2)))
    peak_amplitude = float(np.max(np.abs(signal)))

    # Silence ratio: samples with amplitude below 0.01
    silence_samples = np.abs(signal) < 0.01
    silence_ratio = float(np.mean(silence_samples))

    # Clipping ratio: samples with amplitude >= 0.99
    clipping_samples = np.abs(signal) >= 0.99
    clipping_ratio = float(np.mean(clipping_samples))

    # Estimated SNR calculation
    # Frame size 20ms = 320 samples at 16kHz
    frame_size = int(0.02 * sample_rate)
    if num_samples >= frame_size:
        num_frames = num_samples // frame_size
        frames = signal[: num_frames * frame_size].reshape(num_frames, frame_size)
        frame_powers = np.mean(frames**2, axis=1)

        # Noise power from 10th percentile, signal power from 90th percentile
        noise_power = float(np.percentile(frame_powers, 10))
        signal_power = float(np.percentile(frame_powers, 90))

        if noise_power <= 1e-10 or abs(signal_power - noise_power) < 1e-7:
            estimated_snr_db = 30.0
        else:
            snr_val = signal_power / (noise_power + 1e-10)
            estimated_snr_db = float(10 * np.log10(max(snr_val, 1.0)))
    else:
        estimated_snr_db = 0.0

    # Quality classification logic
    if (
        duration_seconds < settings.MIN_AUDIO_DURATION_SECONDS
        or duration_seconds > settings.MAX_AUDIO_DURATION_SECONDS
        or rms < settings.MIN_RMS
        or silence_ratio > settings.MAX_SILENCE_RATIO
    ):
        classification: QualityClassification = "insufficient"
    elif (
        clipping_ratio > settings.MAX_CLIPPING_RATIO
        or estimated_snr_db < settings.MIN_ESTIMATED_SNR
    ):
        classification: QualityClassification = "degraded"
    else:
        classification: QualityClassification = "good"

    metrics = QualityMetrics(
        duration_seconds=round(duration_seconds, 3),
        rms=round(rms, 5),
        peak_amplitude=round(peak_amplitude, 5),
        silence_ratio=round(silence_ratio, 4),
        clipping_ratio=round(clipping_ratio, 4),
        estimated_snr_db=round(estimated_snr_db, 2),
    )

    return QualityResult(classification=classification, metrics=metrics)
