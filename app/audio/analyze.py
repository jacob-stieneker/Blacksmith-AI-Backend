from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly


class AudioAnalysisError(Exception):
    """Raised when an audio file cannot be analyzed."""


def analyze_audio_file(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise AudioAnalysisError(f"Audio file does not exist: {path}")

    try:
        audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:
        raise AudioAnalysisError(f"Could not read audio file: {exc}") from exc

    return analyze_audio_array(audio, sample_rate)


def analyze_audio_array(audio: np.ndarray, sample_rate: int | float) -> dict[str, Any]:
    if audio is None:
        raise AudioAnalysisError("Audio buffer is missing.")

    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.ndim != 2:
        raise AudioAnalysisError("Audio buffer must be mono or multi-channel audio.")
    if audio.size == 0:
        raise AudioAnalysisError("Audio buffer is empty.")

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    sample_rate = int(sample_rate)

    num_samples = int(audio.shape[0])
    num_channels = int(audio.shape[1])
    duration_seconds = float(num_samples / sample_rate) if sample_rate > 0 else 0.0

    meter = pyln.Meter(sample_rate)
    input_lufs = _safe_measure_lufs(meter, audio)
    input_lra = _safe_measure_lra(meter, audio)
    input_true_peak = _measure_approx_true_peak_chunked(audio, sample_rate)
    input_sample_peak = _measure_sample_peak_dbfs(audio)
    input_rms_dbfs = _measure_rms_dbfs(audio)
    crest_factor_db = _compute_crest_factor_db(input_sample_peak, input_rms_dbfs)

    mono = _mono_mix(audio)
    analysis_signal = _analysis_signal(mono, sample_rate)
    spectral_centroid_hz = _estimate_spectral_centroid_hz(analysis_signal, sample_rate)
    low_band_ratio, mid_band_ratio, high_band_ratio = _estimate_band_energy_ratios(analysis_signal, sample_rate)
    stereo_width = _estimate_stereo_width(audio)

    return {
        "sample_rate": sample_rate,
        "channels": num_channels,
        "duration_seconds": round(duration_seconds, 3),
        "input_lufs": _round_or_none(input_lufs),
        "input_true_peak": _round_or_none(input_true_peak),
        "input_lra": _round_or_none(input_lra),
        "input_sample_peak": _round_or_none(input_sample_peak),
        "input_rms_dbfs": _round_or_none(input_rms_dbfs),
        "crest_factor_db": _round_or_none(crest_factor_db),
        "spectral_centroid_hz": _round_or_none(spectral_centroid_hz),
        "low_band_ratio": _round_or_none(low_band_ratio),
        "mid_band_ratio": _round_or_none(mid_band_ratio),
        "high_band_ratio": _round_or_none(high_band_ratio),
        "stereo_width": _round_or_none(stereo_width),
    }


def _safe_measure_lufs(meter: pyln.Meter, audio: np.ndarray) -> float | None:
    try:
        return float(meter.integrated_loudness(audio))
    except Exception:
        return None


def _safe_measure_lra(meter: pyln.Meter, audio: np.ndarray) -> float | None:
    try:
        return float(meter.loudness_range(audio))
    except Exception:
        return None


def _measure_sample_peak_dbfs(audio: np.ndarray) -> float:
    return _linear_to_db(float(np.max(np.abs(audio))))


def _measure_rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    return _linear_to_db(rms)


def _measure_approx_true_peak_chunked(
    audio: np.ndarray,
    sample_rate: int,
    oversample_factor: int = 2,
    max_chunk_seconds: float = 10.0,
) -> float:
    highest_peak = 0.0
    chunk_size = max(2048, int(sample_rate * max_chunk_seconds))

    for channel_index in range(audio.shape[1]):
        channel = audio[:, channel_index]
        for start in range(0, channel.shape[0], chunk_size):
            block = channel[start : start + chunk_size]
            if block.size == 0:
                continue
            try:
                oversampled = resample_poly(block, oversample_factor, 1)
            except Exception:
                oversampled = block
            highest_peak = max(highest_peak, float(np.max(np.abs(oversampled))))

    return _linear_to_db(highest_peak)


def _compute_crest_factor_db(sample_peak_dbfs: float, rms_dbfs: float) -> float | None:
    if np.isneginf(sample_peak_dbfs) or np.isneginf(rms_dbfs):
        return None
    return float(sample_peak_dbfs - rms_dbfs)


def _mono_mix(audio: np.ndarray) -> np.ndarray:
    if audio.shape[1] == 1:
        return audio[:, 0]
    return np.mean(audio, axis=1, dtype=np.float32)


def _analysis_signal(mono_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if mono_audio.size == 0:
        return mono_audio

    excerpt_length = min(mono_audio.size, sample_rate * 60)
    excerpt = mono_audio[:excerpt_length]

    max_points = 131072
    step = max(1, int(np.ceil(excerpt.size / max_points)))
    return excerpt[::step].astype(np.float32, copy=False)


def _estimate_spectral_centroid_hz(mono_audio: np.ndarray, sample_rate: int) -> float | None:
    if mono_audio.size == 0:
        return None
    spectrum = np.abs(np.fft.rfft(mono_audio * np.hanning(mono_audio.size)))
    freqs = np.fft.rfftfreq(mono_audio.size, d=1.0 / sample_rate)
    denom = float(np.sum(spectrum))
    if denom <= 0.0:
        return None
    return float(np.sum(freqs * spectrum) / denom)


def _estimate_band_energy_ratios(mono_audio: np.ndarray, sample_rate: int) -> tuple[float | None, float | None, float | None]:
    if mono_audio.size == 0:
        return None, None, None

    spectrum = np.abs(np.fft.rfft(mono_audio * np.hanning(mono_audio.size))) ** 2
    freqs = np.fft.rfftfreq(mono_audio.size, d=1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return None, None, None

    low = float(np.sum(spectrum[freqs < 200.0])) / total
    mid = float(np.sum(spectrum[(freqs >= 200.0) & (freqs < 4000.0)])) / total
    high = float(np.sum(spectrum[freqs >= 4000.0])) / total
    return low, mid, high


def _estimate_stereo_width(audio: np.ndarray) -> float | None:
    if audio.shape[1] < 2:
        return 0.0

    left = audio[:, 0]
    right = audio[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    mid_energy = float(np.mean(np.square(mid), dtype=np.float64))
    side_energy = float(np.mean(np.square(side), dtype=np.float64))
    denom = mid_energy + side_energy
    if denom <= 0.0:
        return 0.0

    return float(100.0 * side_energy / denom)


def _linear_to_db(value: float, floor_db: float = -120.0) -> float:
    if value <= 0.0:
        return floor_db
    return float(20.0 * np.log10(value))


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
