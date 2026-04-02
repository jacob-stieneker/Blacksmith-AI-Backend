from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from app.audio.analyze import analyze_audio_array
from app.audio.types import MasteringSettings, clamp

try:
    from pedalboard import (
        Compressor,
        Gain,
        HighShelfFilter,
        HighpassFilter,
        Limiter,
        LowShelfFilter,
        PeakFilter,
        Pedalboard,
    )
except ImportError as exc:
    raise RuntimeError(
        "pedalboard is required for processing. Install it with `pip install pedalboard`."
    ) from exc


@dataclass(frozen=True)
class MasteringRecipe:
    highpass_hz: float
    low_shelf_db: float
    mid_peak_db: float
    presence_peak_db: float
    high_shelf_db: float
    compressor_threshold_db: float
    compressor_ratio: float
    compressor_attack_ms: float
    compressor_release_ms: float
    pre_limiter_gain_db: float
    limiter_threshold_db: float
    limiter_release_ms: float
    target_lufs: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""


def process_audio_file(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    settings: MasteringSettings,
    input_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_wav_path = Path(input_wav_path)
    output_wav_path = Path(output_wav_path)

    try:
        audio, sample_rate = sf.read(str(input_wav_path), always_2d=True, dtype="float32")
    except Exception as exc:
        raise AudioProcessingError(f"Could not read WAV for processing: {exc}") from exc

    if audio.size == 0:
        raise AudioProcessingError("Decoded audio is empty.")

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    settings = settings.normalized()
    input_analysis = input_analysis or analyze_audio_array(audio, sample_rate)
    recipe = build_mastering_recipe(input_analysis, settings)

    board = Pedalboard(
        [
            HighpassFilter(recipe.highpass_hz),
            LowShelfFilter(120.0, recipe.low_shelf_db, 0.707),
            PeakFilter(900.0, recipe.mid_peak_db, 0.9),
            PeakFilter(3500.0, recipe.presence_peak_db, 1.0),
            HighShelfFilter(9500.0, recipe.high_shelf_db, 0.707),
            Compressor(
                recipe.compressor_threshold_db,
                recipe.compressor_ratio,
                recipe.compressor_attack_ms,
                recipe.compressor_release_ms,
            ),
            Gain(recipe.pre_limiter_gain_db),
        ]
    )

    try:
        processed = board(audio, sample_rate)
    except Exception as exc:
        raise AudioProcessingError(f"Pedalboard processing failed: {exc}") from exc

    processed = _normalize_to_target_lufs(processed, sample_rate, settings.target_lufs)
    processed = Limiter(recipe.limiter_threshold_db, recipe.limiter_release_ms)(processed, sample_rate)

    mastered_analysis = analyze_audio_array(processed, sample_rate)

    measured_lufs = mastered_analysis.get("input_lufs")
    if measured_lufs is not None:
        correction_db = clamp(settings.target_lufs - float(measured_lufs), -2.0, 2.0)
        if abs(correction_db) > 0.2:
            processed = _apply_gain(processed, correction_db)
            processed = Limiter(recipe.limiter_threshold_db, recipe.limiter_release_ms)(processed, sample_rate)
            mastered_analysis = analyze_audio_array(processed, sample_rate)

    processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    try:
        sf.write(str(output_wav_path), processed, sample_rate, subtype="PCM_24", format="WAV")
    except Exception as exc:
        raise AudioProcessingError(f"Could not write mastered WAV: {exc}") from exc

    return {
        "recipe": recipe.to_dict(),
        "output_analysis": mastered_analysis,
        "sample_rate": int(sample_rate),
        "channels": int(processed.shape[1] if processed.ndim == 2 else 1),
    }


def build_mastering_recipe(input_analysis: dict[str, Any], settings: MasteringSettings) -> MasteringRecipe:
    low_ratio = float(input_analysis.get("low_band_ratio") or 0.22)
    mid_ratio = float(input_analysis.get("mid_band_ratio") or 0.63)
    high_ratio = float(input_analysis.get("high_band_ratio") or 0.15)
    spectral_centroid_hz = float(input_analysis.get("spectral_centroid_hz") or 2500.0)
    crest_factor_db = float(input_analysis.get("crest_factor_db") or 10.0)
    input_lufs = input_analysis.get("input_lufs")
    input_lufs = float(input_lufs) if input_lufs is not None else -18.0

    warmth_tilt = _map_percent(settings.warmth, -1.8, 1.8)
    brightness_tilt = _map_percent(settings.brightness, -1.8, 1.8)
    punch_shape = _map_percent(settings.punch, -1.0, 1.0)

    auto_low = clamp((0.24 - low_ratio) * 18.0, -1.5, 1.5)
    auto_mid = clamp((0.60 - mid_ratio) * 6.0, -1.0, 1.0)
    auto_high = clamp((0.16 - high_ratio) * 20.0, -1.5, 1.5)

    centroid_brightness = clamp((2800.0 - spectral_centroid_hz) / 1600.0, -0.8, 0.8)

    low_shelf_db = clamp(settings.low_eq + warmth_tilt + auto_low, -6.0, 6.0)
    mid_peak_db = clamp(settings.mid_eq + auto_mid + (warmth_tilt * 0.35) - (brightness_tilt * 0.20), -6.0, 6.0)
    presence_peak_db = clamp((brightness_tilt * 0.55) + centroid_brightness, -3.0, 3.0)
    high_shelf_db = clamp(settings.high_eq + brightness_tilt + auto_high + centroid_brightness, -6.0, 6.0)

    compressor_ratio = clamp(settings.compression + (punch_shape * 0.45), 1.0, 4.0)
    compressor_threshold_db = clamp(-22.0 + ((compressor_ratio - 1.0) * 3.0), -26.0, -12.0)
    compressor_attack_ms = clamp(18.0 + (punch_shape * 10.0), 5.0, 35.0)
    compressor_release_ms = clamp(180.0 - (punch_shape * 60.0), 80.0, 280.0)

    loudness_gap = settings.target_lufs - input_lufs
    density_comp = clamp((10.0 - crest_factor_db) * 0.2, -1.0, 1.0)
    pre_limiter_gain_db = clamp(loudness_gap + density_comp, -6.0, 8.0)

    limiter_threshold_db = -1.0
    limiter_release_ms = 250.0

    return MasteringRecipe(
        highpass_hz=25.0,
        low_shelf_db=round(low_shelf_db, 3),
        mid_peak_db=round(mid_peak_db, 3),
        presence_peak_db=round(presence_peak_db, 3),
        high_shelf_db=round(high_shelf_db, 3),
        compressor_threshold_db=round(compressor_threshold_db, 3),
        compressor_ratio=round(compressor_ratio, 3),
        compressor_attack_ms=round(compressor_attack_ms, 3),
        compressor_release_ms=round(compressor_release_ms, 3),
        pre_limiter_gain_db=round(pre_limiter_gain_db, 3),
        limiter_threshold_db=round(limiter_threshold_db, 3),
        limiter_release_ms=round(limiter_release_ms, 3),
        target_lufs=round(settings.target_lufs, 3),
    )


def _normalize_to_target_lufs(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    try:
        measured = float(meter.integrated_loudness(audio))
    except Exception:
        return audio

    gain_db = clamp(target_lufs - measured, -12.0, 12.0)
    return _apply_gain(audio, gain_db)


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    multiplier = 10.0 ** (gain_db / 20.0)
    return (audio * np.float32(multiplier)).astype(np.float32, copy=False)


def _map_percent(value: int | float, low: float, high: float) -> float:
    normalized = clamp(float(value), 0.0, 100.0) / 100.0
    return low + ((high - low) * normalized)
