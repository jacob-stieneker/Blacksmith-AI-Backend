from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from app.audio.analyze import analyze_audio_array, analyze_audio_file
from app.audio.io import apply_loudnorm_two_pass
from app.audio.types import MasteringSettings, clamp

try:
    from pedalboard import (
        Compressor,
        HighShelfFilter,
        HighpassFilter,
        Limiter,
        LowShelfFilter,
        PeakFilter,
        Pedalboard,
    )
except ImportError as exc:
    raise RuntimeError("pedalboard is required for processing. Install it with `pip install pedalboard`.") from exc


@dataclass(frozen=True)
class MasteringRecipe:
    highpass_hz: float
    low_shelf_db: float
    mid_peak_db: float
    high_shelf_db: float
    compressor_threshold_db: float
    compressor_ratio: float
    compressor_attack_ms: float
    compressor_release_ms: float
    stereo_width_factor: float
    stereo_width_low_cut_hz: float
    target_lufs: float
    target_true_peak_dbtp: float
    target_lra: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioProcessingError(Exception):
    """Raised when audio processing fails."""


COMPRESSION_PRESETS = {
    "low": {"ratio": 1.20, "threshold": -16.0, "attack_ms": 36.0, "release_ms": 230.0},
    "normal": {"ratio": 1.40, "threshold": -18.0, "attack_ms": 28.0, "release_ms": 185.0},
    "high": {"ratio": 1.75, "threshold": -20.0, "attack_ms": 20.0, "release_ms": 150.0},
    "extreme": {"ratio": 2.15, "threshold": -22.0, "attack_ms": 12.0, "release_ms": 120.0},
}

LOUDNESS_PRESETS = {
    "quiet": {"target_lufs": -15.5, "true_peak": -1.0},
    "normal": {"target_lufs": -14.0, "true_peak": -1.0},
    "loud": {"target_lufs": -11.5, "true_peak": -2.0},
    "louder": {"target_lufs": -10.0, "true_peak": -2.0},
}

WIDTH_PRESETS = {
    "narrow": 0.90,
    "normal": 1.00,
    "wide": 1.10,
    "wider": 1.20,
}


def process_audio_file(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    settings: MasteringSettings,
    input_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_wav_path = Path(input_wav_path)
    output_wav_path = Path(output_wav_path)
    pre_loudnorm_path = output_wav_path.with_name("pre_loudnorm.wav")

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
            LowShelfFilter(110.0, recipe.low_shelf_db, 0.707),
            PeakFilter(1050.0, recipe.mid_peak_db, 0.80),
            HighShelfFilter(9000.0, recipe.high_shelf_db, 0.707),
            Compressor(
                recipe.compressor_threshold_db,
                recipe.compressor_ratio,
                recipe.compressor_attack_ms,
                recipe.compressor_release_ms,
            ),
        ]
    )

    try:
        processed = board(audio, sample_rate)
    except Exception as exc:
        raise AudioProcessingError(f"Pedalboard processing failed: {exc}") from exc

    processed = _apply_stereo_width(processed, sample_rate, recipe.stereo_width_factor, recipe.stereo_width_low_cut_hz)

    peak_stats = analyze_audio_array(processed, sample_rate)
    sample_peak = peak_stats.get("input_sample_peak")
    if sample_peak is not None and sample_peak > -1.0:
        processed = _apply_gain(processed, -1.0 - float(sample_peak))

    try:
        processed = Limiter(recipe.target_true_peak_dbtp, 250.0)(processed, sample_rate)
    except Exception as exc:
        raise AudioProcessingError(f"Limiter stage failed: {exc}") from exc

    processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    try:
        sf.write(str(pre_loudnorm_path), processed, sample_rate, subtype="PCM_24", format="WAV")
    except Exception as exc:
        raise AudioProcessingError(f"Could not write pre-loudnorm WAV: {exc}") from exc

    try:
        loudnorm_stats = apply_loudnorm_two_pass(
            pre_loudnorm_path,
            output_wav_path,
            target_i=recipe.target_lufs,
            target_tp=recipe.target_true_peak_dbtp,
            target_lra=recipe.target_lra,
        )
    except Exception as exc:
        raise AudioProcessingError(f"Loudness normalization failed: {exc}") from exc

    try:
        mastered_analysis = analyze_audio_file(output_wav_path)
    except Exception as exc:
        raise AudioProcessingError(f"Could not analyze mastered WAV: {exc}") from exc

    return {
        "recipe": recipe.to_dict(),
        "output_analysis": mastered_analysis,
        "loudnorm": loudnorm_stats,
        "pre_loudnorm_path": str(pre_loudnorm_path),
    }


def build_mastering_recipe(input_analysis: dict[str, Any], settings: MasteringSettings) -> MasteringRecipe:
    low_ratio = float(input_analysis.get("low_band_ratio") or 0.22)
    mid_ratio = float(input_analysis.get("mid_band_ratio") or 0.61)
    high_ratio = float(input_analysis.get("high_band_ratio") or 0.17)
    crest_factor_db = float(input_analysis.get("crest_factor_db") or 10.0)
    input_lra = float(input_analysis.get("input_lra") or 9.0)
    stereo_width = float(input_analysis.get("stereo_width") or 18.0)

    auto_low = clamp((0.23 - low_ratio) * 5.0, -0.8, 0.8)
    auto_mid = clamp((0.60 - mid_ratio) * 3.0, -0.6, 0.6)
    auto_high = clamp((0.17 - high_ratio) * 5.0, -0.8, 0.8)

    low_shelf_db = clamp(settings.low_eq + auto_low, -3.5, 3.5)
    mid_peak_db = clamp(settings.mid_eq + auto_mid, -2.5, 2.5)
    high_shelf_db = clamp(settings.high_eq + auto_high, -3.5, 3.5)

    comp = COMPRESSION_PRESETS[settings.compression_mode].copy()
    if crest_factor_db > 12.5:
        comp["threshold"] -= 1.0
    elif crest_factor_db < 8.0:
        comp["threshold"] += 1.0
        comp["ratio"] = clamp(comp["ratio"] - 0.1, 1.0, 2.4)

    loud = LOUDNESS_PRESETS[settings.loudness_mode]
    width_factor = WIDTH_PRESETS[settings.stereo_width_mode]
    if settings.stereo_width_mode == "normal":
        if stereo_width < 10.0:
            width_factor = 1.06
        elif stereo_width > 32.0:
            width_factor = 0.98

    return MasteringRecipe(
        highpass_hz=24.0,
        low_shelf_db=round(low_shelf_db, 3),
        mid_peak_db=round(mid_peak_db, 3),
        high_shelf_db=round(high_shelf_db, 3),
        compressor_threshold_db=round(comp["threshold"], 3),
        compressor_ratio=round(comp["ratio"], 3),
        compressor_attack_ms=round(comp["attack_ms"], 3),
        compressor_release_ms=round(comp["release_ms"], 3),
        stereo_width_factor=round(width_factor, 3),
        stereo_width_low_cut_hz=180.0,
        target_lufs=round(loud["target_lufs"], 3),
        target_true_peak_dbtp=round(loud["true_peak"], 3),
        target_lra=round(clamp(input_lra, 6.0, 12.0), 3),
    )


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    multiplier = 10.0 ** (gain_db / 20.0)
    return (audio * np.float32(multiplier)).astype(np.float32, copy=False)


def _apply_stereo_width(audio: np.ndarray, sample_rate: int, width_factor: float, low_cut_hz: float) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[1] < 2 or abs(width_factor - 1.0) < 1e-4:
        return audio

    left = audio[:, 0].astype(np.float64, copy=False)
    right = audio[:, 1].astype(np.float64, copy=False)

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    if low_cut_hz > 0.0 and sample_rate > 0:
        norm = low_cut_hz / (sample_rate * 0.5)
        norm = min(max(norm, 1e-5), 0.99)
        sos = butter(2, norm, btype="highpass", output="sos")
        side_high = sosfiltfilt(sos, side)
        side_low = side - side_high
        widened_side = side_low + (side_high * width_factor)
    else:
        widened_side = side * width_factor

    widened_left = mid + widened_side
    widened_right = mid - widened_side
    widened = np.stack([widened_left, widened_right], axis=1)
    widened = np.clip(widened, -1.0, 1.0)
    return widened.astype(np.float32, copy=False)
