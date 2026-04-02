from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.audio.analyze import analyze_audio_array, analyze_audio_file
from app.audio.io import apply_loudnorm_two_pass
from app.audio.types import MasteringSettings, clamp

try:
    from pedalboard import (
        Compressor,
        Distortion,
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
    saturation_drive_db: float
    saturation_mix: float
    target_lufs: float
    target_true_peak_dbtp: float
    target_lra: float

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
            LowShelfFilter(120.0, recipe.low_shelf_db, 0.707),
            PeakFilter(1000.0, recipe.mid_peak_db, 0.85),
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

    if recipe.saturation_mix > 0.0 and recipe.saturation_drive_db > 0.0:
        try:
            saturated = Distortion(drive_db=recipe.saturation_drive_db)(processed, sample_rate)
            processed = ((1.0 - recipe.saturation_mix) * processed) + (recipe.saturation_mix * saturated)
        except Exception as exc:
            raise AudioProcessingError(f"Saturation stage failed: {exc}") from exc

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

    auto_low = clamp((0.23 - low_ratio) * 6.0, -0.75, 0.75)
    auto_mid = clamp((0.60 - mid_ratio) * 3.0, -0.5, 0.5)
    auto_high = clamp((0.17 - high_ratio) * 6.0, -0.75, 0.75)

    low_shelf_db = clamp(settings.low_eq + auto_low, -4.0, 4.0)
    mid_peak_db = clamp(settings.mid_eq + auto_mid, -3.0, 3.0)
    high_shelf_db = clamp(settings.high_eq + auto_high, -4.0, 4.0)

    compressor_ratio = clamp(settings.compression, 1.0, 2.5)
    compressor_attack_ms = clamp(35.0 - ((compressor_ratio - 1.0) * 12.0), 12.0, 35.0)
    compressor_release_ms = clamp(220.0 - ((compressor_ratio - 1.0) * 40.0), 100.0, 220.0)

    dynamic_hint = clamp((crest_factor_db - 10.0) * 0.5, -2.0, 2.0)
    compressor_threshold_db = clamp(-18.0 - ((compressor_ratio - 1.0) * 4.0) + dynamic_hint, -24.0, -12.0)

    saturation_drive_db = round((settings.saturation / 100.0) * 8.0, 3)
    saturation_mix = round((settings.saturation / 100.0) * 0.16, 4)

    target_true_peak_dbtp = -2.0 if settings.target_lufs > -14.0 else -1.0
    target_lra = round(clamp(input_lra, 6.0, 12.0), 3)

    return MasteringRecipe(
        highpass_hz=25.0,
        low_shelf_db=round(low_shelf_db, 3),
        mid_peak_db=round(mid_peak_db, 3),
        high_shelf_db=round(high_shelf_db, 3),
        compressor_threshold_db=round(compressor_threshold_db, 3),
        compressor_ratio=round(compressor_ratio, 3),
        compressor_attack_ms=round(compressor_attack_ms, 3),
        compressor_release_ms=round(compressor_release_ms, 3),
        saturation_drive_db=round(saturation_drive_db, 3),
        saturation_mix=round(saturation_mix, 4),
        target_lufs=round(settings.target_lufs, 3),
        target_true_peak_dbtp=round(target_true_peak_dbtp, 3),
        target_lra=target_lra,
    )


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    multiplier = 10.0 ** (gain_db / 20.0)
    return (audio * np.float32(multiplier)).astype(np.float32, copy=False)
