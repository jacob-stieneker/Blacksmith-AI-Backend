from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.audio.analyze import analyze_audio_file
from app.audio.io import decode_to_wav, render_preview_mp3
from app.audio.process import process_audio_file
from app.audio.types import MasteringSettings
from app.core.config import CLEAN_INTERMEDIATE_FILES


class MasteringEngineError(Exception):
    """Raised when the mastering pipeline fails."""


ProgressCallback = Callable[[str, int, str], None]


def run_mastering_job(
    uploaded_input_path: str | Path,
    job_dir: str | Path,
    settings: MasteringSettings,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    uploaded_input_path = Path(uploaded_input_path)
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    decoded_input_path = job_dir / "decoded_input.wav"
    mastered_wav_path = job_dir / "mastered.wav"
    preview_mp3_path = job_dir / "preview.mp3"

    settings = settings.normalized()
    _emit(progress_callback, "analyze", 25, "Decoding uploaded mix")

    try:
        decode_to_wav(uploaded_input_path, decoded_input_path)

        _emit(progress_callback, "analyze", 40, "Analyzing loudness and tonal balance")
        input_analysis = analyze_audio_file(decoded_input_path)

        _emit(progress_callback, "process", 55, "Building mastering recipe")
        processing_result = process_audio_file(
            decoded_input_path,
            mastered_wav_path,
            settings=settings,
            input_analysis=input_analysis,
        )

        _emit(progress_callback, "process", 82, "Final loudness verification complete")

        _emit(progress_callback, "render", 90, "Rendering mastered preview")
        render_preview_mp3(mastered_wav_path, preview_mp3_path)
    except Exception as exc:
        raise MasteringEngineError(str(exc)) from exc
    finally:
        if CLEAN_INTERMEDIATE_FILES and decoded_input_path.exists():
            decoded_input_path.unlink(missing_ok=True)

        pre_loudnorm_path = job_dir / "pre_loudnorm.wav"
        if CLEAN_INTERMEDIATE_FILES and pre_loudnorm_path.exists():
            pre_loudnorm_path.unlink(missing_ok=True)

    output_analysis = processing_result.get("output_analysis", {})
    _emit(progress_callback, "ready", 100, "Master complete")

    return {
        "preview_filename": preview_mp3_path.name,
        "download_filename": mastered_wav_path.name,
        "input_analysis": input_analysis,
        "output_analysis": output_analysis,
        "recipe": processing_result.get("recipe", {}),
        "settings": settings.to_dict(),
        "loudnorm": processing_result.get("loudnorm", {}),
    }


def _emit(progress_callback: ProgressCallback | None, stage: str, percent: int, message: str) -> None:
    if progress_callback is not None:
        progress_callback(stage, percent, message)
