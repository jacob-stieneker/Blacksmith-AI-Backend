from __future__ import annotations

from pathlib import Path
from typing import Any

from app.audio.analyze import analyze_audio_file
from app.audio.io import decode_to_wav, render_preview_mp3
from app.audio.process import process_audio_file
from app.audio.types import MasteringSettings
from app.core.config import CLEAN_INTERMEDIATE_FILES


class MasteringEngineError(Exception):
    """Raised when the mastering pipeline fails."""


def run_mastering_job(
    uploaded_input_path: str | Path,
    job_dir: str | Path,
    settings: MasteringSettings,
) -> dict[str, Any]:
    uploaded_input_path = Path(uploaded_input_path)
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    decoded_input_path = job_dir / "decoded_input.wav"
    mastered_wav_path = job_dir / "mastered.wav"
    preview_mp3_path = job_dir / "preview.mp3"

    settings = settings.normalized()

    try:
        decode_to_wav(uploaded_input_path, decoded_input_path)
        input_analysis = analyze_audio_file(decoded_input_path)
        processing_result = process_audio_file(
            decoded_input_path,
            mastered_wav_path,
            settings=settings,
            input_analysis=input_analysis,
        )
        render_preview_mp3(mastered_wav_path, preview_mp3_path)
    except Exception as exc:
        raise MasteringEngineError(str(exc)) from exc

    if CLEAN_INTERMEDIATE_FILES and decoded_input_path.exists():
        decoded_input_path.unlink(missing_ok=True)

    output_analysis = processing_result.get("output_analysis", {})

    return {
        "preview_filename": preview_mp3_path.name,
        "download_filename": mastered_wav_path.name,
        "input_analysis": input_analysis,
        "output_analysis": output_analysis,
        "recipe": processing_result.get("recipe", {}),
        "settings": settings.to_dict(),
    }
