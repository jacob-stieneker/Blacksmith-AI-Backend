from __future__ import annotations

import subprocess
from pathlib import Path

from app.core.config import FFMPEG_BINARY, PREVIEW_BITRATE, WORKING_SAMPLE_RATE


class AudioIOError(Exception):
    """Raised when audio import or export fails."""


def decode_to_wav(input_path: str | Path, output_wav_path: str | Path, sample_rate: int | None = None) -> Path:
    input_path = Path(input_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    sr = int(sample_rate or WORKING_SAMPLE_RATE)

    cmd = [
        FFMPEG_BINARY,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sr),
        "-ac",
        "2",
        str(output_wav_path),
    ]
    _run_ffmpeg(cmd, f"Could not decode input file: {input_path}")
    return output_wav_path


def render_preview_mp3(input_wav_path: str | Path, output_mp3_path: str | Path, bitrate: str | None = None) -> Path:
    input_wav_path = Path(input_wav_path)
    output_mp3_path = Path(output_mp3_path)
    output_mp3_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_BINARY,
        "-y",
        "-i",
        str(input_wav_path),
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate or PREVIEW_BITRATE,
        str(output_mp3_path),
    ]
    _run_ffmpeg(cmd, f"Could not create preview MP3 from: {input_wav_path}")
    return output_mp3_path


def _run_ffmpeg(command: list[str], error_prefix: str) -> None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioIOError(
            "ffmpeg was not found. Install ffmpeg or set BSAI_FFMPEG_BINARY to the correct path."
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise AudioIOError(f"{error_prefix}. ffmpeg said: {stderr}")
