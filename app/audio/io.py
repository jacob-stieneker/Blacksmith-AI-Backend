from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

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


def analyze_loudnorm(input_wav_path: str | Path, target_i: float, target_tp: float, target_lra: float) -> dict[str, Any]:
    input_wav_path = Path(input_wav_path)
    cmd = [
        FFMPEG_BINARY,
        "-hide_banner",
        "-nostats",
        "-i",
        str(input_wav_path),
        "-af",
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json",
        "-f",
        "null",
        "-",
    ]
    result = _run_ffmpeg(cmd, f"Could not analyze loudnorm for: {input_wav_path}", return_result=True)
    return _extract_loudnorm_json(result.stderr or "")


def apply_loudnorm_two_pass(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    target_i: float,
    target_tp: float,
    target_lra: float,
) -> dict[str, Any]:
    input_wav_path = Path(input_wav_path)
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    measured = analyze_loudnorm(input_wav_path, target_i, target_tp, target_lra)
    filter_arg = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:"
        f"measured_I={measured['input_i']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:"
        f"linear=true:print_format=summary"
    )

    cmd = [
        FFMPEG_BINARY,
        "-y",
        "-i",
        str(input_wav_path),
        "-af",
        filter_arg,
        "-acodec",
        "pcm_s24le",
        str(output_wav_path),
    ]
    _run_ffmpeg(cmd, f"Could not apply two-pass loudnorm to: {input_wav_path}")
    return measured


def _extract_loudnorm_json(stderr_text: str) -> dict[str, float]:
    match = re.search(r"\{\s*\"input_i\"[\s\S]*?\}\s*$", stderr_text.strip())
    if not match:
        match = re.search(r"\{\s*\"input_i\"[\s\S]*?\}", stderr_text)
    if not match:
        raise AudioIOError("ffmpeg loudnorm analysis did not return JSON statistics.")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise AudioIOError(f"Could not parse loudnorm JSON: {exc}") from exc

    parsed: dict[str, float] = {}
    for key, value in data.items():
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _run_ffmpeg(command: list[str], error_prefix: str, return_result: bool = False):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise AudioIOError(
            "ffmpeg was not found. Install ffmpeg or set BSAI_FFMPEG_BINARY to the correct path."
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise AudioIOError(f"{error_prefix}. ffmpeg said: {stderr}")

    if return_result:
        return result
    return None
