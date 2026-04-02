from __future__ import annotations

import os
from pathlib import Path
from typing import List


APP_ROOT = Path(__file__).resolve().parent.parent.parent
MEDIA_DIR = APP_ROOT / "media"
TEMP_DIR = APP_ROOT / "temp"

MEDIA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_BINARY = os.getenv("BSAI_FFMPEG_BINARY", "ffmpeg")
WORKING_SAMPLE_RATE = int(os.getenv("BSAI_WORKING_SAMPLE_RATE", "48000"))
PREVIEW_BITRATE = os.getenv("BSAI_PREVIEW_BITRATE", "192k")
CLEAN_INTERMEDIATE_FILES = os.getenv("BSAI_CLEAN_INTERMEDIATE_FILES", "true").lower() == "true"


def get_allowed_origins() -> List[str]:
    raw = os.getenv("BSAI_ALLOWED_ORIGINS", "*")
    return [item.strip() for item in raw.split(",") if item.strip()]
